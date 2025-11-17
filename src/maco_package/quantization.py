# =====================================================================
# Imports
# =====================================================================

import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.ao.quantization as tq
from torch.ao.quantization import (
    quantize_fx,
    get_default_qat_qconfig,
    prepare_qat,
    convert,
)
from tqdm import tqdm
# =====================================================================
# Constants
# =====================================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def QuantizePTQ(model, calibration_input=None):
    """
    Post-Training Quantization (FX PTQ).

    Args:
        model: float32 PyTorch model
        calibration_input: single input tensor for tracing/calibration
    Returns:
        quantized_model: int8 FX-graph-mode quantized model (eval mode)
    """
    torch.backends.quantized.engine = "fbgemm"
    qconfig = tq.get_default_qconfig("fbgemm")

    if calibration_input is None:
        calibration_input = torch.randn(1, 3, 224, 224)

    model = model.eval()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        prepared = quantize_fx.prepare_fx(
            model,
            {"": qconfig},
            calibration_input,
        )

        # Calibration run
        prepared(calibration_input)

        # Convert to int8
        quantized = quantize_fx.convert_fx(prepared).eval()

    return quantized



def trainQAT(
    model_fp32,
    train_dataset,
    val_dataset,
    epochs=5,
    batch_size=64,
    lr=1e-4,
    momentum=0.9,
    backend="fbgemm"
):
    """
    Quantization-Aware Training for an arbitrary model.
    Always uses CUDA if available.

    Args:
        model_fp32: FP32 model that supports torch.ao.quantization (should have QuantStub/DequantStub and fuse_model)
        train_dataset: training dataset
        val_dataset: validation dataset
        epochs: number of QAT epochs
        batch_size: batch size
        lr: learning rate
        momentum: SGD momentum
        backend: quantization backend ("fbgemm" for x86)
        
    Returns:
        model_qat  - QAT-trained (fake-quant) model on CPU
        model_int8 - final converted INT8 model (CPU only)
    """

    # ----------------------------
    # Device & quantization engine
    # ----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.quantized.engine = backend

    # ----------------------------
    # DataLoaders
    # ----------------------------
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # ----------------------------
    # Fuse layers if the model supports it
    # ----------------------------
    if hasattr(model_fp32, "fuse_model"):
        model_fp32.eval()
        model_fp32.fuse_model()

    # ----------------------------
    # Prepare QAT
    # ----------------------------
    qconfig = get_default_qat_qconfig(backend)
    model_fp32.qconfig = qconfig
    model_fp32.train()
    # Inject fake-quant observers
    prepare_qat(model_fp32, inplace=True)

    # Move to CUDA for QAT
    model_fp32.to(device)

    model_qat = model_fp32

    # ----------------------------
    # Training setup
    # ----------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model_qat.parameters(), lr=lr, momentum=momentum)

    def train_one_epoch(model, loader):
        model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc="Training", leave=False)

        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{total_loss/(pbar.n+1):.4f}"})

        return total_loss / len(loader)

    def evaluate(model, loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
                preds = model(images).argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / total

    # ----------------------------
    # QAT main loop
    # ----------------------------
    for epoch in range(epochs):
        loss = train_one_epoch(model_qat, train_loader)
        acc = evaluate(model_qat, val_loader)
        print(f"[Epoch {epoch+1}/{epochs}] Loss={loss:.4f} | ValAcc={acc*100:.2f}%")

    # ----------------------------
    # Convert to INT8 (CPU-only)
    # ----------------------------
    model_qat.eval()
    model_qat.cpu()

    model_int8 = convert(model_qat, inplace=False).eval()

    return model_qat, model_int8
