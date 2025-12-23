# =====================================================================
# Imports
# =====================================================================
import warnings
import torch
import torch.nn as nn
import copy
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
# Constants for Normalization 
# =====================================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# =====================================================================
# Quantization Functions
# =====================================================================

def QuantizePTQ(model, calibration_input=None):
    """
    Applies Post-Training Quantization using FX Graph Mode.

    Args:
        model: Float32 PyTorch model to be quantized.
        calibration_input: A representative batch of data for tracing and 
                           calculating observer statistics.
    Returns:
        quantized_model: Int8 FX-graph-mode quantized model in eval mode.
    """
    torch.backends.quantized.engine = "fbgemm"
    qconfig = tq.get_default_qconfig("fbgemm")

    if calibration_input is None:
        calibration_input = torch.randn(1, 3, 224, 224)

    model = model.eval()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Prepare the graph for quantization
        prepared = quantize_fx.prepare_fx(
            model,
            {"": qconfig},
            calibration_input,
        )

        # Calibration pass to determine scale and zero-point
        prepared(calibration_input)

        # Convert the observed graph to actual quantized operations
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
    # 1. Setup Device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- QAT Training Device: {device} ---")

    # 2. Data Loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # 3. Prepare Model
    model_fp32 = copy.deepcopy(model_fp32).cpu()
    model_fp32.eval()
    
    # Fuse Layers
    if hasattr(model_fp32, "fuse_model"):
        try: model_fp32.fuse_model(is_qat=True)
        except: model_fp32.fuse_model()

    model_fp32.train()
    
    # 4. Apply Configuration 
    qconfig = get_default_qat_qconfig(backend)
    model_fp32.qconfig = qconfig
    
    # Force skip connections to be quantized
    for name, module in model_fp32.named_modules():
        if "FloatFunctional" in str(type(module)):
            module.qconfig = qconfig

    # Insert Observers
    prepare_qat(model_fp32, inplace=True)
    
    # 5. Move to GPU
    model_qat = model_fp32.to(device)

    # 6. Training Loop
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model_qat.parameters(), lr=lr, momentum=momentum)

    for epoch in range(epochs):
        model_qat.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            out = model_qat(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{total_loss/(pbar.n+1):.4f}"})
            
        # Validation
        model_qat.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for images, labels in val_loader:

                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                preds = model_qat(images).argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {correct/total*100:.2f}%")

    # 7. Convert (CPU Only)
    model_qat.cpu().eval()

    old_engine = torch.backends.quantized.engine
    torch.backends.quantized.engine = backend
    try:
        model_int8 = convert(model_qat, inplace=False)
    finally:
        torch.backends.quantized.engine = old_engine

    return model_qat, model_int8