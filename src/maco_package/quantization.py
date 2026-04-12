# =====================================================================
# Imports
# =====================================================================
import torch
import torch.nn as nn
import copy
import gc
import warnings
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.ao.quantization import get_default_qat_qconfig_mapping, get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, prepare_qat_fx, convert_fx

# =====================================================================
# Constants
# =====================================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# =====================================================================
# Quantization Functions
# =====================================================================

def QuantizePTQ(model, calibration_loader=None, num_calib_batches=10):
    """
    Applies Post-Training Quantization (PTQ) using FX Graph Mode with calibration.

    Args:
        model (nn.Module): The Float32 model to be quantized.
        calibration_loader (DataLoader, optional): DataLoader providing real data for calibration.
        num_calib_batches (int): Number of batches to use for calibration.

    Returns:
        nn.Module: The quantized Int8 model (FX Graph Module).
    """
    device = next(model.parameters()).device
    model.eval()

    # 1. Setup Configuration
    qconfig_mapping = get_default_qconfig_mapping("fbgemm")

    # 2. Determine Example Input (Safe Check)
    if calibration_loader is not None and torch.is_tensor(calibration_loader):
        # Case A: Input is a Tensor (Single Batch)
        example_input = calibration_loader[0:1].to(device)
    elif calibration_loader is not None:
        # Case B: Input is a DataLoader
        try:
            example_input = next(iter(calibration_loader))[0][0:1].to(device)
        except:
             example_input = torch.randn(1, 3, 224, 224).to(device)
    else:
        # Case C: No input provided
        example_input = torch.randn(1, 3, 224, 224).to(device)

    # Prepare model
    prepared = prepare_fx(model, qconfig_mapping, example_input)

    # 3. Calibration Pass
    if calibration_loader is not None:
        if torch.is_tensor(calibration_loader):
            print("⚖️ Calibrating PTQ on provided Tensor batch...")
            with torch.no_grad():
                prepared(calibration_loader.to(device))
        else:
            print(f"⚖️ Calibrating PTQ on {num_calib_batches} batches...")
            with torch.no_grad():
                for i, (images, _) in enumerate(calibration_loader):
                    if i >= num_calib_batches: break
                    prepared(images.to(device))
    else:
        print("⚠️ WARNING: No loader provided. Calibrating on noise.")
        prepared(example_input)

    # 4. Convert to Int8
    quantized = convert_fx(prepared)
    
    return quantized


def trainQAT(
    model_fp32,
    train_dataset,
    val_dataset,
    epochs=5,
    batch_size=40,
    lr=1e-5,
    momentum=0.9,
    backend="fbgemm",
    patience=None
):
    """
    Executes Quantization Aware Training using FX Graph Mode.

    Args:
        model_fp32 (nn.Module): The pretrained Float32 model.
        train_dataset (Dataset): Training data.
        val_dataset (Dataset): Validation data.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size (lower than standard training to save VRAM).
        lr (float): Learning rate (should be low for QAT).
        momentum (float): Optimizer momentum.
        backend (str): Quantization backend ('fbgemm' or 'qnnpack').
        patience (int, optional): Early stopping patience.

    Returns:
        tuple: (model_qat, model_int8, best_epoch)
            - model_qat: The best QAT model (simulated quantization, on GPU).
            - model_int8: The best converted Int8 model (CPU).
            - best_epoch: The epoch number where best accuracy was achieved.
    """
    # 1. Resource Cleanup
    torch.cuda.empty_cache()
    gc.collect()
    
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

    # 3. Prepare Model for FX
    try:
        example_input = next(iter(train_loader))[0].to("cpu")
    except StopIteration:
        raise ValueError("Train dataset is empty!")

    model_fp32 = copy.deepcopy(model_fp32).to("cpu")
    model_fp32.train()

    qconfig_mapping = get_default_qat_qconfig_mapping(backend)

    try:
        model_qat = prepare_qat_fx(
            model_fp32, 
            qconfig_mapping, 
            example_input
        )
    except Exception as e:
        print(f"❌ FX Tracing Failed: {e}")
        raise e

    model_qat = model_qat.to(device)
    model_qat.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
    model_qat.apply(torch.quantization.enable_observer)
    model_qat.apply(torch.quantization.enable_fake_quant)
    # 4. Training Loop
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model_qat.parameters(), lr=lr, momentum=momentum)

    best_acc = 0.0
    stagnant = 0
    best_epoch = 0
    best_qat_state = copy.deepcopy(model_qat.state_dict())

    print(f"🚀 Starting QAT (Batch Size: {batch_size}, LR: {lr})...")
    
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
        
        acc = correct / total
        avg_loss = total_loss / len(train_loader)
        
        print(f"[Epoch {epoch+1}] loss={avg_loss:.4f} acc={acc*100:.2f}%")

        # Save Best Model
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch + 1
            stagnant = 0
            best_qat_state = copy.deepcopy(model_qat.state_dict())
        else:
            if patience is not None:
                stagnant += 1
                if stagnant >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

    # 5. Conversion
    print(f"🏆 Loading Best Model from Epoch {best_epoch} (Acc: {best_acc*100:.2f}%)")
    
    model_qat.cpu()
    torch.cuda.empty_cache()
    
    model_qat.load_state_dict(best_qat_state)
    model_qat.eval()
    
    try:
        model_int8 = convert_fx(model_qat)
    except Exception as e:
        print(f"⚠️ Conversion Warning: {e}")
        model_int8 = model_qat 

    return model_qat, model_int8, best_epoch