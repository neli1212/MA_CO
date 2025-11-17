# =====================================================================
# Imports
# =====================================================================
import os
import json
from pathlib import Path

import torch
from torch import nn
from torch.ao.quantization import (
    get_default_qat_qconfig,
    prepare_qat,
    quantize_fx,
)

from torchvision import models, transforms
from torchvision.transforms.functional import to_pil_image
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
    resnet50 as resnet50_fp32,          # FP32-ResNet50
)

from torchvision.models.quantization import (
    resnet18 as qresnet18,
    resnet50 as qresnet50,              # optional, aber ok
    resnet50 as resnet50_quant,         # für load_model()
    resnet50 as quant_resnet50,         # für load_resnet50_qatfake()
    resnet50,                           # für load_resnet50_qat_int8() (bare name)
)

from PIL import Image
import torchvision


# =====================================================================
# Constants
# =====================================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# =====================================================================
# Image Preprocessing
# =====================================================================
def PreprocessImagenet(img, return_numpy=False):
    """
    Convert an image into an ImageNet-ready (1,3,224,224) tensor.

    Args:
        img: PIL.Image, file path, or Tensor (CHW or BCHW).
        return_numpy (bool): If True, returns an un-normalized HWC uint8 array.

    Returns:
        torch.Tensor: shape (1,3,224,224) normalized for ImageNet models,
                      unless return_numpy=True (returns numpy array).
    """

    # ------------------------------------------------------
    # 1. Handle Tensor input
    # ------------------------------------------------------
    if isinstance(img, torch.Tensor):
        t = img.clone().detach().cpu()

        if t.ndim == 4:
            t = t.squeeze(0)
        if t.ndim != 3:
            raise ValueError("Tensor input must be CHW or BCHW shape.")

        pil_img = to_pil_image(t)

    else:
        # ------------------------------------------------------
        # 2. Handle PIL image or file path
        # ------------------------------------------------------
        if isinstance(img, str):
            pil_img = Image.open(img).convert("RGB")
        else:
            pil_img = img.convert("RGB")

    # ------------------------------------------------------
    # 3. Standard ImageNet transform
    # ------------------------------------------------------
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    t = tf(pil_img)

    # ------------------------------------------------------
    # 4. Optional: return Numpy image (denormalized)
    # ------------------------------------------------------
    if return_numpy:
        unnorm = (
            t.clone() * torch.tensor(IMAGENET_STD)[:, None, None]
            + torch.tensor(IMAGENET_MEAN)[:, None, None]
        )
        arr = (unnorm.permute(1, 2, 0).numpy().clip(0, 1) * 255).astype("uint8")
        return arr

    return t.unsqueeze(0)


# =====================================================================
# Load a ResNet or Quantization-ready ResNet
# =====================================================================
def load_resnet(name="resnet50", pretrained=True):
    """
    Load an FP32 or quantization-ready ResNet model.

    Args:
        name (str): One of:
            'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
            'qresnet18', 'qresnet50'
        pretrained (bool): Load pretrained ImageNet weights if available.

    Returns:
        torch.nn.Module: Model in eval mode.
    """

    # FP32 architectures
    fp32_models = {
        "resnet18":  (models.resnet18,  ResNet18_Weights.IMAGENET1K_V1),
        "resnet34":  (models.resnet34,  ResNet34_Weights.IMAGENET1K_V1),
        "resnet50":  (models.resnet50,  ResNet50_Weights.IMAGENET1K_V2),
        "resnet101": (models.resnet101, ResNet101_Weights.IMAGENET1K_V2),
        "resnet152": (models.resnet152, ResNet152_Weights.IMAGENET1K_V2),
    }

    # Quantization-ready architectures (not quantized yet!)
    quant_models = {
        "qresnet18": (qresnet18, ResNet18_Weights.IMAGENET1K_V1),
        "qresnet50": (qresnet50, ResNet50_Weights.IMAGENET1K_V2),
    }

    available = {**fp32_models, **quant_models}

    if name not in available:
        raise ValueError(f"Unknown model '{name}'. Available: {list(available.keys())}")

    constructor, weight_enum = available[name]
    weights = weight_enum if pretrained else None

    return constructor(weights=weights).eval()


# =====================================================================
# Load class names for your ImageNet-100 dataset
# =====================================================================
def load_class_names(root):
    """
    Load ImageNet-100 synset-to-name mapping from Labels.json.

    Args:
        root (str or Path): Folder containing Labels.json.

    Returns:
        dict: { "nxxxxx": "Human readable name", ... }
    """
    json_path = Path(root) / "Labels.json"

    with open(json_path, "r") as f:
        return json.load(f)


# =====================================================================
# Save model weights
# =====================================================================
def save_model(model, path):
    """
    Save model weights.

    Args:
        model: torch.nn.Module
        path (str): Destination .pth file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Saved model to: {path}")


def load_model(path, num_classes, device="cpu"):
    """
    Loads your FP32, QAT-fake, or INT8 ResNet50 models.
    All of them come from torchvision.models.quantization.resnet50.
    """

    # Load checkpoint
    state = torch.load(path, map_location="cpu")

    # Build correct architecture
    model = resnet50_quant(weights=None, quantize=False)

    # Replace classifier
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # Load checkpoint
    model.load_state_dict(state, strict=True)

    model.to(device)
    model.eval()

    return model


def load_resnet50_qatfake(path, num_classes, device="cpu", backend="fbgemm"):
    """
    Load a QAT-fake ResNet50 saved via:
        torch.save(model_qat.state_dict(), path)

    This rebuilds the quantizable architecture, fuses layers,
    re-applies QAT preparation, and loads the fake-quant weights.
    """

    # 1. Load state dict
    state = torch.load(path, map_location="cpu")

    # 2. Build quantizable ResNet50 (the SAME type used for training)
    model = quant_resnet50(weights=None, quantize=False)

    # 3. Replace classifier
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # 4. Fuse layers exactly like during trainQAT
    if hasattr(model, "fuse_model"):
        model.eval()
        model.fuse_model()

    # 5. Apply QAT prepare EXACTLY as in trainQAT
    torch.backends.quantized.engine = backend
    model.qconfig = get_default_qat_qconfig(backend)

    model.train()                 # <-- REQUIRED for prepare_qat
    prepare_qat(model, inplace=True)

    # 6. Load state (needs strict=False because QAT adds extra keys)
    missing, unexpected = model.load_state_dict(state, strict=False)

    # 7. Ready
    model.to(device)
    model.eval()
    return model


def load_resnet50_qat_int8(path, num_classes, device="cpu", backend="fbgemm"):
    torch.backends.quantized.engine = backend

    # Build quantized architecture
    model = resnet50(pretrained=False, quantize=True)

    # Replace fully connected layer with INT8 version
    in_features = model.fc.in_features
    model.fc = nn.quantized.Linear(in_features, num_classes)

    # Load INT8 weights
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state, strict=True)

    model.to(device)
    model.eval()
    return model

    return model.eval().to(device)



def load_resnet50_ptq_fx_state(path, num_classes, device="cpu"):
    # --------------------------------------------------
    # 1. Rebuild the FP32 base model
    # --------------------------------------------------
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.eval()

    # --------------------------------------------------
    # 2. Prepare FX PTQ model structure
    # --------------------------------------------------
    qconfig = {"": torch.ao.quantization.get_default_qconfig("fbgemm")}
    example_input = torch.randn(1, 3, 224, 224)

    prepared = quantize_fx.prepare_fx(model, qconfig, example_input)

    # --------------------------------------------------
    # 3. Convert structure to quantized graph
    # --------------------------------------------------
    quantized = quantize_fx.convert_fx(prepared)

    # --------------------------------------------------
    # 4. Load your PTQ INT8 weights
    # --------------------------------------------------
    state = torch.load(path, map_location="cpu")
    quantized.load_state_dict(state)   # strict=False optional

    quantized.to(device)
    quantized.eval()

    return quantized