# =====================================================================
# Imports
# =====================================================================
import platform
import subprocess
import time
import warnings
import copy
from pathlib import Path
import json
import torch
import torch.nn as nn
from torch.ao.quantization import (
    get_default_qat_qconfig,
    prepare_qat,
    convert,
    get_default_qconfig
)
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.quantization import QuantStub, DeQuantStub
import torchvision
from torchvision import models, transforms
from torchvision.models import (
    resnet50 as resnet50_fp32,
    efficientnet_b0,
    densenet121,
    vgg16 as vgg16_standard,
    ResNet50_Weights,
    EfficientNet_B0_Weights,
    DenseNet121_Weights,
    VGG16_Weights,
    MobileNet_V2_Weights
)
from torchvision.models.quantization import (
    resnet50 as resnet50_quant,
    mobilenet_v2 as q_mobilenet
)
import os
from torch.ao.quantization import quantize_fx

# =====================================================================
# Quantizable VGG16 Wrapper
# =====================================================================

class QuantizableVGG16(nn.Module):
    """
    VGG16 model with explicit quant/dequant stubs.
    """
    def __init__(self, num_classes):
        super().__init__()

        self.model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.model.classifier[6] = nn.Linear(4096, num_classes)

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def fuse_model(self):
        """
        Fuses Conv + ReLU layers for quantization.
        """
        for i in range(len(self.model.features) - 1):
            if isinstance(self.model.features[i], nn.Conv2d) and isinstance(self.model.features[i+1], nn.ReLU):
                torch.quantization.fuse_modules(
                    self.model.features,
                    [str(i), str(i+1)],
                    inplace=True
                )

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x


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
    Converts input image to ImageNet-ready tensor.

    Args:
        img: PIL image, file path, or tensor.
        return_numpy (bool): Return uint8 HWC array if True.

    Returns:
        torch.Tensor or numpy.ndarray: Preprocessed image.
    """
    if isinstance(img, torch.Tensor):
        t = img.clone().detach().cpu()

        if t.ndim == 4:
            t = t.squeeze(0)
        if t.ndim != 3:
            raise ValueError("Tensor input must be CHW or BCHW shape.")

        pil_img = to_pil_image(t)
    else:
        if isinstance(img, str):
            pil_img = Image.open(img).convert("RGB")
        else:
            pil_img = img.convert("RGB")

    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    t = tf(pil_img)

    if return_numpy:
        unnorm = (
            t.clone() * torch.tensor(IMAGENET_STD)[:, None, None]
            + torch.tensor(IMAGENET_MEAN)[:, None, None]
        )
        arr = (unnorm.permute(1, 2, 0).numpy().clip(0, 1) * 255).astype("uint8")
        return arr

    return t.unsqueeze(0)


# =====================================================================
# Class Name Utilities
# =====================================================================

def load_class_names(root):
    """
    Loads synset-to-label mapping.

    Args:
        root (str | Path): Dataset root.

    Returns:
        dict: Synset to label mapping.
    """
    json_path = Path(root) / "Labels.json"
    with open(json_path, "r") as f:
        return json.load(f)


# =====================================================================
# Shared Utilities
# =====================================================================

def save_model(model, path):
    """
    Saves model state dictionary.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Saved model to: {path}")


def strip_dropout(model):
    """
    Replaces all Dropout layers with Identity.
    """
    model.eval()
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            parent_name = name.rsplit('.', 1)[0]
            idx = name.split('.')[-1]
            parent = model.get_submodule(parent_name) if parent_name else model

            if isinstance(parent, (nn.ModuleList, nn.Sequential)):
                parent[int(idx)] = nn.Identity()
            else:
                setattr(parent, idx, nn.Identity())
    return model


def safe_fuse(model, is_qat=False):
    """
    Calls fuse_model if available.
    """
    if hasattr(model, "fuse_model"):
        try:
            model.fuse_model(is_qat=is_qat)
        except TypeError:
            model.fuse_model()


# =====================================================================
# Base Model Factories
# =====================================================================

def get_base_resnet50(num_classes):
    model = resnet50_quant(weights="DEFAULT", quantize=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return strip_dropout(model)


def get_base_vgg16(num_classes):
    model = QuantizableVGG16(num_classes)
    return strip_dropout(model)


def get_base_mobilenet(num_classes):
    model = q_mobilenet(weights=MobileNet_V2_Weights.IMAGENET1K_V1, quantize=False)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return strip_dropout(model)


def get_base_efficientnet(num_classes):
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_feats = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_feats, num_classes)
    return strip_dropout(model)


def get_base_densenet(num_classes):
    model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    in_feats = model.classifier.in_features
    model.classifier = nn.Linear(in_feats, num_classes)
    return strip_dropout(model)


BASE_FACTORIES = {
    "resnet50":        get_base_resnet50,
    "vgg16":           get_base_vgg16,
    "mobilenet_v2":    get_base_mobilenet,
    "efficientnet_b0": get_base_efficientnet,
    "densenet121":     get_base_densenet,
}


# =====================================================================
# Quantization Conversions
# =====================================================================

def to_fp16(model):
    """Converts model to FP16."""
    return model.half()


def to_bf16(model):
    """Converts model to BF16."""
    return model.to(torch.bfloat16)


def to_qatfake(model, backend="fbgemm"):
    """Prepares model for fake-QAT."""
    model.eval()
    safe_fuse(model, is_qat=True)
    torch.backends.quantized.engine = backend
    model.qconfig = get_default_qat_qconfig(backend)
    model.train()
    prepare_qat(model, inplace=True)
    return model


def to_int8(model, backend="fbgemm"):
    """Converts QAT-prepared model to INT8."""
    torch.backends.quantized.engine = backend
    model.eval()
    safe_fuse(model, is_qat=True)

    model.train()
    model.qconfig = get_default_qat_qconfig(backend)
    prepare_qat(model, inplace=True)

    model.cpu().eval()
    return convert(model, inplace=False)


def to_ptqfx(model, backend="fbgemm", calibrate=False):
    """Applies FX-based PTQ."""
    model.eval()
    torch.backends.quantized.engine = backend

    qconfig_mapping = torch.ao.quantization.get_default_qconfig_mapping(backend)
    example_input = torch.randn(1, 3, 224, 224)

    prepared = quantize_fx.prepare_fx(model, qconfig_mapping, example_input)

    if calibrate:
        with torch.no_grad():
            prepared(example_input)

    return quantize_fx.convert_fx(prepared).eval()


# =====================================================================
# Unified Loader
# =====================================================================

def load_model_generic(arch, path, num_classes, device="cpu", backend="fbgemm"):
    """
    Builds and loads a model from architecture string.

    Args:
        arch (str): Format 'name-variant'.
        path (str): Checkpoint path.
        num_classes (int): Output classes.
        device (str): Target device.

    Returns:
        nn.Module: Loaded model.
    """
    try:
        model_name, variant = arch.rsplit('-', 1)
    except ValueError:
        raise ValueError(f"Architecture string '{arch}' must be format 'name-variant'")

    if model_name not in BASE_FACTORIES:
        raise ValueError(f"Unknown base architecture: {model_name}")

    model = BASE_FACTORIES[model_name](num_classes)

    if variant == "fp32":
        pass
    elif variant == "fp16":
        model = to_fp16(model)
    elif variant == "bf16":
        model = to_bf16(model)
    elif variant == "qatfake":
        model = to_qatfake(model, backend)
    elif variant == "int8":
        model = to_int8(model, backend)
    elif variant == "ptqfx":
        do_calib = (model_name == "densenet121")
        model = to_ptqfx(model, backend, calibrate=do_calib)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    state = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)

    if missing:
        real_missing = [k for k in missing if "weight" in k or "bias" in k]
        if real_missing:
            print(f"⚠️ Warning: Missing keys: {real_missing[:5]}")

    model.to(device)
    model.eval()
    return model
