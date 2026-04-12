# =====================================================================
# Imports
# =====================================================================
import copy
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx
from torchvision import models, transforms
from torchvision.transforms.functional import to_pil_image
import shap
from shap.maskers import Image as ImageMasker
from lime import lime_image
from torchcam.methods import GradCAM, GradCAMpp, XGradCAM, LayerCAM
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from .utils import PreprocessImagenet
from torch.fx import GraphModule, Interpreter
from maco_package.utils import load_model_generic
import umap
from sklearn.preprocessing import StandardScaler

# =====================================================================
# Constants
# =====================================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# =====================================================================
# 1. Robust Prediction Engine
# =====================================================================

def robust_predict_batch(model, np_imgs, device, model_dtype):
    """
    Converts a batch of images to tensors, casts to the model's precision,
    runs a forward pass, and returns probability distributions.

    Args:
        model (nn.Module): PyTorch model to run inference on.
        np_imgs (list of np.ndarray): List of images in HWC format (0-255, uint8).
        device (str or torch.device): Device to run the model on ('cpu' or 'cuda').
        model_dtype (torch.dtype): Data type of the model (e.g., torch.float32, torch.quint8).

    Returns:
        np.ndarray: Array of shape (N, num_classes) with softmax probabilities.
    """
    batch_tensors = []

    for arr in np_imgs:
        if arr.max() <= 1.0: arr = (arr * 255).astype(np.uint8)
        arr_pil = Image.fromarray(arr.astype("uint8"))
        t = PreprocessImagenet(arr_pil).to(device)
        batch_tensors.append(t)

    x = torch.cat(batch_tensors, dim=0)

    if model_dtype not in [torch.quint8, torch.qint8]:
        x = x.to(model_dtype)

    with torch.no_grad():
        try:
            out = model(x)
        except (RuntimeError, NotImplementedError):
            x_quant = torch.quantize_per_tensor(x.float(), 0.01, 114, torch.quint8)
            out = model(x_quant)

        if hasattr(out, "dequantize"):
            out = out.dequantize()
        probs = out.float().softmax(1).cpu().numpy()

    return probs

# =====================================================================
# 2. SHAP Wrapper
# =====================================================================

def compute_shap(model, pil_img, max_evals=200, device="cpu", plot=True, return_values=False):
    """
    Compute SHAP explanations for a single PIL image with optional plotting.

    Args:
        model (nn.Module): Model to explain.
        pil_img (PIL.Image.Image): Input image.
        max_evals (int): Maximum number of SHAP evaluations.
        device (str): Device to run on ('cpu' or 'cuda').
        plot (bool): Whether to plot the SHAP explanation.
        return_values (bool): Whether to return the raw SHAP values.

    Returns:
        np.ndarray or None: SHAP values for predicted class if return_values=True, else None.
    """
    device = torch.device(device)
    model = model.to(device).eval()

    try: 
        model_dtype = next(model.parameters()).dtype
    except: 
        model_dtype = torch.quint8

    if isinstance(pil_img, (torch.Tensor, torch.FloatTensor, torch.cuda.FloatTensor)):
        t = pil_img.detach().cpu().float()
        if t.dim() == 4: t = t[0]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        t = (t * std + mean).clamp(0, 1)
        img_raw_pil = transforms.ToPILImage()(t).convert("RGB").resize((224, 224))
    else:
        img_raw_pil = pil_img.convert("RGB").resize((224, 224))

    img_np = np.array(img_raw_pil)

    def predict_callback(np_imgs):
        return robust_predict_batch(model, np_imgs, device, model_dtype)

    masker = ImageMasker("inpaint_telea", img_np.shape)
    explainer = shap.Explainer(predict_callback, masker)
    shap_values = explainer(img_np[None], max_evals=max_evals)

    preds = predict_callback(img_np[None])
    pred_cls = int(np.argmax(preds))
    shap_numpy = shap_values.values[0][..., pred_cls]

    if plot:
        plt.figure()
        plt.imshow(img_np)
        plt.imshow(shap_numpy, cmap='jet', alpha=0.5)
        plt.axis('off')
        plt.title(f"SHAP | {model.__class__.__name__}")
        plt.show()

    if return_values: 
        return shap_numpy
        
    return None

# =====================================================================
# 3. LIME Wrapper
# =====================================================================

def compute_lime(model, pil_img, top_labels=1, num_samples=500, device="cpu", plot=True, return_mask=False):
    """
    Compute LIME explanations for a single PIL image.

    Args:
        model (nn.Module): Model to explain.
        pil_img (PIL.Image.Image or torch.Tensor): Input image.
        top_labels (int): Number of top predicted labels to explain.
        num_samples (int): Number of perturbed samples to generate.
        device (str): Device to run on ('cpu' or 'cuda').
        plot (bool): Whether to plot the LIME explanation.
        return_mask (bool): Whether to return the binary mask.

    Returns:
        np.ndarray or None: LIME mask if return_mask=True, else None.
    """
    device = torch.device(device)
    model = model.to(device).eval()
    try: 
        model_dtype = next(model.parameters()).dtype
    except: 
        model_dtype = torch.quint8
    if isinstance(pil_img, torch.Tensor):
        t = pil_img.detach().cpu().float()
        if t.dim() == 4: t = t[0]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        t = (t * std + mean).clamp(0, 1)
        img_raw_pil = transforms.ToPILImage()(t).convert("RGB").resize((224, 224))
    else:
        img_raw_pil = pil_img.convert("RGB").resize((224, 224))

    img_np = np.array(img_raw_pil)
    def predict_callback(np_imgs):
        return robust_predict_batch(model, np_imgs, device, model_dtype)

    explainer = lime_image.LimeImageExplainer(random_state=42)
    
    try:
        explanation = explainer.explain_instance(
            img_np.astype(np.uint8),
            predict_callback,
            top_labels=top_labels,
            hide_color=0,
            num_samples=num_samples
        )

        label = explanation.top_labels[0]
        lime_img, mask = explanation.get_image_and_mask(label, positive_only=True, hide_rest=False)

        if plot:
            plt.imshow(mark_boundaries(lime_img, mask))
            plt.axis("off")
            lbl = model.__class__.__name__
            if "GraphModule" in lbl: lbl = "Graph"
            plt.title(f"LIME | {lbl} | {str(model_dtype).replace('torch.','')}", fontsize=10)

        if return_mask: 
            return mask.astype(np.float32)

    except Exception as e:
        print(f"LIME Error: {e}")
        return None

    return None

def get_model_precision(model):
    """
    Determine the model's dtype, with special handling for quantized layers.

    Args:
        model (nn.Module): Model to check.

    Returns:
        torch.dtype: Model's data type (e.g., torch.float32, torch.quint8).
    """
    for m in model.modules():
        if hasattr(m, 'weight') and callable(m.weight): return torch.quint8
    try: return next(model.parameters()).dtype
    except StopIteration: return torch.quint8

# =====================================================================
# 4. Activation Extraction
# =====================================================================

def get_activations(model, x, target_layer_name):
    """
    Extract activations from a specific layer.

    Args:
        model (nn.Module or GraphModule): Model to extract activations from.
        x (torch.Tensor): Input tensor of shape (1,3,H,W).
        target_layer_name (str): Layer name to extract.

    Returns:
        torch.Tensor: Activation tensor of shape (C,H,W) from target layer.
    """
    storage = {}
    is_graph = isinstance(model, GraphModule)

    def save_act(val):
        if hasattr(val, "dequantize"): val = val.dequantize()
        if isinstance(val, torch.Tensor) and len(val.shape) == 4:
            storage["A"] = val.detach().cpu()

    if is_graph:
        class FXInterpreter(Interpreter):
            def run_node(self, node):
                res = super().run_node(node)
                clean_target = target_layer_name.replace(".", "_")
                if (node.name == clean_target) or \
                   (target_layer_name in str(node.target)) or \
                   (target_layer_name == "features" and "features" in node.name):
                    save_act(res)
                return res
        FXInterpreter(model).run(x.float())
    else:
        def hook_fn(module, input, output): save_act(output)
        target = None
        for name, mod in model.named_modules():
            if name == target_layer_name: target = mod; break
        if not target:
            try: target = model.get_submodule(target_layer_name)
            except: pass
        if not target: raise RuntimeError(f"Layer '{target_layer_name}' not found.")
        handle = target.register_forward_hook(hook_fn)
        try: model(x)
        except RuntimeError as e:
            if "quantized" in str(e):
                x_quant = torch.quantize_per_tensor(x.float(), 0.01, 114, torch.quint8)
                model(x_quant)
            else: raise e
        finally: handle.remove()

    if "A" not in storage: raise RuntimeError(f"No activations found for {target_layer_name}")
    return storage["A"].squeeze(0)

# =====================================================================
# 5. GradCAM Engines
# =====================================================================

def compute_gradcams(model, pil_img, target_layer=None, device="cpu", plot=True, return_maps=False):
    """
    Compute GradCAM, GradCAM++, XGradCAM, and LayerCAM maps.

    Args:
        model (nn.Module): Model to explain.
        pil_img (PIL.Image.Image): Input image.
        target_layer (nn.Module): Layer to hook. Defaults to last conv.
        device (str): Device to run on ('cpu' or 'cuda').
        plot (bool): Whether to plot the CAM maps.
        return_maps (bool): Whether to return raw CAM maps.

    Returns:
        dict or None: Dictionary of CAM maps if return_maps=True, else None.
    """
    net = copy.deepcopy(model).eval().to(device)
    x = PreprocessImagenet(pil_img).to(device).clone().detach().requires_grad_(True)
    if target_layer is None: target_layer = net.layer4[-1]

    cams = {
        "gradcam":   GradCAM(net, target_layer),
        "gradcam++": GradCAMpp(net, target_layer),
        "xgradcam":  XGradCAM(net, target_layer),
        "layercam":  LayerCAM(net, target_layer),
    }

    torch.set_grad_enabled(True)
    logits = net(x)
    cls = int(logits.argmax())

    maps = {}
    cam_names = list(cams.keys())
    for i, (name, cam) in enumerate(cams.items()):
        hmap = cam(cls, logits, retain_graph=(i < len(cam_names)-1))[0]
        hmap = F.interpolate(hmap.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False)[0]
        maps[name] = hmap[0].detach().cpu()

    if plot:
        img_np = PreprocessImagenet(pil_img, return_numpy=True)
        fig, axs = plt.subplots(1, len(maps), figsize=(14, 4))
        if len(maps) == 1: axs = [axs]
        for ax, (name, cam_map) in zip(axs, maps.items()):
            ax.imshow(img_np)
            ax.imshow(cam_map, cmap="jet", alpha=0.45)
            ax.set_title(name)
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    del net
    if return_maps: return maps

# =====================================================================
# 6. EigenCAM
# =====================================================================
def compute_eigencam(model, img_input, target_layer_name, device="cpu", plot=True, 
                     return_map=False, model_label=None):
    """
    Compute EigenCAM heatmap for a single image.

    Args:
        model (nn.Module): Model to explain.
        img_input (PIL.Image.Image or torch.Tensor or str): Input image.
        target_layer_name (str): Layer name to extract features from.
        device (str): Device to run on.
        plot (bool): Whether to visualize overlay.
        return_map (bool): Whether to return heatmap array.
        model_label(str): For Plots

    Returns:
        np.ndarray or PIL.Image.Image: Heatmap or overlay depending on return_map flag.
    """

    if isinstance(img_input, torch.Tensor):
        x = img_input.clone().detach().to(device).float()
        
        if x.dim() == 3: 
            x = x.unsqueeze(0)
        elif x.dim() == 5: 
            x = x.squeeze(0)

        t = img_input.detach().cpu().float()
        if t.dim() == 4: 
            t = t.squeeze(0)
            
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        t = t * std + mean
        t = t.clamp(0, 1)
        pil_img = to_pil_image(t)

    elif isinstance(img_input, str):
        pil_img = Image.open(img_input).convert("RGB")
        x = PreprocessImagenet(pil_img).to(device).float()
    else:
        pil_img = img_input.convert("RGB")
        x = PreprocessImagenet(pil_img).to(device).float()

    model = model.to(device).eval()
    target_dtype = get_model_precision(model)
    if target_dtype != torch.quint8: 
        x = x.to(target_dtype)

    try: 
        A = get_activations(model, x, target_layer_name)
    except Exception as e:
        print(f"EigenCAM Error ({target_layer_name}): {e}")
        return None

    A = A.detach().cpu().float()
    C, H, W = A.shape
    F_flat = A.view(C, -1).numpy()
    F_flat = F_flat - F_flat.mean(axis=1, keepdims=True)

    try:
        _, _, vh = np.linalg.svd(F_flat, full_matrices=False)
        principal_component = vh[0].reshape(H, W)
    except Exception as e:
        print(f"SVD Failed: {e}")
        return None

    if np.abs(np.min(principal_component)) > np.abs(np.max(principal_component)):
        principal_component = -principal_component

    cam = (principal_component - principal_component.min()) / (principal_component.max() - principal_component.min() + 1e-8)
    cam_resized = cv2.resize(cam, (pil_img.width, pil_img.height))

    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.5 * np.array(pil_img) + 0.5 * heatmap).astype(np.uint8)

    if plot:
        title = model_label if model_label else model.__class__.__name__
        if "GraphModule" in title: 
            title = "Graph/PTQ"
        
        plt.imshow(overlay)
        plt.title(f"EigenCAM | {title}\n{target_layer_name}", fontsize=8)
        plt.axis("off")

    if return_map: 
        return cam_resized
    return overlay

# =====================================================================
# 7. Score-CAM Engine
# =====================================================================

def score_cam(model, x, pil_img, target_layer_name, target_class=None, max_channels=32):
    """
    Compute Score-CAM heatmap for a single input image.

    Args:
        model (nn.Module): Model to explain.
        x (torch.Tensor): Preprocessed input tensor (1,3,H,W).
        pil_img (PIL.Image.Image): Original PIL image for overlay.
        target_layer_name (str): Layer name to extract activations from.
        target_class (int, optional): Class index for which to compute CAM. Defaults to predicted class.
        max_channels (int): Maximum number of channels to use from activation map.

    Returns:
        tuple: (overlay_image (np.ndarray), heatmap (np.ndarray))
    """
    model.eval()
    device = x.device
    target_dtype = get_model_precision(model)

    if target_dtype == torch.quint8:
        x_in = x.float()
    else:
        x_in = x.to(target_dtype)

    A = get_activations(model, x_in, target_layer_name)
    A[A < (A.max() * 0.12)] = 0
    C, H, W = A.shape
    k = min(max_channels, C)
    idx = torch.topk(A.view(C, -1).var(dim=1), k).indices
    A_sel = A[idx]
    A_norm = (A_sel - A_sel.min()) / (A_sel.max() - A_sel.min() + 1e-8)
    masks = F.interpolate(A_norm.unsqueeze(1).to(device), size=x.shape[2:], mode="bilinear", align_corners=False)
    scores = []
    with torch.no_grad():
        if target_dtype == torch.quint8:
            try: logits = model(x_in)
            except: logits = model(torch.quantize_per_tensor(x_in, 0.01, 114, torch.quint8))
        else:
            logits = model(x_in)

        if hasattr(logits, "dequantize"): logits = logits.dequantize()
        if target_class is None: target_class = logits.argmax(dim=1).item()

        for i in range(len(A_sel)):
            mk = masks[i]
            if target_dtype != torch.quint8:
                masked_x = x_in * mk.to(target_dtype)
                out = model(masked_x)
            else:
                masked_x_float = x_in.float() * mk.float()
                try: out = model(masked_x_float)
                except: out = model(torch.quantize_per_tensor(masked_x_float, 0.01, 114, torch.quint8))
            if hasattr(out, "dequantize"): out = out.dequantize()
            scores.append(torch.softmax(out.float(), dim=1)[0, target_class].item())

    scores = torch.tensor(scores).to(device)
    cam = torch.sum(scores[:, None, None] * A_sel.to(device).float(), dim=0).cpu().numpy()
    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = cv2.resize(cam, (pil_img.width, pil_img.height))
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = (0.5 * np.array(pil_img) + 0.5 * cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)).astype(np.uint8)

    return overlay, heat

# =====================================================================
# 8. Score-CAM Wrapper
# =====================================================================

def compute_scorecam(model, img_input, target_layer_name="layer4.2", target_class=None,
                     model_label=None, max_channels=256, device="cpu", plot=True,
                     return_map=False, return_overlay=False):
    """
    Wrapper für Score-CAM.
    """
    device = torch.device(device)
    model = model.to(device).eval()

    # --- INPUT HANDLING (Kopie von EigenCAM Logik) ---
    if isinstance(img_input, torch.Tensor):
        # 1. Tensor direkt für das Modell (bereits normalisiert vom Loader)
        x = img_input.clone().detach().to(device).float()
        if x.dim() == 3: x = x.unsqueeze(0)
        elif x.dim() == 5: x = x.squeeze(0) # Failsafe für Video-Batches

        # 2. PIL Image für den Plot erstellen (Denormalisieren für korrekte Farben)
        t = img_input.detach().cpu().float()
        if t.dim() == 4: t = t.squeeze(0)
        
        # ImageNet Denormalisierung für den Plot-Hintergrund
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        t = t * std + mean
        t = t.clamp(0, 1)
        pil_img = to_pil_image(t)
    else:
        # Standard PIL Handling (Normalisierung wird hier durchgeführt)
        pil_img = img_input.convert("RGB")
        x = PreprocessImagenet(pil_img).to(device).float()

    # --- BERECHNUNG ---
    try:
        # Aufruf der Engine mit dem bereits korrekt vorbereiteten Tensor 'x'
        overlay, heat = score_cam(model, x, pil_img, target_layer_name, target_class, max_channels)
    except Exception as e:
        print(f"Error computing Score-CAM for {model_label}: {e}")
        return None

    # Heatmap in Graustufen (0.0 - 1.0) für Metriken (SSIM, Pearson etc.)
    cam_map = cv2.cvtColor(heat, cv2.COLOR_RGB2GRAY).astype("float32") / 255.0

    # --- VISUALISIERUNG ---
    if plot:
        title = model_label if model_label else model.__class__.__name__
        if "GraphModule" in title: title = "Graph/PTQ"
        
        plt.imshow(overlay)
        plt.title(f"Score-CAM | {title}\n{target_layer_name}", fontsize=8)
        plt.axis("off")

    # --- OUTPUT ---
    if return_map and return_overlay: 
        return cam_map, overlay
    elif return_map: 
        return cam_map
    elif return_overlay: 
        return overlay
    else: 
        return None

# =====================================================================
# 9. Feature Extraction and UMAP
# =====================================================================

def get_model_features(model, dataloader, device="cpu", max_samples=None):
    """
    Universal feature extractor that finds the last valid 4D or 2D layer 
    before the classifier, handles quantization, and flattens output.
    """
    model.eval().to(device)
    features = []
    labels = []
    total_extracted = 0
    storage = {}

    # Find the target layer
    target_layer = None
    
    if hasattr(model, 'avgpool'):
        target_layer = model.avgpool
    
    elif hasattr(model, 'layers'):
        if type(model.layers) == torch.nn.modules.module.Module:
            children = list(model.layers.children())
            if children:
                target_layer = children[-1]
            else:
                target_layer = model.layers
        else:
            target_layer = model.layers
        
    elif hasattr(model, 'features'):
        # Für VGG/ResNet-Strukturen
        target_layer = list(model.features.modules())[-1]
    
    else:
        all_mods = [m for m in model.modules() if not isinstance(m, (nn.Sequential, nn.ModuleList))]
        for m in reversed(all_mods):
            if not isinstance(m, nn.Linear):
                target_layer = m
                break

    def hook_fn(module, input, output):
        val = output[0] if isinstance(output, (tuple, list)) else output
        # Handle Quantized Tensors
        if hasattr(val, "dequantize"): 
            val = val.dequantize()
        
        # Global Average Pool if the output is still 4D 
        if len(val.shape) == 4:
            val = torch.mean(val, dim=[2, 3])
            
        # Flatten and move to CPU
        feat = val.detach().cpu().reshape(val.size(0), -1).float()
        storage['feat'] = feat

    handle = target_layer.register_forward_hook(hook_fn)
    
    # Precision Check
    try:
        model_dtype = next(model.parameters()).dtype
    except StopIteration:
        model_dtype = torch.quint8

    with torch.no_grad():
        for imgs, lbls in dataloader:
            if max_samples is not None and total_extracted >= max_samples:
                break

            x = imgs.to(device)
            # Handle float variants (fp32, fp16, bf16)
            if model_dtype in [torch.float32, torch.float16, torch.bfloat16]:
                x = x.to(model_dtype)
            
            try:
                model(x)
            except Exception:
                # Handle Quantized INT8 models requiring quint8 input
                x_quant = torch.quantize_per_tensor(x.cpu().float(), 0.1, 0, torch.quint8).to(device)
                model(x_quant)

            features.append(storage['feat'])
            labels.append(lbls.cpu())
            total_extracted += imgs.size(0)

    handle.remove()
    
    final_features = torch.cat(features).numpy()
    final_labels = torch.cat(labels).numpy()

    if max_samples is not None:
        final_features = final_features[:max_samples]
        final_labels = final_labels[:max_samples]

    return final_features, final_labels

def compute_umap(features, labels, n_neighbors=15, min_dist=0.1, metric='cosine'):
    """
    Reduce high-dimensional features to 2D using UMAP.

    Args:
        features (np.ndarray): Feature matrix of shape (N, D).
        labels (np.ndarray): Class labels.
        n_neighbors (int): Number of neighbors for UMAP.
        min_dist (float): Minimum distance between points in UMAP embedding.
        metric (str): Distance metric for UMAP.

    Returns:
        np.ndarray: 2D UMAP embedding of shape (N,2).
    """
    scaled_features = StandardScaler().fit_transform(features)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=42)
    embedding = reducer.fit_transform(scaled_features)
    return embedding
