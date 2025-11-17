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
from torchcam.methods import (
    GradCAM,
    GradCAMpp,
    XGradCAM,
    LayerCAM,
)
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from .utils import PreprocessImagenet

# =====================================================================
# Constants
# =====================================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def compute_shap(model, pil_img, max_evals=200, device="cpu",
                 plot=True, return_values=True):
    """
    Compute (and optionally plot) SHAP explanations.
    
    Returns:
        SHAP values (H, W) for the predicted class if return_values=True.
    """
    import shap, numpy as np, copy, torch
    from shap.maskers import Image as ImageMasker
    from PIL import Image

    # ---- prepare model ----
    net = copy.deepcopy(model).eval().to(device)

    # ---- preprocess once ----
    img_np = PreprocessImagenet(pil_img, return_numpy=True)      # HWC uint8
    img_tensor = PreprocessImagenet(pil_img).to(device)          # (1,3,224,224)

    # ---- prediction wrapper for SHAP ----
    def predict_np(np_imgs):
        batch = []
        for arr in np_imgs:
            arr_pil = Image.fromarray(arr.astype("uint8"))
            batch.append(PreprocessImagenet(arr_pil))
        x = torch.cat(batch).to(device)
        with torch.no_grad():
            out = net(x)
        return out.softmax(1).cpu().numpy()

    # ---- build SHAP masker ----
    mean_px = (np.array(IMAGENET_MEAN) * 255).astype(np.float32)
    baseline = np.ones_like(img_np, dtype=np.float32) * mean_px
    masker = ImageMasker(baseline, img_np.shape)

    # ---- SHAP Explainer ----
    explainer = shap.Explainer(predict_np, masker)
    shap_values = explainer(img_np[None], max_evals=max_evals)

    # ---- select predicted class ----
    pred_cls = int(np.argmax(predict_np(img_np[None])))
    vals = np.array(shap_values.values[0][..., pred_cls], dtype=np.float32)

    # ---- plot ----
    if plot:
        shap.image_plot([vals], img_np)

    # ---- return SHAP array ----
    del net
    if return_values:
        return vals

def compute_lime(
    model,
    pil_img,
    top_labels=1,
    num_samples=1000,
    device="cpu",
    plot=True,
    return_mask=True
):
    """
    Compute (and optionally plot) a LIME explanation.

    Returns:
        mask (H, W) if return_mask=True
    """
    import copy, torch, numpy as np
    from lime import lime_image
    from skimage.segmentation import mark_boundaries
    import matplotlib.pyplot as plt
    from PIL import Image

    # ---- Clone model to ensure no unwanted mutation ----
    net = copy.deepcopy(model).eval().to(device)

    # ---- Preprocess image ----
    img_np = PreprocessImagenet(pil_img, return_numpy=True)      # HWC uint8
    img_tensor = PreprocessImagenet(pil_img).to(device)          # (1,3,224,224)

    # ---- prediction wrapper for LIME (expects numpy HWC) ----
    def predict_np(np_imgs):
        batch = []
        for arr in np_imgs:
            arr_pil = Image.fromarray(arr.astype("uint8"))
            batch.append(PreprocessImagenet(arr_pil))             # normalized CHW
        x = torch.cat(batch, dim=0).to(device)

        with torch.no_grad():
            out = net(x)

        return out.softmax(1).cpu().numpy()

    # ---- Run LIME ----
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_np.astype(np.uint8),
        predict_np,
        top_labels=top_labels,
        hide_color=0,
        num_samples=num_samples,
    )

    # ---- Extract LIME mask ----
    label = explanation.top_labels[0]
    lime_img, mask = explanation.get_image_and_mask(
        label,
        positive_only=True,
        hide_rest=False
    )

    # ---- Plot ----
    if plot:
        plt.imshow(mark_boundaries(lime_img, mask))
        plt.axis("off")
        plt.show()

    del net

    # ---- Return LIME mask (optional) ----
    if return_mask:
        return mask


def compute_gradcams(
    model,
    pil_img,
    target_layer=None,
    device="cpu",
    plot=True,
    return_maps=False
):
    """
    Compute multiple gradient-based CAM maps (GradCAM, GradCAM++, XGradCAM, LayerCAM).

    Args:
        model: PyTorch model
        pil_img: PIL image, path, or tensor
        target_layer: layer to hook; default = last conv (ResNet-like)
        device: "cpu" or "cuda"
        plot: visualize overlays
        return_maps: return dict of CAM maps

    Returns:
        dict: {method_name: heatmap_tensor} if return_maps=True
    """

    # ---------------------------------------------------------
    # Clone model
    # ---------------------------------------------------------
    net = copy.deepcopy(model).eval().to(device)

    # ---------------------------------------------------------
    # Preprocess image
    # ---------------------------------------------------------
    x = PreprocessImagenet(pil_img).to(device)  # (1,3,224,224)
    x = x.clone().detach().requires_grad_(True)

    # ---------------------------------------------------------
    # Default layer selection for ResNet
    # ---------------------------------------------------------
    if target_layer is None:
        # Works for most torchvision ResNets
        target_layer = net.layer4[-1]

    # ---------------------------------------------------------
    # Initialize CAM methods
    # ---------------------------------------------------------
    cams = {
        "gradcam":   GradCAM(net, target_layer),
        "gradcam++": GradCAMpp(net, target_layer),
        "xgradcam":  XGradCAM(net, target_layer),
        "layercam":  LayerCAM(net, target_layer),
    }

    # ---------------------------------------------------------
    # Run forward pass
    # ---------------------------------------------------------
    torch.set_grad_enabled(True)
    logits = net(x)
    cls = int(logits.argmax())

    # ---------------------------------------------------------
    # Compute CAM maps
    # ---------------------------------------------------------
    maps = {}
    cam_names = list(cams.keys())

    for i, (name, cam) in enumerate(cams.items()):
        # retain_graph=True for all except last
        hmap = cam(cls, logits, retain_graph=(i < len(cam_names)-1))[0]

        # resize to 224×224
        hmap = F.interpolate(
            hmap.unsqueeze(0),
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )[0]  # shape (1, H, W)
        maps[name] = hmap[0].detach().cpu()

    # ---------------------------------------------------------
    # Plot image + overlays
    # ---------------------------------------------------------
    if plot:
        import numpy as np
        from PIL import Image

        # original image (HWC uint8) for overlay
        img_np = PreprocessImagenet(pil_img, return_numpy=True)

        fig, axs = plt.subplots(1, len(maps), figsize=(14, 4))

        if len(maps) == 1:
            axs = [axs]

        for ax, (name, cam_map) in zip(axs, maps.items()):
            ax.imshow(img_np)                            # show original image
            ax.imshow(cam_map, cmap="jet", alpha=0.45)   # overlay CAM
            ax.set_title(name)
            ax.axis("off")

        plt.tight_layout()
        plt.show()

    del net

    # ---------------------------------------------------------
    # Optional return of raw maps
    # ---------------------------------------------------------
    if return_maps:
        return maps


def compute_eigencam(
    model,
    img_input,
    target_layer_name,
    device="cpu",
    plot=True,
    return_map=False,
):
    # ---------------------------------------------
    # 1. Accept: PIL image, file path, or Tensor
    # ---------------------------------------------
    if isinstance(img_input, torch.Tensor):
        t = img_input.detach().cpu()
        if t.dim() == 4:
            t = t.squeeze(0)
        pil_img = to_pil_image(t)

    elif isinstance(img_input, str):
        pil_img = Image.open(img_input).convert("RGB")

    else:
        pil_img = img_input.convert("RGB")

    # ---------------------------------------------
    # 2. Preprocess (exact same as your original)
    # ---------------------------------------------
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    x = preprocess(pil_img).unsqueeze(0)

    # ---------------------------------------------
    # 3. Run model on CPU (INT8 requirement)
    # ---------------------------------------------
    device = torch.device("cpu")
    model = model.to(device).eval()
    x = x.to(device)

    # ---------------------------------------------
    # 4. Find target layer (your original logic)
    # ---------------------------------------------
    def find_layer(m, name):
        for n, module in m.named_modules():
            if n == name:
                return module
        raise RuntimeError(f"Layer {name} not found.")

    target_layer = find_layer(model, target_layer_name)

    # ---------------------------------------------
    # 5. Hook activations (your original code)
    # ---------------------------------------------
    activations = {}

    def hook_fn(m, i, o):
        if hasattr(o, "dequantize"):
            o = o.dequantize()
        activations["value"] = o.detach().cpu()

    handle = target_layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        _ = model(x)

    handle.remove()

    # EXACT original behavior: crash with KeyError if missing
    A = activations["value"].squeeze(0)

    # ---------------------------------------------
    # 6. EigenCAM (your exact math)
    # ---------------------------------------------
    C, H, W = A.shape

    F_flat = A.reshape(C, -1).numpy()
    F_flat = F_flat - F_flat.mean()

    _, _, vh = np.linalg.svd(F_flat, full_matrices=False)
    pc = vh[0].reshape(H, W)

    if pc.mean() < 0:
        pc = -pc

    cam = (pc - pc.min()) / (pc.max() - pc.min() + 1e-8)
    cam_resized = cv2.resize(cam, (pil_img.width, pil_img.height))

    heatmap = cv2.applyColorMap(
        (cam_resized * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = (
        0.5 * np.array(pil_img) +
        0.5 * heatmap
    ).astype(np.uint8)

    # ---------------------------------------------
    # 7. Plot
    # ---------------------------------------------
    if plot:
        plt.imshow(overlay)
        plt.title(f"EigenCAM — {target_layer_name}")
        plt.axis("off")
        plt.show()

    # ---------------------------------------------
    # 8. Return map if requested
    # ---------------------------------------------
    if return_map:
        return cam_resized


class CamTap(nn.Module):
    def forward(self, x):
        return x

def insert_cam_tap_fx(model: fx.GraphModule,
                      tap_name="cam_tap",
                      pool_name="avgpool") -> fx.GraphModule:

    if hasattr(model, tap_name):
        return model   # prevent double insertion

    gm = model
    gm.add_submodule(tap_name, CamTap())
    graph = gm.graph

    # find avgpool
    avgpool_node = None
    for node in graph.nodes:
        if node.op == "call_module" and node.target == pool_name:
            avgpool_node = node
            break

    if avgpool_node is None:
        raise RuntimeError("avgpool node not found")

    source = avgpool_node.args[0]

    # insert tap before avgpool
    with graph.inserting_before(avgpool_node):
        tap_node = graph.call_module(tap_name, args=(source,))

    avgpool_node.replace_input_with(source, tap_node)

    graph.lint()
    gm.recompile()
    return gm




def get_target_layer(model, target_layer_name):
    # INT8 CAM-TAP
    if target_layer_name == "cam_tap" and hasattr(model, "cam_tap"):
        return model.cam_tap

    # FP32
    for name, m in model.named_modules():
        if name == target_layer_name:
            return m

    raise RuntimeError(f"Layer '{target_layer_name}' not found.")




def get_activations(model, x, target_layer_name):
    layer = get_target_layer(model, target_layer_name)
    acts = {}

    def hook(m, i, o):
        if hasattr(o, "dequantize"):
            o = o.dequantize()
        acts["A"] = o.detach().cpu()

    h = layer.register_forward_hook(hook)
    with torch.no_grad():
        logits = model(x)
    h.remove()

    if "A" not in acts:
        raise RuntimeError("Hook did not run")

    return acts["A"].squeeze(0), logits
# ======================================================================




def score_cam(model, x, pil_img, target_layer_name, target_class=None, max_channels=256):
    model.eval()
    device = x.device

    A, logits = get_activations(model, x, target_layer_name)
    C, H, W = A.shape

    if target_class is None:
        target_class = logits.argmax(dim=1).item()

    # select channels
    flat = A.view(C, -1)
    var = flat.var(dim=1)
    k = min(max_channels, C)
    idx = torch.topk(var, k).indices
    A_sel = A[idx]
    C_sel = A_sel.shape[0]

    # normalize
    A_min = A_sel.view(C_sel, -1).min(dim=1)[0].view(C_sel,1,1)
    A_max = A_sel.view(C_sel, -1).max(dim=1)[0].view(C_sel,1,1)
    A_norm = (A_sel - A_min) / (A_max - A_min + 1e-8)
    A_norm = torch.clamp(A_norm, 0, 1)

    # upsample
    masks = F.interpolate(
        A_norm.unsqueeze(1).to(device),
        size=x.shape[2:], 
        mode="bilinear",
        align_corners=False
    )

    # score masks
    scores = []
    with torch.no_grad():
        for i in range(C_sel):
            mk = masks[i]
            if mk.std() < 1e-6:
                scores.append(0.0)
                continue
            masked_x = (x.float() * mk.float()).to(device)
            out = model(masked_x)
            scores.append(out[0, target_class].item())

    scores = torch.tensor(scores).clamp(min=0)
    if scores.sum() == 0:
        scores = torch.ones_like(scores)
    alphas = scores / (scores.sum() + 1e-8)

    cam = torch.sum(alphas[:,None,None] * A_sel, dim=0).numpy()
    if cam.mean() < 0:
        cam = -cam

    # overlay
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = cv2.resize(cam, (pil_img.width, pil_img.height))
    cam = (cam * 255).astype(np.uint8)

    heat = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    overlay = (0.5 * np.array(pil_img) + 0.5 * heat).astype(np.uint8)

    return overlay, heat
def compute_scorecam(
    model,
    img_input,
    target_layer_name="layer4.2",
    target_class=None,
    model_label=None,
    max_channels=256,
    device="cpu",
    plot=True,
    return_map=False,
    return_overlay=False,
):
    """
    Score-CAM wrapper using your PreprocessImagenet() and score_cam().
    Returns clean CAM map if return_map=True.
    """

    # ------------------------------------------------------------
    # 1. Convert input to PIL.Image
    # ------------------------------------------------------------
    if isinstance(img_input, torch.Tensor):
        t = img_input.detach().cpu()
        if t.dim() == 4:
            t = t[0]
        pil_img = to_pil_image(t)
    elif isinstance(img_input, str):
        pil_img = Image.open(img_input).convert("RGB")
    else:
        pil_img = img_input.convert("RGB")

    # ------------------------------------------------------------
    # 2. Preprocess
    # ------------------------------------------------------------
    device = torch.device("cpu")
    x = PreprocessImagenet(pil_img).to(device)

    net = model.to(device).eval()

    # ------------------------------------------------------------
    # 3. Score-CAM
    # ------------------------------------------------------------
    overlay, heat = score_cam(
        net,
        x,
        pil_img,
        target_layer_name=target_layer_name,
        target_class=target_class,
        max_channels=max_channels,
    )

    # Convert RGB heatmap → grayscale CAM [0,1]
    cam_map = cv2.cvtColor(heat, cv2.COLOR_RGB2GRAY).astype("float32") / 255.0

    # ------------------------------------------------------------
    # 4. Plot
    # ------------------------------------------------------------
    if plot:
        title = model_label if model_label else net.__class__.__name__
        plt.imshow(overlay)
        plt.title(f"Score-CAM | {title} | layer={target_layer_name}")
        plt.axis("off")
        plt.show()

    # ------------------------------------------------------------
    # 5. Return
    # ------------------------------------------------------------
    if return_map and return_overlay:
        return cam_map, overlay
    elif return_map:
        return cam_map
    elif return_overlay:
        return overlay
    else:
        return None