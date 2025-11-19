# ------------------------------------------------------------
# Imports 
# ------------------------------------------------------------

import os
import time
import psutil
import torch
import numpy as np
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm
import umap
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
import torch.nn.functional as F


def evaluate_top1(model, dataloader, device):
    """
    Compute Top-1 accuracy over the given dataloader.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def measure_latency(model, dataloader, device, warmup=20, runs=200):
    """
    Measure average inference latency in milliseconds.
    Warmup iterations are skipped from measurement.
    """
    model.eval()
    times = []

    # Warmup runs
    with torch.no_grad():
        for i, (x, _) in enumerate(dataloader):
            if i >= warmup:
                break
            x = x.to(device)
            _ = model(x)

    # Timed runs
    with torch.no_grad():
        for i, (x, _) in enumerate(tqdm(dataloader, total=runs, desc="Measuring Latency", leave=False)):
            if i >= runs:
                break

            x = x.to(device)
            start = time.time()
            _ = model(x)
            end = time.time()

            times.append((end - start) * 1000)  # ms

    return sum(times) / len(times)


def measure_throughput(model, dataloader, device, runs=200):
    """
    Measure model throughput (images per second).
    """
    model.eval()
    count = 0
    start = time.time()

    with torch.no_grad():
        for i, (x, _) in enumerate(tqdm(dataloader, total=runs, desc="Measuring Throughput", leave=False)):
            if i >= runs:
                break
            x = x.to(device)
            _ = model(x)
            count += x.size(0)

    duration = time.time() - start
    return count / duration


def file_size_mb(path):
    """Return file size in megabytes."""
    return os.path.getsize(path) / (1024 * 1024)


def normalize(cam):
    """
    Normalize CAM to [0,1].
    """
    cam = cam.astype(np.float32)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam


def binarize(cam, threshold=0.8):
    """
    Binarize heatmap using fixed threshold.
    """
    return (cam >= threshold).astype(np.float32)


def compare_cams_advanced(cam_a, cam_b):
    """
    Compare two CAM heatmaps using several similarity metrics.

    cam_a, cam_b: 2D numpy arrays (H, W)

    Returns:
        dict containing:
            ssim  – structural similarity
            corr  – Pearson correlation
            dice  – Dice similarity of binarized maps
            iou   – IoU of binarized maps
            emd   – Earth Mover's Distance (1D on flattened maps)
            js    – Jensen–Shannon divergence
    """
    cam_a = normalize(cam_a)
    cam_b = normalize(cam_b)

    # SSIM --------------------------------------------------------
    ssim_score = ssim(cam_a, cam_b, data_range=1.0)

    # Pearson correlation -----------------------------------------
    corr = np.corrcoef(cam_a.flatten(), cam_b.flatten())[0, 1]

    # Dice ---------------------------------------------------------
    A = binarize(cam_a)
    B = binarize(cam_b)
    intersection = (A * B).sum()
    dice = (2 * intersection) / (A.sum() + B.sum() + 1e-8)

    # IoU ----------------------------------------------------------
    union = A.sum() + B.sum() - intersection
    iou = intersection / (union + 1e-8)

    # Earth Mover's Distance --------------------------------------
    emd = wasserstein_distance(cam_a.flatten(), cam_b.flatten())

    # Jensen–Shannon divergence -----------------------------------
    pa = cam_a.flatten() + 1e-12
    pb = cam_b.flatten() + 1e-12
    pa /= pa.sum()
    pb /= pb.sum()
    js = jensenshannon(pa, pb)

    return {
        "ssim": float(ssim_score),
        "corr": float(corr),
        "dice": float(dice),
        "iou": float(iou),
        "emd": float(emd),
        "js": float(js)
    }


def compare_shap(shap_a, shap_b, top_k=0.2):
    """
    Compare two SHAP explanations (superpixel importance maps).

    Inputs:
        shap_a, shap_b: 2D numpy arrays (H, W)
        top_k: fraction of most important pixels to compare

    Returns:
        dict with:
            spearman        – Spearman rank correlation
            sign_agreement  – percentage of same sign (+/-)
            topk_overlap    – Jaccard overlap of top-k pixels
    """

    a = shap_a.flatten().astype(np.float32)
    b = shap_b.flatten().astype(np.float32)

    # Spearman ----------------------------------------------------
    spearman, _ = spearmanr(a, b)

    # Sign agreement ----------------------------------------------
    sign_a = np.sign(a)
    sign_b = np.sign(b)

    mask_nonzero = (sign_a != 0) | (sign_b != 0)
    sign_agree = (sign_a[mask_nonzero] == sign_b[mask_nonzero]).mean()

    # Top-k overlap ------------------------------------------------
    k = int(len(a) * top_k)

    idx_a = np.argsort(-np.abs(a))[:k]
    idx_b = np.argsort(-np.abs(b))[:k]

    set_a = set(idx_a.tolist())
    set_b = set(idx_b.tolist())
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    topk_overlap = intersection / (union + 1e-8)

    return {
        "spearman": float(spearman),
        "sign_agreement": float(sign_agree),
        "topk_overlap": float(topk_overlap)
    }


def compare_lime(mask_a, mask_b):
    """
    Compare two LIME masks (binary relevance maps).

    mask_a, mask_b: 2D arrays with values 0/1

    Returns:
        dict with:
            iou         – intersection-over-union
            precision   – predicted relevant pixels that match the base mask
            recall      – recovered relevant pixels from base mask
            agreement   – pixel-wise agreement
    """

    A = (mask_a > 0).astype(np.uint8)
    B = (mask_b > 0).astype(np.uint8)

    intersection = np.logical_and(A, B).sum()
    union = np.logical_or(A, B).sum()
    iou = intersection / (union + 1e-8)

    precision = intersection / (B.sum() + 1e-8)
    recall = intersection / (A.sum() + 1e-8)

    agreement = (A == B).mean()

    return {
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "agreement": float(agreement)
    }


def run_umap_visualization(feats_fp32, feats_qat, feats_ptq, labels_fp32):
    """
    Compute and plot UMAP embedding to compare feature spaces
    of FP32, QAT INT8, and PTQ INT8 models.
    """
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine")

    all_feats = np.concatenate([feats_fp32, feats_qat, feats_ptq], axis=0)
    embedding = reducer.fit_transform(all_feats)

    n_fp32 = len(feats_fp32)
    n_qat  = len(feats_qat)

    emb_fp32 = embedding[:n_fp32]
    emb_qat  = embedding[n_fp32:n_fp32+n_qat]
    emb_ptq  = embedding[n_fp32+n_qat:]

    plt.figure(figsize=(10, 7))
    plt.scatter(emb_fp32[:,0], emb_fp32[:,1], s=5, alpha=0.6, label="FP32")
    plt.scatter(emb_qat[:,0],  emb_qat[:,1],  s=5, alpha=0.6, label="QAT INT8")
    plt.scatter(emb_ptq[:,0],  emb_ptq[:,1],  s=5, alpha=0.6, label="PTQ INT8")

    plt.title("UMAP Feature Space – FP32 vs QAT vs PTQ")
    plt.legend()
    plt.show()


def extract_features2(model, dataloader, device="cpu"):
    """
    Extract final feature vectors from a model by hooking into its avgpool layer.
    For quantized models, input to avgpool is manually dequantized.
    """
    model.to(device).eval()
    feats = []
    labels = []

    activation = {}

    def hook_fn(module, input, output):
        """
        Ignore 'output' (QUInt8 issue) and manually compute avgpool
        using the float version of input[0].
        """
        x = input[0]  # (B, C, H, W) before avgpool

        if x.is_quantized:
            int_repr = x.int_repr().float()
            scale = x.q_scale()
            zero_point = x.q_zero_point()
            x = (int_repr - zero_point) * scale
        else:
            x = x.detach().float().cpu()

        avg = F.adaptive_avg_pool2d(x, (1, 1))
        activation["feat"] = avg.reshape(avg.size(0), -1).numpy()

    handle = model.avgpool.register_forward_hook(hook_fn)

    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Extracting", leave=False):
            x = x.to(device)
            _ = model(x)
            batch_feats = activation["feat"]
            feats.append(batch_feats)
            labels.append(y.numpy())

    handle.remove()

    feats = np.concatenate(feats, axis=0)
    labels = np.concatenate(labels, axis=0)
    return feats, labels


def compute_class_shifts(feats_fp32, feats_qat, feats_ptq, labels):
    """
    For each class:
      - compute the mean L2 distance between FP32 and QAT features
      - compute the mean L2 distance between FP32 and PTQ features
      - compute centroid L2 distances as well 

    Returns a dict: {class_id: {...metrics...}}
    """
    classes = np.unique(labels)
    results = {}

    # global check
    assert feats_fp32.shape == feats_qat.shape == feats_ptq.shape
    assert len(labels) == feats_fp32.shape[0]

    for c in classes:
        idx = (labels == c)

        f32_c = feats_fp32[idx]
        qat_c = feats_qat[idx]
        ptq_c = feats_ptq[idx]

        # 1) sample-wise mean L2 distance to FP32
        d_qat_samples = np.linalg.norm(f32_c - qat_c, axis=1).mean()
        d_ptq_samples = np.linalg.norm(f32_c - ptq_c, axis=1).mean()

        # 2) centroid distances 
        mu_f32 = f32_c.mean(axis=0)
        mu_qat = qat_c.mean(axis=0)
        mu_ptq = ptq_c.mean(axis=0)

        d_qat_centroid = np.linalg.norm(mu_f32 - mu_qat)
        d_ptq_centroid = np.linalg.norm(mu_f32 - mu_ptq)

        results[int(c)] = {
            "mean_L2_FP32_vs_QAT": float(d_qat_samples),
            "mean_L2_FP32_vs_PTQ": float(d_ptq_samples),
            "centroid_L2_FP32_vs_QAT": float(d_qat_centroid),
            "centroid_L2_FP32_vs_PTQ": float(d_ptq_centroid),
            "num_samples": int(idx.sum()),
        }

    return results