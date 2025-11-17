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
def evaluate_top1(model, dataloader, device):
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
    model.eval()
    times = []


    with torch.no_grad():
        for i, (x, _) in enumerate(dataloader):
            if i >= warmup:
                break
            x = x.to(device)
            _ = model(x)

  
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
    return os.path.getsize(path) / (1024 * 1024)
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance


def normalize(cam):
    cam = cam.astype(np.float32)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam


def binarize(cam, threshold=0.8):
    return (cam >= threshold).astype(np.float32)


def compare_cams_advanced(cam_a, cam_b):
    """
    cam_a, cam_b: 2D numpy arrays (H, W)
    Rückgabe: dict mit allen Metriken
    """
    cam_a = normalize(cam_a)
    cam_b = normalize(cam_b)

    # SSIM --------------------------------------------------------
    ssim_score = ssim(cam_a, cam_b, data_range=1.0)

    # Pearson-Korrelation -----------------------------------------
    corr = np.corrcoef(cam_a.flatten(), cam_b.flatten())[0, 1]

    # Dice ---------------------------------------------------------
    A = binarize(cam_a)
    B = binarize(cam_b)
    intersection = (A * B).sum()
    dice = (2 * intersection) / (A.sum() + B.sum() + 1e-8)

    # IoU ----------------------------------------------------------
    union = A.sum() + B.sum() - intersection
    iou = intersection / (union + 1e-8)

    # Earth Mover's Distance (1D version on flattened maps) --------
    emd = wasserstein_distance(cam_a.flatten(), cam_b.flatten())

    # Jensen–Shannon Divergence -----------------------------------
    # Heatmaps als Wahrscheinlichkeitsverteilung
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
        shap_a, shap_b: 2D numpy arrays (H, W) of SHAP values
        top_k: percentage of most important regions to compare (0.2 = 20%)

    Returns:
        dict with:
            spearman: rank correlation between importance rankings
            sign_agreement: % of pixels with same sign (+/-)
            topk_overlap: Jaccard overlap of top-k important pixels
    """

    # Flatten superpixel maps (SHAP works over superpixels, not pixel structure)
    a = shap_a.flatten().astype(np.float32)
    b = shap_b.flatten().astype(np.float32)

    # ------------------------------
    # 1. Spearman Rank-Korrelation
    # ------------------------------
    spearman, _ = spearmanr(a, b)

    # ------------------------------
    # 2. Vorzeichen-Übereinstimmung
    # ------------------------------
    sign_a = np.sign(a)
    sign_b = np.sign(b)

    # ignore exact zeros
    mask_nonzero = (sign_a != 0) | (sign_b != 0)
    sign_agree = (sign_a[mask_nonzero] == sign_b[mask_nonzero]).mean()

    # ------------------------------
    # 3. Top-k Overlap (wichtigste Regionen)
    # ------------------------------
    k = int(len(a) * top_k)

    # indices sorted by absolute importance
    idx_a = np.argsort(-np.abs(a))[:k]
    idx_b = np.argsort(-np.abs(b))[:k]

    # Jaccard Overlap
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
    Compare two LIME masks (binary segmentation maps).

    mask_a, mask_b: 2D numpy arrays with 0/1

    Returns:
        dict with:
            iou            – intersection-over-union of relevant segments
            precision      – % of predicted relevant segments that match base
            recall         – % of base relevant segments recovered
            agreement      – % of all pixels with same importance status
    """

    # Ensure binary masks (in case LIME returned segments > 1)
    A = (mask_a > 0).astype(np.uint8)
    B = (mask_b > 0).astype(np.uint8)

    # ------------------------------
    # 1. IoU (Intersection-over-Union)
    # ------------------------------
    intersection = np.logical_and(A, B).sum()
    union = np.logical_or(A, B).sum()
    iou = intersection / (union + 1e-8)

    # ------------------------------
    # 2. Precision / Recall
    # ------------------------------
    # precision: B relevance matches A relevance
    precision = intersection / (B.sum() + 1e-8)

    # recall: A relevance recovered by B
    recall = intersection / (A.sum() + 1e-8)

    # ------------------------------
    # 3. Agreement
    # ------------------------------
    agreement = (A == B).mean()

    return {
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "agreement": float(agreement)
    }


def run_umap_visualization(feats_fp32, feats_qat, feats_ptq, labels_fp32):
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine")

    all_feats = np.concatenate([feats_fp32, feats_qat, feats_ptq], axis=0)
    embedding = reducer.fit_transform(all_feats)

    n_fp32 = len(feats_fp32)
    n_qat  = len(feats_qat)

    emb_fp32 = embedding[:n_fp32]
    emb_qat  = embedding[n_fp32:n_fp32+n_qat]
    emb_ptq  = embedding[n_fp32+n_qat:]

    # Plot
    plt.figure(figsize=(10, 7))
    plt.scatter(emb_fp32[:,0], emb_fp32[:,1], s=5, alpha=0.6, label="FP32")
    plt.scatter(emb_qat[:,0],  emb_qat[:,1],  s=5, alpha=0.6, label="QAT INT8")
    plt.scatter(emb_ptq[:,0],  emb_ptq[:,1],  s=5, alpha=0.6, label="PTQ INT8")

    plt.title("UMAP Feature Space – FP32 vs QAT vs PTQ")
    plt.legend()
    plt.show()


def extract_features2(model, dataloader, device="cpu"):
    model.to(device).eval()
    feats = []
    labels = []

    activation = {}

    def hook_fn(module, input, output):
        """
        Wir ignorieren 'output' (QUInt8-Problem) und
        berechnen unser eigenes avgpool-Ergebnis aus 'input[0]'.
        """
        x = input[0]  # Tensor vor avgpool, Shape: (B, C, H, W)

        # Falls quantisiert: dequantisieren über int_repr/scale/zero_point
        if x.is_quantized:
            int_repr = x.int_repr().float()      # (B, C, H, W) auf CPU
            scale = x.q_scale()
            zero_point = x.q_zero_point()
            x = (int_repr - zero_point) * scale  # float32 auf CPU
        else:
            x = x.detach().float().cpu()

        # Jetzt selbst avgpool auf float ausführen → (B, C, 1, 1)
        avg = F.adaptive_avg_pool2d(x, (1, 1))

        # (B, C, 1, 1) → (B, C)
        activation["feat"] = avg.reshape(avg.size(0), -1).numpy()

    # Hook: immer auf avgpool (bei FP32 und quantisierten Modellen vorhanden)
    handle = model.avgpool.register_forward_hook(hook_fn)

    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Extracting", leave=False):
            x = x.to(device)
            _ = model(x)  # Hook füllt activation["feat"]

            batch_feats = activation["feat"]          # (B, C)
            feats.append(batch_feats)
            labels.append(y.numpy())

    handle.remove()

    feats = np.concatenate(feats, axis=0)
    labels = np.concatenate(labels, axis=0)
    return feats, labels