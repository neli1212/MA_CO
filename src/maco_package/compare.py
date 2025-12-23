# =====================================================================
# Imports
# =====================================================================
import os
import numpy as np
import torch
from scipy.stats import spearmanr, wasserstein_distance
from scipy.spatial.distance import jensenshannon
from skimage.metrics import structural_similarity as ssim
from maco_package.xai import (
    compute_scorecam, compute_eigencam, compute_shap, compute_lime
)
from maco_package.utils import load_model_generic
from tqdm import tqdm
# =====================================================================
# 1. COMPREHENSIVE METRICS LIBRARY
# =====================================================================

def normalize_map(cam):
    """Normalize 0-1 for fair comparison."""
    cam = cam.astype(np.float32)
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

def binarize_map(cam, threshold=0.3):
    """Binarize heatmap for overlap metrics."""
    return (cam > threshold).astype(np.float32)

def compare_heatmaps(map_a, map_b):
    """
    Used for: ScoreCAM, EigenCAM
    Metrics: SSIM, Corr, Dice, IoU, EMD, JS-Div
    """
    # Resize B to match A
    if map_a.shape != map_b.shape:
        import cv2
        map_b = cv2.resize(map_b, (map_a.shape[1], map_a.shape[0]))

    a = normalize_map(map_a)
    b = normalize_map(map_b)
    
    # Continuous Metrics
    score_ssim = ssim(a, b, data_range=1.0)
    score_corr = np.corrcoef(a.flatten(), b.flatten())[0, 1]
    score_emd  = wasserstein_distance(a.flatten(), b.flatten())
    
    # JS Divergence (Treat as probability dist)
    pa = a.flatten() + 1e-12; pa /= pa.sum()
    pb = b.flatten() + 1e-12; pb /= pb.sum()
    score_js = jensenshannon(pa, pb)

    # Binary Metrics (Dice / IoU)
    bin_a = binarize_map(a)
    bin_b = binarize_map(b)
    intersection = (bin_a * bin_b).sum()
    union = (bin_a + bin_b).sum() - intersection
    
    score_dice = (2 * intersection) / (bin_a.sum() + bin_b.sum() + 1e-8)
    score_iou  = intersection / (union + 1e-8)

    return {
        "ssim": float(score_ssim),
        "corr": float(score_corr),
        "emd":  float(score_emd),
        "js":   float(score_js),
        "dice": float(score_dice),
        "iou":  float(score_iou)
    }

def compare_attributions(attr_a, attr_b, top_k_percent=0.2):
    """
    Used for: SHAP
    Metrics: Spearman, Sign Agreement, Top-K Overlap
    """
    a = attr_a.flatten()
    b = attr_b.flatten()

    # Spearman
    score_spearman, _ = spearmanr(a, b)
    if np.isnan(score_spearman): score_spearman = 0.0

    # Sign Agreement
    sign_a = np.sign(a)
    sign_b = np.sign(b)
    # Compare only where values are non-zero to avoid noise
    mask = (sign_a != 0) | (sign_b != 0)
    if mask.sum() > 0:
        score_sign = (sign_a[mask] == sign_b[mask]).mean()
    else:
        score_sign = 1.0

    # Top-K Overlap
    k = max(1, int(len(a) * top_k_percent))
    idx_a = set(np.argsort(-np.abs(a))[:k])
    idx_b = set(np.argsort(-np.abs(b))[:k])
    score_overlap = len(idx_a & idx_b) / k

    return {
        "spearman": float(score_spearman),
        "sign_agreement": float(score_sign),
        "topk_overlap": float(score_overlap)
    }

def compare_masks(mask_a, mask_b):
    """
    Used for: LIME
    Metrics: IoU, Precision, Recall, Agreement
    """
    # LIME masks are usually segmentation indices, treat as binary overlap of the region
    a = (mask_a > 0).astype(np.uint8)
    b = (mask_b > 0).astype(np.uint8)

    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    
    iou = intersection / (union + 1e-8)
    precision = intersection / (b.sum() + 1e-8) 
    recall = intersection / (a.sum() + 1e-8)    
    agreement = (a == b).mean()

    return {
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "agreement": float(agreement)
    }
import numpy as np
from scipy.spatial.distance import euclidean

def compare_umap_embeddings(emb_fp32, emb_var, labels):
    """
    Compares the FP32 embedding to a variant (INT8/FP16) embedding.
    Returns the average drift distance per class.
    """
    classes = np.unique(labels)
    drifts = []

    # 1. Normalize embeddings 
    e32 = (emb_fp32 - emb_fp32.min()) / (emb_fp32.max() - emb_fp32.min())
    evar = (emb_var - emb_var.min()) / (emb_var.max() - emb_var.min())

    for c in classes:
        mask = (labels == c)
        
        if not np.any(mask): continue
        
        # Calculate the 'Center of Gravity' for this class in both maps
        centroid_32 = e32[mask].mean(axis=0)
        centroid_var = evar[mask].mean(axis=0)
        
        # Calculate Euclidean Distance between the two centers
        drift = euclidean(centroid_32, centroid_var)
        drifts.append(drift)

    avg_drift = np.mean(drifts)

    std_32 = np.std(e32)
    std_var = np.std(evar)
    density_change = std_var / std_32

    return {
        "avg_logic_drift": float(avg_drift),
        "density_ratio": float(density_change)
    }
# =====================================================================
# 2. COMPARATOR PIPELINE CLASS 
# =====================================================================
class XAIComparator:
    def __init__(self, base_path, num_classes=100):
        self.base_path = base_path
        self.num_classes = num_classes
        self.results = {} 

    def load_model(self, arch, tag, suffix, device):
        path = os.path.join(self.base_path, arch, f"{arch}_{suffix}.pth")
        
        if not os.path.exists(path):
            path = os.path.join(self.base_path, arch, f"{arch}_ft1_{suffix}.pth")
            
        if not os.path.exists(path):
            return None
            
        try:
            from maco_package.utils import load_model_generic
            return load_model_generic(f"{arch}-{tag}", path, self.num_classes, device=device)
        except Exception as e:
            print(f"      ⚠️ Load Failed {tag}: {e}")
            return None

    def run(self, arch, sample_images, methods, val_loader, epochs_ft=1, epochs_qat=3):
        """
        Main execution loop for XAI comparison.
        Now includes Global Logic Drift via UMAP.
        """
        from maco_package.xai import get_model_features, compute_umap
        from sklearn.preprocessing import StandardScaler
        import umap

        print(f"\n⚡ Processing Architecture: {arch.upper()}")
        
        # 1. Initialize results
        self.results[arch] = {} 

        # 2. Load Baseline (The "Golden" Model)
        baseline_suffix = f"ft{epochs_ft}_finetuned"
        model_fp32 = self.load_model(arch, "fp32", baseline_suffix, "cuda")
        
        if model_fp32 is None:
            print(f"   ❌ Baseline {baseline_suffix} missing. Skipping {arch}.")
            return

        # --- UMAP GLOBAL LOGIC DRIFT (FP32 BASELINE) ---
        print("   Extracting Global Features for UMAP (FP32 Baseline)...")
        # Extract features for UMAP (Max 1000 for speed)
        feats_32, lbls_32 = get_model_features(model_fp32, val_loader, device="cuda", max_samples=None)
        
        # Fit the Baseline projection space
        scaler = StandardScaler()
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        
        feats_32_scaled = scaler.fit_transform(feats_32)
        emb_32 = reducer.fit_transform(feats_32_scaled)

        # 3. Determine Layer Mapping for Baseline
        layer_map = {
            "resnet50": "layer4.2",
            "vgg16": "model.features.28",
            "mobilenet_v2": "features.18.0",
            "efficientnet_b0": "features.8",
            "densenet121": "features.norm5"
        }
        layer_fp32 = layer_map.get(arch, "features")

        # 4. Generate Baseline Maps for per-image methods
        baselines = {m: [] for m in methods}
        print("   Generating Baseline Explanations (FP32)...")
        
        for img in tqdm(sample_images, desc="Baselines", leave=False):
            try:
                if "scorecam" in methods:
                    res = compute_scorecam(model_fp32, img, layer_fp32, device="cuda", plot=False, return_map=True)
                    baselines["scorecam"].append(res)
                if "eigencam" in methods:
                    res = compute_eigencam(model_fp32, img, layer_fp32, device="cuda", plot=False, return_map=True)
                    baselines["eigencam"].append(res)
                if "shap" in methods:
                    res = compute_shap(model_fp32, img, max_evals=100, device="cuda", plot=False, return_values=True)
                    baselines["shap"].append(res)
                if "lime" in methods:
                    res = compute_lime(model_fp32, img, num_samples=250, device="cuda", plot=False, return_mask=True)
                    baselines["lime"].append(res)
            except Exception as e:
                print(f"   ⚠️ Baseline gen failed: {e}")
                for m in methods:
                    if len(baselines[m]) < (len(baselines["scorecam"]) if "scorecam" in methods else 1):
                        baselines[m].append(None)

        # 5. Define Variants
        variants = [
            ("fp16",  f"ft{epochs_ft}_fp16", "cuda"),
            ("bf16",  f"ft{epochs_ft}_bf16", "cuda"),
            ("int8",  f"ft{epochs_ft}_qat{epochs_qat}_int8", "cpu"),
            ("ptqfx", f"ft{epochs_ft}_ptqfx", "cpu")
        ]

        # 6. Iterate Variants & Compare
        for tag, suffix, dev in variants:
            print(f"\n   👉 Comparing Baseline vs {tag.upper()} ({dev})")
            
            model_var = self.load_model(arch, tag, suffix, dev)
            if model_var is None: continue

            self.results[arch][tag] = {m: {} for m in methods}

            # --- UMAP GLOBAL LOGIC DRIFT--
            try:
                print(f"      [UMAP] Extracting Variant Features...", end="\r")
                feats_var, _ = get_model_features(model_var, val_loader, device=dev, max_samples=None)
                

                feats_var_scaled = scaler.transform(feats_var)
                emb_var = reducer.transform(feats_var_scaled)

                from maco_package.compare import compare_umap_embeddings
                self.results[arch][tag]["umap"] = compare_umap_embeddings(emb_32, emb_var, lbls_32)
                print(f"      [UMAP] Logic Drift: {self.results[arch][tag]['umap']['avg_logic_drift']:.4f}")
            except Exception as e:
                print(f"      ⚠️ UMAP Failed for {tag}: {e}")

            # --- PER-IMAGE LOOP (CAM, LIME, SHAP) ---
            layer_var = layer_fp32
            if arch == "resnet50" and tag in ["int8", "ptqfx"]: layer_var = "layer4.2.conv3"
            if arch == "efficientnet_b0" and tag == ["ptqfx"]: layer_var = "features"

            for i, img in enumerate(tqdm(sample_images, desc=f"Processing {tag}", unit="img")):
                try:
                    if "scorecam" in methods and baselines["scorecam"][i] is not None:
                        res = compute_scorecam(model_var, img, layer_var, device=dev, plot=False, return_map=True)
                        if res is not None:
                            metrics = compare_heatmaps(baselines["scorecam"][i], res)
                            for k, v in metrics.items(): self.results[arch][tag]["scorecam"].setdefault(k, []).append(v)

                    if "eigencam" in methods and baselines["eigencam"][i] is not None:
                        res = compute_eigencam(model_var, img, layer_var, device=dev, plot=False, return_map=True)
                        if res is not None:
                            metrics = compare_heatmaps(baselines["eigencam"][i], res)
                            for k, v in metrics.items(): self.results[arch][tag]["eigencam"].setdefault(k, []).append(v)

                    if "shap" in methods and baselines["shap"][i] is not None:
                        res = compute_shap(model_var, img, max_evals=100, device=dev, plot=False, return_values=True)
                        if res is not None:
                            metrics = compare_attributions(baselines["shap"][i], res)
                            for k, v in metrics.items(): self.results[arch][tag]["shap"].setdefault(k, []).append(v)
                    
                    if "lime" in methods and baselines["lime"][i] is not None:
                        res = compute_lime(model_var, img, num_samples=250, device=dev, plot=False, return_mask=True)
                        if res is not None:
                            metrics = compare_masks(baselines["lime"][i], res)
                            for k, v in metrics.items(): self.results[arch][tag]["lime"].setdefault(k, []).append(v)

                except Exception:
                    continue
            
            del model_var
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        del model_fp32
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def generate_report(self, filename="xai_report.txt"):
        with open(filename, "w") as f:
            f.write("XAI ROBUSTNESS & LOGIC DRIFT REPORT\n")
            f.write("==================================\n")
            
            for arch, variants in self.results.items():
                f.write(f"\nArchitecture: {arch.upper()}\n")
                f.write("="*80 + "\n")
                
                for tag, methods_data in variants.items():
                    f.write(f"  Variant: {tag}\n")
                    f.write(f"  {'-'*70}\n")

                    # Print UMAP Drift Metrics
                    if "umap" in methods_data:
                        u = methods_data["umap"]
                        f.write(f"    [LOGIC DRIFT (UMAP)]\n")
                        f.write(f"      Avg Logic Drift: {u['avg_logic_drift']:.4f}\n")
                        f.write(f"      Density Ratio:   {u['density_ratio']:.4f}\n\n")
                    
                    # Print ScoreCAM / EigenCAM
                    for m_name in ["scorecam", "eigencam"]:
                        if m_name in methods_data and methods_data[m_name]:
                            metrics = methods_data[m_name]
                            f.write(f"    [{m_name.upper()}]\n")
                            row1 = f"      SSIM: {np.mean(metrics.get('ssim', [0])):.3f} | "
                            row1 += f"Corr: {np.mean(metrics.get('corr', [0])):.3f} | "
                            row1 += f"JS: {np.mean(metrics.get('js', [0])):.3f}\n"
                            row2 = f"      Dice: {np.mean(metrics.get('dice', [0])):.3f} | "
                            row2 += f"IoU:  {np.mean(metrics.get('iou', [0])):.3f}  | "
                            row2 += f"EMD: {np.mean(metrics.get('emd', [0])):.3f}\n"
                            f.write(row1 + row2 + "\n")

                    # Print SHAP
                    if "shap" in methods_data and methods_data["shap"]:
                        metrics = methods_data["shap"]
                        f.write(f"    [SHAP]\n")
                        row = f"      Spearman: {np.mean(metrics.get('spearman', [0])):.3f} | "
                        row += f"Sign Agree: {np.mean(metrics.get('sign_agreement', [0])):.3f} | "
                        row += f"Top-K: {np.mean(metrics.get('topk_overlap', [0])):.3f}\n"
                        f.write(row + "\n")

                    # Print LIME
                    if "lime" in methods_data and methods_data["lime"]:
                        metrics = methods_data["lime"]
                        f.write(f"    [LIME]\n")
                        row = f"      IoU: {np.mean(metrics.get('iou', [0])):.3f} | "
                        row += f"Agreement: {np.mean(metrics.get('agreement', [0])):.3f} | "
                        row += f"Prec/Rec: {np.mean(metrics.get('precision', [0])):.2f}/{np.mean(metrics.get('recall', [0])):.2f}\n"
                        f.write(row + "\n")
                        
        print(f"✅ Report saved to {filename}")