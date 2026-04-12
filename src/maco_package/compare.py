# =====================================================================
# Imports
# =====================================================================
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from scipy.stats import spearmanr, wasserstein_distance
from skimage.metrics import structural_similarity as ssim
from maco_package.xai import (
    compute_scorecam,
    compute_eigencam,
    compute_shap,
    compute_lime,
    compute_umap
)
from maco_package.utils import load_model_generic
from tqdm import tqdm
from scipy.spatial import procrustes
from sklearn.manifold import trustworthiness

# =====================================================================
# 1. METRICS 
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
    Metrics: SSIM, Pearson Correlation, Wasserstein Distance (EMD), IoU
    """
    if map_a.shape != map_b.shape:
        import cv2
        map_b = cv2.resize(map_b, (map_a.shape[1], map_a.shape[0]))
    a = normalize_map(map_a)
    b = normalize_map(map_b)

    # Continuous Metrics
    score_ssim = ssim(a, b, data_range=1.0)
    score_corr = np.corrcoef(a.flatten(), b.flatten())[0, 1]
    score_emd = wasserstein_distance(a.flatten(), b.flatten())

    # Binary Metrics (IoU)
    bin_a = binarize_map(a)
    bin_b = binarize_map(b)
    intersection = (bin_a * bin_b).sum()
    union = (bin_a + bin_b).sum() - intersection
    score_iou = intersection / (union + 1e-8)

    return {
        "ssim": float(score_ssim),
        "corr": float(score_corr),
        "emd": float(score_emd),
        "iou": float(score_iou)
    }

def compare_attributions_signed_rank(attr_a, attr_b, top_k_percent=0.2):
    a = attr_a.flatten()
    b = attr_b.flatten()
    # Spearman
    score_spearman, _ = spearmanr(a, b)
    if np.isnan(score_spearman):
        score_spearman = 0.0
    # top k
    k = max(1, int(len(a) * top_k_percent))
    
    idx_a = np.argsort(-np.abs(a))[:k]
    idx_b = np.argsort(-np.abs(b))[:k]
    
    matches = 0
    for i in range(k):
        if idx_a[i] == idx_b[i]:
            if np.sign(a[idx_a[i]]) == np.sign(b[idx_b[i]]):
                matches += 1
                
    score_signed_rank = matches / k

    return {
        "spearman": float(score_spearman),
        "topk_overlap": float(score_signed_rank)
    }

def compare_masks(mask_a, mask_b):
    """
    Used for: LIME
    Metrics: IoU
    """
    a = (mask_a > 0).astype(np.uint8)
    b = (mask_b > 0).astype(np.uint8)
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    iou = intersection / (union + 1e-8)
    return {
        "iou": float(iou)
    }

def compare_umap_embeddings(emb_fp32, emb_var, labels):
    """
    Used for: UMAO
    Metrics: Procrustes disparity, Trustworthiness, Mean Centroid Displacement.
    """

    # Procrustes 
    mtx1, mtx2, disparity = procrustes(emb_fp32, emb_var)

    # Trustworthiness 
    t_score = trustworthiness(emb_fp32, emb_var, n_neighbors=15)

    # Mean Centroid Displacement
    unique_labels = np.unique(labels)
    drifts = []
    for label in unique_labels:
        mask = (labels == label)
        if np.any(mask):
            c32 = mtx1[mask].mean(axis=0)
            cvar = mtx2[mask].mean(axis=0)
            drifts.append(np.linalg.norm(c32 - cvar))
    
    mean_drift = float(np.mean(drifts)) if drifts else 0.0

    return {
        "procrustes_disparity": float(disparity),
        "trustworthiness": float(t_score),
        "centroid_drift": mean_drift
    }


# =====================================================================
# 2. COMPARATOR PIPELINE CLASS 
# =====================================================================

class XAIComparator:
    def __init__(self, base_path, num_classes=100):
        self.base_path = base_path
        self.num_classes = num_classes
        self.results = {}          # stores aggregated sums and counts

    def load_model(self, arch, tag, suffix, device):
        arch_dir = os.path.join(self.base_path, arch)
        if not os.path.exists(arch_dir):
            print(f" ⚠️ Directory missing: {arch_dir}")
            return None
        # Search for the file containing the suffix
        all_files = os.listdir(arch_dir)
        matches = [f for f in all_files if suffix in f and f.endswith(".pth")]
        if not matches:
            print(f" ⚠️ No file found for {arch} with suffix: {suffix}")
            return None
        # Use the first valid match
        path = os.path.join(arch_dir, matches[0])
        try:
            from maco_package.utils import load_model_generic
            return load_model_generic(f"{arch}-{tag}", path, self.num_classes, device=device)
        except Exception as e:
            print(f" ⚠️ Load Failed {tag}: {e}")
            return None

    def run(self, arch, val_loader, num_images, methods, epochs_ft=1, epochs_qat=3, max_samples_umap=None):
        """
        Main execution loop for XAI comparison – processes images in batches.
        """
        from maco_package.xai import get_model_features
        from sklearn.preprocessing import StandardScaler
        import umap

        print("\n" + "="*80)
        print(f"[ARCH] {arch.upper()} – XAI comparison started")
        print("="*80)

        self.results[arch] = {}

        # Load Baseline
        model_fp32 = self.load_model(arch, "fp32", "fp32_polished", "cuda")
        model_fp32_cpu = self.load_model(arch, "fp32", "fp32_polished", "cpu") if "eigencam" in methods else None
        if model_fp32 is None:
            print(f" ❌ Baseline fp32_polished missing. Skipping {arch}.")
            return
        print(" ✓ Baseline model loaded.")

        # Determine Layer Mapping for Baseline and Variants
        layer_map = {
            "resnet50": {
                "fp32": "layer4.2.relu2", 
                "fp16": "layer4.2.relu2", 
                "bf16": "layer4.2.relu2",
                "int8": "layer4.2.conv3", 
                "ptqfx": "layer4.2.conv3"
            },
            "vgg16": {
                "fp32": "features.29", 
                "fp16": "features.29", 
                "bf16": "features.29",
                "int8": "features.28", 
                "ptqfx": "features.28"
            },
            "mobilenet_v2": {
                "fp32": "features.18.2", 
                "fp16": "features.18.2", 
                "bf16": "features.18.2",
                "int8": "features.18.0", 
                "ptqfx": "features.18.0"
            },
            "densenet121": {
                "fp32": "features.denseblock4.denselayer16.conv2", 
                "fp16": "features.denseblock4.denselayer16.conv2", 
                "bf16": "features.denseblock4.denselayer16.conv2",
                "int8": "features.norm5", 
                "ptqfx": "features.norm5"
            },
            "googlenet": {
                "fp32": "inception5b.branch4.1.relu", 
                "fp16": "inception5b.branch4.1.relu", 
                "bf16": "inception5b.branch4.1.relu",
                "int8": "inception5b.branch4.1.conv", 
                "ptqfx": "inception5b.branch4.1.conv"
            },
            "mnasnet": {
                "fp32": "layers.16", 
                "fp16": "layers.16", 
                "bf16": "layers.16",
                "int8": "layers.14", 
                "ptqfx": "layers.14"
            }
        }
        layer_fp32 = layer_map.get(arch, {}).get("fp32", "features")

        # Variant definitions
        variants = [
            ("fp16", "p16", "cuda"),
            ("bf16", "bf16", "cuda"),
            ("int8", "int8", "cpu"),
            ("ptqfx", "ptqfx", "cpu")
        ]
        variant_tags = [tag for tag, _, _ in variants]
        print(f"[INFO] Variants scheduled: {variant_tags}")

        # Pre-load all variant models
        variant_models = {}
        for tag, suffix, dev in variants:
            print(f"[LOAD] Model variant={tag} device={dev}")
            model_var = self.load_model(arch, tag, suffix, dev)
            if model_var is not None:
                variant_models[tag] = (model_var, dev)
            else:
                print(f" ⚠️ Could not load {tag}, skipping.")

        # ========== UMAP  ==========
        if "umap" in methods:
            print("\n" + "="*60)
            print("[UMAP] FEATURE EXTRACTION ")
            print("="*60)
            print("[UMAP] Extracting baseline feature embeddings")
            # Baseline features
            feats_32, labels_32 = get_model_features(model_fp32, val_loader, device="cuda", max_samples=max_samples_umap)
            emb_32 = compute_umap(feats_32, labels_32)

            for tag, (model_var, dev) in variant_models.items():
                print(f"[UMAP] variant={tag} device={dev}")
                try:
                    feats_var, labels_var = get_model_features(model_var, val_loader, device=dev, max_samples=max_samples_umap)
                    emb_var = compute_umap(feats_var, labels_var)
                    umap_metrics = compare_umap_embeddings(emb_32, emb_var, labels_32)
                    self.results[arch].setdefault(tag, {})["umap"] = umap_metrics
                except Exception as e:
                    print(f" ⚠️ UMAP failed for {tag}: {e}")
        else:
            print(" Skipping UMAP (not requested).")

        # ========== Per-image XAI methods ==========
        per_image_methods = [m for m in methods if m in ["scorecam", "eigencam", "shap", "lime"]]
        if per_image_methods:
            print("\n" + "="*60)
            print(f" PER-IMAGE XAI METHODS: {per_image_methods}")
            print("="*60)
            print(f"[XAI] image comparisons started | max_images={num_images} | batch_size={val_loader.batch_size}")

            for tag in variant_models.keys():
                self.results[arch].setdefault(tag, {})
                for m in per_image_methods:
                    self.results[arch][tag].setdefault(m, {})

            processed = 0
            pbar = tqdm(
                total=num_images,
                desc=f"{arch}",
                unit="img",
                dynamic_ncols=True
            )

            for batch_idx, (images, labels) in enumerate(val_loader):
                if processed >= num_images:
                    break

                batch_size = images.size(0)
                images_cpu = images.cpu()

                for i in range(batch_size):
                    if processed >= num_images:
                        break

                    img_tensor = images_cpu[i]

                    # --- Compute baseline explanations ---
                    baseline_explanations = {}
                    try:
                        if "scorecam" in per_image_methods:
                            res = compute_scorecam(model_fp32, img_tensor, layer_fp32, device="cuda", plot=False, return_map=True)
                            baseline_explanations["scorecam"] = res
                        if "eigencam" in per_image_methods:
                            baseline_explanations["eigencam_cuda"] = compute_eigencam(model_fp32, img_tensor, layer_fp32, device="cuda", plot=False, return_map=True)
                            baseline_explanations["eigencam_cpu"] = compute_eigencam(model_fp32_cpu, img_tensor, layer_fp32, device="cpu", plot=False, return_map=True)
                        if "shap" in per_image_methods:
                            res = compute_shap(model_fp32, img_tensor, max_evals=100, device="cuda", plot=False, return_values=True)
                            baseline_explanations["shap"] = res
                        if "lime" in per_image_methods:
                            res = compute_lime(model_fp32, img_tensor, num_samples=100, device="cuda", plot=False, return_mask=True)
                            baseline_explanations["lime"] = res
                    except Exception as e:
                        print(f" ⚠️ Baseline generation failed for image {processed}: {e}")
                        processed += 1
                        pbar.update(1)
                        continue

                    # --- For each variant, compute and compare ---
                    for tag, (model_var, dev) in variant_models.items():
                        layer_var = layer_map.get(arch, {}).get(tag, "features")

                        for method in per_image_methods:
                            if method == "eigencam":
                                baseline_map = True  
                            else:
                                baseline_map = baseline_explanations.get(method)
                            pbar.set_postfix({
                                "variant": tag,
                                "method": method
                            })

                            if baseline_map is None:
                                continue
                            
                            try:
                                if method == "scorecam":
                                    var_map = compute_scorecam(model_var, img_tensor, layer_var, device=dev, plot=False, return_map=True)
                                    if var_map is not None:
                                        metrics = compare_heatmaps(baseline_map, var_map)
                                elif method == "eigencam":
                                    ref_key = "eigencam_cpu" if dev == "cpu" else "eigencam_cuda"
                                    ref_map = baseline_explanations.get(ref_key) 
                                    var_map = compute_eigencam(model_var, img_tensor, layer_var, device=dev, plot=False, return_map=True)
                                    if var_map is not None and ref_map is not None:
                                        metrics = compare_heatmaps(ref_map, var_map)
                                elif method == "shap":
                                    var_attr = compute_shap(model_var, img_tensor, max_evals=100, device=dev, plot=False, return_values=True)
                                    if var_attr is not None:
                                        metrics = compare_attributions(baseline_map, var_attr)
                                elif method == "lime":
                                    var_mask = compute_lime(model_var, img_tensor, num_samples=100, device=dev, plot=False, return_mask=True)
                                    if var_mask is not None:
                                        metrics = compare_masks(baseline_map, var_mask)
                                else:
                                    continue

                                # Accumulate metrics 
                                for metric_name, value in metrics.items():
                                    d = self.results[arch][tag][method].setdefault(metric_name, {"sum": 0.0, "count": 0})
                                    d["sum"] += value
                                    d["count"] += 1

                            except Exception:
                                continue

                    processed += 1
                    pbar.update(1)

            pbar.close()
        else:
            print(" No per-image XAI methods requested, skipping image-wise comparisons.")

        # Clean up models
        del model_fp32
        for tag, (model_var, dev) in variant_models.items():
            del model_var
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate_report(self, filename="xai_report.txt"):
        """
        Generate report exactly in the original format.
        Computes means from aggregated sums/counts.
        """
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
                        p_dist = u.get('procrustes_disparity', 0)
                        t_score = u.get('trustworthiness', 0)
                        c_drift = u.get('centroid_drift', 0)
                        f.write(f"    [LOGIC DRIFT (UMAP)]\n")
                        f.write(f"      Procrustes: {p_dist:.4f} | Trustworthiness: {t_score:.4f} | Mean Class Drift: {c_drift:.4f}\n\n")
                    
                    def get_mean(method_dict, metric, default=0.0):
                        if metric in method_dict:
                            agg = method_dict[metric]
                            if agg["count"] > 0:
                                return agg["sum"] / agg["count"]
                        return default

                    # Print ScoreCAM / EigenCAM
                    for m_name in ["scorecam", "eigencam"]:
                        if m_name in methods_data and methods_data[m_name]:
                            mdict = methods_data[m_name]
                            f.write(f"    [{m_name.upper()}]\n")
                            ssim_val = get_mean(mdict, "ssim")
                            corr_val = get_mean(mdict, "corr")
                            emd_val = get_mean(mdict, "emd")
                            iou_val = get_mean(mdict, "iou")
                            row = f"      SSIM: {ssim_val:.3f} | Pearson: {corr_val:.3f} | "
                            row += f"Wasserstein: {emd_val:.3f} | IoU: {iou_val:.3f}\n"
                            f.write(row + "\n")

                    # Print SHAP
                    if "shap" in methods_data and methods_data["shap"]:
                        mdict = methods_data["shap"]
                        f.write(f"    [SHAP]\n")
                        spearman_val = get_mean(mdict, "spearman")
                        topk_val = get_mean(mdict, "topk_overlap")
                        row = f"      Spearman: {spearman_val:.3f} | Top-K Overlap: {topk_val:.3f}\n"
                        f.write(row + "\n")

                    # Print LIME
                    if "lime" in methods_data and methods_data["lime"]:
                        mdict = methods_data["lime"]
                        f.write(f"    [LIME]\n")
                        iou_val = get_mean(mdict, "iou")
                        row = f"      IoU: {iou_val:.3f}\n"
                        f.write(row + "\n")
                        
        print(f"✅ Report saved to {filename}")