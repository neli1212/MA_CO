# =====================================================================
# Imports
# =====================================================================
import argparse
import torch
import os
import random
import numpy as np
import warnings
from maco_package.data import build_loaders
from maco_package.compare import XAIComparator

# =====================================================================
# Environment Configuration
# =====================================================================
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# =====================================================================
# Main XAI Evaluation Pipeline (Batch‑Processed, Optional UMAP)
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="XAI Consistency Evaluation Pipeline")
    parser.add_argument("--model_type", type=str, default="resnet50",
                        choices=["vgg16", "resnet50", "mobilenet_v2", "googlenet", "densenet121","mnasnet"])
    parser.add_argument("--root", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--epochs_ft", type=int, default=1, help="Finetune epoch suffix in filename")
    parser.add_argument("--epochs_qat", type=int, default=3, help="QAT epoch suffix in filename")
    parser.add_argument("--save_dir", type=str, default=os.path.join(os.getcwd(), "saved_models"),
                        help="Directory where models are stored")
    parser.add_argument("--img_count", type=int, default=1000, help="Number of images to evaluate (can be thousands)")
    parser.add_argument("--methods", nargs="+", default=["scorecam", "eigencam", "lime"],
                        help="XAI methods to evaluate: scorecam, eigencam, shap, lime, umap")
    parser.add_argument("--umap_samples", type=int, default=None, help="Samples for UMAP (if umap in methods)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for data loading")
    args = parser.parse_args()

    # 1. Setup Data
    print(f"\n--- Loading Data from {args.root} ---")
    _, val_loader, class_to_idx = build_loaders(args.root, batch_size=args.batch_size)
    num_classes = len(class_to_idx)

    # 2. Setup Paths
    save_dir = os.path.abspath(args.save_dir)
    report_name = f"{args.model_type}_XAI_CONSISTENCY.txt"
    report_path = os.path.join(save_dir, args.model_type, report_name)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    # 3. Initialize Comparator
    comparator = XAIComparator(base_path=save_dir, num_classes=num_classes)
    print(f"--- Starting XAI Analysis for {args.model_type.upper()} ---")
    print(f"Requested methods: {args.methods}")

    # 4. Execution (batch‑processed, optional UMAP)
    try:
        comparator.run(
            arch=args.model_type,
            val_loader=val_loader,
            num_images=args.img_count,
            methods=args.methods,
            epochs_ft=args.epochs_ft,
            epochs_qat=args.epochs_qat,
            max_samples_umap=args.umap_samples
        )

        # Generate Report
        comparator.generate_report(filename=report_path)
        print(f"\n--- Pipeline Complete ---")
        print(f"Final XAI Consistency Report saved at: {report_path}")

    except Exception as e:
        print(f"\n❌ Pipeline Error: {e}")
        raise

if __name__ == "__main__":
    main()