# =====================================================================
# Imports
# =====================================================================
import argparse
import torch
import os
import random
import numpy as np
import warnings
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from maco_package.data import build_loaders 
from maco_package.utils import load_model_generic
from maco_package.compare import XAIComparator

# =====================================================================
# Environment Configuration
# =====================================================================
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# =====================================================================
# Helper Functions
# =====================================================================

def get_image_samples(val_loader, count):
    random.seed(42) #For consistency
    all_indices = list(range(len(val_loader.dataset)))
    random_indices = random.sample(all_indices, min(count, len(all_indices)))
    
    samples = []
    to_pil = transforms.ToPILImage()
    for idx in random_indices:
        tensor, _ = val_loader.dataset[idx]
        samples.append(to_pil(tensor))
    return samples

# =====================================================================
# Main XAI Evaluation Pipeline
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="XAI Consistency Evaluation Pipeline")
    parser.add_argument("--model_type", type=str, default="resnet50",
                        choices=["vgg16", "resnet50", "mobilenet_v2", "efficientnet_b0", "densenet121"])
    parser.add_argument("--root", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--epochs_ft", type=int, default=1, help="Finetune epoch suffix in filename")
    parser.add_argument("--epochs_qat", type=int, default=3, help="QAT epoch suffix in filename")
    parser.add_argument("--save_dir", type=str, default=os.path.join("..", "..", "saved_models"), 
                        help="Directory where models are stored")
    parser.add_argument("--img_count", type=int, default=5, 
                        help="Number of random images to evaluate")
    parser.add_argument("--methods", nargs="+", default=["scorecam", "eigencam", "lime"],
                        help="XAI methods to evaluate (scorecam, eigencam, shap, lime)")
    args = parser.parse_args()

    # 1. Setup Data
    print(f"\n--- Loading Data from {args.root} ---")
    _, val_loader, class_to_idx = build_loaders(args.root, batch_size=32)
    num_classes = len(class_to_idx)
    
    # 2. Randomly Sample Images for Heatmaps (ScoreCAM/LIME/etc)
    print(f"Selecting {args.img_count} random images for heatmap evaluation...")
    test_images = get_image_samples(val_loader, args.img_count)

    # 3. Setup Paths
    save_dir = os.path.abspath(args.save_dir)
    report_name = f"{args.model_type}_XAI_CONSISTENCY.txt"
    report_path = os.path.join(save_dir, args.model_type, report_name)

    # 4. Initialize Comparator
    comparator = XAIComparator(base_path=save_dir, num_classes=num_classes)

    print(f"--- Starting XAI Analysis for {args.model_type.upper()} ---")
    
    # 5. Execution
    try:
        comparator.run(
            arch=args.model_type,
            sample_images=test_images,
            methods=args.methods,
            val_loader=val_loader,  
            epochs_ft=args.epochs_ft,
            epochs_qat=args.epochs_qat
        )
        
        # Generate Report
        comparator.generate_report(filename=report_path)
        
        print(f"\n--- Pipeline Complete ---")
        print(f"Final XAI Consistency Report saved at: {report_path}")
        
    except Exception as e:
        print(f"\n❌ Pipeline Error: {e}")
    


if __name__ == "__main__":
    main()