# =====================================================================
# Imports
# =====================================================================
import argparse
import torch
import torch.nn as nn
import os
import copy
import warnings
import time
import platform
from tqdm import tqdm
import cpuinfo
import subprocess
from maco_package.quantization import QuantizePTQ, trainQAT
from maco_package.data import build_loaders, finetuneModel
from maco_package.utils import (
    save_model,
    load_model_generic,
    BASE_FACTORIES #
)

# =====================================================================
# Environment Configuration
# =====================================================================
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# =====================================================================
# Helper Functions
# =====================================================================

def get_gpu_name():
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
        ).decode().strip()
    except:
        return "N/A"

def get_cpu_name():
    try:
        return cpuinfo.get_cpu_info()["brand_raw"]
    except:
        return platform.processor()

def evaluate_model(model, data_loader, device="cpu"):
    """
    Computes accuracy and latency. Handles dtype casting for FP16/BF16.
    """
    model.to(device)
    model.eval()

    # Determine model dtype (Important for half-precision variants)
    try:
        model_dtype = next(model.parameters()).dtype
    except StopIteration:
        model_dtype = torch.float32

    correct = 0
    total = 0
    latencies = []

    pbar = tqdm(data_loader, desc=f"Eval ({device})", unit="batch", leave=False)

    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Cast inputs to model's precision
            if images.dtype != model_dtype:
                images = images.to(model_dtype)

            start = time.perf_counter()
            outputs = model(images)
            end = time.perf_counter()

            latencies.append(((end - start) / images.size(0)) * 1000)

            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return 100 * correct / total, sum(latencies) / len(latencies)

def get_save_path(base_dir, model_type, ft_e, qat_e, suffix):
    arch_dir = os.path.join(base_dir, model_type)
    os.makedirs(arch_dir, exist_ok=True)
    
    if suffix in ["finetuned", "ptqfx", "fp16", "bf16"]:
        fname = f"{model_type}_ft{ft_e}_{suffix}.pth"
    else:
        fname = f"{model_type}_ft{ft_e}_qat{qat_e}_{suffix}.pth"
    return os.path.join(arch_dir, fname)

# =====================================================================
# Main Execution Pipeline
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Model Compression Pipeline")
    parser.add_argument("--model_type", type=str, default="resnet50",
                        choices=list(BASE_FACTORIES.keys())) # Dynamically use your factory keys
    parser.add_argument("--root", type=str, required=True, help="Data root directory")
    parser.add_argument("--epochs_ft", type=int, default=1)
    parser.add_argument("--epochs_qat", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="saved_models")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    args = parser.parse_args()

    print(f"\n🚀 Starting Pipeline for {args.model_type.upper()}")
    train_loader, val_loader, class_to_idx = build_loaders(args.root, batch_size=32)
    num_classes = len(class_to_idx)

    calib_images, _ = next(iter(train_loader))
    calib_images = calib_images.cpu()

    model_paths = {}

    # 1. FP32 FINETUNING
    ft_path = get_save_path(args.save_dir, args.model_type, args.epochs_ft, args.epochs_qat, "finetuned")
    
    if os.path.exists(ft_path) and not args.force:
        print(f"📦 Loading existing FP32 weights: {ft_path}")
        model_fp32 = load_model_generic(f"{args.model_type}-fp32", ft_path, num_classes, device="cpu")
    else:
        print(f"🛠 Training FP32 Baseline...")
        # USING YOUR NEW FACTORY LOGIC
        model_fp32 = BASE_FACTORIES[args.model_type](num_classes)
        model_fp32 = finetuneModel(
            model_fp32, train_loader.dataset, val_loader.dataset,
            num_classes, epochs=args.epochs_ft, full_finetune=False
        )
        save_model(model_fp32, ft_path)

    # Base for all subsequent quantizations
    model_fp32_clean = copy.deepcopy(model_fp32).cpu().eval()
    model_paths["FP32_Finetuned"] = (ft_path, f"{args.model_type}-fp32")

    # 2. PTQ (Static Quantization)
    ptq_path = get_save_path(args.save_dir, args.model_type, args.epochs_ft, args.epochs_qat, "ptqfx")
    if not (os.path.exists(ptq_path) and not args.force):
        print("⚡ Applying PTQ...")
        # Always use a clean copy of the finetuned FP32 weights
        model_ptq = QuantizePTQ(copy.deepcopy(model_fp32_clean), calibration_input=calib_images)
        save_model(model_ptq, ptq_path)
    model_paths["INT8_PTQ"] = (ptq_path, f"{args.model_type}-ptqfx")

    # 3. QAT (Quantization Aware Training)
    qat_path = get_save_path(args.save_dir, args.model_type, args.epochs_ft, args.epochs_qat, "int8")
    if not (os.path.exists(qat_path) and not args.force):
        print("🧠 Applying QAT...")
        qat_fp32_base = copy.deepcopy(model_fp32_clean)
        _, model_int8 = trainQAT(
            qat_fp32_base, train_loader.dataset, val_loader.dataset, epochs=args.epochs_qat
        )
        save_model(model_int8, qat_path)
    model_paths["INT8_QAT"] = (qat_path, f"{args.model_type}-int8")

    # 4. FP16 & BF16 EXPORTS
    fp16_path = get_save_path(args.save_dir, args.model_type, args.epochs_ft, 0, "fp16")
    if not os.path.exists(fp16_path):
        save_model(copy.deepcopy(model_fp32_clean).half(), fp16_path)
    model_paths["FP16"] = (fp16_path, f"{args.model_type}-fp16")

    bf16_path = get_save_path(args.save_dir, args.model_type, args.epochs_ft, 0, "bf16")
    if not os.path.exists(bf16_path):
        save_model(copy.deepcopy(model_fp32_clean).to(torch.bfloat16), bf16_path)
    model_paths["BF16"] = (bf16_path, f"{args.model_type}-bf16")

    # 5. EVALUATION
    if not args.skip_eval:
        report_fname = f"{args.model_type}_performance_report.txt"
        report_path = os.path.join(args.save_dir, args.model_type, report_fname)
        
        cpu_info, gpu_info = get_cpu_name(), get_gpu_name()

        with open(report_path, "w") as f:
            f.write(f"PERFORMANCE REPORT: {args.model_type.upper()}\n")
            f.write(f"CPU: {cpu_info} | GPU: {gpu_info}\n")
            f.write("=" * 95 + "\n")
            header = f"{'Variant':<18} | {'Device':<8} | {'Size(MB)':<10} | {'Acc(%)':<10} | {'Lat(ms/img)':<15}\n"
            f.write(header)
            f.write("-" * 95 + "\n")

        # Define tasks: (Label, Device)
        eval_tasks = [
            ("FP32_Finetuned", "cpu"),
            ("INT8_PTQ", "cpu"),
            ("INT8_QAT", "cpu"),
        ]
        if torch.cuda.is_available():
            eval_tasks += [
                ("FP32_Finetuned", "cuda"),
                ("FP16", "cuda"),
                ("BF16", "cuda"),
            ]

        for label, device in eval_tasks:
            path, model_tag = model_paths[label]
            size_mb = os.path.getsize(path) / (1024 * 1024)

            try:
                # load_model_generic handles the internal conversion (to_fp16, to_int8, etc.)
                model = load_model_generic(model_tag, path, num_classes, device=device)
                acc, latency = evaluate_model(model, val_loader, device=device)

                result_line = f"{label:<18} | {device:<8} | {size_mb:<10.2f} | {acc:<10.2f} | {latency:<15.4f}\n"
                with open(report_path, "a") as f:
                    f.write(result_line)
                
                print(f"✅ Evaluated {label} on {device}")
                del model
                if device == "cuda": torch.cuda.empty_cache()

            except Exception as e:
                print(f"❌ Failed {label} on {device}: {e}")

        print(f"\n📊 Summary Report: {report_path}")

if __name__ == "__main__":
    main()