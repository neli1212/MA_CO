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
    Computes accuracy and latency. 
    """
    model.to(device)
    model.eval()

 

    try:
        model_dtype = next(model.parameters()).dtype
    except StopIteration:
        model_dtype = torch.float32

    correct = 0
    total = 0
    latencies = []

    print(f"🔥 Warming up {device}...")
    dummy_input, _ = next(iter(data_loader))
    dummy_input = dummy_input.to(device).to(dtype=model_dtype)
    if device == "cuda":
        dummy_input = dummy_input.to(memory_format=torch.channels_last)
    
    with torch.no_grad():
        for _ in range(10): 
            _ = model(dummy_input)
    if device == "cuda":
        torch.cuda.synchronize()

    pbar = tqdm(data_loader, desc=f"Eval ({device})", unit="batch", leave=False)

    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            if images.dtype != model_dtype:
                images = images.to(dtype=model_dtype)
            
            if device == "cuda":
                images = images.to(memory_format=torch.channels_last)
            if device == "cuda":
                torch.cuda.synchronize() 
            
            start = time.perf_counter()
            outputs = model(images)
            
            if device == "cuda":
                torch.cuda.synchronize() 
            
            end = time.perf_counter()

            latencies.append(((end - start) / images.size(0)) * 1000)

            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return 100 * correct / total, sum(latencies) / len(latencies)

def get_save_path(base_dir, model_type, ft_e, qat_e, suffix):
    arch_dir = os.path.join(base_dir, model_type)
    os.makedirs(arch_dir, exist_ok=True)
    
    if suffix in ["finetuned", "ptqfx", "fp16", "bf16", "fp32_polished"]:
        fname = f"{model_type}_ft{ft_e}_{suffix}.pth"
    else:
        fname = f"{model_type}_ft{ft_e}_qat{qat_e}_{suffix}.pth"
    return os.path.join(arch_dir, fname)

# =====================================================================
# Main Execution Pipeline
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Model Compression Pipeline")
    parser.add_argument("--model_type", type=str, default="resnet50", choices=list(BASE_FACTORIES.keys()))
    parser.add_argument("--root", type=str, required=True, help="Data root directory")
    parser.add_argument("--resume", type=str, default=None, help="Path to existing FP32 model to skip initial training")
    parser.add_argument("--epochs_ft", type=int, default=1)
    parser.add_argument("--epochs_qat", type=int, default=1)
    parser.add_argument("--patience", type=int, default=None)
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


    # 1. LOAD STANDARD UNPOLISHED MODEL 
    if args.resume and os.path.exists(args.resume):
        print(f"⏩ SKIPPING Step 1. Loading your existing model: {args.resume}")
        model_unpolished = load_model_generic(f"{args.model_type}-fp32", args.resume, num_classes, device="cpu")
        ft_path = args.resume 
    else:
        ft_path = get_save_path(args.save_dir, args.model_type, args.epochs_ft, args.epochs_qat, "finetuned")
        if os.path.exists(ft_path) and not args.force:
            print(f"📦 Loading existing FP32 weights: {ft_path}")
            model_unpolished = load_model_generic(f"{args.model_type}-fp32", ft_path, num_classes, device="cpu")
        else:
            print(f"🛠 Training FP32 Standard (Unpolished)...")
            model_unpolished = BASE_FACTORIES[args.model_type](num_classes)
            model_unpolished, _ = finetuneModel(
                model_unpolished, train_loader.dataset, val_loader.dataset,
                num_classes, epochs=args.epochs_ft, full_finetune=False,
                patience=args.patience
            )
            save_model(model_unpolished, ft_path)


    # 2. POLISH FP32 (LR: 1e-5 same as QAT) 
    polished_path = get_save_path(args.save_dir, args.model_type, args.epochs_ft, 0, "fp32_polished")
    
    if os.path.exists(polished_path) and not args.force:
        print(f"📦 Loading existing Polished FP32: {polished_path}")
        model_polished = load_model_generic(f"{args.model_type}-fp32", polished_path, num_classes, device="cpu")
    else:
        print(f"✨ Polishing FP32 Baseline...")
        model_polished = copy.deepcopy(model_unpolished)
        model_polished, _ = finetuneModel(
            model_polished, 
            train_loader.dataset, 
            val_loader.dataset,
            num_classes, 
            epochs=args.epochs_qat,       
            lr=1e-5,        
            patience=args.patience
        )
        save_model(model_polished, polished_path)

    model_paths["FP32_Polished"] = (polished_path, f"{args.model_type}-fp32")
    
    # --- PTQ ---
    ptq_path = get_save_path(args.save_dir, args.model_type, args.epochs_ft, 0, "ptqfx")
    if not (os.path.exists(ptq_path) and not args.force):
        print("⚡ Applying PTQ to Polished FP32...")
        model_ptq = QuantizePTQ(copy.deepcopy(model_polished), calibration_loader=calib_images)
        save_model(model_ptq, ptq_path)
    model_paths["INT8_PTQ"] = (ptq_path, f"{args.model_type}-ptqfx")


    # 3. QAT same LR as POLISH FP32
    qat_path = get_save_path(args.save_dir, args.model_type, args.epochs_ft, args.epochs_qat, "int8")
    
    if os.path.exists(qat_path) and not args.force:
        print(f"📦 Loading existing QAT weights: {qat_path}")
    else:
        print("🧠 Applying QAT to Unpolished FP32...")
        qat_input = copy.deepcopy(model_unpolished)
        
        _, model_int8_qat, _ = trainQAT(
            qat_input, train_loader.dataset, val_loader.dataset, 
            epochs=args.epochs_qat, patience=args.patience, lr=1e-5
        )
        save_model(model_int8_qat, qat_path)
    model_paths["INT8_QAT"] = (qat_path, f"{args.model_type}-int8")

    # 4. FP16 & BF16 EXPORTS
    fp16_path = get_save_path(args.save_dir, args.model_type, args.epochs_ft, 0, "fp16")
    if not os.path.exists(fp16_path):
        save_model(copy.deepcopy(model_polished).half(), fp16_path)
    model_paths["FP16"] = (fp16_path, f"{args.model_type}-fp16")

    bf16_path = get_save_path(args.save_dir, args.model_type, args.epochs_ft, 0, "bf16")
    if not os.path.exists(bf16_path):
        save_model(copy.deepcopy(model_polished).to(torch.bfloat16), bf16_path)
    model_paths["BF16"] = (bf16_path, f"{args.model_type}-bf16")

    # 5. EVALUATION
    if not args.skip_eval:
        report_fname = f"{args.model_type}_performance_report.txt"
        report_path = os.path.join(args.save_dir, args.model_type, report_fname)
        
        cpu_info, gpu_info = get_cpu_name(), get_gpu_name()

        print(f"\n📉 Starting Final Evaluation...")
        with open(report_path, "w") as f:
            f.write(f"PERFORMANCE REPORT: {args.model_type.upper()}\n")
            f.write(f"CPU: {cpu_info} | GPU: {gpu_info}\n")
            f.write("=" * 95 + "\n")
            header = f"{'Variant':<18} | {'Device':<8} | {'Size(MB)':<10} | {'Acc(%)':<10} | {'Lat(ms/img)':<15}\n"
            f.write(header)
            f.write("-" * 95 + "\n")

        eval_tasks = [
            ("FP32_Polished", "cpu"), 
            ("INT8_PTQ", "cpu"),
            ("INT8_QAT", "cpu"),
        ]
        
        if torch.cuda.is_available():
            eval_tasks += [
                ("FP32_Polished", "cuda"),
                ("FP16", "cuda"),
                ("BF16", "cuda"),
            ]

        for label, device in eval_tasks:
            if label not in model_paths:
                continue

            path, model_tag = model_paths[label]
            if not os.path.exists(path):
                print(f"⚠️ Skipping {label}: File not found ({path})")
                continue

            size_mb = os.path.getsize(path) / (1024 * 1024)

            try:
                # Load appropriate model
                model = load_model_generic(model_tag, path, num_classes, device=device)
                
                # Run eval
                acc, latency = evaluate_model(model, val_loader, device=device)

                result_line = f"{label:<18} | {device:<8} | {size_mb:<10.2f} | {acc:<10.2f} | {latency:<15.4f}\n"
                
                # Append to report immediately
                with open(report_path, "a") as f:
                    f.write(result_line)
                
                print(f"✅ Evaluated {label} on {device}: Acc={acc:.2f}%")
                
                # Cleanup
                del model
                if device == "cuda": torch.cuda.empty_cache()

            except Exception as e:
                print(f"❌ Failed {label} on {device}: {e}")

        print(f"\n📊 Summary Report saved to: {report_path}")
if __name__ == "__main__":
    main()