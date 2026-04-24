"""
批量计算图像质量指标
========================================
功能:
    自动遍历结果目录下的所有子文件夹，调用 metrics_calc_all.py 计算 6 种指标
    （PSNR, SSIM, LPIPS, NIQE, MUSIQ, BRISQUE），每个子文件夹生成独立的结果文件。

使用方法:
    python scripts/metrics/batch_eval_metrics.py ^
        --results_dir ./results/Experiment_A ^
        --gt_dir ./datasets/LOL-v2/Real_captured/Test/Normal ^
        --use_gpu

参数:
    --results_dir  必选，包含多个结果子文件夹的根目录
    --gt_dir       必选，参考图像 (GT) 目录
    --img_ext      图像扩展名，默认 png
    --use_gpu      使用 GPU 加速 (推荐)
    --output_root  指标结果存放根目录，默认 results/metrics

输出:
    results/metrics/{实验名}/{子文件夹名}.txt
"""

import os
import argparse
import sys
from natsort import natsorted

# Add the metrics script directory to path to allow importing metrics_calc_all
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from metrics_calc_all import measure_all
except ImportError:
    print("Error: Could not import measure_all from metrics_calc_all.py")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Batch calculate metrics for all result subdirectories")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory containing subfolders of inference results")
    parser.add_argument("--gt_dir", type=str, required=True, help="Directory containing ground truth images")
    parser.add_argument("--img_ext", type=str, default="png", help="Image extension (default: png)")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for evaluation (recommended for MUSIQ/LPIPS)")
    parser.add_argument("--output_root", type=str, default=os.path.join("results", "metrics"), help="Root directory to save metric results")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory {args.results_dir} does not exist.")
        return

    if not os.path.exists(args.gt_dir):
        print(f"Error: GT directory {args.gt_dir} does not exist.")
        return

    # Create output metrics directory
    # If results_dir is 'results/MyExp', we want output to be 'results/metrics/MyExp'
    results_dir_name = os.path.basename(os.path.normpath(args.results_dir))
    metrics_save_dir = os.path.join(args.output_root, results_dir_name)
    os.makedirs(metrics_save_dir, exist_ok=True)
    
    # Get all subdirectories in results_dir
    subdirs = [d for d in os.listdir(args.results_dir) if os.path.isdir(os.path.join(args.results_dir, d))]
    subdirs = natsorted(subdirs)
    
    if not subdirs:
        # Check if results_dir itself contains images (maybe no subdirs)
        imgs = [f for f in os.listdir(args.results_dir) if f.lower().endswith(args.img_ext.lower())]
        if imgs:
            print(f"No subdirectories found, but images found in {args.results_dir}. Processing as a single result set.")
            save_path = os.path.join(metrics_save_dir, "metrics.txt")
            measure_all(args.gt_dir, args.results_dir, img_ext=args.img_ext, use_gpu=args.use_gpu, save_path=save_path)
            return
        else:
            print(f"No subdirectories or images found in {args.results_dir}.")
            return

    print(f"Found {len(subdirs)} result subdirectories. Starting batch evaluation...")
    
    for subdir in subdirs:
        curr_res_dir = os.path.join(args.results_dir, subdir)
        save_path = os.path.join(metrics_save_dir, f"{subdir}.txt")
        
        print(f"\n>>>> Evaluating: {subdir}")
        print(f">>>> Result Dir: {curr_res_dir}")
        print(f">>>> Save Path:  {save_path}")
        
        try:
            measure_all(
                args.gt_dir,
                curr_res_dir,
                img_ext=args.img_ext,
                use_gpu=args.use_gpu,
                save_path=save_path
            )
        except Exception as e:
            print(f"Error evaluating {subdir}: {e}")

    print(f"\nBatch evaluation completed! Metrics saved to: {metrics_save_dir}")

if __name__ == "__main__":
    main()
