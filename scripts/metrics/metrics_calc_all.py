"""
一键计算全部图像质量评价指标 (单目录)
========================================
功能:
    计算 6 种图像质量评价指标:
    - 全参考 (FR): PSNR↑, SSIM↑, LPIPS↓  — 需要与 GT 对比
    - 无参考 (NR): NIQE↓, MUSIQ↑, BRISQUE↓ — 仅评估增强图像

使用方法:
    python scripts/metrics/metrics_calc_all.py ^
        --dirA ./datasets/LOL-v2/Real_captured/Test/Normal ^
        --dirB ./results/Experiment_A/net_g_44000 ^
        --type png --use_gpu ^
        --save_txt metrics_results_all.txt

参数:
    --dirA      必选，参考图像 (GT) 目录
    --dirB      必选，增强图像 (模型输出) 目录
    --type      图像扩展名，默认 png
    --use_gpu   使用 GPU 加速 (推荐，MUSIQ/LPIPS 较慢)
    --save_txt  结果保存路径，默认 metrics_results_all.txt

依赖:
    pip install opencv-python numpy torch torchvision natsort tqdm scikit-image lpips pyiqa brisque
"""

import argparse
import glob
import os
import time
import warnings
from collections import OrderedDict

import cv2
import numpy as np
import torch
from natsort import natsorted
from tqdm import tqdm

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips

# 忽略特定警告以保持输出整洁
warnings.filterwarnings("ignore")

# 设置 Hugging Face 镜像站，加速模型权重下载
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

try:
    import pyiqa

    HAVE_PYIQA = True
except ImportError:
    HAVE_PYIQA = False
    print("Warning: pyiqa library not found. Install with: pip install pyiqa")

try:
    from brisque import BRISQUE

    HAVE_BRISQUE = True
except ImportError:
    HAVE_BRISQUE = False
    print("Warning: brisque library not found. Install with: pip install brisque")


class Measure:
    def __init__(self, net="alex", use_gpu=False):
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.model = lpips.LPIPS(net=net)
        self.model.to(self.device)

    def measure(self, imgA, imgB):
        return [float(f(imgA, imgB)) for f in [self.psnr, self.ssim, self.lpips]]

    def lpips(self, imgA, imgB, model=None):
        tA = self.t(imgA).to(self.device)
        tB = self.t(imgB).to(self.device)
        dist01 = self.model.forward(tA, tB).item()
        return dist01

    def ssim(self, imgA, imgB, gray_scale=False):
        if gray_scale:
            score, diff = ssim(
                cv2.cvtColor(imgA, cv2.COLOR_RGB2GRAY),
                cv2.cvtColor(imgB, cv2.COLOR_RGB2GRAY),
                full=True,
                channel_axis=True,
            )
        else:
            score, diff = ssim(imgA, imgB, full=True, channel_axis=2)
        return score

    def psnr(self, imgA, imgB):
        psnr_val = psnr(imgA, imgB)
        return psnr_val

    def t(self, img):
        def to_4d(img):
            assert len(img.shape) == 3
            assert img.dtype == np.uint8
            img_new = np.expand_dims(img, axis=0)
            assert len(img_new.shape) == 4
            return img_new

        def to_CHW(img):
            return np.transpose(img, [2, 0, 1])

        def to_tensor(img):
            return torch.Tensor(img)

        return to_tensor(to_4d(to_CHW(img))) / 127.5 - 1


def calculate_niqe_pyiqa(img_rgb, niqe_metric_obj, device):
    """
    使用 pyiqa 库计算 NIQE
    """
    if not HAVE_PYIQA or img_rgb is None or niqe_metric_obj is None:
        return None
    try:
        img_tensor = (
            torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        )
        img_tensor = img_tensor.to(device)
        with torch.no_grad():
            niqe_value = niqe_metric_obj(img_tensor)
        return niqe_value.item()
    except Exception as e:
        print(f"Error calculating NIQE: {e}")
        return None


def calculate_brisque(img_bgr, brisque_obj):
    """
    使用预初始化的对象计算 BRISQUE
    """
    if not HAVE_BRISQUE or img_bgr is None:
        return None

    try:
        score = brisque_obj.score(img_bgr)
        return score
    except Exception as e:
        # print(f"Error calculating BRISQUE: {e}")
        return None


def calculate_musiq_pyiqa(img_rgb, musiq_metric_obj, device):
    """
    使用 pyiqa 库计算 MUSIQ
    """
    if not HAVE_PYIQA or img_rgb is None or musiq_metric_obj is None:
        return None
    try:
        img_tensor = (
            torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        )
        img_tensor = img_tensor.to(device)
        with torch.no_grad():
            musiq_value = musiq_metric_obj(img_tensor)
        return musiq_value.item()
    except Exception as e:
        print(f"Error calculating MUSIQ: {e}")
        return None


def fiFindByWildcard(wildcard):
    return natsorted(glob.glob(wildcard, recursive=True))


def imread(path):
    return cv2.imread(path)[:, :, [2, 1, 0]]


def format_dict_result(res):
    psnr = res.get("psnr", 0.0)
    ssim = res.get("ssim", 0.0)
    lpips = res.get("lpips", 0.0)

    niqe_str = (
        f"{res.get('niqe'):8.4f}" if res.get("niqe") is not None else f"{'NaN':>8}"
    )
    musiq_str = (
        f"{res.get('musiq'):8.4f}" if res.get("musiq") is not None else f"{'NaN':>8}"
    )
    brisque_str = (
        f"{res.get('brisque'):8.4f}"
        if res.get("brisque") is not None
        else f"{'NaN':>8}"
    )

    return (
        f"{psnr:8.2f} {ssim:8.4f} {lpips:8.4f} | {niqe_str} {musiq_str} {brisque_str}"
    )


def measure_all(dirA, dirB, img_ext="png", use_gpu=False, save_path=None):
    t_init = time.time()

    paths_A = fiFindByWildcard(os.path.join(dirA, f"*.{img_ext}"))
    paths_B = fiFindByWildcard(os.path.join(dirB, f"*.{img_ext}"))

    if len(paths_A) != len(paths_B):
        print(
            f"Warning: Directory file counts don't match: {len(paths_A)} vs {len(paths_B)}."
        )
        print("Matching files by name...")
        dict_A = {os.path.basename(p): p for p in paths_A}
        dict_B = {os.path.basename(p): p for p in paths_B}
        common_names = set(dict_A.keys()).intersection(set(dict_B.keys()))
        common_names = natsorted(list(common_names))

        paths_A = [dict_A[name] for name in common_names]
        paths_B = [dict_B[name] for name in common_names]
        print(f"Found {len(paths_A)} matching pairs.")

    if len(paths_A) == 0:
        raise ValueError("No matching image pairs found.")

    header = f"{'Reference':<25} {'Enhanced':<25} {'PSNR(dB)':>10} {'SSIM':>8} {'LPIPS':>8} | {'NIQE':>8} {'MUSIQ':>8} {'BRISQUE':>8} {'Time(s)':>8}"

    measure_ref = Measure(use_gpu=use_gpu)
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    niqe_metric = pyiqa.create_metric("niqe", device=device) if HAVE_PYIQA else None
    musiq_metric = pyiqa.create_metric("musiq", device=device) if HAVE_PYIQA else None
    brisque_obj = BRISQUE(url=False) if HAVE_BRISQUE else None

    results = []
    saver = None
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        saver = open(save_path, "w", encoding="utf-8")
        saver.write("Reference (Full-Reference) Metrics: PSNR, SSIM, LPIPS\n")
        saver.write(
            "No-Reference Metrics (Evaluated on Enhanced): NIQE, MUSIQ, BRISQUE\n"
        )
        saver.write("=" * len(header) + "\n")
        saver.write(header + "\n")
        saver.write("-" * len(header) + "\n")

    iterator = tqdm(
        zip(paths_A, paths_B),
        total=len(paths_A),
        desc="评估进度",
        unit="pair",
    )

    valid_niqe_scores = []
    valid_musiq_scores = []
    valid_brisque_scores = []

    for pathA, pathB in iterator:
        result = OrderedDict()

        t_start = time.time()

        # Read images
        imgA_rgb = imread(pathA)
        imgB_bgr = cv2.imread(pathB)

        if imgB_bgr is None or imgA_rgb is None:
            print(f"\nFailed to read image pairs: {pathA} or {pathB}")
            continue

        imgB_rgb = cv2.cvtColor(imgB_bgr, cv2.COLOR_BGR2RGB)

        # Full-Reference Metrics
        result["psnr"], result["ssim"], result["lpips"] = measure_ref.measure(
            imgA_rgb, imgB_rgb
        )

        # No-Reference Metrics (on Enhanced Image imgB)
        niqe_score = calculate_niqe_pyiqa(imgB_rgb, niqe_metric, device)
        musiq_score = calculate_musiq_pyiqa(imgB_rgb, musiq_metric, device)
        brisque_score = calculate_brisque(imgB_bgr, brisque_obj)

        result["niqe"] = niqe_score
        result["musiq"] = musiq_score
        result["brisque"] = brisque_score

        if niqe_score is not None:
            valid_niqe_scores.append(niqe_score)
        if musiq_score is not None:
            valid_musiq_scores.append(musiq_score)
        if brisque_score is not None:
            valid_brisque_scores.append(brisque_score)

        d_time = time.time() - t_start

        filename_A = os.path.basename(pathA)
        filename_B = os.path.basename(pathB)

        if len(filename_A) > 23:
            filename_A = filename_A[:20] + "..."
        if len(filename_B) > 23:
            filename_B = filename_B[:20] + "..."

        if saver:
            saver.write(
                f"{filename_A:<25} {filename_B:<25} "
                f"{format_dict_result(result)} {d_time:8.2f}\n"
            )

        results.append(result)

    psnr_avg = np.mean([res["psnr"] for res in results])
    ssim_avg = np.mean([res["ssim"] for res in results])
    lpips_avg = np.mean([res["lpips"] for res in results])

    niqe_avg = np.mean(valid_niqe_scores) if valid_niqe_scores else 0.0
    musiq_avg = np.mean(valid_musiq_scores) if valid_musiq_scores else 0.0
    brisque_avg = np.mean(valid_brisque_scores) if valid_brisque_scores else 0.0

    total_time = time.time() - t_init

    summary_res = {
        "psnr": psnr_avg,
        "ssim": ssim_avg,
        "lpips": lpips_avg,
        "niqe": niqe_avg if valid_niqe_scores else None,
        "musiq": musiq_avg if valid_musiq_scores else None,
        "brisque": brisque_avg if valid_brisque_scores else None,
    }

    if saver:
        saver.write("-" * len(header) + "\n")
        saver.write(
            f"{'Average':<25} {'Count: ' + str(len(results)):<25} {format_dict_result(summary_res)} {total_time:8.1f}\n"
        )
        saver.write("\n" + "=" * 80 + "\n")
        saver.write("Detailed No-Reference Metrics Stats:\n")
        if valid_niqe_scores:
            saver.write(
                f"NIQE:    {np.mean(valid_niqe_scores):.4f} ± {np.std(valid_niqe_scores):.4f}\n"
            )
        if valid_musiq_scores:
            saver.write(
                f"MUSIQ:   {np.mean(valid_musiq_scores):.4f} ± {np.std(valid_musiq_scores):.4f}\n"
            )
        if valid_brisque_scores:
            saver.write(
                f"BRISQUE: {np.mean(valid_brisque_scores):.4f} ± {np.std(valid_brisque_scores):.4f}\n"
            )

        saver.write(f"\nProcessed {len(results)} image pairs in {total_time:0.1f}s.\n")
        saver.close()

    summary_header = f"{'Summary':<25} {'Count':<25} {'PSNR(dB)':>10} {'SSIM':>8} {'LPIPS':>8} | {'NIQE':>8} {'MUSIQ':>8} {'BRISQUE':>8} {'Time(s)':>8}"
    print("\n" + summary_header)
    print("-" * len(summary_header))
    print(
        f"{'Average':<25} {len(results):<25} {format_dict_result(summary_res)} {total_time:8.1f}"
    )
    print(f"Processed {len(results)} image pairs in {total_time:0.1f}s.")

    print("\n" + "=" * 80)
    print("AVERAGE SCORES (Detailed):")
    print("=" * 80)
    print(f"PSNR (Higher is better):  {psnr_avg:.4f}")
    print(f"SSIM (Higher is better):  {ssim_avg:.4f}")
    print(f"LPIPS (Lower is better):  {lpips_avg:.4f}")

    if valid_niqe_scores:
        print(
            f"NIQE (Lower is better):   {np.mean(valid_niqe_scores):.4f} ± {np.std(valid_niqe_scores):.4f}"
        )
    else:
        print("NIQE:                      Not calculated")

    if valid_musiq_scores:
        print(
            f"MUSIQ (Higher is better): {np.mean(valid_musiq_scores):.4f} ± {np.std(valid_musiq_scores):.4f}"
        )
    else:
        print("MUSIQ:                     Not calculated")

    if valid_brisque_scores:
        print(
            f"BRISQUE (Lower is bet.):  {np.mean(valid_brisque_scores):.4f} ± {np.std(valid_brisque_scores):.4f}"
        )
    else:
        print("BRISQUE:                   Not calculated")
    print("=" * 80)
    if save_path:
        print(f"Results saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate Full-Reference and No-Reference image quality metrics"
    )
    parser.add_argument(
        "--dirA",
        default=r"C:\Users\CTS\Desktop\send_single\1",
        type=str,
        help="Directory containing reference images",
    )
    parser.add_argument(
        "--dirB",
        default=r"C:\Users\CTS\Desktop\send_single\2",
        type=str,
        help="Directory containing enhanced images",
    )
    parser.add_argument("--type", default="png", help="Image extensions, e.g., png")
    parser.add_argument(
        "--use_gpu", action="store_true", default=False, help="Use GPU for evaluation"
    )
    parser.add_argument(
        "--save_txt",
        type=str,
        default="metrics_results_all.txt",
        help="Path to save the combined metrics calculation results.",
    )
    args = parser.parse_args()

    dirA = args.dirA
    dirB = args.dirB
    img_ext = args.type
    use_gpu = args.use_gpu
    save_txt = args.save_txt

    if len(dirA) > 0 and len(dirB) > 0:
        measure_all(
            dirA,
            dirB,
            img_ext=img_ext,
            use_gpu=use_gpu,
            save_path=save_txt,
        )
    else:
        print("Error: Both --dirA and --dirB must be provided.")
