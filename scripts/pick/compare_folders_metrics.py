"""
跨文件夹 PSNR/SSIM 对比
========================================
功能:
    比较一个父目录下多个子文件夹中同名图像相对于 GT 的 PSNR 和 SSIM。
    结果保存为两个 CSV 文件 ({prefix}_PSNR.csv, {prefix}_SSIM.csv)，
    行是文件名，列是不同子文件夹 (模型/实验)。

使用方法:
    python scripts/pick/compare_folders_metrics.py ^
        --parent_dir ./results/Experiment_A ^
        --gt_path D:/Datasets/LOL-v2/Real_captured/Test/Normal ^
        --type png

参数:
    --parent_dir  必选，包含多个结果子文件夹的父目录
    --gt_path     GT 目录 (绝对路径=外部GT; 文件夹名=父目录内子文件夹; 缺省=第一个子文件夹)
    --type        图像扩展名，默认 png
    --save_csv    CSV 保存路径，默认 {parent_dir}/metrics_comparison.csv

输出:
    {save_csv}_PSNR.csv, {save_csv}_SSIM.csv
"""

import glob
import os
import argparse
import csv

import numpy as np
import cv2
from natsort import natsorted
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def imread(path):
    """读取图像，BGR转RGB"""
    return cv2.imread(path)[:, :, [2, 1, 0]]


def calculate_psnr(imgA, imgB):
    """计算PSNR"""
    return psnr(imgA, imgB)


def calculate_ssim(imgA, imgB):
    """计算SSIM"""
    score, _ = ssim(imgA, imgB, full=True, channel_axis=2)
    return score


def extract_core_name(filename):
    """
    提取文件的核心名称，去除常见前缀
    例如: normal_001.png -> 001
          normal001.png -> 001
          low_001.png -> 001
          low001.png -> 001
          001.png -> 001
    """
    # 去除扩展名
    name_without_ext = os.path.splitext(filename)[0]

    # 尝试移除常见前缀（带下划线和不带下划线）
    prefixes = [
        "normal_",
        "normal",
        "low_",
        "low",
        "high_",
        "high",
        "gt_",
        "gt",
        "ref_",
        "ref",
    ]
    for prefix in prefixes:
        if name_without_ext.lower().startswith(prefix.lower()):
            core = name_without_ext[len(prefix) :]
            # 确保提取出的核心名称不为空
            if core:
                return core

    return name_without_ext


def find_images_in_folder(folder_path, img_ext="png"):
    """在文件夹中查找所有图像文件"""
    pattern = os.path.join(folder_path, f"*.{img_ext}")
    return natsorted(glob.glob(pattern))


def build_filename_mapping(folder_paths, img_ext="png"):
    """
    构建文件名映射，基于核心名称匹配不同文件夹中的文件
    返回: (common_core_names, folder_mappings)
    - common_core_names: 所有文件夹共有的核心名称列表
    - folder_mappings: {folder_path: {core_name: actual_filename}}
    """
    folder_mappings = {}

    # 为每个文件夹建立核心名称到实际文件名的映射
    for folder in folder_paths:
        images = find_images_in_folder(folder, img_ext)
        mapping = {}
        for img_path in images:
            filename = os.path.basename(img_path)
            core_name = extract_core_name(filename)
            mapping[core_name] = filename
        folder_mappings[folder] = mapping

    # 找出所有文件夹共有的核心名称
    if not folder_mappings:
        return [], {}

    core_name_sets = [set(mapping.keys()) for mapping in folder_mappings.values()]
    common_core_names = set.intersection(*core_name_sets)

    return natsorted(list(common_core_names)), folder_mappings


def compare_folders(parent_dir, reference_folder=None, img_ext="png", save_csv=None):
    """
    比较多个文件夹下同名文件的PSNR和SSIM

    Args:
        parent_dir: 包含多个子文件夹的父目录
        reference_folder: GT文件夹路径（可选）
                         - 如果是绝对路径，作为外部GT文件夹使用
                         - 如果是文件夹名，从parent_dir的子文件夹中查找
                         - 如果为None，使用第一个子文件夹作为GT
        img_ext: 图像文件扩展名
        save_csv: CSV保存路径
    """
    # 获取所有子文件夹
    subfolders = [
        f for f in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, f))
    ]
    subfolders = natsorted(subfolders)

    if len(subfolders) < 1:
        print(f"错误：至少需要1个子文件夹，当前有 {len(subfolders)} 个")
        return

    print(f"找到 {len(subfolders)} 个文件夹：{subfolders}")

    # 获取文件夹的完整路径
    folder_paths = {name: os.path.join(parent_dir, name) for name in subfolders}

    # 判断reference_folder是否为外部路径
    is_external_reference = False
    ref_folder_path = None
    ref_folder_name = None

    if reference_folder:
        # 检查是否为绝对路径
        if os.path.isabs(reference_folder):
            # 外部GT文件夹
            if not os.path.isdir(reference_folder):
                print(f"错误：指定的外部GT文件夹不存在：{reference_folder}")
                return
            is_external_reference = True
            ref_folder_path = reference_folder
            ref_folder_name = os.path.basename(reference_folder)
            compare_folders_list = subfolders  # 所有子文件夹都与外部GT比较
            print(f"使用外部GT文件夹：{reference_folder}")
        else:
            # 内部GT文件夹
            if reference_folder not in subfolders:
                print(
                    f"警告：指定的GT文件夹 '{reference_folder}' 不存在于parent_dir中，使用第一个文件夹作为GT"
                )
                ref_folder_name = subfolders[0]
            else:
                ref_folder_name = reference_folder
            ref_folder_path = folder_paths[ref_folder_name]
            compare_folders_list = [f for f in subfolders if f != ref_folder_name]
    else:
        # 默认使用第一个文件夹作为GT
        if len(subfolders) < 2:
            print(
                f"错误：至少需要2个子文件夹进行比较（当没有指定外部GT时），当前只有 {len(subfolders)} 个"
            )
            return
        ref_folder_name = subfolders[0]
        ref_folder_path = folder_paths[ref_folder_name]
        compare_folders_list = subfolders[1:]

    # 获取共有的文件名（使用核心名称匹配）
    if is_external_reference:
        # 需要包含外部GT文件夹
        all_folders = [ref_folder_path] + [
            folder_paths[f] for f in compare_folders_list
        ]
    else:
        all_folders = [
            folder_paths[f] for f in [ref_folder_name] + compare_folders_list
        ]

    common_core_names, filename_mappings = build_filename_mapping(all_folders, img_ext)

    if not common_core_names:
        print("错误：没有找到共同的文件")
        print("\n提示：请检查各文件夹中的文件命名是否对应")
        print("支持的命名模式：normal_xxx.png, low_xxx.png, xxx.png等")
        # 显示每个文件夹的样例文件
        print("\n各文件夹的样例文件：")
        for folder in all_folders[:3]:  # 只显示前3个文件夹
            images = find_images_in_folder(folder, img_ext)[:3]
            if images:
                folder_name = (
                    os.path.basename(folder)
                    if folder != ref_folder_path
                    else "GT文件夹"
                )
                print(f"  {folder_name}: {[os.path.basename(img) for img in images]}")
        return

    print(f"找到 {len(common_core_names)} 个共同文件（基于核心名称匹配）")
    print(
        f"GT文件夹：{ref_folder_name if not is_external_reference else reference_folder}"
    )
    print(f"比较文件夹：{compare_folders_list}")

    # 准备CSV数据 - 分别存储PSNR和SSIM
    psnr_data = []
    ssim_data = []

    # CSV表头
    header = ["Filename"] + compare_folders_list
    psnr_data.append(header.copy())
    ssim_data.append(header.copy())

    # 计算每个文件的指标
    print("\n开始计算指标...")
    for core_name in tqdm(common_core_names, desc="处理进度"):
        psnr_row = [core_name]
        ssim_row = [core_name]

        # 读取GT图像（使用映射的实际文件名）
        ref_actual_filename = filename_mappings[ref_folder_path][core_name]
        ref_img_path = os.path.join(ref_folder_path, ref_actual_filename)
        try:
            ref_img = imread(ref_img_path)
        except Exception as e:
            print(f"警告：无法读取GT图像 {ref_img_path}: {e}")
            continue

        # 与每个文件夹进行比较
        for folder_name in compare_folders_list:
            # 使用映射的实际文件名
            compare_actual_filename = filename_mappings[folder_paths[folder_name]][
                core_name
            ]
            compare_img_path = os.path.join(
                folder_paths[folder_name], compare_actual_filename
            )

            try:
                compare_img = imread(compare_img_path)

                # 确保图像尺寸一致
                if ref_img.shape != compare_img.shape:
                    print(
                        f"警告：图像尺寸不一致 {core_name} - {ref_folder_name} vs {folder_name}"
                    )
                    psnr_row.append("N/A")
                    ssim_row.append("N/A")
                    continue

                # 计算PSNR和SSIM
                psnr_val = calculate_psnr(ref_img, compare_img)
                ssim_val = calculate_ssim(ref_img, compare_img)

                psnr_row.append(f"{psnr_val:.4f}")
                ssim_row.append(f"{ssim_val:.4f}")

            except Exception as e:
                print(f"警告：处理图像时出错 {core_name} - {folder_name}: {e}")
                psnr_row.append("N/A")
                ssim_row.append("N/A")

        psnr_data.append(psnr_row)
        ssim_data.append(ssim_row)

    # 计算平均值
    psnr_avg_row = ["Average"]
    ssim_avg_row = ["Average"]

    for i, folder_name in enumerate(compare_folders_list):
        col_idx = i + 1  # 第一列是Filename，所以从1开始

        psnr_values = []
        ssim_values = []

        # 从PSNR数据中收集值
        for row in psnr_data[1:]:  # 跳过表头
            try:
                if row[col_idx] != "N/A":
                    psnr_values.append(float(row[col_idx]))
            except (ValueError, IndexError):
                pass

        # 从SSIM数据中收集值
        for row in ssim_data[1:]:  # 跳过表头
            try:
                if row[col_idx] != "N/A":
                    ssim_values.append(float(row[col_idx]))
            except (ValueError, IndexError):
                pass

        avg_psnr = np.mean(psnr_values) if psnr_values else 0
        avg_ssim = np.mean(ssim_values) if ssim_values else 0

        psnr_avg_row.append(f"{avg_psnr:.4f}")
        ssim_avg_row.append(f"{avg_ssim:.4f}")

    psnr_data.append(psnr_avg_row)
    ssim_data.append(ssim_avg_row)

    # 保存CSV文件
    if save_csv:
        # 生成两个CSV文件的路径
        base_path = os.path.splitext(save_csv)[0]
        psnr_csv_path = f"{base_path}_PSNR.csv"
        ssim_csv_path = f"{base_path}_SSIM.csv"

        os.makedirs(os.path.dirname(save_csv) or ".", exist_ok=True)

        # 保存PSNR CSV
        with open(psnr_csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerows(psnr_data)
        print(f"\nPSNR结果已保存到：{psnr_csv_path}")

        # 保存SSIM CSV
        with open(ssim_csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerows(ssim_data)
        print(f"SSIM结果已保存到：{ssim_csv_path}")

    # 打印摘要
    print("\n" + "=" * 80)
    print("摘要统计")
    print("=" * 80)
    print(
        f"GT文件夹：{ref_folder_name if not is_external_reference else reference_folder}"
    )
    print(f"处理文件数：{len(common_core_names)}")
    print("-" * 80)
    for i, folder_name in enumerate(compare_folders_list):
        col_idx = i + 1
        print(
            f"{folder_name:30s} PSNR: {psnr_avg_row[col_idx]:>8s} dB    SSIM: {ssim_avg_row[col_idx]:>8s}"
        )
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="比较多个文件夹下同名文件的PSNR和SSIM指标"
    )
    parser.add_argument(
        "--parent_dir", type=str, required=True, help="包含多个子文件夹的父目录路径"
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        default=r"D:\Workspace\Datasets\LOL-v2\Real_captured\Test\Normal",
        help="GT文件夹路径或名称。可以是：1) 绝对路径（外部GT文件夹） 2) parent_dir下的子文件夹名 3) None使用第一个子文件夹作为GT",
    )
    parser.add_argument(
        "--type", type=str, default="png", help="图像文件扩展名（默认：png）"
    )
    parser.add_argument(
        "--save_csv",
        type=str,
        default=None,
        help="CSV文件保存路径（默认：parent_dir/metrics_comparison.csv）",
    )

    args = parser.parse_args()

    # 设置默认CSV保存路径
    if args.save_csv is None:
        args.save_csv = os.path.join(args.parent_dir, "metrics_comparison.csv")

    compare_folders(
        parent_dir=args.parent_dir,
        reference_folder=args.gt_path,
        img_ext=args.type,
        save_csv=args.save_csv,
    )
