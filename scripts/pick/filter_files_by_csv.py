"""
根据 CSV 过滤文件夹中的文件
========================================
功能:
    读取 CSV 中的 Filename 列，保留文件夹中匹配的文件，删除不匹配的文件。
    默认预览模式 (dry-run)，加 --execute 才实际删除。

使用方法:
    # 预览（不实际删除）
    python scripts/pick/filter_files_by_csv.py ^
        --csv metrics_comparison_PSNR.csv ^
        --folders folder1 folder2

    # 递归处理子目录并实际删除
    python scripts/pick/filter_files_by_csv.py ^
        --csv metrics_comparison_PSNR.csv ^
        --folders folder1 folder2 --recursive --execute

参数:
    --csv        必选，包含 Filename 列的 CSV 文件
    --folders    必选，要处理的文件夹 (可多个)
    --recursive  递归处理子目录
    --execute    实际执行删除 (缺省为预览模式)
"""

import os
import csv
import argparse
from pathlib import Path
from typing import Set


def load_filenames_from_csv(csv_path: str) -> Set[str]:
    """
    从CSV文件中读取Filename列的所有值
    
    Args:
        csv_path: CSV文件路径
        
    Returns:
        包含所有filename数字的集合
    """
    filenames = set()
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get('Filename', '').strip()
            if filename:
                filenames.add(filename)
    
    print(f"从CSV中读取到 {len(filenames)} 个文件名")
    print(f"文件名列表: {sorted(filenames)}")
    return filenames


def should_keep_file(filename: str, target_filenames: Set[str]) -> bool:
    """
    判断文件是否应该保留
    
    Args:
        filename: 文件名
        target_filenames: 目标文件名集合
        
    Returns:
        如果文件名包含目标文件名中的任何一个数字，返回True
    """
    for target in target_filenames:
        if target in filename:
            return True
    return False


def filter_folder(folder_path: str, target_filenames: Set[str], dry_run: bool = True, recursive: bool = False) -> tuple:
    """
    过滤文件夹中的文件
    
    Args:
        folder_path: 文件夹路径
        target_filenames: 目标文件名集合
        dry_run: 如果为True，只显示将要删除的文件，不实际删除
        recursive: 如果为True，递归处理所有子目录
        
    Returns:
        (保留的文件数, 删除的文件数)
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"警告: 文件夹不存在: {folder_path}")
        return 0, 0
    
    if not folder.is_dir():
        print(f"警告: 路径不是文件夹: {folder_path}")
        return 0, 0
    
    kept_count = 0
    deleted_count = 0
    
    print(f"\n处理文件夹: {folder_path} {'(递归)' if recursive else '(非递归)'}")
    print("-" * 80)
    
    # 遍历文件夹中的所有文件
    if recursive:
        # 递归遍历所有子目录
        items = folder.rglob('*')
    else:
        # 只遍历当前目录
        items = folder.iterdir()
    
    for item in items:
        if item.is_file():
            filename = item.name
            # 对于递归模式，显示相对路径
            display_name = str(item.relative_to(folder)) if recursive else filename
            
            if should_keep_file(filename, target_filenames):
                print(f"✓ 保留: {display_name}")
                kept_count += 1
            else:
                if dry_run:
                    print(f"✗ 将删除: {display_name}")
                else:
                    print(f"✗ 删除: {display_name}")
                    item.unlink()
                deleted_count += 1
    
    print(f"\n文件夹 {folder_path} 统计:")
    print(f"  保留文件: {kept_count}")
    print(f"  {'将要' if dry_run else '已'}删除文件: {deleted_count}")
    
    return kept_count, deleted_count


def main():
    parser = argparse.ArgumentParser(
        description='根据CSV文件中的Filename列过滤文件夹中的文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 预览模式（不实际删除）
  python filter_files_by_csv.py --csv metrics.csv --folders folder1 folder2
  
  # 递归处理所有子目录（预览）
  python filter_files_by_csv.py --csv metrics.csv --folders folder1 folder2 --recursive
  
  # 实际删除文件
  python filter_files_by_csv.py --csv metrics.csv --folders folder1 folder2 --execute
  
  # 递归删除
  python filter_files_by_csv.py --csv metrics.csv --folders folder1 folder2 --recursive --execute
        """
    )
    
    parser.add_argument(
        '--csv',
        required=True,
        help='CSV文件路径（包含Filename列）'
    )
    
    parser.add_argument(
        '--folders',
        nargs='+',
        required=True,
        help='要处理的文件夹路径（可以指定多个）'
    )
    
    parser.add_argument(
        '--execute',
        action='store_true',
        help='实际执行删除操作（默认为预览模式）'
    )
    
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='递归处理所有子目录（默认只处理当前目录）'
    )
    
    args = parser.parse_args()
    
    # 读取CSV文件
    print("=" * 80)
    print("步骤 1: 读取CSV文件")
    print("=" * 80)
    target_filenames = load_filenames_from_csv(args.csv)
    
    if not target_filenames:
        print("错误: CSV文件中没有找到任何Filename")
        return
    
    # 处理每个文件夹
    print("\n" + "=" * 80)
    print(f"步骤 2: 过滤文件夹 ({'预览模式' if not args.execute else '执行模式'})")
    print("=" * 80)
    
    if not args.execute:
        print("⚠️  当前为预览模式，不会实际删除文件")
        print("⚠️  如需实际删除，请添加 --execute 参数")
    else:
        print("⚠️  警告: 将实际删除不匹配的文件！")
    
    total_kept = 0
    total_deleted = 0
    
    for folder in args.folders:
        kept, deleted = filter_folder(folder, target_filenames, dry_run=not args.execute, recursive=args.recursive)
        total_kept += kept
        total_deleted += deleted
    
    # 总结
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    print(f"处理文件夹数: {len(args.folders)}")
    print(f"保留文件总数: {total_kept}")
    print(f"{'将要' if not args.execute else '已'}删除文件总数: {total_deleted}")
    
    if not args.execute and total_deleted > 0:
        print("\n如确认删除，请运行:")
        folders_str = ' '.join(args.folders)
        recursive_flag = ' --recursive' if args.recursive else ''
        print(f"python {os.path.basename(__file__)} --csv {args.csv} --folders {folders_str}{recursive_flag} --execute")


if __name__ == '__main__':
    main()
