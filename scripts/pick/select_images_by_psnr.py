#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按 PSNR 排名一致性挑选代表性图片
========================================
功能:
    读取 compare_folders_metrics.py 生成的 PSNR CSV，基于各模型的平均 PSNR 排名，
    筛选出单张图片 PSNR 排序与整体排名一致的代表性样本（用于论文展示）。

使用方法:
    直接修改 main() 中的 csv_path 后运行:
    python scripts/pick/select_images_by_psnr.py

输出:
    - {stem}_selected_strict.csv   严格符合排名的图片
    - {stem}_selected_partial.csv  部分符合排名的图片 (得分>=0.8)
    - {stem}_selected_all_scores.csv  所有图片及其一致性得分
    - {stem}_selected_report.txt   详细分析报告

说明:
    - 一致性得分 1.0 表示完全符合整体排名顺序
    - 如果严格符合的图片不足 10 张，自动补充得分第二高的图片
"""

import pandas as pd
from pathlib import Path


def get_model_ranking(df):
    """
    获取基于平均PSNR的模型排名（从高到低）
    
    Args:
        df: DataFrame containing PSNR metrics
        
    Returns:
        tuple: (ranking list, average values)
    """
    # 获取平均值行
    avg_row = df[df['Filename'] == 'Average']
    if avg_row.empty:
        raise ValueError("CSV文件中未找到'Average'行")
    
    # 提取各模型的平均PSNR
    avg_values = avg_row.iloc[0, 1:].astype(float)
    
    # 按PSNR从高到低排序
    ranking = avg_values.sort_values(ascending=False).index.tolist()
    
    return ranking, avg_values


def check_image_consistency(row, ranking):
    """
    检查单张图片的PSNR值是否符合整体排名顺序
    
    Args:
        row: 图片的PSNR数据行
        ranking: 模型排名列表（从高到低）
        
    Returns:
        bool: 是否符合排名顺序
    """
    # 获取该图片各模型的PSNR值
    psnr_values = row[ranking].astype(float).values
    
    # 检查是否严格按照降序排列
    is_consistent = all(psnr_values[i] >= psnr_values[i+1] 
                       for i in range(len(psnr_values)-1))
    
    return is_consistent


def calculate_ranking_score(row, ranking):
    """
    计算图片的排名一致性得分（允许部分符合）
    
    Args:
        row: 图片的PSNR数据行
        ranking: 模型排名列表（从高到低）
        
    Returns:
        float: 一致性得分 (0-1之间)
    """
    psnr_values = row[ranking].astype(float).values
    
    # 计算正确排序的对数
    correct_pairs = sum(1 for i in range(len(psnr_values)-1) 
                       if psnr_values[i] >= psnr_values[i+1])
    
    total_pairs = len(psnr_values) - 1
    score = correct_pairs / total_pairs if total_pairs > 0 else 0
    
    return score


def select_representative_images(csv_path, output_path=None, 
                                 consistency_threshold=1.0,
                                 min_score=0.8):
    """
    选择符合整体排名的代表性图片
    
    Args:
        csv_path: 输入CSV文件路径
        output_path: 输出CSV文件路径（如果为None，自动生成）
        consistency_threshold: 严格一致性阈值 (1.0表示完全符合)
        min_score: 最小一致性得分阈值
        
    Returns:
        tuple: (严格符合的DataFrame, 部分符合的DataFrame)
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    print(f"总图片数: {len(df) - 1}")  # 减去Average行
    print("\n" + "="*60)
    
    # 获取模型排名
    ranking, avg_values = get_model_ranking(df)
    
    print("基于平均PSNR的模型排名 (从高到低):")
    for i, model in enumerate(ranking, 1):
        print(f"  {i}. {model}: {avg_values[model]:.4f}")
    
    print("\n" + "="*60)
    
    # 分离数据行和平均值行
    data_df = df[df['Filename'] != 'Average'].copy()
    
    # 计算每张图片的一致性
    data_df['consistency_score'] = data_df.apply(
        lambda row: calculate_ranking_score(row, ranking), axis=1
    )
    data_df['is_strictly_consistent'] = data_df.apply(
        lambda row: check_image_consistency(row, ranking), axis=1
    )
    
    # 筛选符合要求的图片
    strictly_consistent = data_df[data_df['is_strictly_consistent']].copy()
    partially_consistent = data_df[
        (~data_df['is_strictly_consistent']) & 
        (data_df['consistency_score'] >= min_score)
    ].copy()
    
    print(f"\n严格符合排名的图片数: {len(strictly_consistent)}")
    print(f"部分符合排名的图片数 (得分>={min_score}): {len(partially_consistent)}")
    
    # 统计信息
    print("\n一致性得分统计:")
    print(f"  平均得分: {data_df['consistency_score'].mean():.3f}")
    print(f"  中位数: {data_df['consistency_score'].median():.3f}")
    print(f"  标准差: {data_df['consistency_score'].std():.3f}")
    
    # 保存结果
    if output_path is None:
        csv_dir = Path(csv_path).parent
        base_name = Path(csv_path).stem
        output_path = csv_dir / f"{base_name}_selected.csv"
    
    output_path = Path(output_path)
    
    # 保存严格符合的图片（如果不足10张，补充得分第二高的图片）
    strict_output = output_path.parent / f"{output_path.stem}_strict{output_path.suffix}"
    
    # 准备输出数据
    if len(strictly_consistent) > 0:
        output_df = strictly_consistent.copy()
        output_df['note'] = '严格符合'
    else:
        # 如果没有严格符合的图片，创建空DataFrame
        output_df = pd.DataFrame(columns=list(data_df.columns) + ['note'])
    
    # 如果严格符合的图片少于10张，补充得分第二高的图片
    if len(strictly_consistent) < 10:
        # 获取所有唯一的得分值（降序排列）
        unique_scores = sorted(data_df['consistency_score'].unique(), reverse=True)
        
        # 找到严格符合图片的最高得分（应该是1.0）
        max_score = strictly_consistent['consistency_score'].max() if len(strictly_consistent) > 0 else 1.0
        
        # 找到第二高的得分
        second_scores = [s for s in unique_scores if s < max_score]
        
        if second_scores:
            second_highest_score = second_scores[0]
            # 获取得分第二高的所有图片
            second_tier = data_df[data_df['consistency_score'] == second_highest_score].copy()
            second_tier['note'] = f'补充(得分={second_highest_score:.2f})'
            
            # 合并数据
            output_df = pd.concat([output_df, second_tier], ignore_index=True)
            
            print(f"\n注意: 严格符合的图片仅{len(strictly_consistent)}张，已补充{len(second_tier)}张得分第二高的图片(得分={second_highest_score:.2f})")
    
    # 重新排列列，将注释列放在最后
    cols = ['Filename'] + ranking + ['consistency_score', 'is_strictly_consistent', 'note']
    output_df[cols].to_csv(strict_output, index=False)
    print(f"\n严格符合的图片已保存到: {strict_output}")
    
    # 显示示例
    print(f"\n前{min(10, len(output_df))}个图片示例:")
    display_cols = ['Filename'] + ranking + ['note']
    print(output_df[display_cols].head(10).to_string(index=False))
    
    # 保存部分符合的图片
    if len(partially_consistent) > 0:
        partial_output = output_path.parent / f"{output_path.stem}_partial{output_path.suffix}"
        cols = ['Filename'] + ranking + ['consistency_score', 'is_strictly_consistent']
        partially_consistent[cols].to_csv(partial_output, index=False)
        print(f"\n部分符合的图片已保存到: {partial_output}")
        
        # 显示前几个示例
        print("\n前5个部分符合排名的图片示例:")
        print(partially_consistent[['Filename'] + ranking + ['consistency_score']].head().to_string(index=False))
    
    # 保存所有图片及其得分
    all_output = output_path.parent / f"{output_path.stem}_all_scores{output_path.suffix}"
    cols = ['Filename'] + ranking + ['consistency_score', 'is_strictly_consistent']
    data_df[cols].to_csv(all_output, index=False)
    print(f"\n所有图片及其得分已保存到: {all_output}")
    
    # 生成详细分析报告
    generate_analysis_report(data_df, ranking, avg_values, output_path)
    
    return strictly_consistent, partially_consistent


def generate_analysis_report(data_df, ranking, avg_values, output_path):
    """
    生成详细的分析报告
    
    Args:
        data_df: 包含所有图片数据的DataFrame
        ranking: 模型排名列表
        avg_values: 平均PSNR值
        output_path: 输出路径
    """
    report_path = Path(output_path).parent / f"{Path(output_path).stem}_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("图片排名一致性分析报告\n")
        f.write("="*70 + "\n\n")
        
        f.write("1. 整体模型排名 (基于平均PSNR，从高到低):\n")
        f.write("-" * 50 + "\n")
        for i, model in enumerate(ranking, 1):
            f.write(f"   {i}. {model:8s}: {avg_values[model]:.4f}\n")
        
        f.write("\n2. 图片一致性统计:\n")
        f.write("-" * 50 + "\n")
        f.write(f"   总图片数: {len(data_df)}\n")
        f.write(f"   严格符合排名: {sum(data_df['is_strictly_consistent'])} "
                f"({sum(data_df['is_strictly_consistent'])/len(data_df)*100:.1f}%)\n")
        f.write(f"   部分符合排名 (得分>=0.8): {sum(data_df['consistency_score'] >= 0.8)} "
                f"({sum(data_df['consistency_score'] >= 0.8)/len(data_df)*100:.1f}%)\n")
        
        f.write("\n3. 一致性得分分布:\n")
        f.write("-" * 50 + "\n")
        f.write(f"   平均得分: {data_df['consistency_score'].mean():.3f}\n")
        f.write(f"   中位数: {data_df['consistency_score'].median():.3f}\n")
        f.write(f"   标准差: {data_df['consistency_score'].std():.3f}\n")
        f.write(f"   最小值: {data_df['consistency_score'].min():.3f}\n")
        f.write(f"   最大值: {data_df['consistency_score'].max():.3f}\n")
        
        # 得分分布
        f.write("\n   得分分布:\n")
        for threshold in [1.0, 0.8, 0.6, 0.4, 0.2]:
            count = sum(data_df['consistency_score'] >= threshold)
            percentage = count / len(data_df) * 100
            f.write(f"     得分 >= {threshold:.1f}: {count:3d} ({percentage:5.1f}%)\n")
        
        f.write("\n4. 最佳代表性图片 (完全符合排名):\n")
        f.write("-" * 50 + "\n")
        strict_images = data_df[data_df['is_strictly_consistent']]
        if len(strict_images) > 0:
            # 按照第一个模型的PSNR排序
            strict_images_sorted = strict_images.sort_values(by=ranking[0], ascending=False)
            for idx, row in strict_images_sorted.head(10).iterrows():
                f.write(f"   {row['Filename']}: ")
                f.write(" -> ".join([f"{row[model]:.2f}" for model in ranking]))
                f.write("\n")
        else:
            f.write("   无完全符合排名的图片\n")
        
        f.write("\n5. 模型间PSNR相关性分析:\n")
        f.write("-" * 50 + "\n")
        correlation_matrix = data_df[ranking].corr()
        for i, model1 in enumerate(ranking):
            for model2 in ranking[i+1:]:
                corr = correlation_matrix.loc[model1, model2]
                f.write(f"   {model1} vs {model2}: {corr:.3f}\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"\n详细分析报告已保存到: {report_path}")


def main():
    """主函数"""
    # CSV文件路径
    csv_path = r"d:\Workspace\A_Projects\Thesis\LLIE\Restormer_LLIE\metrics_comparison_PSNR.csv"
    
    # 检查文件是否存在
    if not Path(csv_path).exists():
        print(f"错误: 文件不存在 - {csv_path}")
        return
    
    print("正在分析PSNR数据...")
    print("="*60)
    
    # 选择代表性图片
    strictly_consistent, partially_consistent = select_representative_images(
        csv_path=csv_path,
        consistency_threshold=1.0,  # 严格一致性
        min_score=0.8  # 部分符合的最低得分
    )
    
    print("\n" + "="*60)
    print("分析完成!")


if __name__ == "__main__":
    main()
