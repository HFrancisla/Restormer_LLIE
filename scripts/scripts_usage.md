# Scripts 使用指南

`scripts/` 目录下的辅助脚本，覆盖 FLOPs 计算、指标评价、图片筛选、注意力可视化四类任务。  
所有命令均在 **项目根目录** 下运行。

---

## 目录总览

```text
scripts/
├── flops/
│   └── calc_flops.py              # FLOPs & 参数量计算
├── metrics/
│   ├── metrics_calc_all.py        # 单目录 6 指标计算 (核心)
│   ├── batch_eval_metrics.py      # 批量目录 6 指标计算
│   └── eval_metrics.ps1           # PowerShell 一键评价入口
├── pick/
│   ├── compare_folders_metrics.py # 跨文件夹 PSNR/SSIM 对比 → CSV
│   ├── select_images_by_psnr.py   # 按排名一致性挑选代表图
│   └── filter_files_by_csv.py     # 按 CSV 过滤/删除文件
└── visualization/
    ├── common.py                  # 公共模块 (不可直接运行)
    ├── visualize_attention.py     # Encoder 任意层/块注意力可视化
    ├── visualize_first_block.py   # Level1 Block0 完整流程可视化
    ├── visualize_encoder_all.ps1  # 全量 Encoder 可视化批处理
    └── visualize_first_block_all.ps1  # 全量 Block0 可视化批处理
```

---

## 1. FLOPs & 参数量

```powershell
python scripts/flops/calc_flops.py
```

- 无需参数，输入尺寸在脚本内 `input_sizes` 中配置
- 结果追加写入 `Flops&Params.txt`

---

## 2. 指标评价

### 2.1 单目录计算 (metrics_calc_all.py)

计算 PSNR↑ / SSIM↑ / LPIPS↓ / NIQE↓ / MUSIQ↑ / BRISQUE↓ 共 6 项指标。

```powershell
python scripts/metrics/metrics_calc_all.py `
    --dirA ./datasets/LOL-v2/Real_captured/Test/Normal `
    --dirB ./results/Exp/net_g_44000 `
    --type png --use_gpu --save_txt results.txt
```

### 2.2 批量目录计算 (batch_eval_metrics.py)

自动遍历 `--results_dir` 下所有子文件夹并逐一计算。

```powershell
python scripts/metrics/batch_eval_metrics.py `
    --results_dir ./results/Experiment_A `
    --gt_dir ./datasets/LOL-v2/Real_captured/Test/Normal `
    --use_gpu
```

输出: `results/metrics/{实验名}/{子文件夹}.txt`

### 2.3 PowerShell 一键评价 (eval_metrics.ps1)

对 `batch_eval_metrics.py` 的封装，提供默认 GT 路径。

```powershell
.\scripts\metrics\eval_metrics.ps1 -ResultsDir .\results\Experiment_A
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-ResultsDir` | 推理结果根目录 | (必选) |
| `-GtDir` | GT 图像目录 | LOL-v2 Real Test Normal |
| `-UseGpu` | GPU 加速 | `$true` |
| `-ImgExt` | 图像扩展名 | `png` |

---

## 3. 图片筛选

### 3.1 跨文件夹 PSNR/SSIM 对比 (compare_folders_metrics.py)

```powershell
python scripts/pick/compare_folders_metrics.py `
    --parent_dir ./results/Experiment_A `
    --gt_path D:/Datasets/LOL-v2/Real_captured/Test/Normal
```

输出: `{parent_dir}/metrics_comparison_PSNR.csv` 和 `_SSIM.csv`

### 3.2 按排名一致性挑选代表图 (select_images_by_psnr.py)

```powershell
# 修改 main() 中的 csv_path 后运行
python scripts/pick/select_images_by_psnr.py
```

输出: `_strict.csv` / `_partial.csv` / `_all_scores.csv` / `_report.txt`

### 3.3 按 CSV 过滤文件 (filter_files_by_csv.py)

```powershell
# 预览模式
python scripts/pick/filter_files_by_csv.py `
    --csv metrics_comparison_PSNR.csv --folders folder1 folder2

# 实际删除
python scripts/pick/filter_files_by_csv.py `
    --csv metrics_comparison_PSNR.csv --folders folder1 folder2 --execute
```

---

## 4. 注意力可视化

### 4.1 Encoder 任意层/块 (visualize_attention.py)

```powershell
python scripts/visualization/visualize_attention.py `
    --image ./low00323.png `
    --checkpoint ./experiments/Restormer_128_2_60k_MDTA/net_g_44000.pth `
    --attn_type MDTA --level 1 --block 0
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--image` | 输入图像 | (必选) |
| `--checkpoint` | 模型权重 | (必选) |
| `--attn_type` | MDTA / HTA / WTA | (必选) |
| `--level` | 层级 1-3, Latent=4 | `1` |
| `--block` | Block 索引, -1=最后 | `-1` |
| `--resize` | 是否 resize | 不 resize |

### 4.2 Level1 Block0 完整流程 (visualize_first_block.py)

```powershell
python scripts/visualization/visualize_first_block.py `
    --image ./low00323.png `
    --checkpoint ./experiments/Restormer_128_2_60k_MDTA/net_g_44000.pth `
    --attn_type MDTA
```

生成 3×4 总览图 + Q/K/V/Attention Map 等单独高清图。

### 4.3 批处理脚本

```powershell
# Encoder 全量可视化 (多层多块)
.\scripts\visualization\visualize_encoder_all.ps1

# Block0 全量可视化 (多图 × 多注意力)
.\scripts\visualization\visualize_first_block_all.ps1
```

运行前在脚本内配置图像路径和权重路径。
