# Restormer LLIE 训练与测试脚本使用指南

本文档包含了 Restormer LLIE 项目的完整训练、测试和指标计算流程说明。

---

## 1. 训练脚本

### PowerShell 方式（推荐）

```powershell
# 设置使用的 GPU 设备编号（例如：使用第 2 块 GPU）
$env:CUDA_VISIBLE_DEVICES = "2"

# 启动训练
python basicsr\train.py -opt LLIE\Options\Restormer_128_2_60k.yml
```

### CMD 方式

```cmd
# 设置使用的 GPU 设备编号（例如：使用第 2 块 GPU）
set CUDA_VISIBLE_DEVICES=2

# 启动训练
python basicsr\train.py -opt LLIE\Options\Restormer_128_2_60k.yml
```

### Bash/Linux 方式

```bash
# 设置使用的 GPU 设备编号
export CUDA_VISIBLE_DEVICES=2

# 启动训练
python basicsr/train.py -opt LLIE/Options/Restormer_128_2_60k.yml
```

**参数说明：**

- `-opt`: 训练配置文件路径，包含网络结构、数据集路径、训练超参数等设置

**注意事项：**

- 确保数据集已正确放置在配置文件指定的目录
- 根据实际 GPU 数量和显存大小调整 `CUDA_VISIBLE_DEVICES` 和批处理大小
- 训练日志和检查点会保存在 `experiments/` 目录下

---

## 2. 测试脚本 (PowerShell 批量测试)

使用 `test.ps1` 脚本可以自动遍历指定权重目录下的所有 `.pth` 文件进行推理测试。

### 基本用法

```powershell
# 批量测试权重目录中的所有模型
.\LLIE\test.ps1 -WeightsDir ".\experiments\Restormer_128_2_60k_MDTA" -Opt "LLIE\Options\Restormer_128_2_60k.yml"
```

### 参数说明

- `-WeightsDir`: 包含 `.pth` 权重文件的目录，脚本会逐一测试其中的所有权重
- `-Opt`: 网络结构配置文件路径
- `-InputDir`: 待增强的图像目录（可选，默认指向 LOL-v2 Real Test Low）
- `-BaseResultDir`: 结果保存根目录（可选，默认在 `results/` 下以权重文件夹命名）

**注意事项：**

- 脚本会自动为每个权重在结果根目录下创建一个独立的子文件夹
- 确保您的开发环境已激活并可以运行 `python`
- 如果只需测试单个权重，可以创建一个只包含该权重的临时文件夹

---

## 3. 指标评价脚本 (PowerShell 一键评价)

使用 `eval_metrics.ps1` 脚本可以一次性计算 6 种图像质量指标（PSNR, SSIM, LPIPS, NIQE, MUSIQ, BRISQUE）。

### 基本用法

```powershell
# 对指定的结果目录进行批量指标评价
.\scripts\metrics\eval_metrics.ps1 -ResultsDir ".\results\Restormer_128_2_60k_MDTA"
```

### 参数说明

- `-ResultsDir`: 推理结果根目录（包含各权重结果的子目录）
- `-GtDir`: 地面真值（GT）图像目录（可选，默认指向 LOL-v2 Normal）
- `-UseGpu`: 是否使用 GPU 计算指标（可选，默认：$true）
- `-ImgExt`: 图像扩展名（可选，默认：png）

### 计算指标说明

1. **全参考指标 (Full-Reference)**: 需要与 GT 对比
   - **PSNR**: 越高越好
   - **SSIM**: 结构相似性，越高越好 (0-1)
   - **LPIPS**: 感知相似性，**越低越好**
2. **无参考指标 (No-Reference)**: 仅评估增强图像质量
   - **NIQE**: **越低越好**
   - **MUSIQ**: 越高越好 (0-100)
   - **BRISQUE**: **越低越好**

**注意事项：**

- 评价结果将按子目录名称保存在 `results/metrics/{实验名}/` 目录下
- 首次运行会自动从 Hugging Face 镜像站下载深度学习模型权重
- 建议开启 GPU 加速，因为 MUSIQ 和 LPIPS 在 CPU 上运行较慢

---

---

## 4. 完整工作流程

### 步骤 1: 训练模型 (PowerShell)

```powershell
$env:CUDA_VISIBLE_DEVICES = "0"
python basicsr\train.py -opt LLIE\Options\Restormer_128_2_60k.yml
```

### 步骤 2: 批量推理测试 (PowerShell)

```powershell
.\LLIE\test.ps1 -WeightsDir ".\experiments\Restormer_128_2_60k_MDTA" -Opt "LLIE\Options\Restormer_128_2_60k.yml"
```

### 步骤 3: 批量指标评价 (PowerShell)

```powershell
.\scripts\metrics\eval_metrics.ps1 -ResultsDir ".\results\Restormer_128_2_60k_MDTA"
```

---

## 5. 常见问题

### Q1: 训练时显存不足怎么办？

- 减小配置文件中的 `batch_size`
- 减小图像块大小 `gt_size`
- 使用更小的网络配置

### Q2: 测试时找不到模型权重？

- 检查 `--weights` 参数路径是否正确
- 确认训练已完成并保存了检查点
- 查看 `experiments/` 目录下的模型文件

### Q3: 指标计算时文件数量不匹配？

- 确保两个目录中的图像文件名完全一致
- 检查图像文件扩展名是否正确
- 确认没有隐藏文件或系统文件

### Q4: NIQE/MUSIQ/BRISQUE 计算失败？

- 确认已安装 `pyiqa` 和 `brisque` 库
- 检查网络连接，首次运行需要下载模型权重
- 如果下载慢，确认脚本已设置 HF_ENDPOINT 镜像

### Q5: 如何选择最佳检查点？

- 查看训练日志中的验证指标
- 测试多个检查点（如 40k, 50k, 60k）
- 综合考虑全参考和无参考指标

---

## 6. 目录结构参考

```
Restormer_LLIE/
├── datasets/
│   └── LOL-v2/
│       ├── Real_captured/
│       │   ├── Train/
│       │   │   ├── Low/
│       │   │   └── Normal/
│       │   └── Test/
│       │       ├── Low/
│       │       └── Normal/
│       └── Synthetic/
│           ├── Train/
│           └── Test/
├── LLIE/
│   ├── Options/
│   │   └── Restormer_128_2_60k.yml
│   ├── test.py
│   ├── metrics_calc_1.py  # 全参考指标
│   └── metrics_calc_2.py  # 无参考指标
├── experiments/
│   └── Restormer_128_2_60k_MDTA/
│       ├── models/
│       │   ├── net_g_20000.pth
│       │   ├── net_g_40000.pth
│       │   └── net_g_60000.pth
│       └── training_states/
└── results/
    └── Restormer_128_2_60k_MDTA_44000/
```
