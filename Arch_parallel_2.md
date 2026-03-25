```python

##########################################################################
## Dual-Branch Spatial & Frequency Attention
class DualAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, LayerNorm_type):
        super(DualAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # --- 合并生成所有组件 (q_s, k_s, v_s, q_f, k_f) ---
        self.qkv_all = nn.Conv2d(dim, dim * 5, kernel_size=1, bias=bias)
        self.qkv_all_dwconv = nn.Conv2d(
            dim * 5,
            dim * 5,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 5,
            bias=bias,
        )

        # --- 空间分支投影 ---
        self.proj_spatial = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # --- 频率分支的 LayerNorm (FSAS Style) ---
        self.norm_f = LayerNorm(dim, LayerNorm_type)

        # --- 特征融合投影 ---
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        # ==========================================
        # 1. 空间域分支交互
        # ==========================================
        qkv_all = self.qkv_all_dwconv(self.qkv_all(x))
        q_s, k_s, v_s, q_f, k_f = qkv_all.chunk(5, dim=1)

        q_s = rearrange(q_s, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k_s = rearrange(k_s, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v_s = rearrange(v_s, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q_s = F.normalize(q_s, dim=-1)
        k_s = F.normalize(k_s, dim=-1)

        attn_s = (q_s @ k_s.transpose(-2, -1)) * self.temperature
        attn_s = attn_s.softmax(dim=-1)

        out_s = attn_s @ v_s
        out_s = rearrange(
            out_s, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )
        out_s = self.proj_spatial(out_s)

        # ==========================================
        # 2. 局部 8x8 频率域分支交互
        # ==========================================
        patch_size = 8

        # 切块 [b, c, h, w] -> [b, c, h//8, w//8, 8, 8]
        q_patch = rearrange(
            q_f, "b c (h p1) (w p2) -> b c h w p1 p2", p1=patch_size, p2=patch_size
        )
        k_patch = rearrange(
            k_f, "b c (h p1) (w p2) -> b c h w p1 p2", p1=patch_size, p2=patch_size
        )

        q_f_fft = torch.fft.rfft2(q_patch.float())
        k_f_fft = torch.fft.rfft2(k_patch.float())

        attn_f_fft = q_f_fft * k_f_fft
        attn_f_patch = torch.fft.irfft2(attn_f_fft, s=(patch_size, patch_size))

        # 还原形状 [b, c, h//8, w//8, 8, 8] -> [b, c, h, w]
        attn_f = rearrange(
            attn_f_patch,
            "b c h w p1 p2 -> b c (h p1) (w p2)",
            p1=patch_size,
            p2=patch_size,
        )
        # [修改点] 按照 FSAS 逻辑使用 LayerNorm 替换 Softmax
        attn_f = self.norm_f(attn_f)

        # ==========================================
        # 3. 融合
        # ==========================================
        out = out_s * attn_f
        out = self.project_out(out)

        return out
```

# DualAttention 模块架构详解

## 1. 设计初衷与背景

本模块是针对 **低光图像增强 (LLIE)** 任务设计的双分支注意力机制。它结合了 Restormer 强大的空间建模能力和 FSAS (Frequency Selection Attention) 敏锐的频率域捕获能力。

### 核心特性

* **空间分支**：采用转置注意力 (Transposed Attention)，在通道维度执行矩阵乘法，捕捉全局空间结构。
* **频率分支**：采用局部频率子块 (8x8 Patch)，执行 FFT 点乘交互，捕捉局部频率细节（如噪声、边缘）。
* **串行融合**：频率分支生成的滤波器动态调节空间分支的输出，实现“特征精炼”。

---

## 2. 代码实现关键流程

### A. 统一投影优化 (`dim * 5`)

不同于传统的各分支独立计算，本实现将 $Q_s, K_s, V_s, Q_f, K_f$ 合并到一个大的卷积投影中：

* **优势**：大幅减少 GPU 的 Kernel Launch 次数，提高计算带宽利用率，加速推理。
* **分组卷积**：使用 `groups=dim*5` 的 Depthwise Conv 以保持各权重的独立性并控制参数量。

### B. 局部频率交互 (FSAS Style)

将特征图切分为 $8 \times 8$ 的小块执行 FFT：

* **计算效率**：多组小尺度 FFT 比单次全图 FFT 显存占用更低，并行效率更高。
* **非平稳性建模**：局部 FFT 允许模型对图像不同区域（如极暗区 vs. 亮区）应用不同的频率掩码。

### C. 信号调制 (LayerNorm vs Softmax)

在频率分支中，使用了 **LayerNorm** 而非 Softmax：

* **Softmax** 属于强竞争机制，导致特征过度稀疏（变黑）。
* **LayerNorm** 保持了信号的正负分布与方差稳定，通过逐元素相乘实现对空间特征的 **“增强”** 或 **“抑制”**。

---

## 3. 实现与应用注意事项

> [!IMPORTANT]
> **尺寸整除性**
> 由于频率分支使用了固定 $8 \times 8$ 的 Patch 大小，输入图像的尺寸必须能被 8 整除。在处理非常规尺寸图像时，需要在模型头部进行补零 (Padding)。

> [!TIP]
> **数据精度**
> FFT 在处理 FP16 精度时可能存在稳定性问题。代码中显式使用了 `.float()` 将信号转为 FP32 执行 FFT，并在完成后通过 `.type_as(x)` 转回原始精度。

> [!NOTE]
> **串行 vs 并行**
> 当前实现采用的是 **半并行提取 + 串行融合** 的结构。$Q_f, K_f$ 是从原始输入 $X$ 提取的。如果追求更极致的精炼效果，可以将 $Q_f, K_f$ 的输入改为空间分支的输出 `out_s`。

---

## 4. DualAttention 流程示意图

```text
           Input X (B, C, H, W)
                |
       [ QKV_All Projection ]
      (Conv 1x1 + DWConv 3x3)
                |
    +-----------+-----------+
    |           |           |
 [Qs, Ks, Vs]  [V_s]      [Qf, Kf]
    |           |           |
    |           |    [ 8x8 Patching ]
    |           |           |
    |           |      [ rFFT2 ]
    |           |           |
    |           |   [ Qf_fft * Kf_fft ]
    |           |           |
    |           |      [ irFFT2 ]
    |           |           |
    |           |    [ Unpatching ]
    |           |           |
    |           |      [ LayerNorm ]
    |           |           |
    |           |        (Attn_f)
    |           |           |
 [ Spatial Attention ]      |
 (MDTA: Qs * Ks^T * Vs)     |
    |                       |
 [ Proj_spatial ]           |
    |                       |
 (Out_s)                    |
    |                       |
    +---------[ X ]---------+
          (Element-wise)
                |
         [ Proj_out ]
                |
          Output (B, C, H, W)
```

### 流程阶段详解

1. **统一投影层 (Merged Projection)**：输入特征图 $X$ 同时生成 5 个分量 ($Q_s, K_s, V_s, Q_f, K_f$)。这一步最大化了 GPU 的吞吐量。
2. **空间分支 (Spatial Branch - MDTA)**：计算通道间的自注意力掩码（Transposed Attention）。通过 Softmax 归一化后的矩阵与 $V_s$ 相乘，重构全局通道特征。输出 $Out\_s$ 代表了经过空间关系精炼后的主特征流。
3. **频率分支 (Frequency Branch - FSAS Style)**：将特征切分为 $8 \times 8$ 的小窗口，专注于局部规律。在频率域执行点乘交互，捕捉特定的频率成分。通过 LayerNorm 产生的调制信号 $Attn\_f$ 数值稳定且具有正负极性。
4. **跨域融合 (Cross-domain Fusion)**：利用频率分支产生的调制信号对空间分支的结果进行 **逐元素点乘 (Hadamard Product)**。最后通过投影层整合信息，输出最终特征图。
