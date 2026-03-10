# 尺度自适应的空频混合网络架构 (Scale-Aware Hybrid Restormer+FFTformer)

本文档详细记录了在低光照图像增强（LLIE）任务中，为了兼顾 **Restormer（空域局部精度）** 和 **FFTformer（频域全局统筹）** 两者的优势而设计的深度融合与架构调优方案。

---

## 1. 核心设计动机与现状痛点

在处理低光照图像增强任务时，模型需要解决两个棘手的问题：

1. **宏观的光斑色偏与低频光照退化**：这需要极大的“全局感受野”。
2. **微观的暗光噪点与边缘模糊**：这需要对像素和空间特征有极其细腻的“通道级聚焦”。

**目前的痛点（缝合怪的代价）：**
如果直接在原始 4 层 U-Net 的每一个 `TransformerBlock` 内部都**强行并行**原版 Restormer 的空间模块（MDTA/GDFN）和 FFTformer 的频域模块（FSAS/DFFN），会引发致命的显存 OOM 危机以及无法接受的运算时间：

- 浅层高分辨率特征做全图 FFT 极度挥霍算力，且 $8 \times 8$ 的 Patch 在初期（如 $128 \times 128$ 的图中）几乎只能看到像素级高频噪声，无法抓取宏观长距离低频退化，与 3x3 卷积功能重复。
- 深层如果全套搬运，特征尺寸又太小（如 $16 \times 16$），失去了做 2D FFT 抓取空间“波动”的物理意义。

**核心解决思路：Scale-Aware（尺度感应）分级策略**
并非所有层都适合频域，也并非所有层都需要空间流。合理的双轨设计应顺应下采样的分辨率阶段，进行战术分配。

---

## 2. 网络层级开关机制 (Level-wise Strategy)

基于原始网络在各个特征级（Level）的分辨率与通道数量变化，我们设计了四种不同算力焦点的运行模式：

| 网络下采样层级 | 输入尺寸/通道 | Attention 注意力开关 | FFN 前馈网络开关 | 设计意图 (Rational) |
| :--- | :--- | :--- | :--- | :--- |
| **Encoder L1 & L2** | $H \times W$<br>大尺寸 / 小通道 | **MDTA** (通道空间)<br>~~FSAS~~ (关) | **DWConv** (空间)<br>~~FFT Filter~~ (关) | 高分辨率区域全面弃用频域！利用纯空域架构，以最低成本提取局部纹理边缘，保护高频细节。 |
| **Encoder L3** | $\frac{H}{4} \times \frac{W}{4}$<br>中等尺寸 / 中等通道 | **MDTA** (通道空间)<br>~~FSAS~~ (关) | **DWConv** (开)<br>**FFT Filter** (开) | 准备过渡期。利用空域继续提取局部信息，但此时 $8 \times 8$ 的 FFT Filter 视野扩大，开始介入并主导剔除低频色偏。 |
| **Latent (瓶颈 L4)** | $\frac{H}{8} \times \frac{W}{8}$<br>极小尺寸 / 极致通道 | **MDTA** (通道空间)<br>~~FSAS~~ (关) | **DWConv** (开)<br>**FFT Filter** (开) | 尺寸极小时 FFT 注意力无用武之地，纯靠 MDTA 榨干海量通道（如384维）的语义交互。双开 FFN 保持深层照度修正。 |
| **Decoder L3 ~ L1** | 对应逐层上采样放大 | **双轨全开**<br>(MDTA + FSAS) | **双域门控**<br>(DWConv + FFT) | 特征重建阶段！融合跳跃连接层后解开性能封印。频域统筹大局，空域重建锋利边缘，达到极限恢复画质。 |

---

## 3. 核心代码模块实现与重构

为了支撑以上按层开关切分的网路结构，重构了 `basicsr/models/archs/restormer_arch.py` 内部的两大核心模块：

### 3.1 动态双域前馈网络 (`DD_GDFN`)

通过在初始化时传入 `use_spatial` 和 `use_freq` 布尔值，动态分配内部通道的乘积比例，坚决不分配冗余内存：

```python
class DD_GDFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, use_spatial=True, use_freq=True):
        super(DD_GDFN, self).__init__()
        self.use_spatial = use_spatial
        self.use_freq = use_freq
        self.patch_size = 8
        hidden_features = int(dim * ffn_expansion_factor)

        # 动态计算膨胀倍数：门控信号(1) + 空间流(1，如果开启) + 频域流(1，如果开启)
        multiplier = 1 + int(self.use_spatial) + int(self.use_freq)
        self.project_in = nn.Conv2d(dim, hidden_features * multiplier, kernel_size=1, bias=bias)

        if self.use_spatial:
            self.dwconv_spatial = nn.Conv2d(...)
        if self.use_freq:
            self.fft_weight = nn.Parameter(...) # FFT 在这里学习不同 patch 位置的频域全局退化分布

    def forward(self, x):
        # 拆分特征用于不同的处理支线
        features = self.project_in(x).chunk(1 + int(self.use_spatial) + int(self.use_freq), dim=1)
        gate, idx, out_fused = features[0], 1, 0
        
        if self.use_spatial:
            out_fused += self.dwconv_spatial(features[idx])
            idx += 1
        if self.use_freq:
            # 执行频域 FFT 与参数系数 self.fft_weight 的点乘滤波
            out_fused += fft_filter(features[idx], self.fft_weight)
            
        # 激活过后的门控统一缩放
        return self.project_out(F.gelu(gate) * out_fused)
```

### 3.2 自适应控制阀的 `TransformerBlock`

重构 Block 控制流，使得它不再强绑定某种注意力方式，还能控制对加权权重的声明。

```python
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type,
                 use_spatial_attn=True, use_freq_attn=True, 
                 use_spatial_ffn=True, use_freq_ffn=True):
        # ...
        if self.use_spatial_attn and self.use_freq_attn:
            self.weight_attn = nn.Parameter(torch.ones(2))      # 当双开时，学习一个融合占比阈值
            
        self.ffn = DD_GDFN(..., use_spatial=use_spatial_ffn, use_freq=use_freq_ffn)
        
    def forward(self, x):
        # 根据所给的开关分别跑对应 Attention 并且如果双开则进行 Softmax(weight_attn) 加权融合
        # 最后送入自我管理开关的 ffn ...
        return x + attn_out + self.ffn(...)
```

### 3.3 主架构网络 `Restormer` 的分配构建

依靠开关参数，对深达 4 层 U-Net 的海量模块进行上述分级战术布置，让前期模型“减负”，后期模型“火力全开”。

```python
self.encoder_level1 = nn.Sequential(*[TransformerBlock(...,
    use_spatial_attn=True, use_freq_attn=False, use_spatial_ffn=True, use_freq_ffn=False
) for _ in range(num_blocks[0])]) # 切断一切 FFT 运算！

self.encoder_level3 = nn.Sequential(*[TransformerBlock(...,
    use_spatial_attn=True, use_freq_attn=False, use_spatial_ffn=True, use_freq_ffn=True
) for _ in range(num_blocks[2])]) # 保留通道注意力，打开 FFN 的频域处理

# 恢复重建阶层（Decoder 3, 2, 1 及后续 Refinement 将四路全部设定为 True）
```

---

## 4. 双域混合损失函数设计 (Dual-Domain Supervision)

除了在网络架构前向传播上的“空频融合”，后向传播的梯度求导（Loss）同样需要匹配双领域特征，以保证训练时不出现“偏科”（比如网络只更新空域部分，偷懒不学频域权重）的现象。

因此，引入了兼具空域和频域的双边损失函数（Dual-Domain Loss）：

1. **空域保真度 (Spatial Domain: L1 Loss)**：
   用来主导图像像素级（如 `128x128`）的恢复与重建。保证基础结构的信噪比、均方误差符合人眼直观感知。作为主导力量，权重（`loss_weight`）设为 `1.0`。

2. **频域结构感应 (Frequency Domain: FFT Loss)**：
   对网络输出（`pred`）和真实图像（`target`）分别进行 2D 傅里叶变换，提取出实部（`real`）和虚部（`imag`），进而在复数域张量空间直接算 `L1Loss`。
   - **理论支撑**：频域极具代表性的高低频分布，强制在 Loss 层面拉平两者的“振幅（Amplitude）”和“相位（Phase）”。在低光照任务中，“相位”不对齐极容易造成严重的区域色偏黑斑，仅仅依靠空域 Loss（只求像素间差异和最小）网络难以矫正全局色偏。
   - 强迫深层（如 Level 3, L4）和 Decoder 各个级段带有 `fft_weight` 参数的 `DD_GDFN` 和 `FSAS` 全速收敛。在此作为微调信号与辅助约束，权重配置通常为 `0.1`。

**集成应用代码层 (*image_restoration_model.py*)：**
混合梯度汇聚，在引擎最底层累加损失直接后馈：

```python
total_loss = l_pix
if self.cri_fft is not None:
    loss_dict['l_fft'] = l_fft
    total_loss += l_fft  # 混合空域与频域梯度！
    
total_loss.backward()
```

---

## 5. 总结与优化成效

这种融合重构，完美兼容了原始两篇顶会论文（Restormer 和 FFTformer）的核心架构意图（FFTformer 作者刻意在 Encoder 抹去 FSAS 注意力层同样印证了这点）。

**实际收益：**

- **性能解放**：抛弃了不合场景下的冗余频域计算，高分辨率层级的处理提速明显，且极大释放了被吞噬显存资源（Batch Size 可大幅回升）。
- **指标提点**：模型能精准把捉何种特征尺度该采用“空间局部挖掘（Spatial）”亦或者“全局暗光环境滤波（Frequency）”，特征耦合现象（即各种机制各管各的却不协调甚至效果抵消）得到彻底改善。
