### 1. 将 LayerNorm 替换为 RMSNorm

* **怎么改：** Restormer 在 MDTA（注意力模块）和 GDFN（前馈网络）之前都使用了标准的 LayerNorm。你可以直接将其替换为 RMSNorm。
* **效果与原因：** 图像修复任务通常输入的是高分辨率图像（如 4K），特征图占据巨大的显存。RMSNorm 去掉了计算均值和方差平移的过程，这不仅能**加快前向和反向传播速度**，还能**节省一定的显存带宽**。在低级视觉（Low-level vision）任务中，特征的绝对均值往往没那么重要，RMSNorm 完全可以胜任。

### 2. 将 GDFN 升级为 SwiGLU 变体 (需保留卷积)

* **Restormer 的现状：** Restormer 的论文会发现，它提出的 **GDFN (Gated-Dconv Feed-Forward Network)** 本身就是一个门控网络！它将特征一分为二，其中一半经过 GELU 激活后与另一半相乘。这其实已经有了 LLaMA 里 GatedFFN 的雏形。
* **怎么改：**
  * 将 GDFN 中的激活函数从 **GELU 替换为 Swish/SiLU**（即变成 SwiGLU 的形态）。
  * **关键警告：** LLM 中的 SwiGLU 是纯线性层（Linear），但你在魔改 Restormer 时，**绝对不能丢掉里面的 3x3 Depthwise 卷积 (Dconv)**。因为对于图像修复任务，局部空间上下文（Spatial Inductive Bias）至关重要。纯 Linear 层无法捕捉相邻像素的瑕疵。
* **效果与原因：** Swish/SiLU 配合门控机制通常比 GELU 拥有更平滑的梯度和更强的非线性表达能力，能够在不增加额外计算量的情况下，提升网络对图像纹理和高频细节的恢复能力。

### 3. 将 MDTA 修改为 GQA 版本的通道注意力 (最具挑战性但也最有趣)

* **Restormer 的现状：** Restormer 为了解决高分辨率图像算力爆炸的问题，提出了 **MDTA (Multi-Dconv Head Transposed Attention)**。它不是在空间维度 $H \times W$ 上算注意力，而是在**通道维度 $C$** 上算注意力。它的计算复杂度是 $O(HW \cdot C^2)$，而不是标准注意力的 $O((HW)^2 \cdot C)$。
* **如何引入 GQA：**
  * 在 LLM 中，GQA 是将多个 Query 头共享一组 Key 和 Value 头，以减少 KV Cache。**但图像修复通常是一次性前向传播（非自回归），没有 KV Cache 机制。**
  * 在 Restormer 中引入 GQA 的目的不再是省 Cache，而是**减少模型参数量和 FLOPs（浮点运算次数）**。
  * 具体做法：假设你有 8 个 Query 头（对应通道的各个子空间），你可以只生成 2 个 Key 头和 2 个 Value 头。然后让相邻的 4 个 Query 头共享同一组 K 和 V 来计算通道注意力图（大小为 $C \times C$）。
* **效果与原因：** $K$ 和 $V$ 的生成层（通常是 $1 \times 1$ 卷积）通道数大幅减少。这可以在保持 MDTA 强大全局特征聚合能力的同时，进一步压榨网络的计算冗余，让模型变得更轻量（适合部署到手机端进行实时图像处理）。

---
