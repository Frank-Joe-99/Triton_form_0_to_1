# Layer Normalization and RMS Normalization

本章对应 Triton 官方教程 **Layer Normalization**，覆盖 LayerNorm 的前向与反向
传播，并以此学习 Triton 中的行级 reduction 和并行梯度归约。

另外，本章还增加了 RMSNorm 变体（比Layer Normalization计算过程简单一些），
主要聚焦于 RMSNorm 的前向传播，并对照 LayerNorm 理解归一化设计的取舍。

## RMS Normalization

本章对应 Triton 官方教程 **Layer Normalization** 中的 RMSNorm 变体，覆盖
RMSNorm 的前向与反向传播，并对照 LayerNorm 理解归一化设计的取舍。

### 学习目标

- 从 RMSNorm 公式推导 forward 只需 `mean(x^2)` 与 reciprocal rms，无需均值
- 对比 LayerNorm，理解去掉均值中心化后 backward 中 `dx` 的简化
- 理解 SRAM 容量如何限制单行可处理的最大 feature size

### 核心公式

$$
y = \frac{x}{\sqrt{\text{mean}(x^2)+\epsilon}}\odot w
$$

与 LayerNorm 相比，RMSNorm 省去了均值中心化与偏置项。Forward 只需保存每行的
rstd，backward 复用它，避免重新计算统计量。

### 文件

- `RMS-norm.ipynb`：中文导读、forward 内核



## Layer Normalization

### 学习目标

- 从 LayerNorm 公式推导 forward 所需的 mean、variance 和 reciprocal std
- 在一个 program 中处理一行并执行 `tl.sum` reduction
- 理解 backward 中 `dx`、`dw`、`db` 的数据依赖
- 使用 partial buffer 与 lock 对跨 program 的权重梯度进行归约
- 理解 SRAM 容量如何限制单行可处理的最大 feature size

### 核心公式

$$
y = \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}\odot w+b
$$

Forward 保存每行的 mean 与 reciprocal standard deviation，backward 复用它们，
避免重新计算统计量。`dx` 可按行独立完成，但 `dw` 和 `db` 需要汇总所有行的贡献，
因此需要额外的并行归约设计。

### 文件

- `layer-norm.ipynb`：中文导读、forward/backward 内核、自定义 autograd 与 benchmark

### 阅读重点

1. 先推导 `x_hat`、`c1`、`c2`，再阅读 backward kernel。
2. 区分“行内 reduction”和“跨 program reduction”。
3. 观察 BLOCK_SIZE 为什么取大于 N 的最小二次幂。

### 官方资料

- [Layer Normalization](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)




### 阅读重点

1. 对照 LayerNorm 的 `x_hat`、`c1`、`c2` 推导 RMSNorm 的 `dx`，确认 `c2` 项为何消失。
2. 区分“行内 reduction”和“跨 program reduction”，确认 `dw` 归约与 LayerNorm 完全一致。
3. 观察 BLOCK_SIZE 为什么取大于 N 的最小二次幂，以及 SRAM 对单行 feature size 的限制。