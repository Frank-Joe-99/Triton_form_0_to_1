# Fused Attention

本章对应 Triton 官方教程 **Fused Attention**，实现 FlashAttention v2 的 Triton
版本，涵盖 causal/non-causal forward 与 backward。

## 学习目标

- 理解标准 attention 为什么会物化一个 `N × N` 的 score 矩阵
- 使用分块计算和 online softmax 降低显存读写
- 理解 running maximum、normalizer 与数值稳定性的关系
- 使用 causal mask 限制当前位置只能关注历史 token
- 认识 forward/backward 的分块方式、重计算策略与性能度量

## 核心思路

标准公式为 `softmax(QK^T * scale) V`。FlashAttention 不把完整 score 和
probability 矩阵写入全局内存，而是在片上逐块更新 softmax 状态和输出累加器。
因此数学结果不变，但 I/O 复杂度显著降低。

## 文件

- `fused-attention.ipynb`：中文阅读路线、官方完整实现、测试与 benchmark

## 阅读重点

1. 先单独理解 online softmax，再进入 kernel。
2. 标注 Q/K/V 每个 tile 的 shape 和被复用的维度。
3. 分开阅读 forward、backward preprocess、`dK/dV` 与 `dQ`。

## 注意

本章实现较长且对 GPU 架构、dtype、head dimension 有明确限制。建议先掌握矩阵乘法、
softmax 和 LayerNorm reduction，再逐段阅读。

## 官方资料

- [Fused Attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)

