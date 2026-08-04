# Low-Memory Dropout

本章对应 Triton 官方教程 **Low-Memory Dropout**，实现只保存一个 `int32` seed
的低内存 dropout，而不是保存与输入等大的布尔 mask。

## 学习目标

- 理解训练阶段 dropout 的缩放因子 `1 / (1 - p)`
- 比较“显式保存 mask”和“由 seed 重建 mask”的状态开销
- 使用 `tl.rand(seed, offsets)` 为每个元素生成可复现的伪随机数
- 理解并行随机数中 seed 与 counter/offset 的职责
- 继续练习一维 grid、mask load/store 和 host wrapper

## 核心思路

同一组 `(seed, offset)` 会生成同一个随机数，所以 backward 或重算时无需保存
完整 dropout mask。只要保留 seed，并让每个逻辑元素始终使用稳定且唯一的 offset，
就能重建相同的 keep/drop 决策。

这是一种典型的“用计算换状态”方法：减少显存占用和额外的内存流量，代价是重新
生成伪随机数。

## 文件

- `low-memory-dropout.ipynb`：中文导读、baseline、seed 版内核和练习

## 阅读重点

1. 区分 `p`（丢弃概率）和 `1-p`（保留概率）。
2. 思考 offset 为什么必须在整个 tensor 内唯一。
3. 比较显式 mask 版本与 seed 版本需要保存的状态。

## 官方资料

- [Low-Memory Dropout](https://triton-lang.org/main/getting-started/tutorials/04-low-memory-dropout.html)

