# Matrix Multiplication

本章对应 Triton 官方教程 **Matrix Multiplication**，目标是实现 FP16 矩阵乘法
`C = A @ B`，并理解高性能 GEMM 内核最重要的组织方式。

## 学习目标

- 将矩阵划分为二维 tile，并让一个 Triton program 负责一个输出 tile
- 使用 shape、stride 和广播构造二维指针块
- 沿 K 维循环，通过 `tl.dot` 在 FP32 累加器中完成分块乘加
- 使用 grouped ordering 提高 L2 cache 的数据复用率
- 使用 `triton.autotune` 为不同的 `(M, N, K)` 搜索元参数
- 理解如何在写回前融合激活函数

## 核心思路

对于 `A[M, K] @ B[K, N]`，每个 program 计算 `C` 的一个
`BLOCK_SIZE_M × BLOCK_SIZE_N` 区块。内核沿 K 维依次加载 A、B 的子块，
将乘加结果累积在片上，最后一次性写回全局内存。

普通 row-major 的 program 调度会较快离开刚刚使用过的数据。官方教程使用
`GROUP_SIZE_M` 对输出块重新编号，让相邻 program 更可能复用同一批 A、B tile。

## 文件

- `matrix-multiplication.ipynb`：中文学习导读、官方实现、正确性检查与性能基准

## 阅读重点

1. 先手算一个输出 tile 所需的 A/B 指针形状。
2. 再观察 `offs_am[:, None]` 和 `offs_bn[None, :]` 如何产生二维广播。
3. 最后比较 row-major 与 grouped ordering 的 program 访问顺序。

## 官方资料

- [Matrix Multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)

