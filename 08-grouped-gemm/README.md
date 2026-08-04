# Group GEMM

本章对应 Triton 官方教程 **Group GEMM**。目标是用固定数量的 CTA 处理一组形状
可能不同的矩阵乘法，并把任务调度放在 device 端完成。

## 学习目标

- 区分 batched GEMM 与 grouped GEMM
- 使用指针数组、shape 数组和 leading dimension 描述多个 GEMM
- 让固定数量的 persistent CTA 在 device 端领取输出 tile
- 理解静态调度、tile 编号和跨问题边界的映射
- 认识支持硬件上的 TMA descriptor 版本

## 核心思路

Host 端把每个 GEMM 的 A/B/C 地址、`M/N/K` 和 stride 打包到 device tensor。
固定数量的 program 遍历“所有 GEMM 的全部输出 tile”，先确定当前 tile 属于哪个
问题，再执行分块乘法。这样可以减少大量小 GEMM 各自 launch 的开销，并提高 GPU
占用率。

## 文件

- `grouped-gemm.ipynb`：中文导读、普通 grouped GEMM、TMA 版本、测试与 benchmark

## 阅读重点

1. 先弄清 host 端的 pointer/size/stride 三组元数据。
2. 再跟踪一个 program 如何跨越多个 GEMM 的 tile 区间。
3. 对比固定 CTA 数与“每个 tile 一个 program”的启动方式。

## 官方资料

- [Group GEMM](https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html)

