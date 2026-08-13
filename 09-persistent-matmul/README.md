# Persistent Matmul

本章对应 Triton 官方教程 **Persistent Matmul**，比较普通、persistent、TMA 和
tensor descriptor 等多种矩阵乘法实现。

## 学习目标

- 理解 persistent kernel 与普通“一 tile 一 program”启动方式的区别
- 将 program 数量限制为 SM 数量，并在 kernel 内循环处理多个输出 tile
- 比较 FP16 与受支持硬件上的 FP8 路径
- 认识 Tensor Memory Accelerator 和 tensor descriptor
- 使用 Proton profiler 分析 kernel，而不仅是记录单个耗时

## 核心思路

普通 matmul 为每个输出 tile 启动一个 program；persistent matmul 只启动接近 GPU
SM 数量的 program，让每个 program 在 kernel 内处理多个 tile。它可以降低调度开销，
但也使负载均衡、寄存器/共享内存占用和 tile 顺序更重要。

## 文件

- `persistent-matmul.ipynb`：中文导读、各实现版本、验证、benchmark 与 profiler

## 阅读重点

1. 找到“总 tile 数”和“实际启动 program 数”。
2. 跟踪 `tile_id += NUM_SMS` 的 persistent 调度模式。
3. 把 TMA 看作一种数据搬运机制，先与 persistent 调度概念分开理解。

## 硬件提示

官方说明该教程可能在共享内存较小的设备（例如 RTX 4090）上失败；FP8/TMA 路径也
有更高的计算能力要求。

## 官方资料

- [Persistent Matmul](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html)

