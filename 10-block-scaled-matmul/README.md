# Block Scaled Matrix Multiplication

本章对应 Triton 官方教程 **Block Scaled Matrix Multiplication**，介绍 FP4/FP8
OCP microscaling 格式下的块缩放矩阵乘法。

## 学习目标

- 理解低精度值与 block scale 共同表示数值的方式
- 区分 MXFP4、MXFP8、NVFP4 及其 scale 类型
- 理解 scale factor 沿 K 维的广播粒度 `VEC_SIZE`
- 学习 NVIDIA scale preshuffling/packed layout 的动机
- 使用 `tl.dot_scaled` 调用硬件加速的块缩放乘法
- 认识 NVIDIA Blackwell 与 AMD CDNA4 路径的差异

## 核心公式

$$
C = (A \odot scale_A) @ (B \odot scale_B)
$$

scale 并非逐元素存储，而是由一小段低精度值共享。为了让 tensor core 的快速 K 循环
连续读取 scale，物理布局通常不是直观的二维 row-major，需要预先重排和打包。

## 文件

- `block-scaled-matmul.ipynb`：中文导读、scale 布局、CUDA/AMD kernel、验证与 profiler

## 阅读重点

1. 先在逻辑二维布局中理解 scale，再阅读 packed 五维布局。
2. 明确数据 block size、scale vector size 和 MMA tile 是三个不同概念。
3. 把本章当作硬件专项教程；不支持该指令的 GPU 无法运行是预期行为。

## 官方资料

- [Block Scaled Matrix Multiplication](https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html)

