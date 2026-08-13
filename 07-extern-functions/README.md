# Libdevice External Functions

本章对应 Triton 官方教程 **Libdevice (`tl.extra.libdevice`) function**，演示如何
在 Triton kernel 中调用设备端数学库函数。

## 学习目标

- 使用 `triton.language.extra.libdevice` 调用外部设备函数
- 了解 Triton 如何根据输入 dtype 选择对应的函数实现
- 以 `asin` 为例完成 load、外部函数调用和 store
- 认识默认 libdevice 路径与自定义 `extern_libs` 的区别
- 理解 CUDA libdevice 与 AMD OCML/OCKL 的后端差异

## 核心思路

Triton 负责 kernel 的并行组织和内存访问，libdevice 提供经过硬件后端优化的数学
函数。`libdevice.asin(x)` 会根据 `x` 的类型选择单精度或双精度底层实现，使 kernel
无需手写复杂数学近似。

## 文件

- `extern-functions.ipynb`：中文导读、`asin` kernel、默认与自定义库路径示例

## 阅读重点

1. 外部函数调用不会改变 Triton 的 grid/offset/mask 模式。
2. 自定义库路径通常依赖 Triton 安装目录，不应硬编码为个人绝对路径。
3. 对比 `libdevice.asin` 与 `tl.libdevice`/`tl.math` API 的版本差异。

## 官方资料

- [Libdevice external functions](https://triton-lang.org/main/getting-started/tutorials/07-extern-functions.html)

