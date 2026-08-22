# Triton from 0 to 1

基于 [Triton 官方教程](https://triton-lang.org/main/getting-started/tutorials/index.html)
整理的学习笔记。每章包含一个概览 README 和一个可继续记录思考的 Jupyter notebook。

## Learning roadmap

- [x] 安装与 Triton 编程模型简介
- [x] Vector Addition
- [x] Fused Softmax
- [x] Matrix Multiplication
- [x] Low-Memory Dropout
- [x] Layer Normalization
- [ ] Fused Attention （施工中）
- [ ] Libdevice External Functions
- [ ] Group GEMM
- [ ] Persistent Matmul
- [ ] Block Scaled Matrix Multiplication

勾选表示实际学习进度，而不是仓库中是否已经准备好学习材料。

## Repository structure

| Directory | Topic | 主要内容 |
|---|---|---|
| `notes/` | Concept notes | 独立概念笔记 |
| `00-getting-start/` | Getting Started | 环境安装与 Triton 简介 |
| `01-vector-add/` | Vector Addition | program id、offset、mask、benchmark |
| `02-fused-softmax/` | Fused Softmax | kernel fusion 与 reduction |
| `03-matrix-multiplication/` | Matrix Multiplication | tiled GEMM、L2 grouping、autotune |
| `04-low-memory-dropout/` | Low-Memory Dropout | 并行伪随机数与 seed 重建 mask |
| `05-layer-norm/` | Layer Normalization | forward/backward 与并行归约 |
| `06-fused-attention/` | Fused Attention | FlashAttention v2 与 online softmax |
| `07-extern-functions/` | Libdevice | 外部设备数学函数 |
| `08-grouped-gemm/` | Group GEMM | 多 GEMM 的 device-side 调度 |
| `09-persistent-matmul/` | Persistent Matmul | persistent scheduling、TMA、profiling |
| `10-block-scaled-matmul/` | Block Scaled Matmul | FP4/FP8 microscaling 与 `tl.dot_scaled` |

## Notebook usage

新章节 notebook 以 Triton 官方当前版本为代码基线，并额外加入：

- 中文学习导读与建议阅读顺序
- 核心 shape、调度和内存访问的心智模型
- 章节末尾的思考题与可继续填写的学习记录
- 官方正确性检查和性能基准代码（未在本仓库环境运行）

建议按编号顺序学习。高级章节依赖特定 GPU 架构；遇到 TMA、FP8、FP4 或
block-scaled 指令不受支持时，可先完成代码阅读，不必把硬件限制当作实现错误。

## Expected environment

- Python
- PyTorch
- Triton
- CUDA 或 ROCm 支持的 GPU
