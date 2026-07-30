# The first Triton program: Vector ADD

# Vector Addition

## Learning objectives

- Understand the Triton programming model
- Understand program IDs
- Understand block-based memory access
- Learn masked load and store operations

## Kernel structure

The vector-addition kernel assigns one Triton program to one block of
elements. Each program calculates its starting offsets using
`tl.program_id(axis=0)`.

## Files

- `vector_add.py`: Triton implementation
- `benchmark.py`: Performance comparison with PyTorch