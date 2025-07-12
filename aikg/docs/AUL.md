# AUL: AI Unity Language

## Introduction

AUL (AI Unity Language) is a unified expression language designed specifically for LLM-assisted AI kernel generation. Its purpose is to enable various AI models to more simply and clearly design kernel algorithms, optimize scheduling solutions, and interface with multiple hardware backends.

## Core Philosophy

- AUL is a Python-like language that concisely describes the core design of a kernel without requiring strict syntax.
- AUL is for LLM understanding and does not need to be actually executed.
- AUL syntax can be customized according to hardware requirements.
- AUL code pursues simplicity and avoids redundancy.
- AUL can be flexibly extended with new operations.

## Important Constraints

- **MUST** only use the AUL syntax defined in this document.
- **MUST** validate the code to avoid using undefined syntax, Python, backend languages, or third-party libraries.
- **FORBIDDEN** to use Python's `with` statement or other syntactic structures not defined in AUL.

---

# Part 1: General AUL

This part defines the basic syntax and general functions of AUL, applicable to all hardware platforms.

## Basic Data Types

| Type | Description | Usage Example |
|-----|-----|---------|
| TensorPtr | Tensor pointer | `input: U.TensorPtr` |
| BufferPtr | Memory pointer | `buffer: U.BufferPtr` |
| Tile | Data block unit | `tile = U.Tile(shape=(M, N, K), dtype=U.dtype)` |

## Kernel Function Signature

```python
# Standard AUL kernel function signature
def kernel_name(input1: U.TensorPtr, input2: U.TensorPtr, output: U.TensorPtr):
    # Function body
```

**Parameter Description:**
- `input/output`: The starting address for Tensor data input/output.

**Note:** The Shape information used should be hardcoded directly into the AUL function body according to the values in the Tiling function.

## Basic Operation Functions

```python
# Create Tile example
tile = U.Tile(shape=(M, N), dtype=U.dtype)

# Data copy operation example
U.data_copy(dst=dst_buffer, src=src_buffer)

# Data fill example
scalar_127 = U.FilledTile((M, N), U.float32, U.VecBuf, value=127.0)
```

## Basic Calculation Operations

- Arithmetic operations: `+`, `-`, `*`, `/`
- Comparison operations: `==`, `!=`, `>`, `<`, `>=`, `<=`
- Indexing operations: `tensor[start:end]`, `tile[start:end]`

> **Important:** Hardware backends can extend calculation operations (see AUL-NPU and other extension parts).

---

# Part 2: AUL-NPU Extension

This part extends AUL by adding specific functions and optimizations for Neural Processing Units (NPUs).

## NPU Memory Hierarchy

| Level | Description |
|-----|-----|
| GlobalMem | Global memory, large capacity but slow access |
| VecBuf | Vector buffer, high-speed cache for vector computations |
| CubeBuf | Matrix multiplication memory, for optimizing matrix operations |

## NPU Parallel Control

```python
# Example of getting the unique ID of the current core
core_idx = U.get_core_idx()  # Range is [0, core_num-1]

# Software Pipelining loop
# U.Pipelined is used to indicate that loop iterations can be overlapped to hide latency.
# The iterations parameter specifies the total number of logical iterations.
for iter_idx in U.Pipelined(iterations=LOOP_COUNT):
    # Operations within the loop body
    # Data loading and computation are expected to be scheduled for overlap by the backend.
    pass
```

**Core API Description:**
- `get_core_idx()`: Gets the current core ID for multi-core parallel processing.
- `U.Pipelined(iterations=N)`: Defines a loop of N iterations, **strongly suggesting** that iterations should be executed with overlap.

> **Important:** For a detailed explanation of pipelining, please refer to the "NPU Architecture and Pipelining Concepts for AUL" section at the end of the document.

## NPU Memory Operations

```python
# Example of creating a Tile at a specific location
tile = U.Tile(shape=(M, N), dtype=U.dtype, pos=U.VecBuf)

# Example of data copy between specific memory locations
U.data_copy(dst=dst_buffer, src=src_buffer, src_pos=U.GlobalMem, dst_pos=U.VecBuf)
```

**Parameter Description:**
- `U.Tile(...)`: Creates a Tensor with a fixed-size tile.
- `U.data_copy(...)`: Copies data between different memory spaces.

## NPU Tile-level Calculation Operations

AUL uses a generic vector interface, specifying the operation with the `op` parameter for flexible extension:

### Vector Binary Operations

```python
# Vector binary operation example
U.vbinary_op(op="add", dst=dst_tile, src1=tile_a, src2=tile_b) # Addition
U.vbinary_op(op="mul", dst=dst_tile, src1=tile_a, src2=tile_b) # Multiplication

# Supported operation types include: add, sub, mul, div, etc.
# Extended binary operation example
U.vbinary_op(op="xxx", dst=dst_tile, src1=tile_a, src2=tile_b)
```

### Vector Unary Operations

```python
# Vector unary operation example
U.vunary_op(op="sqrt", dst=dst_tile, src=src_tile) # Square root
U.vunary_op(op="exp", dst=dst_tile, src=src_tile)  # Exponent
U.vunary_op(op="log", dst=dst_tile, src=src_tile)  # Logarithm
U.vunary_op(op="relu", dst=dst_tile, src=src_tile) # ReLU
U.vunary_op(op="abs", dst=dst_tile, src=src_tile)  # Absolute value

# Extended unary operation example
U.vunary_op(op="xxx", dst=dst_tile, src=src_tile)
```

### Vector Reduction Operations

```python
# Vector reduction operation example
U.vreduce_op(op="sum", dst=dst_tile, src=src_tile, axis=-1) # Sum reduction
U.vreduce_op(op="max", dst=dst_tile, src=src_tile, axis=0) # Max reduction

# Note:
# 1. The mean operation requires sum followed by div.
```

### Matrix Operations

```python
# Matrix multiplication example
U.matmul_op(dst=dst_tile, src1=a_tile, src2=b_tile)

# Note: Data types of src1 and src2 must be the same.
```

### Vector-Scalar Operations

```python
# Vector-scalar operation example
U.vectorscalar_op(op="adds", dst=dst_tile, src=src_tile, factor=3.14) # Add scalar
U.vectorscalar_op(op="muls", dst=dst_tile, src=src_tile, factor=2.0)  # Multiply by scalar
```

---

# AUL Syntax Cheat Sheet

## General AUL Cheat Sheet

| Category | Syntax/API | Description |
|------|----------|------|
| **Types** | `U.TensorPtr`, `U.BufferPtr`, `U.Tile` | Basic data types |
| **Data Ops** | `U.Tile(shape, dtype)` | Create data tile |
|  | `U.data_copy(dst, src)` | Data copy |
|  | `U.FilledTile(shape, dtype, value)` | Data fill |
| **Calc Ops** | `+`, `-`, `*`, `/` | Basic arithmetic |
|  | `==`, `!=`, `>`, `<`, `>=`, `<=` | Comparison |
| **Indexing** | `[]` | Data indexing |


## Extended AUL-NPU Cheat Sheet

| Category | Syntax/API | Description |
|------|----------|------|
| **Memory Loc** | `GlobalMem`, `VecBuf`, `CubeBuf` | NPU memory hierarchy |
| **HW Control** | `U.get_core_idx()` | Get core ID |
|  | `U.Pipelined(iterations=N)` | Pipelined loop |
| **Ext Data Ops** | `U.Tile(shape, dtype, pos)` | Extended Tile creation |
|  | `U.data_copy(dst, src, src_pos, dst_pos)`| Extended data copy |
| **Vec Binary** | `U.vbinary_op(op="add\|mul\|div\|sub\|...", dst, src1, src2)` | Vector binary ops |
| **Vec Unary** | `U.vunary_op(op="sqrt\|exp\|log\|relu\|abs\|...", dst, src)` | Vector unary ops |
| **Vec Reduce** | `U.vreduce_op(op="sum\|max\|min\|...", dst, src, axis=-1)` | Vector reduction ops |
| **Vec Matrix** | `U.matmul_op(dst, src1, src2)` | Matrix multiplication |
| **Vec Scalar** | `U.vectorscalar_op(op="adds\|muls\|maxs\|mins\|...", dst, src, factor)` | Vector-scalar ops |
| **Scalar-Scalar**| `+`, `-`, `*`, `/`, `==`, `!=`, `>`, `<`, `>=`, `<=` | Basic scalar ops |


## AUL Programming Guidelines

1. **Design Algorithm First**: Determine the general AUL logic.
2. **Then Hardware Optimization**: Add optimizations for the target hardware's AUL.
3. **Keep it Simple**: Avoid redundant code.
4. **Use Memory Correctly**: Clearly define data movement paths.
5. **Utilize Operation Functions**: Use the `op` parameter to specify operations.

**Special Note:** AUL is designed for LLMs; it is a descriptive language, not an executable one. The goal is to simplify kernel description, reduce platform differences, and retain optimization capabilities.

---

# Summary

As a unified AI kernel expression language, the main value of AUL lies in:

1. **Simplified Description**: Concise and intuitive kernel description.
2. **Cross-Platform**: Reduces hardware differences.
3. **Retained Optimization**: Allows for hardware-specific optimizations.
4. **LLM-Friendly**: Designed for LLMs with non-strict syntax.

AUL helps LLMs achieve a consistent development experience on different hardware, balancing performance and flexibility.

---

# NPU Architecture and Pipelining Concepts for AUL

## NPU Hardware Overview

This section introduces the typical hardware composition of an NPU, providing background for understanding NPU-related features in AUL (like memory locations and pipelining).

### Overall Structure and Core Components

-   **Global Memory (GlobalMem):** A large-capacity, relatively slow memory shared across the entire NPU chip (e.g., 40GB capacity). All persistent Tensor data is usually stored here.
-   **NPU Core:** The chip contains multiple (e.g., 8) physical processing cores, which are the main units for executing computations. Code is typically executed in parallel on these cores.

### Internal Resources of an NPU Core

Each NPU core typically includes the following key parts:

-   **Vector Buffer (VecBuf):** A high-speed, small-capacity on-chip memory dedicated to a core (e.g., 256KB). This is where the compute units directly operate on data. Data must be loaded from GlobalMem to here before computation. There may be alignment requirements for data storage addresses (e.g., 256 Bytes).

-   **Execution Units:** Specialized hardware units that operate in parallel:
    *   **Load Unit:** Responsible for transferring data *from* GlobalMem *to* the core's VecBuf.
    *   **Store Unit:** Responsible for transferring data *from* the VecBuf back *to* GlobalMem.
    *   **Vector Compute Unit:** Performs the actual computations (e.g., vector addition, multiplication, reduction) on data residing in the VecBuf. May only support contiguous or masked access patterns.
    *   **Scalar Compute Unit:** Similar to a small CPU, it can perform scalar operations for control flow or auxiliary calculations.

### Execution Model and Parallelism

-   **Data Flow Path:** A typical data path is: GlobalMem → (Load Unit) → VecBuf → (Compute Unit) → VecBuf → (Store Unit) → GlobalMem.
-   **Core Parallelism:** The core idea of NPU design is parallelism. Different NPU cores can work in parallel.
-   **Unit Parallelism:** The load, store, and compute units within a core **can also operate concurrently**. For example, the load unit can fetch the next data block while the compute unit is processing the current one.
-   **Data Dependency and Synchronization:** For data processed using only *one set of buffers*, there is a strict serial dependency: `Load` must complete before the `Compute` for that data can begin; `Compute` must complete before `Store` can write back the result.
-   **Synchronization Mechanism:** The underlying hardware typically uses mechanisms like `set_flag / wait_flag` for synchronization. If not managed properly, this can lead to unit stalls.

## AUL Pipeline Expression

AUL provides high-level abstractions to simplify pipeline programming:

-   `U.Pipelined(iterations=N)`: A high-level instruction that informs the LLM/compiler that the loop iterations are expected to be executed in a pipelined manner, leveraging the hardware's parallel capabilities.

## AUL-NPU Pipeline Example: Vector Addition

```python
import aul as U

@sub_kernel
def vector_add_pipelined(A: U.TensorPtr, B: U.TensorPtr, C: U.TensorPtr):
    
    # 1. Parse configuration parameters
    TILE_LEN = 256
    LOOP_COUNT = 5
    total_len = 10240
    BLOCK_DIM = 8
    
    # 2. Get core ID and calculate data range
    core_idx = U.get_core_idx()
    len_per_core = total_len // BLOCK_DIM
    start_idx = core_idx * len_per_core
    end_idx = start_idx + len_per_core
    
    # 3. Implement pipeline using Pipelined loop
    for i in U.Pipelined(iterations=LOOP_COUNT):
        # 3.1 Calculate data index for the current iteration
        current_start = start_idx + i * TILE_LEN
        current_end = current_start + TILE_LEN
        
        # 3.2 Create Tiles
        a_tile = U.Tile(shape=(TILE_LEN,), dtype=A.dtype, pos=U.VecBuf)
        b_tile = U.Tile(shape=(TILE_LEN,), dtype=B.dtype, pos=U.VecBuf)
        c_tile = U.Tile(shape=(TILE_LEN,), dtype=C.dtype, pos=U.VecBuf)
        
        # 3.3 Pipeline Stage 1: Load input data
        U.data_copy(dst=a_tile, src=A[current_start:current_end],
                    src_pos=U.GlobalMem, dst_pos=U.VecBuf)
        U.data_copy(dst=b_tile, src=B[current_start:current_end],
                    src_pos=U.GlobalMem, dst_pos=U.VecBuf)
        
        # 3.4 Pipeline Stage 2: Perform computation
        U.vbinary_op(op="add", dst=c_tile, src1=a_tile, src2=b_tile)
        
        # 3.5 Pipeline Stage 3: Write back result
        U.data_copy(dst=C[current_start:current_end], src=c_tile,
                    src_pos=U.VecBuf, dst_pos=U.GlobalMem)
```

---

# Practical Guide: Best Practices for AUL Code Generation

## Conversion Steps

1. **Understand the Task**: Fully understand the functional logic of the kernel.
2. **Design the Base Algorithm**: Use general AUL to design the basic algorithm flow.
3. **Add Hardware Optimizations**: Add optimizations based on target hardware features, such as pipelining.
4. **Validate the Code**: Check for compliance with AUL syntax specifications.

## Common Mistakes to Avoid

1. **Using Undefined APIs**: Only use APIs defined in the documentation; do not "invent" new ones.
2. **Ignoring Memory Hierarchy**: Be explicit about data movement between different memory levels.
3. **Incorrect Optimization Timing**: Ensure correctness before optimizing.
4. **Inappropriate Batching**: Consider the kernel input size and divide it into reasonable batch sizes.
5. **Attempting to Implement Double Buffering**: Double buffering will be implemented in a later compiler stage; it does not need to be considered in AUL code design.
6. **Operating on Tensors of Different Sizes**: Verify Tensor sizes against the pre-allocated Tiles before computation.
7. **Tiling a Dimension Involved in a Reduction**: Be careful about the dimension of the tiling loop; try to avoid tiling an axis involved in a reduction operation.
8. **Broadcasting**: AUL does not support broadcasting; pay attention to the sizes of the tensors in an operation.
9. **Tiling Method**: If data size exceeds storage limits, it needs to be tiled again. Ensure the new tiling is on the same axis as the inter-core tiling. For example, if inter-core parallelism splits the second axis, the `for` loop should also tile the second axis.
10. **Unsupported Syntax**: AUL does not support the `.shape` operation; please write out shapes explicitly.

## Checklist

- [ ] Function signature is correct and includes necessary parameters.
- [ ] Memory operations explicitly specify source and destination locations.
- [ ] Optimizations, like multi-core parallelism, are used appropriately.
- [ ] No undefined operations or syntax are used.
- [ ] Code structure is clear and the logic is correct. 