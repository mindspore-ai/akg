# Torch-MLIR 补丁说明

本目录包含应用于 Torch-MLIR 代码库的补丁，用于适配 ms_inferrt 项目的构建需求。

## 补丁列表

### 001-build-isolate-symbols.patch

**描述：**
为 TorchMLIR Python 绑定添加符号隔离，避免与其他 MLIR Python 绑定的符号冲突。

**问题：**
当 Torch-MLIR 作为其他项目的一部分构建时，其 Python 绑定可能与其他 MLIR Python 绑定（如来自 LLVM/MLIR 主项目的绑定）产生符号冲突，导致运行时链接错误或符号解析问题。

**解决方案：**

- 在 Linux 平台上为 `TorchMLIRAggregateCAPI` 目标添加链接器选项 `--exclude-libs,ALL`
- 该选项会排除库中的所有符号，避免与其他库的符号冲突
- 确保 Torch-MLIR 的 Python 绑定可以与其他 MLIR 组件共存

**修改的文件：**

- `python/CMakeLists.txt`

### 002-build-embedded.patch

**描述：**
添加嵌入式构建模式支持，允许 Torch-MLIR 作为其他项目的嵌入式组件构建。

**问题：**
当 Torch-MLIR 作为其他项目的一部分嵌入构建时，某些初始化代码（如 Python site 初始化）可能不适用或会导致冲突。需要一种方式来区分独立构建和嵌入式构建。

**解决方案：**

- 添加 `TORCH_MLIR_BUILD_EMBEDDED` CMake 选项
- 当启用该选项时，跳过 `TorchMLIRSiteInitialize` 和 `TorchMLIRPythonSources.Tools` 组件的声明和包含
- 跳过 Tools 组件是为了避免 torch-mlir 的 `tools/opt/__main__.py` 被打包到嵌入项目中，该脚本调用的是 `torch-mlir-opt` 而非嵌入项目自己的 opt 工具
- 在构建逻辑中添加第三个分支来处理嵌入式构建场景
- 确保嵌入式构建时不会执行不必要的初始化操作

**修改的文件：**

- `CMakeLists.txt`
- `python/CMakeLists.txt`

### 003-build-remove-tests.patch

**描述：**
移除测试相关的构建目标和子目录，简化构建过程。

**问题：**
在将 Torch-MLIR 作为依赖项集成到其他项目时，通常不需要构建和运行 Torch-MLIR 的测试套件。这些测试会增加构建时间，并且可能引入额外的依赖项。

**解决方案：**

- 移除 `check-torch-mlir-all` 自定义目标及其依赖关系
- 移除 `add_subdirectory(test)` 调用，不构建测试目录
- 移除各个子项目中的测试相关依赖
- 保留核心功能构建，减少构建时间和复杂度

**修改的文件：**

- `CMakeLists.txt`
- `projects/pt1/CMakeLists.txt`
- `projects/pt1/python/CMakeLists.txt`
- `projects/pt1/python/test/CMakeLists.txt`

### 004-disable-torch-to-linalg.patch

**描述：**
添加 CMake 选项以禁用 Torch-to-Linalg 转换支持，允许构建轻量级的 Torch-MLIR。

**问题：**
在某些使用场景中，不需要将 Torch IR 转换为 Linalg dialect。强制构建 Linalg 相关的转换代码会增加编译时间和二进制大小，同时引入不必要的依赖关系。

**解决方案：**

- 添加 `TORCH_MLIR_ENABLE_LINALG` CMake 选项（默认为 ON）
- 使用条件编译宏保护 Linalg 相关的头文件包含和代码
- 在 CMakeLists.txt 中根据选项决定是否构建 TorchToLinalg 子目录
- 使用 `#ifdef TORCH_MLIR_ENABLE_LINALG` 保护 Linalg 转换 Pass 的注册和使用
- 允许用户在不需要 Linalg 支持时跳过相关代码的构建

**修改的文件：**

- `CMakeLists.txt`
- `include/torch-mlir/Conversion/Passes.td`
- `lib/Conversion/CMakeLists.txt`
- `lib/Conversion/Passes.cpp`
- `lib/Dialect/TorchConversion/Transforms/Passes.cpp`

### 005-fix-compilation-errors.patch

**描述：**
修复 Torch-MLIR 在较旧版本 LLVM 上的编译兼容性问题。

**问题：**
Torch-MLIR 跟踪的是 LLVM 主干的较新版本，而本项目使用的是 LLVM 19.1.7（2025年1月12日的 commit）。由于 LLVM/MLIR 在新版本中重命名或修改了多个 API，直接编译会产生大量编译错误。

**解决方案：**

将 torch-mlir 中使用的新版 LLVM/MLIR API 降级回旧版本 API，主要涉及以下变更：

1. **Pattern Rewrite API 重命名**：
  - `applyPatternsGreedily` → `applyPatternsAndFoldGreedily`
  - `applyOpPatternsGreedily` → `applyOpPatternsAndFold`

2. **GreedyRewriteConfig 配置方式变化**：
  - `config.setUseTopDownTraversal(true)` → `config.useTopDownTraversal = true`
  - `config.setMaxIterations(...)` → `config.maxIterations = ...`
  - `config.setStrictness(...)` → `config.strictMode = ...`

3. **DataFlow Analysis API 重命名**：
  - `GenericLatticeAnchorBase` → `GenericProgramPointBase`
  - `LatticeAnchor` → `ProgramPoint`
  - `registerAnchorKind` → `registerPointKind`
  - `getLatticeAnchor` → `getProgramPoint`
  - `getProgramPointAfter(op)` → `ProgramPoint(op)`

4. **Bufferization API 变化**：
  - `bufferization::ToBufferOp` → `bufferization::ToMemrefOp`

5. **OpConversionPattern API 变化**：
  - `OneToNOpAdaptor` → `OpAdaptor`

6. **FunctionOpInterface API 变化**：
  - `func.eraseArguments()` 返回类型从 `LogicalResult` 变为 `void`

7. **其他 API 调整**：
  - `getBackwardSlice()` 不再返回 `LogicalResult`
  - APInt 构造函数需要显式类型转换以避免编译错误

**修改的文件：**

- `lib/Dialect/TMTensor/Transforms/Bufferize.cpp`
- `lib/Dialect/TMTensor/Transforms/ConvertToLoops.cpp`
- `lib/Dialect/Torch/Transforms/AdjustCallingConventions.cpp`
- `lib/Dialect/Torch/Transforms/DecomposeComplexOps.cpp`
- `lib/Dialect/Torch/Transforms/FuseQuantizedOps.cpp`
- `lib/Dialect/Torch/Transforms/GlobalizeObjectGraph.cpp`
- `lib/Dialect/Torch/Transforms/InlineGlobalSlots.cpp`
- `lib/Dialect/Torch/Transforms/MatchQuantizedOps.cpp`
- `lib/Dialect/Torch/Transforms/MaximizeValueSemantics.cpp`
- `lib/Dialect/Torch/Transforms/PrepareForGlobalizeObjectGraph.cpp`
- `lib/Dialect/Torch/Transforms/RecomposeComplexOps.cpp`
- `lib/Dialect/Torch/Transforms/RestructureNonConstantAxes.cpp`
- `lib/Dialect/Torch/Transforms/ScalarizeShapes.cpp`
- `lib/Dialect/Torch/Transforms/SimplifyDtypeCalculations.cpp`
- `lib/Dialect/Torch/Transforms/SimplifyShapeCalculations.cpp`
- `lib/Dialect/TorchConversion/Transforms/BackendTypeConversionPasses.cpp`
- `lib/Dialect/TorchConversion/Transforms/UnpackQuantTensor.cpp`
- `lib/RefBackend/RefBackend.cpp`

### 006-disable-aten-fold-constant.patch

**描述：**  
禁用 `aten.ones`、`aten.zeros`、`aten.full`、`prim.NumToTensor.Scalar` 和 `aten.clone` 的 fold 方法；同时为 `aten.reciprocal` 增加 canonicalizer（`reciprocal(sqrt(x)) -> rsqrt(x)`），使该组合在 canonicalization 后保持期望形态。

**问题：**  
在 mfusion 的 `fuse_and_optimize` 流程中会执行 `canonicalize`。Torch-MLIR 中若 `aten.ones`、`aten.zeros`、`aten.full`、`prim.NumToTensor.Scalar` 启用了 `hasFolder` 并实现了 `fold()`，当输入为编译期常量时会被折叠为 `torch.vtensor.literal`，导致后续 pipeline 中出现 literal，与当前处理方式不符。若 `aten.clone` 启用了 `hasFolder` 并实现了 `fold()`，可能会将本来用于保持布局语义的 clone 操作提前折叠掉，影响后续算子输入形态。另外，`reciprocal(sqrt(x))` 需要在 Torch 侧稳定规范化为 `rsqrt(x)`，以便后续优化链路统一识别。

**解决方案：**

1. **TableGen（.td）**：
    - 在 `include/torch-mlir/Dialect/Torch/IR/GeneratedTorchOps.td` 中移除 `AtenOnesOp`、`AtenZerosOp`、`AtenFullOp`、`PrimNumToTensorScalarOp` 和 `AtenCloneOp` 的 `let hasFolder = 1;`，使 TableGen 不再为这些 Op 声明 fold。
    - 为 `AtenReciprocalOp` 增加 `let hasCanonicalizer = 1;`，启用 reciprocal 的规范化模式注册。
2. **C++ 实现（.cpp）**：
    - 在 `lib/Dialect/Torch/IR/TorchOps.cpp` 中删除 `AtenOnesOp::fold`、`AtenZerosOp::fold`、`AtenFullOp::fold`、`PrimNumToTensorScalarOp::fold` 和 `AtenCloneOp::fold` 的完整实现块，与 `.td` 中取消 `hasFolder` 保持一致，避免链接未定义符号。
    - 新增 `AtenReciprocalOp::getCanonicalizationPatterns`，将 `reciprocal(sqrt(x))` 规范化为 `rsqrt(x)`。

**修改的文件：**

- `include/torch-mlir/Dialect/Torch/IR/GeneratedTorchOps.td`
- `lib/Dialect/Torch/IR/TorchOps.cpp`

### 007-disable-adaptive-avg-pool2d-canonicalizer.patch

**描述：**  
禁用 `aten._adaptive_avg_pool2d` 的规范化器，防止其在 canonicalization 过程中被替换为 `aten.adaptive_avg_pool2d`。

**问题：**  
在 mfusion 的 `fuse_and_optimize` 流程中会执行 `canonicalize`。Torch-MLIR 中 `aten._adaptive_avg_pool2d` 操作的规范化器会将其替换为 `aten.adaptive_avg_pool2d`，而`aten.adaptive_avg_pool2d`算子已经被decompose处理过，在当前阶段不应该出现，这会导致后续执行报错。

**解决方案：**

1. **TableGen（.td）**：在 `include/torch-mlir/Dialect/Torch/IR/GeneratedTorchOps.td` 中移除 `Aten_AdaptiveAvgPool2dOp` 的 `let hasCanonicalizer = 1;`，使 TableGen 不再为该 Op 生成规范化器相关代码。
2. **C++ 实现（.cpp）**：在 `lib/Dialect/Torch/IR/TorchOps.cpp` 中删除 `Aten_AdaptiveAvgPool2dOp::getCanonicalizationPatterns` 方法的完整实现块，与 .td 中取消 hasCanonicalizer 保持一致。

**修改的文件：**

- `include/torch-mlir/Dialect/Torch/IR/GeneratedTorchOps.td`
- `lib/Dialect/Torch/IR/TorchOps.cpp`

### 008-canonicalize-aten-permute-negative-dims.patch

**描述：**
让 `aten.permute` 在使用负维度索引时既能通过 verifier，又能在 canonicalization 阶段被规范化为正维度写法，避免后续转换流程处理负维度时出错。

**问题：**
在 mfusion 的转换流程中，`aten.permute` 可能携带负维度索引（例如 rank 4 的张量使用 `-2` 作为维度索引）。这里真正的“unknown 维度”应当由“不是编译期常量”来表示，而不是把常量 `-1` 特判成 unknown；否则 verifier 会错误地跳过合法的负维度语义，或者在负维度还未归一化时直接报出 `observed invalid index in permutation (-2)` 之类的错误。即使后续有 canonicalize，也无法保证一定先于 verifier 执行。

**解决方案：**

1. **TableGen（.td）**：在 `include/torch-mlir/Dialect/Torch/IR/GeneratedTorchOps.td` 中，为 `AtenPermuteOp` 增加 `hasCanonicalizer` 方法。
2. **C++ 实现（.cpp）**：在 `lib/Dialect/Torch/IR/TorchOps.cpp` 中同时修改 verifier 和 canonicalizer：
   - 在 `AtenPermuteOp::verify()` 中，保留 `!fromIsSet` 作为“unknown 维度”的判断；对编译期常量维度则先执行 `toPositiveDim(from, outRank)`，再进行范围检查、重复维度检查以及输入/输出 shape 一致性校验，使 `-1`、`-2` 这类合法负维度写法能够被正确接受。
   - 新增 `AtenPermuteOp::getCanonicalizationPatterns`，当 `dims` 是常量整数列表时，将其中的负维度统一转换为正维度，并重建 `PrimListConstructOp` 与新的 `aten.permute`。
   - 这样处理后，IR 创建阶段不会因为负维度提前失败，后续 canonicalize 也能把常量负维度稳定收敛为统一形式。

**修改的文件：**

- `include/torch-mlir/Dialect/Torch/IR/GeneratedTorchOps.td`
- `lib/Dialect/Torch/IR/TorchOps.cpp`

## 应用顺序

这些补丁应按数字顺序应用（001 → 002 → ...）。
