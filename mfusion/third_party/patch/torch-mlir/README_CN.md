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
禁用 `aten.ones`、`aten.zeros`、`aten.full` 的常量折叠，使这三类 Op 在 canonicalization 时不再被折叠为 `torch.vtensor.literal`。

**问题：**  
在 mfusion 的 `fuse_and_optimize` 流程中会执行 `canonicalize`。Torch-MLIR 中若上述 Op 启用了 `hasFolder` 并实现了 `fold()`，当输入为编译期常量时会被折叠为 `torch.vtensor.literal`，导致后续 pipeline 中出现 literal，与当前处理方式不符。

**解决方案：**

1. **TableGen（.td）**：在 `include/torch-mlir/Dialect/Torch/IR/GeneratedTorchOps.td` 中移除 `AtenOnesOp`、`AtenZerosOp`、`AtenFullOp` 的 `let hasFolder = 1;`，使 TableGen 不再为这三类 Op 声明 fold。
2. **C++ 实现（.cpp）**：在 `lib/Dialect/Torch/IR/TorchOps.cpp` 中删除 `AtenOnesOp::fold`、`AtenZerosOp::fold`、`AtenFullOp::fold` 的完整实现块（约 130 行），与 .td 中取消 hasFolder 保持一致，避免链接未定义符号。

**修改的文件：**
- `include/torch-mlir/Dialect/Torch/IR/GeneratedTorchOps.td`
- `lib/Dialect/Torch/IR/TorchOps.cpp`

## 应用顺序

这些补丁应按数字顺序应用（001 → 002 → ...）。
