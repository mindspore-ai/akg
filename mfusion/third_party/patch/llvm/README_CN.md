# LLVM 补丁说明

本目录包含应用于 LLVM/MLIR 代码库的补丁，用于适配 ms_inferrt 项目的构建需求。

## 补丁列表

### 001-install-pdll-files.patch

**描述：**
在 MLIR 安装规则中添加 `*.pdll` 文件的安装支持，确保 PDLL（Pattern Description Language for LLVM）文件被正确安装到目标目录。

**问题：**
MLIR 的 CMake 安装规则（`mlir/CMakeLists.txt`）在安装 include 目录时，只包含了特定类型的文件（`*.def`, `*.h`, `*.inc`, `*.td`），**没有包含 `*.pdll` 文件**。这导致在使用 install 产物构建其他项目（如 mopt）时，无法找到 PDLL 文件（如 `mlir/Transforms/DialectConversion.pdll`），从而无法使用 PDLL 功能。

**解决方案：**

- 在两个 `install(DIRECTORY ...)` 规则中都添加 `PATTERN "*.pdll"` 模式
- 确保所有 PDLL 文件都会被正确安装到install目录

**修改的文件：**

- `mlir/CMakeLists.txt`

**技术细节：**

PDLL 是 MLIR 中用于描述模式匹配和转换的领域特定语言。这些 `.pdll` 文件需要被包含在安装产物中，以便其他项目可以在编译时引用它们（例如通过 `#include "mlir/Transforms/DialectConversion.pdll"`）。

## 应用顺序

这些补丁应按数字顺序应用（001 → 002 → ...）。
