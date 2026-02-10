# 任务装配策略

## 依赖追踪机制

`trace_dependencies` 工具通过 AST 分析自动检测函数的所有依赖：

### 1. Import 别名解析

从文件顶部的 `import` 语句和顶级赋值中构建 `{别名: 源模块}` 映射：

```python
# import torch._prims_common as utils → {"utils": "torch._prims_common"}
# from torch._decomp import register_decomposition → {"register_decomposition": "torch._decomp"}
```

### 2. 依赖分类

- **同文件依赖**：在同一文件中定义的函数/类，直接提取
- **外部调用**：通过 import 别名引用的其他模块函数
  - 公共 API（如 `torch.tensor()`）→ 保留 import
  - 内部 API（含 `_` 前缀模块）→ 需内联

### 3. 外部调用处理

对于需要内联的外部调用：
1. 通过源模块路径定位原始文件
2. 使用 `read_function` 读取完整实现
3. 检查函数签名，确保参数一致
4. 将函数体内联到输出文件

## 装配策略

### 排除式（大文件）

```
文件总函数 - 不需要的函数 = 输出
```

适用于：目标函数依赖大量同文件函数时

### 选择性（精确提取）

```
入口函数 + 依赖函数列表 = 输出
```

适用于：依赖关系清晰、函数数量有限时

### 完整嵌入（小文件）

```
整个文件内容 = 输出
```

适用于：文件较小、函数间紧密耦合时

## 验证流程

```
1. 语法检查 → AST 可解析
2. 结构检查 → Model 类 + get_inputs + get_init_inputs 存在
3. 运行时检查 → 实例化 → forward → 无 NaN/Inf
4. 参考对比 → 多组输入与原始函数输出比对
```
