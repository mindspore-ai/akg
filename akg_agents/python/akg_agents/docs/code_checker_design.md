# CodeChecker 代码检查器设计方案

## 1. 背景与问题

### 1.1 当前痛点

在 AIKG 代码生成流程中，Coder Agent 生成 Triton 代码时经常会犯一些**简单的语法错误**，例如：
- 使用 `break`、`continue`、`return` 等禁止的控制流语句
- 在 Ascend 后端使用 `while` 循环（不支持）
- 在 Ascend 后端使用 `tl.where`（不支持）
- Python 语法错误（括号不匹配、缩进错误等）

这些问题在 `suggestion_docs.md` 中已明确说明，但 LLM 仍经常犯错。

### 1.2 当前流程的问题

```
Designer → Coder → Verifier (1分钟) → 发现语法错误 → Conductor 分析 → Coder 修复
```

**问题**：Verifier 每次执行需要约 1 分钟，而这些语法错误完全可以通过静态分析在 Verifier 之前检测出来。

## 2. 解决方案

### 2.1 设计目标

1. **提前拦截**：在 Verifier 之前检测出简单的语法错误
2. **快速反馈**：静态检查 < 1 秒，LLM 检查 < 15 秒
3. **自动修复**：检测到错误后自动回到 Coder 进行修复
4. **双模式支持**：
   - **静态模式**：已知场景（如 triton-ascend），使用规则匹配，零成本
   - **LLM 模式**：未知场景，使用 LLM 分析，灵活性高
5. **作用域感知**：只在 Triton kernel 函数内部应用禁止规则，避免误报

### 2.2 优化后的流程

```
代码输入
    │
    ▼
┌─────────────────────────┐
│  Python 语法检查        │  ← 使用 ast.parse()，检测括号/缩进等错误
│  （所有 DSL 都执行）     │
└─────────────────────────┘
    │
    ├── 有语法错误 ──┬── LLM 路径 → 直接返回错误（节省 LLM 调用成本）
    │               │
    │               └── 静态检查路径 → 继续做规则检查，最后汇总
    │
    └── 无语法错误 ──┬── LLM 路径 → 调用 LLM 分析
                    │
                    └── 静态检查路径 → 只做规则检查
                            │
                            ▼
                    ┌─────────────────────┐
                    │  识别 Kernel 范围   │  ← 通过 @triton.jit 等装饰器
                    └─────────────────────┘
                            │
                            ▼
                    ┌─────────────────────┐
                    │  应用 kernel_only   │  ← 只在 kernel 内检查
                    │  规则               │     禁止的控制流/API
                    └─────────────────────┘
```

### 2.3 检查模式选择策略

```python
def should_use_static_check(dsl: str, backend: str) -> bool:
    """判断是否使用静态检查"""
    # 已知场景：使用静态规则检查（快速、准确、零成本）
    known_scenarios = [
        ("triton_ascend", "ascend"),
        ("triton_cuda", "cuda"),
    ]
    return (dsl.lower(), backend.lower()) in known_scenarios
```

## 3. 检查规则

### 3.1 Python 语法检查（所有 DSL）

使用 `ast.parse()` 进行语法检查，可以检测：
- ✅ 括号不匹配（如 `"(" was never closed`）
- ✅ 缩进错误（如 `unindent does not match any outer indentation level`）
- ✅ 关键字拼写错误
- ✅ 冒号、逗号等符号遗漏

### 3.2 Kernel 范围识别

通过识别装饰器来确定 Triton kernel 函数的范围：
- `@triton.jit`
- `@jit`
- `@triton.autotune`
- `@triton.heuristics`

**作用**：`kernel_only=True` 的规则只在 kernel 函数内部检查，避免误报普通 Python 函数中的合法语法。

### 3.3 通用规则（所有 Triton 后端）

| 规则类型 | 检查项 | 说明 | kernel_only |
|---------|-------|------|-------------|
| 控制流 | `return` | 禁止在 kernel 中使用 return | ✅ |
| 控制流 | `break` | 禁止在 kernel 中使用 break | ✅ |
| 控制流 | `continue` | 禁止在 kernel 中使用 continue | ✅ |
| 语法 | `lambda` | 禁止在 kernel 中使用 lambda 表达式 | ✅ |

### 3.4 Ascend 特有规则

| 规则类型 | 检查项 | 说明 | 建议替代方案 | kernel_only |
|---------|-------|------|-------------|-------------|
| 控制流 | `while` | 禁止 while 循环 | 使用 `for range + if` | ✅ |
| API | `tl.where` | 禁止用于复杂内存偏移 | 使用 `if-else` 静态分支 | ✅ |
| API | `tl.float16(x)` | 禁止构造函数式转换 | 使用 `.to(tl.float16)` | ✅ |

### 3.5 已移除的规则

| 规则 | 移除原因 |
|------|---------|
| `direct_slice` | 正则匹配会误报普通 Python 列表索引（如 `shape[0]`），静态分析无法区分 Triton 张量和普通 Python 对象 |

## 4. 架构设计

### 4.1 文件结构

```
akg_agents/
├── core/
│   └── checker/
│       ├── __init__.py
│       └── code_checker.py      # CodeChecker 类
├── utils/langgraph/
│   ├── nodes.py                 # 添加 create_code_checker_node
│   └── routers.py               # 添加 create_code_checker_router
├── workflows/
│   └── coder_only_workflow.py   # 修改流程（第一阶段）
│   └── default_workflow.py      # 后续修改
└── resources/prompts/
    └── checker/
        └── code_check.j2        # LLM 检查模式的 prompt 模板
```

### 4.2 类设计

```python
class CodeChecker:
    """代码检查器：支持静态规则检查和 LLM 检查两种模式"""
    
    # Triton kernel 装饰器模式
    KERNEL_DECORATOR_PATTERNS = [
        r'@triton\.jit',
        r'@jit',
        r'@triton\.autotune',
        r'@triton\.heuristics',
    ]
    
    def __init__(self, backend: str, dsl: str, config: dict = None):
        self.backend = backend
        self.dsl = dsl
        self.config = config or {}
        self._kernel_decorator_re = re.compile('|'.join(self.KERNEL_DECORATOR_PATTERNS))
    
    async def check(self, code: str, task_info: dict = None) -> Tuple[bool, str, List[Dict]]:
        """
        检查代码
        
        检查流程：
        1. Python 语法检查（所有 DSL）
        2. 根据 DSL 选择静态检查或 LLM 检查
        3. 静态检查：语法错误和规则错误汇总
        4. LLM 检查：有语法错误直接返回，不调用 LLM
        
        Returns:
            Tuple[bool, str, List[Dict]]: 
                - passed: 是否通过检查
                - error_message: 格式化的错误信息
                - errors: 详细错误列表
        """
        # 第一步：Python 语法检查
        syntax_errors = self._check_python_syntax(code)
        
        if self._should_use_static_check():
            # 静态检查路径：语法错误和规则错误一起汇总
            return self._static_check(code, syntax_errors)
        else:
            # LLM 路径：有语法错误直接返回，节省 LLM 调用成本
            if syntax_errors:
                return False, self._format_errors(syntax_errors), syntax_errors
            return await self._llm_check(code, task_info)
    
    def _check_python_syntax(self, code: str) -> List[Dict]:
        """使用 ast.parse() 检查 Python 语法错误"""
        try:
            ast.parse(code)
            return []
        except SyntaxError as e:
            return [{
                "line": e.lineno or 0,
                "error_type": "syntax_error",
                "detail": f"Python 语法错误: {e.msg}",
                "suggestion": "请检查括号、引号是否匹配，缩进是否正确",
                "code_snippet": ""
            }]
    
    def _find_kernel_ranges(self, code: str) -> List[Tuple[int, int]]:
        """找出代码中所有 Triton kernel 函数的行范围"""
        # 通过装饰器识别 kernel 函数，追踪缩进确定函数边界
        ...
    
    def _is_in_kernel(self, line_num: int, kernel_ranges: List[Tuple[int, int]]) -> bool:
        """检查某行是否在 kernel 函数内"""
        ...
    
    def _static_check(self, code: str, syntax_errors: List[Dict] = None) -> Tuple[bool, str, List[Dict]]:
        """静态规则检查（快速、零成本）"""
        errors = list(syntax_errors) if syntax_errors else []
        
        # 找出所有 kernel 函数的范围
        kernel_ranges = self._find_kernel_ranges(code)
        
        for line_num, line in enumerate(lines, 1):
            in_kernel = self._is_in_kernel(line_num, kernel_ranges)
            
            for rule_name, rule_config in rules.items():
                # kernel_only 规则但不在 kernel 内，跳过
                if rule_config.get("kernel_only", False) and not in_kernel:
                    continue
                
                # 应用规则检查
                ...
        
        return passed, error_message, errors
```

### 4.3 LangGraph 节点设计

```python
# NodeFactory 中添加
@staticmethod
def create_code_checker_node(checker: CodeChecker, trace_instance, config: dict):
    async def code_checker_node(state: KernelGenState) -> dict:
        code = state.get("coder_code", "")
        passed, error_message, errors = await checker.check(code, state)
        
        return {
            "code_check_passed": passed,
            "code_check_errors": error_message,
            "code_check_details": errors,
            "step_count": state.get("step_count", 0) + 1,
            "agent_history": ["code_checker"]
        }
    return code_checker_node
```

### 4.4 路由设计

```python
# RouterFactory 中添加
@staticmethod
def create_code_checker_router():
    async def route_after_code_checker(state: KernelGenState) -> str:
        if state.get("code_check_passed", True):
            return "verifier"  # 检查通过，进入 verifier
        else:
            return "coder"     # 检查失败，回到 coder 修复
    return route_after_code_checker
```

## 5. 实施计划与变更清单

### 第一阶段（已完成 ✅）

#### 新增文件
| 文件路径 | 说明 |
|---------|------|
| `core/checker/__init__.py` | CodeChecker 模块入口 |
| `core/checker/code_checker.py` | CodeChecker 核心实现 |
| `resources/prompts/checker/code_check.j2` | LLM 检查模式的 prompt 模板 |
| `docs/code_checker_design.md` | 本设计文档 |

#### 修改文件
| 文件路径 | 修改内容 |
|---------|---------|
| `utils/langgraph/state.py` | 添加 `code_check_passed`, `code_check_errors`, `code_check_details` 字段 |
| `utils/langgraph/nodes.py` | 添加 `create_code_checker_node` 方法；修改 `coder_node` 处理 code_check_errors |
| `utils/langgraph/routers.py` | 添加 `create_code_checker_router` 方法 |
| `workflows/coder_only_workflow.py` | 添加 CodeChecker 节点到工作流 |
| `core/agent/coder.py` | 在 `input_data` 中添加 `code_check_errors` 字段 |
| `resources/prompts/coder/codegen.j2` | 添加 `code_check_errors` 显示模板 |

### 第二阶段（已完成 ✅）

1. ✅ **Python 语法检查**：使用 `ast.parse()` 检测语法错误
2. ✅ **Kernel 范围识别**：只在 kernel 内部应用 kernel_only 规则
3. ✅ **作用域感知**：避免误报普通 Python 函数中的合法语法
4. ✅ **规则优化**：简化建议，移除误报率高的规则

### 第三阶段（后续）
1. ⬜ 修改 `DefaultWorkflow` 添加检查节点
2. ⬜ 添加更多静态检查规则
3. ⬜ 优化 LLM 检查模式的 prompt

### 配置选项

在 `config.yaml` 中可配置：

```yaml
# 是否启用 CodeChecker（默认 true）
enable_code_checker: true

# CodeChecker 检查失败后最大重试次数（默认 2）
max_code_check_retries: 2

# LLM 检查模式的模型（可选，默认使用 conductor 的模型）
agent_model_config:
  code_checker: "your_model_name"
```

## 6. 预期效果

| 指标 | 优化前 | 优化后 |
|-----|-------|-------|
| 简单语法错误发现时间 | ~60秒（Verifier） | ~0秒（静态检查） |
| Python 语法错误检测 | Verifier 执行后 | 即时检测 |
| 修复循环开销 | Verifier 执行 + Conductor 分析 | 仅 Coder 重新生成 |
| 资源消耗 | 每次都执行 Verifier | 静态检查零成本 |
| LLM 调用成本 | 语法错误也调用 LLM | 语法错误直接返回 |

## 7. 错误输出示例

### 7.1 语法错误

```
============================================================
⚠️ 代码静态检查发现以下问题（错误由 CodeChecker 识别），请修复后重新生成：
============================================================

【问题 1/1】第 7 行
  错误类型: syntax_error
  错误描述: Python 语法错误: '(' was never closed（第 24 列）
  错误代码上下文（第 4-9 行）：
         5 | @triton.jit
         6 | def my_kernel(input_ptr, output_ptr, ...):
  >>>    7 |     pid = tl.program_id(0
         8 |     offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
  推荐修复方法：
    请检查第 7 行的语法：
      - 检查括号、引号是否匹配
      - 检查缩进是否正确
      ...
============================================================
```

### 7.2 规则错误

```
============================================================
⚠️ 代码静态检查发现以下问题（错误由 CodeChecker 识别），请修复后重新生成：
============================================================

【问题 1/1】第 10 行
  错误类型: forbidden_keywords/break
  错误描述: 禁止在 kernel 中使用 break 语句
  错误代码上下文（第 7-11 行）：
         7 |     pid = tl.program_id(0)
         8 |     for i in range(10):
         9 |         if i > 5:
  >>>   10 |             break
        11 | 
  推荐修复方法：
    Triton kernel 不支持 break 语句。
    请重新设计循环逻辑，通过以下方式避免使用 break：
      - 使用 mask 控制每次迭代的 tl.load/tl.store 是否实际执行
      - 使用 if 条件判断跳过不需要执行的逻辑
    
    参考文档：suggestion_docs.md 第4节「API使用限制与替代方案」
============================================================
```

## 8. 参考文档

- `suggestion_docs.md`: Triton 开发规范和限制说明
- `basic_docs.md`: Triton 编程基础教程
- `nodes.py`: LangGraph 节点工厂
- `routers.py`: LangGraph 路由工厂
