# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
CodeChecker: 代码检查器

支持两种检查模式：
1. 静态规则检查：已知场景（如 triton-ascend），使用规则匹配，零成本
2. LLM 检查：未知场景，使用 LLM 分析，灵活性高
"""

import re
import ast
import logging
import os
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from ai_kernel_generator import get_project_root

logger = logging.getLogger(__name__)


@dataclass
class CheckError:
    """检查错误信息"""
    line: int
    error_type: str
    detail: str
    suggestion: str
    code_snippet: str


class CodeChecker:
    """
    代码检查器：支持静态规则检查和 LLM 检查两种模式
    
    使用场景：
    - 在 Coder 生成代码后、Verifier 验证前，进行快速的代码检查
    - 检测常见的语法错误，避免浪费 Verifier 的执行时间
    """
    
    # 已知场景配置：这些场景使用静态规则检查
    KNOWN_SCENARIOS = {
        ("triton_ascend", "ascend"),
        ("triton_cuda", "cuda"),
    }
    
    # 通用禁止规则（所有 Triton 后端）
    # kernel_only: True 表示仅在 kernel 函数内检查，False 表示全局检查
    COMMON_RULES = {
        "forbidden_keywords": {
            "return": {
                "pattern": r'(?<!["\'])\breturn\b(?!["\'])',
                "detail": "禁止在 kernel 中使用 return 语句",
                "suggestion": """Triton kernel 不支持 return 语句。
请重新设计 kernel 逻辑，通过以下方式避免使用 return：
  - 使用 mask 控制 tl.load/tl.store 的执行范围
  - 使用 if-else 分支处理不同条件
  - 将需要提前退出的逻辑转换为条件执行

参考文档：suggestion_docs.md 第4节「API使用限制与替代方案」""",
                "kernel_only": True  # 仅在 kernel 内检查
            },
            "break": {
                "pattern": r'(?<!["\'])\bbreak\b(?!["\'])',
                "detail": "禁止在 kernel 中使用 break 语句",
                "suggestion": """Triton kernel 不支持 break 语句。
请重新设计循环逻辑，通过以下方式避免使用 break：
  - 使用 mask 控制每次迭代的 tl.load/tl.store 是否实际执行
  - 使用 if 条件判断跳过不需要执行的逻辑
  - 循环仍会执行所有迭代，但通过 mask 控制实际操作

参考文档：suggestion_docs.md 第4节「API使用限制与替代方案」""",
                "kernel_only": True  # 仅在 kernel 内检查
            },
            "continue": {
                "pattern": r'(?<!["\'])\bcontinue\b(?!["\'])',
                "detail": "禁止在 kernel 中使用 continue 语句",
                "suggestion": """Triton kernel 不支持 continue 语句。
请重新设计循环逻辑，通过以下方式避免使用 continue：
  - 使用 if 条件判断包裹需要执行的代码块
  - 使用 mask 控制 tl.load/tl.store 的执行

参考文档：suggestion_docs.md 第4节「API使用限制与替代方案」""",
                "kernel_only": True  # 仅在 kernel 内检查
            },
        },
        "forbidden_syntax": {
            "lambda": {
                "pattern": r'\blambda\s+',
                "detail": "禁止在 kernel 中使用 lambda 表达式",
                "suggestion": """Triton kernel 不支持 lambda 表达式。
推荐写法：直接使用内联代码替代 lambda
  # ❌ 错误写法
  func = lambda x: x * 2
  result = func(data)
  
  # ✅ 正确写法：内联计算
  result = data * 2

参考文档：suggestion_docs.md 第4节「API使用限制与替代方案」""",
                "kernel_only": True  # 仅在 kernel 内检查
            },
        }
    }
    
    # Ascend 特有规则
    ASCEND_RULES = {
        "forbidden_keywords": {
            "while": {
                "pattern": r'(?<!["\'])\bwhile\b(?!["\'])',
                "detail": "Ascend 后端 kernel 中禁止使用 while 循环",
                "suggestion": """使用 for range + if 替代 while 循环。
推荐写法：
  # ❌ 错误写法（循环上限是编译时常量）
  i = 0
  while i < N_ITERS:  # N_ITERS 是编译时常量
      # 处理逻辑
      i += 1
  
  # ✅ 正确写法1：循环上限是编译时常量
  for i in range(N_ITERS):  # N_ITERS: tl.constexpr
      # 处理逻辑
  
  # ❌ 错误写法（循环上限是运行时动态值）
  i = 0
  while i < n_iters:  # n_iters 是运行时动态值
      offset = i * TILE + tl.arange(0, TILE)
      data = tl.load(ptr + offset)
      i += 1
  
  # ✅ 正确写法2：循环上限是运行时动态值
  for i in range(MAX_ITERS):  # MAX_ITERS: tl.constexpr（需足够大）
      if i < n_iters:  # n_iters 是运行时参数
          offset = i * TILE + tl.arange(0, TILE)
          data = tl.load(ptr + offset)
          tl.store(ptr + offset, data * 2)""",
                "kernel_only": True  # 仅在 kernel 内检查
            },
        },
        "forbidden_api": {
            "tl.where": {
                "pattern": r'\btl\.where\s*\(',
                "detail": "Ascend 后端 kernel 中禁止使用 tl.where（特别是用于计算内存偏移时）",
                "suggestion": """使用 if-else 静态分支替代 tl.where。
推荐写法：
  # ❌ 错误写法：在内存偏移中使用 tl.where
  offset = tl.where(condition, offset1, offset2)
  data = tl.load(ptr + offset)
  
  # ✅ 正确写法：使用 if-else 静态分支
  if condition:
      data = tl.load(ptr + offset1)
  else:
      data = tl.load(ptr + offset2)
  
注意：Ascend 后端对 tl.where 生成的复杂指针运算支持不完全，应使用 if-else 静态分支处理。""",
                "kernel_only": True  # 仅在 kernel 内检查
            },
            "tl.float16_conversion": {
                "pattern": r'\btl\.float16\s*\(',
                "detail": "kernel 中禁止使用 tl.float16(x) 类型转换语法",
                "suggestion": """使用 .to(tl.float16) 方法进行类型转换。
推荐写法：
  # ❌ 错误写法
  result = tl.float16(scalar)
  
  # ✅ 正确写法
  result = scalar.to(tl.float16)
  
注意：Triton 仅支持 to(type) 方法进行类型转换，不支持构造函数式转换。""",
                "kernel_only": True  # 仅在 kernel 内检查
            },
        },
    }
    
    # CUDA 特有规则（目前为空，后续可扩展）
    CUDA_RULES = {}
    
    # Triton kernel 装饰器模式
    KERNEL_DECORATOR_PATTERNS = [
        r'@triton\.jit',
        r'@jit',
        r'@triton\.autotune',
        r'@triton\.heuristics',
    ]
    
    def __init__(self, backend: str, dsl: str, config: Optional[dict] = None):
        """
        初始化 CodeChecker
        
        Args:
            backend: 后端类型（如 'ascend', 'cuda'）
            dsl: DSL 类型（如 'triton_ascend', 'triton_cuda'）
            config: 配置字典，包含 LLM 相关配置
        """
        self.backend = backend.lower() if backend else ""
        self.dsl = dsl.lower() if dsl else ""
        self.config = config or {}
        self._compiled_rules = None
        self._kernel_decorator_re = re.compile('|'.join(self.KERNEL_DECORATOR_PATTERNS))
        self.root_dir = get_project_root()
        
        logger.info(f"CodeChecker initialized: backend={self.backend}, dsl={self.dsl}")
    
    def _should_use_static_check(self) -> bool:
        """
        判断是否使用静态检查
        
        Returns:
            bool: True 表示使用静态规则检查，False 表示使用 LLM 检查
        """
        return (self.dsl, self.backend) in self.KNOWN_SCENARIOS
    
    def _find_kernel_ranges(self, code: str) -> List[Tuple[int, int]]:
        """
        找出代码中所有 Triton kernel 函数的行范围
        
        通过识别 @triton.jit、@jit 等装饰器来确定 kernel 函数，
        并通过缩进追踪确定函数的结束位置。
        
        Args:
            code: 完整的代码字符串
        
        Returns:
            List[Tuple[int, int]]: [(start_line, end_line), ...] 
                每个元组表示一个 kernel 的行范围（1-based，左闭右开）
        """
        lines = code.split('\n')
        kernel_ranges = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # 检测到 kernel 装饰器
            if self._kernel_decorator_re.search(stripped):
                decorator_start = i
                
                # 跳过连续的装饰器行（可能有多个装饰器叠加）
                while i < len(lines) and lines[i].strip().startswith('@'):
                    i += 1
                
                # 下一行应该是 def 函数定义
                if i < len(lines) and lines[i].strip().startswith('def '):
                    kernel_start = decorator_start + 1  # 1-based，从装饰器开始
                    func_def_line = lines[i]
                    kernel_indent = len(func_def_line) - len(func_def_line.lstrip())
                    
                    # 从函数定义的下一行开始找函数结束位置
                    i += 1
                    while i < len(lines):
                        current_line = lines[i]
                        # 跳过空行
                        if not current_line.strip():
                            i += 1
                            continue
                        # 非空行，检查缩进
                        current_indent = len(current_line) - len(current_line.lstrip())
                        if current_indent <= kernel_indent:
                            # 缩进变小或相等，函数结束
                            break
                        i += 1
                    
                    kernel_end = i + 1  # 1-based，左闭右开
                    kernel_ranges.append((kernel_start, kernel_end))
                    logger.debug(f"Found kernel range: lines {kernel_start}-{kernel_end-1}")
                    continue
            
            i += 1
        
        if kernel_ranges:
            logger.info(f"CodeChecker: Found {len(kernel_ranges)} kernel function(s)")
        
        return kernel_ranges
    
    def _is_in_kernel(self, line_num: int, kernel_ranges: List[Tuple[int, int]]) -> bool:
        """
        检查某行是否在 Triton kernel 函数内
        
        Args:
            line_num: 行号（1-based）
            kernel_ranges: kernel 函数的行范围列表
        
        Returns:
            bool: True 表示在 kernel 内，False 表示在 kernel 外
        """
        for start, end in kernel_ranges:
            if start <= line_num < end:
                return True
        return False
    
    def _check_python_syntax(self, code: str) -> List[Dict]:
        """
        检查 Python 代码语法错误
        
        使用 ast.parse() 进行语法检查，可以检测：
        - 括号不匹配（如 "(" was never closed）
        - 缩进错误
        - 关键字拼写错误
        - 其他语法错误
        
        Args:
            code: 要检查的代码
        
        Returns:
            List[Dict]: 语法错误列表，每个错误包含 line, error_type, detail, suggestion, code_snippet
        """
        errors = []
        try:
            ast.parse(code)
        except SyntaxError as e:
            line_num = e.lineno or 0
            # 获取错误行的代码片段
            code_lines = code.split('\n')
            code_snippet = ""
            if 0 < line_num <= len(code_lines):
                code_snippet = code_lines[line_num - 1].rstrip()
            
            # 构建详细的错误信息
            error_msg = e.msg or "语法错误"
            if e.offset:
                error_msg += f"（第 {e.offset} 列）"
            
            errors.append({
                "line": line_num,
                "error_type": "syntax_error",
                "detail": f"Python 语法错误: {error_msg}",
                "suggestion": f"""请检查第 {line_num} 行的语法：
  - 检查括号、引号是否匹配
  - 检查缩进是否正确
  - 检查关键字拼写是否正确
  - 检查冒号、逗号等符号是否遗漏""",
                "code_snippet": code_snippet
            })
            
            logger.warning(f"CodeChecker: Python syntax error at line {line_num}: {error_msg}")
        
        return errors
    
    def _get_rules(self) -> Dict:
        """
        根据 backend/dsl 获取适用的检查规则
        
        Returns:
            Dict: 合并后的规则字典
        """
        if self._compiled_rules is not None:
            return self._compiled_rules
        
        # 从通用规则开始
        rules = {}
        for category, items in self.COMMON_RULES.items():
            rules[category] = dict(items)
        
        # 添加 Ascend 特有规则
        if "ascend" in self.backend or "ascend" in self.dsl:
            for category, items in self.ASCEND_RULES.items():
                if category not in rules:
                    rules[category] = {}
                rules[category].update(items)
        
        # 添加 CUDA 特有规则
        if "cuda" in self.backend or "cuda" in self.dsl:
            for category, items in self.CUDA_RULES.items():
                if category not in rules:
                    rules[category] = {}
                rules[category].update(items)
        
        self._compiled_rules = rules
        return rules
    
    async def check(self, code: str, task_info: Optional[dict] = None) -> Tuple[bool, str, List[Dict]]:
        """
        检查代码
        
        Args:
            code: 要检查的代码
            task_info: 任务信息（LLM 模式使用）
        
        Returns:
            Tuple[bool, str, List[Dict]]:
                - passed: 是否通过检查
                - error_message: 格式化的错误信息（用于传递给 Coder）
                - errors: 详细错误列表
        """
        if not code or not code.strip():
            logger.warning("CodeChecker: Empty code provided")
            return True, "", []
        
        # 第一步：Python 语法检查（对所有 DSL 都适用）
        syntax_errors = self._check_python_syntax(code)
        
        if self._should_use_static_check():
            # 静态检查路径：语法错误和规则错误一起汇总
            logger.info(f"CodeChecker: Using static check for {self.dsl}/{self.backend}")
            return self._static_check(code, syntax_errors)
        else:
            # LLM 检查路径：如果有语法错误，直接返回，不调用 LLM（节省成本）
            if syntax_errors:
                logger.warning(f"CodeChecker: Found {len(syntax_errors)} syntax error(s), skipping LLM check")
                code_lines = code.split('\n')
                error_message = self._format_errors(syntax_errors, code_lines)
                return False, error_message, syntax_errors
            
            logger.info(f"CodeChecker: Using LLM check for {self.dsl}/{self.backend}")
            return await self._llm_check(code, task_info)
    
    def _static_check(self, code: str, syntax_errors: Optional[List[Dict]] = None) -> Tuple[bool, str, List[Dict]]:
        """
        静态规则检查（快速、零成本）
        
        支持作用域感知：
        - kernel_only=True 的规则仅在 Triton kernel 函数内检查
        - kernel_only=False 或未设置的规则在全局检查
        
        Args:
            code: 要检查的代码
            syntax_errors: 预先检测到的语法错误列表（可选）
        
        Returns:
            Tuple[bool, str, List[Dict]]: (是否通过, 错误信息, 错误详情列表)
        """
        # 从语法错误开始（如果有的话）
        errors: List[Dict] = list(syntax_errors) if syntax_errors else []
        lines = code.split('\n')
        rules = self._get_rules()
        
        # 找出所有 kernel 函数的范围
        kernel_ranges = self._find_kernel_ranges(code)
        
        for line_num, line in enumerate(lines, 1):
            # 跳过注释行和空行
            stripped_line = line.strip()
            if not stripped_line or stripped_line.startswith('#'):
                continue
            
            # 判断当前行是否在 kernel 内
            in_kernel = self._is_in_kernel(line_num, kernel_ranges)
            
            # 检查所有规则类别
            for category, items in rules.items():
                for rule_name, rule_config in items.items():
                    # 检查规则作用域
                    is_kernel_only = rule_config.get("kernel_only", False)
                    if is_kernel_only and not in_kernel:
                        # kernel_only 规则但不在 kernel 内，跳过
                        continue
                    
                    pattern = rule_config["pattern"]
                    if re.search(pattern, line):
                        # 检查是否在字符串中（简单判断）
                        if self._is_in_string(line, pattern):
                            continue
                        
                        errors.append({
                            "line": line_num,
                            "error_type": f"{category}/{rule_name}",
                            "detail": rule_config["detail"],
                            "suggestion": rule_config["suggestion"],
                            "code_snippet": line.rstrip()  # 保存完整的错误行代码（保留原始缩进）
                        })
        
        passed = len(errors) == 0
        error_message = self._format_errors(errors, lines) if errors else ""
        
        if errors:
            logger.warning(f"CodeChecker: Found {len(errors)} issues")
            # 打印所有错误（不限制数量）
            for err in errors:
                logger.warning(f"  Line {err['line']}: {err['detail']}")
            # 打印格式化的错误信息（包含代码上下文）
            logger.warning(f"\n{error_message}")
        else:
            logger.info("CodeChecker: Static check passed")
        
        return passed, error_message, errors
    
    def _is_in_string(self, line: str, pattern: str) -> bool:
        """
        简单判断匹配是否在字符串内
        
        注意：这是一个简化的实现，可能有误判
        """
        # 找到匹配位置
        match = re.search(pattern, line)
        if not match:
            return False
        
        pos = match.start()
        
        # 统计匹配位置之前的引号数量
        before = line[:pos]
        single_quotes = before.count("'") - before.count("\\'")
        double_quotes = before.count('"') - before.count('\\"')
        
        # 如果引号数量为奇数，说明在字符串内
        return (single_quotes % 2 == 1) or (double_quotes % 2 == 1)
    
    def _load_doc(self, doc_path: str) -> str:
        """
        加载文档内容（参考 AgentBase 的实现）
        
        Args:
            doc_path: 文档文件的相对路径（如 "basic_docs.md" 或 "suggestion_docs.md"）
        
        Returns:
            str: 文档内容，如果文件不存在则返回空字符串
        """
        try:
            # 解析文档路径（参考 AgentBase._resolve_configurable_doc_path）
            resolved_path = self._resolve_doc_path(doc_path)
            
            # 拼接完整路径
            full_path = os.path.join(self.root_dir, resolved_path)
            
            if not os.path.exists(full_path):
                logger.warning(f"CodeChecker: Doc not found: {full_path}")
                return ""
            
            logger.info(f"CodeChecker: Loading doc: {full_path}")
            
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.warning(f"CodeChecker: Failed to load doc {doc_path}: {e}")
            return ""
    
    def _resolve_doc_path(self, doc_path: str) -> str:
        """
        解析文档路径（参考 AgentBase._resolve_configurable_doc_path）
        
        Args:
            doc_path: 原始文档路径（文件名或相对路径）
        
        Returns:
            str: 解析后的文档路径
        """
        try:
            # 检查是否有传入的config
            if not self.config:
                # 如果没有配置，使用默认路径（根据 dsl 推断）
                docs_subdir = f"{self.dsl}_docs"
                return os.path.join("resources", "docs", docs_subdir, doc_path)
            
            # 尝试从docs_dir配置中获取对应agent的文档目录
            docs_dir_config = self.config.get('docs_dir', {})
            # CodeChecker 使用 coder 的文档目录（因为它们检查的是 coder 生成的代码）
            if 'coder' in docs_dir_config:
                docs_dir = docs_dir_config['coder']
                return os.path.join(docs_dir, doc_path)
            
            # 如果没有配置，使用默认路径
            docs_subdir = f"{self.dsl}_docs"
            return os.path.join("resources", "docs", docs_subdir, doc_path)
            
        except Exception as e:
            logger.warning(f"CodeChecker: Failed to resolve doc path {doc_path}: {e}")
            # 降级到默认路径
            docs_subdir = f"{self.dsl}_docs"
            return os.path.join("resources", "docs", docs_subdir, doc_path)
    
    async def _llm_check(self, code: str, task_info: Optional[dict]) -> Tuple[bool, str, List[Dict]]:
        """
        LLM 分析检查（灵活、有成本）
        
        Args:
            code: 要检查的代码
            task_info: 任务信息
        
        Returns:
            Tuple[bool, str, List[Dict]]: (是否通过, 错误信息, 错误详情列表)
        """
        try:
            from ai_kernel_generator.core.llm.model_loader import create_model
            from ai_kernel_generator.utils.common_utils import get_prompt_path
            from jinja2 import Template
            import os
            
            # 加载 prompt 模板
            prompt_path = os.path.join(get_prompt_path(), "checker", "code_check.j2")
            if not os.path.exists(prompt_path):
                logger.warning(f"CodeChecker: LLM prompt template not found: {prompt_path}")
                logger.warning("CodeChecker: Falling back to static check")
                return self._static_check(code)
            
            with open(prompt_path, 'r', encoding='utf-8') as f:
                template = Template(f.read())
            
            # 加载文档（参考 coder 的方式）
            basic_docs = self._load_doc("basic_docs.md")
            suggestion_docs = self._load_doc("suggestion_docs.md")
            
            # 获取 expert_suggestion（如果有）
            expert_suggestion = ""
            if task_info:
                expert_suggestion = task_info.get("expert_suggestion", "")
            
            # 渲染 prompt
            prompt = template.render(
                code=code[:5000],  # 限制代码长度
                dsl=self.dsl,
                backend=self.backend,
                basic_docs=basic_docs,
                suggestion_docs=suggestion_docs,
                expert_suggestion=expert_suggestion[:2000] if expert_suggestion else ""
            )
            
            # 调用 LLM
            model_config = self.config.get("agent_model_config", {})
            model_name = model_config.get("code_checker") or model_config.get("conductor")
            
            if not model_name:
                logger.warning("CodeChecker: No LLM model configured, falling back to static check")
                return self._static_check(code)
            
            model = create_model(model_name)
            
            # 调用模型
            if hasattr(model, 'ainvoke'):
                response = await model.ainvoke(prompt)
                response_text = response.content if hasattr(response, 'content') else str(response)
            else:
                completion = await model.chat.completions.create(
                    model=model.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,  # 低温度，更确定性
                    max_tokens=2000,
                )
                response_text = completion.choices[0].message.content
            
            # 解析 LLM 响应（传递代码行列表用于格式化）
            code_lines = code.split('\n')
            return self._parse_llm_response(response_text, code_lines)
            
        except Exception as e:
            logger.error(f"CodeChecker: LLM check failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            # 失败时回退到静态检查
            logger.warning("CodeChecker: Falling back to static check")
            return self._static_check(code)
    
    def _parse_llm_response(self, response: str, code_lines: Optional[List[str]] = None) -> Tuple[bool, str, List[Dict]]:
        """
        解析 LLM 响应
        
        期望 LLM 返回 JSON 格式：
        {
            "passed": true/false,
            "errors": [
                {"line": 10, "detail": "...", "suggestion": "..."}
            ]
        }
        
        Args:
            response: LLM 响应文本
            code_lines: 完整的代码行列表（用于格式化错误信息）
        """
        import json
        
        try:
            # 尝试提取 JSON
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                result = json.loads(json_match.group())
                passed = result.get("passed", True)
                errors = result.get("errors", [])
                
                # 格式化错误
                formatted_errors = []
                for err in errors:
                    line_num = err.get("line", 0)
                    # 如果 LLM 没有提供 code_snippet，从代码行中提取
                    code_snippet = err.get("code_snippet", "")
                    if not code_snippet and code_lines and 0 < line_num <= len(code_lines):
                        code_snippet = code_lines[line_num - 1].rstrip()
                    
                    formatted_errors.append({
                        "line": line_num,
                        "error_type": "llm_analysis",
                        "detail": err.get("detail", ""),
                        "suggestion": err.get("suggestion", ""),
                        "code_snippet": code_snippet
                    })
                
                error_message = self._format_errors(formatted_errors, code_lines) if formatted_errors else ""
                return passed, error_message, formatted_errors
        except json.JSONDecodeError:
            pass
        
        # 如果解析失败，简单判断
        response_lower = response.lower()
        if "no issues" in response_lower or "passed" in response_lower or "correct" in response_lower:
            return True, "", []
        elif "error" in response_lower or "issue" in response_lower or "problem" in response_lower:
            error_dict = {
                "line": 0,
                "error_type": "llm_analysis",
                "detail": response[:500],
                "suggestion": "请根据上述分析修改代码",
                "code_snippet": ""
            }
            error_message = self._format_errors([error_dict], code_lines)
            return False, error_message, [error_dict]
        
        # 默认通过
        return True, "", []
    
    def _format_errors(self, errors: List[Dict], code_lines: Optional[List[str]] = None) -> str:
        """
        格式化错误信息，便于传递给 Coder
        
        Args:
            errors: 错误列表
            code_lines: 完整的代码行列表（用于显示上下文），如果为 None 则不显示上下文
        
        Returns:
            str: 格式化后的错误信息
        """
        if not errors:
            return ""
        
        lines = [
            "=" * 60,
            "⚠️ 代码静态检查发现以下问题（错误由 CodeChecker 识别），请修复后重新生成：",
            "=" * 60,
            ""
        ]
        
        # 确保包含所有错误
        for i, err in enumerate(errors, 1):
            error_line = err['line']
            lines.append(f"【问题 {i}/{len(errors)}】第 {error_line} 行")
            lines.append(f"  错误类型: {err.get('error_type', 'unknown')}")
            lines.append(f"  错误描述: {err['detail']}")
            
            # 显示错误行的代码上下文（前后±3行）
            if code_lines is not None:
                # 计算上下文范围（前后各3行，共7行）
                # 注意：确保不越界
                start_line = max(1, error_line - 3)  # 至少从第1行开始
                end_line = min(len(code_lines), error_line + 3)  # 不超过总行数
                
                lines.append(f"  错误代码上下文（第 {start_line}-{end_line} 行）：")
                for ctx_line_num in range(start_line, end_line + 1):
                    ctx_line = code_lines[ctx_line_num - 1]  # 转换为0-based索引
                    # 标记错误行（使用 >>> 标记）
                    if ctx_line_num == error_line:
                        lines.append(f"  >>> {ctx_line_num:4d} | {ctx_line}")
                    else:
                        lines.append(f"      {ctx_line_num:4d} | {ctx_line}")
            elif err.get('code_snippet'):
                # 如果没有代码行列表，至少显示错误行的完整代码
                lines.append(f"  错误代码（第 {error_line} 行）: {err['code_snippet']}")
            
            # 显示修复建议
            if err.get('suggestion'):
                lines.append(f"  推荐修复方法：")
                suggestion_lines = err['suggestion'].strip().split('\n')
                for sug_line in suggestion_lines:
                    lines.append(f"    {sug_line}")
            
            lines.append("")
        
        lines.append("=" * 60)
        lines.append(f"共发现 {len(errors)} 个问题，请严格按照 Triton 编程规范修改上述问题。")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def get_check_summary(self, errors: List[Dict]) -> str:
        """
        获取检查摘要（简短版本，用于日志）
        
        Args:
            errors: 错误列表
        
        Returns:
            str: 简短的摘要信息
        """
        if not errors:
            return "✅ 代码检查通过"
        
        error_types = set(err.get("error_type", "unknown") for err in errors)
        return f"❌ 发现 {len(errors)} 个问题: {', '.join(error_types)}"

