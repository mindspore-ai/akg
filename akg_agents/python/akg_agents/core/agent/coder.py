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

# DEPRECATED: 此模块已废弃，新版请使用 akg_agents.op.agents.kernel_gen.KernelGen
# 当所有引用方（langgraph_task.py 等）迁移完成后将删除此文件
import warnings
warnings.warn(
    "akg_agents.core.agent.coder is deprecated, use akg_agents.op.agents.kernel_gen instead",
    DeprecationWarning,
    stacklevel=2,
)

import logging
import re
from typing import Tuple, List
from pathlib import Path
from akg_agents.op.database.coder_database import CoderDatabase
from akg_agents.op.utils.triton_ascend_api_docs import (
    resolve_triton_ascend_api_docs,
)
from akg_agents.utils.common_utils import ParserFactory, remove_copyright_from_text, get_md5_hash
from akg_agents.utils.hardware_utils import get_hardware_doc
from akg_agents.op.utils.swft_docs_loader import get_swft_docs_content
from akg_agents.core_v2.agents import AgentBase, register_agent
from akg_agents import get_project_root
from akg_agents.core.extractor_torch import extract_kernelbench_shapes_dtypes

logger = logging.getLogger(__name__)


@register_agent
class Coder(AgentBase):
    def __init__(self,
                 op_name: str,
                 task_desc: str,
                 dsl: str,
                 framework: str,
                 backend: str,
                 arch: str = "",
                 workflow_config_path: str = None,  # 已废弃，保留用于向后兼容
                 parser_config_path: str = None,    # 新的 parser 配置路径
                 config: dict = None,
                 source_backend: str = None,  # 源后端（如 cuda），用于跨后端转换场景
                 source_arch: str = None):    # 源架构（如 a100），用于跨后端转换场景
        self.op_name = op_name
        self.task_desc = remove_copyright_from_text(task_desc)
        self.dsl = dsl
        self.framework = framework
        self.backend = backend
        self.arch = arch
        self.workflow_config_path = workflow_config_path  # 保留用于向后兼容
        self.parser_config_path = parser_config_path  # 新的配置路径
        self.config = config
        self.source_backend = source_backend  # 跨后端转换时的源后端
        self.source_arch = source_arch  # 跨后端转换时的源架构
        self.codegen_step_count = 0
        self.api_step_count = 0

        self.sample_num = 3 # RAG默认样本数
        
        # RAG检索结果缓存，避免相同任务参数重复检索
        # key: 任务参数的hash, value: 检索结果字符串
        self._rag_cache = {}

        # 从config中获取model_config
        if config:
            self.model_config = config.get("agent_model_config", {})
        else:
            raise ValueError("config is required for Coder")

        context = {
            "dsl": dsl,
            "op_name": op_name,
            "framework": framework,
            "backend": backend,
            "arch": arch,
            "task_desc": task_desc,
            "source_backend": source_backend,
            "source_arch": source_arch,
        }
        if config and config.get("session_id"):
            context["session_id"] = config["session_id"]
        super().__init__(context=context, config=config)

        self.format_instructions = ""

        if "triton_cuda" in self.dsl or "triton_ascend" in self.dsl:
            if self.dsl == "triton_cuda":
                self.func_name = f"{self.op_name}_triton_cuda_{self.framework}"
            elif self.dsl == "triton_ascend":
                self.func_name = f"{self.op_name}_triton_ascend_{self.framework}"
            else:
                # 兼容旧代码，如果dsl包含triton_cuda或triton_ascend但不是精确匹配
                self.func_name = f"{self.op_name}_{self.dsl}_{self.framework}"
        else:
            self.func_name = f"{self.op_name}_{self.dsl}_{self.framework}"

        # 初始化coder生成模板
        self.coder_prompt = self.load_template("coder/codegen.j2")
        self.api_docs_prompt = self.load_template("utils/api_gen_template.j2")
        self.user_examples_prompt = self.load_template("utils/examples_compression_template.j2")

        # 准备基础文档数据
        self.base_doc = {
            "op_name": self.op_name,
            "task_desc": self.task_desc,
            "framework": self.framework,
            "dsl": self.dsl,
            "func_name": self.func_name,
            "format_instructions": self.format_instructions,

            "api_docs": self._load_api_docs_initial(),
            "api_debug_docs": self.load_doc("api/api_debug.md"),
            "dsl_basic_docs": self.load_doc("basic_docs.md"),
            "expert_suggestion": self.load_doc("suggestion_docs.md"),
            "expert_suggestion_debug": self.load_doc("suggestion_docsdebug.md"),
            "backend": self.backend,

            # 可选参数
            "hardware_docs": get_hardware_doc(self.backend, self.arch),
            "arch_name": self.arch,
            
            # 跨后端转换参数
            "source_backend": self.source_backend,  # 源后端（如 cuda -> ascend）
            "source_arch": self.source_arch,        # 源架构（如 a100 -> ascend910b4）
        }

    def _load_api_docs_initial(self) -> str:
        if self.dsl == "triton_ascend":
            return ""
        return self.load_doc("api/api.md")

    async def _ensure_api_docs_loaded(self) -> str:
        if self.dsl != "triton_ascend":
            return self.base_doc["api_docs"]

        if not self.base_doc["api_docs"]:
            self.base_doc["api_docs"] = await resolve_triton_ascend_api_docs(
                backend=self.backend,
                arch=self.arch,
            )
        return self.base_doc["api_docs"]
        
        ## 添加详细算子信息
        try:
            from ai_kernel_generator.core.extractor_torch import extract_kernelbench_shapes_dtypes
            meta = extract_kernelbench_shapes_dtypes(self.base_doc["task_desc"], device="cuda")
            add_info = ""
            print("=== Inputs ===")
            for x in meta["inputs"]:
                print(x)
                add_info += x.__str__() + "\n"

            print("=== Parameters ===")
            for k, v in meta["parameters"].items():
                print(k, v)
                add_info += f"{k}: {v}\n"

            if meta.get("graph_tensors"):
                print("=== Graph tensors (node outputs) ===")
                for t in meta["graph_tensors"][:20]:
                    print(t)
                    add_info += t.__str__() + "\n"

            self.base_doc["task_desc"] += "\n\n\n## 算子参数信息\n" + add_info
        except Exception as e:
            logger.warning(f"Failed to extract shapes and dtypes: {e}")


    def _load_user_examples(self) -> str:
        """
        根据framework加载对应的DSL示例代码

        Returns:
            str: 示例代码内容，如果找不到对应示例则返回空字符串
        """
        if not self.framework:
            logger.warning("framework为空，无法加载示例代码")
            return ""

        # 使用配置化的文档路径
        try:
            # 从config中获取coder的docs_dir
            if not self.config:
                raise ValueError("No config provided. Cannot resolve document path.")

            docs_dir_config = self.config.get('docs_dir', {})
            if 'coder' not in docs_dir_config:
                raise ValueError("No doc directory configured for coder agent.")

            coder_docs_dir = docs_dir_config['coder']
            base_dir = Path(get_project_root()) / coder_docs_dir / "examples"

        except Exception as e:
            logger.warning(f"Failed to resolve configurable doc path: {e}, using fallback path")
            # 降级到硬编码路径（根据DSL类型选择）
            docs_subdir = f"{self.dsl}_docs"
            base_dir = Path(get_project_root()) / "op" / "resources" / "docs" / docs_subdir / "examples"

        if not base_dir.exists():
            logger.warning(f"Triton示例目录不存在: {base_dir}, 返回空字符串")
            return ""

        all_code = []
        # 支持多种文件格式：py, md, txt等
        supported_extensions = ['*.py', '*.md', '*.txt']

        for extension in supported_extensions:
            # 使用glob模式匹配framework开头的文件
            for file_path in base_dir.glob(f"{self.framework}_{extension}"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        if content:
                            # 根据文件类型添加不同的注释标识
                            if file_path.suffix == '.py':
                                all_code.append(f"# Python File: {file_path.name}\n{content}\n")
                            elif file_path.suffix == '.md':
                                all_code.append(f"# Markdown File: {file_path.name}\n{content}\n")
                            elif file_path.suffix == '.txt':
                                all_code.append(f"# Text File: {file_path.name}\n{content}\n")
                            else:
                                all_code.append(f"# File: {file_path.name}\n{content}\n")
                except Exception as e:
                    logger.warning(f"读取示例文件 {file_path} 时发生错误: {str(e)}")
                    continue

        if not all_code:
            logger.warning(f"未找到{self.framework}相关的示例文件")
            return ""

        return "\n".join(all_code)

    @staticmethod
    def _extract_code(raw_output: str, dsl: str = "") -> str:
        """从 LLM 输出中提取纯代码。

        支持从 ```python / ```cpp / ```cuda 等代码块提取。
        若模型直接输出纯代码（无 markdown 包裹），则返回 strip 后的内容。
        """
        code = raw_output.strip()
        # 容错：如果模型用了 ```xxx ... ``` 包裹，提取内部内容
        pattern = r'```(?:python|cpp|cuda|c\+\+|cuda-c|triton)?\s*\n(.*?)```'
        matches = re.findall(pattern, code, re.DOTALL)
        if matches:
            code = max(matches, key=len).strip()
        return code

    async def _generate_api_docs(self, sketch: str, conductor_suggestion: str, task_info: dict) -> str:
        """
        生成API文档，如果原始API文档过长则使用LLM进行内容压缩

        Args:
            sketch: AUL代码作为sketch
            conductor_suggestion: Conductor建议
            task_info: 任务信息字典

        Returns:
            str: 适合的API文档内容
        """
        api_docs = await self._ensure_api_docs_loaded()

        if len(api_docs) > 6000:  # 如果api文档过长，使用llm进行content压缩
            api_parser = ParserFactory.get_api_parser()
            format_api_instructions = api_parser.get_format_instructions()
            api_input_data = {
                **self.base_doc,
                "api_docs": api_docs,
                "sketch": sketch,  # AUL代码作为sketch
                "llm_suggestions": conductor_suggestion,  # Conductor建议
                "error_log": task_info.get('verifier_error', ''),
                "format_instructions": format_api_instructions
            }

            self.api_step_count += 1
            to_update_api_details = {
                "agent_name": "api",
                "hash": task_info.get("task_id", "Api") + "@" + str(self.api_step_count),
                "task_id": "",
                "step": self.api_step_count,
            }
            self.context.update(to_update_api_details)

            api_docs_json, _, _ = await self.run_llm(self.api_docs_prompt, api_input_data, self.model_config.get("api_generator") or "standard")
            parsed_content = api_parser.parse(api_docs_json)
            api_docs_suitable = "\n\n".join(
                f"API name: {name}\nAPI description:{desc}\nAPI implement：\n{impl}"
                for name, desc, impl in zip(
                    parsed_content.api_name,
                    parsed_content.api_desc,
                    parsed_content.api_example
                )
            )
        else:
            api_docs_suitable = api_docs

        return api_docs_suitable

    def load_doc(self, doc_path: str) -> str:
        """
        重写load_doc方法，特殊处理swft后端
        逻辑：
        1. 常规后端：使用标准的文档读取方式
        2. swft后端：
           - 如果指定的文档路径存在，就读取该文档
           - 如果指定的文档路径不存在，则默认使用swft的本地文档

        Args:
            doc_path (str): 文档文件的相对路径

        Returns:
            str: 文档内容
        """
        # 检查是否是swft后端
        if self.dsl.lower() == "swft":
            logger.info(f"检测到swft后端，尝试读取文档: {doc_path}")

            try:
                # 首先尝试使用父类的标准方法读取指定文档
                standard_content = super().load_doc(doc_path)
                if standard_content:
                    logger.info(f"成功读取到swft的指定文档: {doc_path}")
                    return standard_content
                else:
                    logger.info(f"swft指定文档为空，使用默认的swft本地文档")
                    return get_swft_docs_content()

            except Exception as e:
                logger.warning(f"读取swft指定文档失败: {str(e)}，使用默认的swft本地文档")
                return get_swft_docs_content()
        else:
            # 对于其他后端，使用父类的标准方法
            return super().load_doc(doc_path)

    async def _select_optimal_examples(self, task_info) -> str:
        """
        智能选择最优的示例代码，避免prompt过长

        策略：
        1. 有Designer手写优化建议：复用Designer的RAG结果，提取impl_code
        2. 无Designer手写优化建议：根据rag参数决定是否使用RAG检索

        Returns:
            str: 选择后的示例代码
        """
        rag_enabled = self.config.get("rag", False)
        if rag_enabled:
            handwrite_suggestions = task_info.get("handwrite_suggestions", [])
            if handwrite_suggestions:
                logger.info(f"[Coder] Using Designer results (handwrite_suggestions found)")
                return self._reuse_designer_rag_results(handwrite_suggestions)
            else:
                logger.info(f"[Coder] RAG enabled, using _independent_rag_for_impl_code()")
                return await self._independent_rag_for_impl_code()
        else:
            # rag=False时，直接使用本地示例
            logger.info(f"[Coder] RAG disabled (rag=False), using local examples")
            return self._load_user_examples()

    def _reuse_designer_rag_results(self, handwrite_suggestions: list) -> str:
        """复用Designer阶段的RAG结果，提取impl_code"""
        all_code = []
        
        for suggestion in handwrite_suggestions:
            name = suggestion.get("name", "")
            impl_code = suggestion.get("impl_code", "")
            
            if impl_code:
                all_code.append(f"# Reference Implementation: {name}\n{impl_code}\n")
        
        if all_code:
            logger.info(f"Successfully loaded {len(all_code)} reference implementations")
        
        return "\n".join(all_code)

    async def _independent_rag_for_impl_code(self) -> str:
        """优先尝试RAG检索impl_code，失败后自动降级到本地示例
        
        使用缓存机制避免相同任务参数的重复检索
        """
        # 生成缓存key，基于任务参数
        cache_key = get_md5_hash(
            framework_code=self.task_desc,
            backend=self.backend,
            arch=self.arch,
            dsl=self.dsl,
            framework=self.framework,
            sample_num=self.sample_num
        )
        
        # 检查缓存
        if cache_key in self._rag_cache:
            logger.info(f"Using cached RAG retrieval result for task: {self.op_name}")
            return self._rag_cache[cache_key]
        
        # 尝试创建 CoderDatabase，如果失败则让异常向上传播
        # VectorStore 初始化时会检查依赖，如果缺少会抛出明确的错误信息
        database = CoderDatabase(config=self.config)
        
        try:
            await database.auto_update(
                dsl=self.dsl,
                framework=self.framework,
                backend=self.backend,
                arch=self.arch,
                ref_type="impl",
                update_mode="skip"
            )
            
            selected_pairs = await database.samples(
                output_content=["name", "impl_code"],
                framework_code=self.task_desc,
                backend=self.backend,
                arch=self.arch,
                dsl=self.dsl,
                framework=self.framework,
                sample_num=self.sample_num
            )
            
            # 如果RAG检索没有结果，降级到本地示例
            if not selected_pairs:
                logger.warning("RAG retrieval found no relevant implementations, using local examples")
                result = self._load_user_examples()
                # 缓存结果（即使是降级到本地示例，也缓存，避免重复判断）
                self._rag_cache[cache_key] = result
                return result
            
            all_code = []
            for pair in selected_pairs:
                name = pair.get("name", "")
                impl_code = pair.get("impl_code", "")
                
                if impl_code:
                    all_code.append(f"# Reference Implementation: {name}\n{impl_code}\n")
            
            result = "\n".join(all_code)
            logger.info(f"RAG retrieved {len(all_code)} reference implementations")
            
            # 缓存检索结果
            self._rag_cache[cache_key] = result
            return result
        except Exception as e:
            # RAG检索失败，降级到本地示例
            logger.warning(f"RAG retrieval failed: {e}, using local examples")
            result = self._load_user_examples()
            # 缓存降级结果，避免重复尝试
            self._rag_cache[cache_key] = result
            return result


    @staticmethod
    def _extract_applied_strategy_ids(sketch: str) -> List[str]:
        """从草图中的 Applied Meta-Strategies 块提取策略 ID 列表。"""
        if not sketch:
            return []

        match = re.search(r"Applied\s+Meta-Strategies:\s*\[([^\]]*)\]", sketch, flags=re.IGNORECASE | re.MULTILINE)
        if not match:
            return []

        raw_ids = match.group(1)
        strategy_ids: List[str] = []
        for item in raw_ids.split(","):
            pid = item.strip().strip("'\"")
            if pid:
                strategy_ids.append(pid)
        return strategy_ids

    def _build_strategy_glossary(self, sketch: str) -> str:
        """构建包含实现层约束的策略词典，供 Coder 保持策略一致性。"""
        strategy_ids = self._extract_applied_strategy_ids(sketch)
        if not strategy_ids:
            return ""

        from akg_agents.core.meta_prompt.manager import MetaPromptManager

        # 关键：必须使用当前任务的 arch/dsl 初始化，避免默认配置造成策略失真。
        manager = MetaPromptManager(arch=self.arch, dsl=self.dsl)

        lines: List[str] = [
            "## 术语解释 (Strategy Glossary)",
            f"**目标硬件**: {self.arch or 'default'}",
            f"**DSL**: {self.dsl}",
            f"**实现模式**: {manager.realization_mode}",
            "",
        ]

        unresolved_ids: List[str] = []
        for strategy_id in strategy_ids:
            prompt = manager.prompts.get(strategy_id)
            if not prompt:
                unresolved_ids.append(strategy_id)
                continue

            lines.append(f"- **{strategy_id}** ({prompt.category}) | implementation_type: {prompt.implementation_type}")
            lines.append(f"  - **核心意图**: {prompt.architectural_intent}")
            lines.append("  - **实现逻辑**:")
            lines.append(manager.render_prompt_logic(prompt))
            lines.append("")

        if unresolved_ids:
            lines.append(f"- **未解析策略ID**: {', '.join(unresolved_ids)}")

        return "\n".join(lines).strip()

    async def run(self, task_info: dict) -> Tuple[str, str, str]:
        """执行代码生成

        Args:
            task_info: 任务信息字典，包含当前所有代码和状态

        Returns:
            Tuple[str, str, str]: 生成的代码、提示信息和推理过程
        """
        try:
            # 从task_info中获取代码信息
            sketch = task_info.get('designer_code', '')

            # 从task_info中获取conductor的建议
            conductor_suggestion = task_info.get('conductor_suggestion', '')

            # 获取api文档
            api_docs_suitable = await self._generate_api_docs(sketch, conductor_suggestion, task_info)

            # 智能选择最优的示例代码
            dsl_examples = await self._select_optimal_examples(task_info)

            # 策略锚点解析与加强：从 sketch 中提取策略并生成实现层词典
            strategy_glossary = ""
            if "Applied Meta-Strategies:" in sketch:
                try:
                    strategy_glossary = self._build_strategy_glossary(sketch)
                    if strategy_glossary:
                        logger.debug("Strategy glossary generated with current arch/dsl context.")
                except Exception as e:
                    logger.warning(f"Failed to generate strategy glossary: {e}")
            print("strategy_glossary:", strategy_glossary)

            # ============ FIX: 强化 Debug 阶段的策略保持 ============
            # 如果处于 Debug 模式（有错误日志），且存在优化策略，强制提醒 Coder 保持策略
            if strategy_glossary and (task_info.get('verifier_error') or conductor_suggestion):
                glossary_reminder = "\n\n[CRITICAL OPTIMIZATION CONSTRAINT]\nYou are essentially debugging, BUT you must STRICTLY PRESERVE the architectural strategies defined in the 'Strategy Glossary' above (e.g., Split-K, specific tiling).\nDO NOT simplify the code to a naive implementation just to fix the bug. Fix the bug WITHIN the constraints of the high-performance strategy."
                if conductor_suggestion:
                   conductor_suggestion += glossary_reminder
                else:
                   conductor_suggestion = glossary_reminder
            # ============ Hint模式：参数范围已在sketch的"设计适用范围"注释中 ============
            enable_hint_mode = self.config.get("enable_hint_mode", False)
            has_space_config = "space_config_code" in task_info and task_info.get("space_config_code")
            has_param_space = enable_hint_mode and has_space_config
                      
            # 基于base_doc构建输入，只更新变化的部分
            input_data = {
                **self.base_doc,
                "sketch": sketch,  # sketch中已包含"设计适用范围"注释（含hint信息）
                "llm_suggestions": conductor_suggestion,  # Conductor建议
                "coder_code": task_info.get('coder_code', ''),
                "error_log": task_info.get('verifier_error', ''),
                "code_check_errors": task_info.get('code_check_errors', ''),  # CodeChecker静态检查错误
                "api_docs_suitable": api_docs_suitable,
                "dsl_examples": dsl_examples,
                "strategy_glossary": strategy_glossary,  # 注入策略锚点词典
                "enable_llm_range_inference": self.config.get("enable_llm_range_inference", False),  # LLM推理模式
                "enable_hint_mode": enable_hint_mode,  # Hint模式
                "has_param_space": has_param_space,  # 是否有参数空间
                "user_requirements": task_info.get('user_requirements', ''),  # 用户额外需求（来自 ReAct）
            }

            # 执行LLM生成前更新context，确保正确性
            self.codegen_step_count += 1
            to_update_codegen_details = {
                "agent_name": "coder",
                "hash": task_info.get("task_id", "Coder") + "@" + str(self.codegen_step_count),
                "task_id": task_info.get("task_id", ""),
                "step": self.codegen_step_count,
                "workflow_name": task_info.get("workflow_name", ""),
            }
            self.context.update(to_update_codegen_details)

            raw_output, prompt, reasoning = await self.run_llm(
                self.coder_prompt, input_data, self.model_config.get("coder") or "standard"
            )
            # 去 JSON 化：从 LLM 输出中提取纯代码
            generated_code = self._extract_code(raw_output, self.dsl)
            return generated_code, prompt, reasoning
        except Exception as e:
            logger.error(f"Exception in coder.run: {type(e).__name__}: {e}")
            raise
