"""
通用 LLM 驱动的 Skill 选择器

两阶段筛选：
1. 粗筛：基于 metadata 标签快速过滤
2. 精筛：使用 LLM 根据任务和历史智能选择

架构设计：
- SelectionContext: 通用选择上下文（领域无关）
- SkillSelector: 通用选择器，支持自定义过滤器（可插拔）

扩展方式：
- 继承 SelectionContext 添加领域特定字段
- 实现自定义过滤器函数
- 继承 SkillSelector 实现领域特定选择器

示例：
    # 文档生成场景
    context = SelectionContext(
        task_type="document_generation",
        custom_fields={"doc_type": "api", "language": "python"}
    )
    
    # 自定义过滤器
    def doc_filter(skill, context):
        doc_type = context.custom_fields.get("doc_type")
        return skill.metadata.get("doc_types", "").split(",") if doc_type else True
    
    selector = SkillSelector(custom_filters=[doc_filter])
    selected = selector.select(skills, context, llm_func)
"""

from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field
import logging

from .metadata import SkillMetadata, SkillLevel

logger = logging.getLogger(__name__)


@dataclass
class SelectionContext:
    """通用 Skill 选择上下文（领域无关）
    
    适用于任何领域的 Skill 选择，通过 custom_fields 扩展领域特定信息。
    
    字段说明：
        task_type: 任务类型（如："document_generation", "test_generation"）
        framework: 使用的框架（如："pytest", "sphinx"）
        optimization_goal: 优化目标（如："speed", "quality"）
        custom_fields: 自定义字段字典，用于传递领域特定信息
    
    示例：
        # 文档生成
        context = SelectionContext(
            task_type="document_generation",
            custom_fields={
                "doc_type": "api",
                "language": "python",
                "format": "markdown"
            }
        )
        
        # 测试生成
        context = SelectionContext(
            task_type="test_generation",
            custom_fields={
                "test_type": "unit",
                "coverage_target": 80,
                "framework": "pytest"
            }
        )
    """
    # 通用任务字段
    task_type: Optional[str] = None          # 任务类型
    framework: Optional[str] = None          # 框架
    optimization_goal: Optional[str] = None  # 优化目标
    
    # 自定义扩展字段（用于任意领域）
    custom_fields: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            "task_type": self.task_type,
            "framework": self.framework,
            "optimization_goal": self.optimization_goal
        }
        result.update(self.custom_fields)
        return {k: v for k, v in result.items() if v is not None}


class SkillSelector:
    """
    通用 Skill 选择器
    
    提供两阶段筛选：
    1. 粗筛（coarse_filter）：基于 metadata 快速过滤
    2. 精筛（fine_select）：使用 LLM 智能选择
    
    支持自定义过滤器，可适配不同领域：
    - 算子生成：使用 OperatorFilterMixin
    - 文档生成：自定义 doc_type 过滤器
    - 测试生成：自定义 test_type 过滤器
    """
    
    def __init__(self, custom_filters: Optional[List[Callable]] = None):
        """初始化选择器
        
        Args:
            custom_filters: 自定义过滤器列表，每个过滤器签名为：
                filter_func(skill: SkillMetadata, context: SelectionContext) -> bool
        """
        self.custom_filters = custom_filters or []
    
    def coarse_filter(
        self,
        skills: List[SkillMetadata],
        context: SelectionContext,
        level: Optional[SkillLevel] = None
    ) -> List[SkillMetadata]:
        """
        阶段 1：粗筛（基于 metadata 标签）
        
        快速过滤掉明显不相关的 Skill，减少 LLM 输入 Token
        
        Args:
            skills: 所有候选 Skill
            context: 选择上下文
            level: 指定只从某个层级筛选（可选）
        
        Returns:
            过滤后的候选 Skill 列表
        """
        candidates = []
        
        for skill in skills:
            # 通用过滤：层级匹配
            if level and skill.level != level:
                continue
            
            # 应用自定义过滤器
            if self.custom_filters:
                passed = True
                for filter_func in self.custom_filters:
                    if not filter_func(skill, context):
                        passed = False
                        break
                if not passed:
                    continue
            
            candidates.append(skill)
        
        logger.info(f"粗筛: {len(skills)} -> {len(candidates)} skills")
        return candidates
    
    def build_llm_prompt(
        self,
        candidates: List[SkillMetadata],
        context: SelectionContext,
        history: Optional[List[Dict]] = None
    ) -> str:
        """
        构建 LLM 精筛的 prompt
        
        Args:
            candidates: 粗筛后的候选 Skill
            context: 选择上下文
            history: 执行历史（可选）
        
        Returns:
            LLM prompt 字符串
        """
        # 辅助函数：转义大括号，避免被 PromptTemplate 误解析
        def escape_braces(text: str) -> str:
            return text.replace('{', '{{').replace('}', '}}')
        
        # 格式化上下文
        context_str = "\n".join([
            f"- {k}: {escape_braces(str(v))}" for k, v in context.to_dict().items()
        ])
        
        # 格式化候选 Skill
        skills_str = ""
        for i, skill in enumerate(candidates, 1):
            level_str = f"L{skill.level.value}" if skill.level else "?"
            skills_str += f"\n{i}. **{escape_braces(skill.name)}** ({level_str})\n"
            skills_str += f"   描述: {escape_braces(skill.description)}\n"
            if skill.metadata:
                skills_str += f"   标签: {escape_braces(str(skill.metadata))}\n"
        
        # 格式化历史（如果有）
        history_str = ""
        if history:
            history_str = "\n## 执行历史\n"
            for i, h in enumerate(history, 1):
                history_str += f"{i}. {escape_braces(h.get('action', '未知操作'))}\n"
                history_str += f"   结果: {escape_braces(str(h.get('result', '未知')))}\n"
                if 'error' in h:
                    history_str += f"   错误: {escape_braces(str(h['error']))}\n"
        
        # 构建完整 prompt
        # 注意：JSON 示例中的大括号需要转义为 {{ 和 }}，避免被 PromptTemplate 当作格式化占位符
        prompt = (
            "你是一个 Skill 选择专家，负责根据任务需求选择最相关的 Skill。\n\n"
            "## 任务上下文\n"
            f"{context_str}\n"
            f"{history_str}\n"
            "## 候选 Skill（已粗筛）\n"
            f"{skills_str}\n\n"
            "## 任务\n"
            "请分析任务需求和候选 Skill，选择最相关的 1-3 个 Skill。\n\n"
            "## 输出格式\n"
            "请返回 JSON 格式：\n"
            '```json\n'
            '{{\n'
            '  "selected": ["skill-name-1", "skill-name-2"],\n'
            '  "reason": "选择这些 Skill 的理由"\n'
            '}}\n'
            '```\n\n'
            "注意：\n"
            "- 只返回 JSON，不要其他内容\n"
            "- selected 数组包含选中的 Skill 名称\n"
            "- reason 简要说明选择理由\n"
        )
        
        return prompt
    
    def parse_llm_response(
        self,
        llm_response: str,
        candidates: List[SkillMetadata]
    ) -> List[SkillMetadata]:
        """
        解析 LLM 响应，提取选中的 Skill（参考 common_utils.py 的稳健解析）
        
        Args:
            llm_response: LLM 的响应文本
            candidates: 候选 Skill 列表
        
        Returns:
            选中的 Skill 列表
        """
        import json
        import re
        
        # 策略 1: 直接解析
        try:
            result = json.loads(llm_response)
            return self._extract_skills_from_result(result, candidates)
        except json.JSONDecodeError:
            logger.debug("直接解析失败，尝试提取 JSON 块")
        
        # 策略 2: 提取 ```json 代码块
        json_blocks = re.findall(r'```json\s*(.*?)\s*```', llm_response, re.DOTALL)
        for block in reversed(json_blocks):
            try:
                result = json.loads(block.strip())
                return self._extract_skills_from_result(result, candidates)
            except json.JSONDecodeError:
                continue
        
        # 策略 3: 提取通用代码块
        code_blocks = re.findall(r'```\s*(.*?)\s*```', llm_response, re.DOTALL)
        for block in reversed(code_blocks):
            block = block.strip()
            if block.startswith('{') and block.endswith('}'):
                try:
                    result = json.loads(block)
                    return self._extract_skills_from_result(result, candidates)
                except json.JSONDecodeError:
                    continue
        
        # 策略 4: 提取花括号包围的内容
        json_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
        matches = re.findall(json_pattern, llm_response, re.DOTALL)
        for match in reversed(matches):
            try:
                result = json.loads(match)
                if isinstance(result, dict) and 'selected' in result:
                    return self._extract_skills_from_result(result, candidates)
            except json.JSONDecodeError:
                continue
        
        # 所有策略都失败
        logger.error(f"无法从 LLM 响应中提取有效的 JSON: {llm_response[:200]}...")
        return []
    
    def _extract_skills_from_result(
        self,
        result: dict,
        candidates: List[SkillMetadata]
    ) -> List[SkillMetadata]:
        """从解析结果中提取 Skill"""
        selected_names = result.get("selected", [])
        reason = result.get("reason", "未提供理由")
        
        logger.info(f"LLM 选择理由: {reason}")
        
        # 根据名称查找 Skill
        selected_skills = []
        skill_dict = {s.name: s for s in candidates}
        
        for name in selected_names:
            if name in skill_dict:
                selected_skills.append(skill_dict[name])
                logger.info(f"选中 Skill: {name}")
            else:
                logger.warning(f"Skill 不在候选列表中: {name}")
        
        return selected_skills
    
    def select(
        self,
        all_skills: List[SkillMetadata],
        context: SelectionContext,
        llm_generate_func: Optional[Any] = None,
        history: Optional[List[Dict]] = None,
        level: Optional[SkillLevel] = None
    ) -> List[SkillMetadata]:
        """
        完整的两阶段选择流程
        
        Args:
            all_skills: 所有可用的 Skill
            context: 选择上下文
            llm_generate_func: LLM 生成函数（接受 prompt 返回文本）
            history: 执行历史
            level: 指定只从某个层级筛选（可选）
        
        Returns:
            最终选中的 Skill 列表
        """
        # 阶段 1：粗筛
        candidates = self.coarse_filter(all_skills, context, level)
        
        if not candidates:
            logger.warning("粗筛后无候选 Skill")
            return []
        
        # 如果没有提供 LLM 函数，直接返回粗筛结果
        if llm_generate_func is None:
            logger.info("未提供 LLM 函数，返回粗筛结果")
            return candidates
        
        # 阶段 2：LLM 精筛
        prompt = self.build_llm_prompt(candidates, context, history)
        
        try:
            llm_response = llm_generate_func(prompt)
            selected = self.parse_llm_response(llm_response, candidates)
            
            if not selected:
                logger.warning("LLM 未选中任何 Skill，返回粗筛结果")
                return candidates
            
            return selected
        
        except Exception as e:
            logger.error(f"LLM 精筛失败: {e}，返回粗筛结果")
            return candidates
    
    def get_children_skills(
        self,
        parent_skill: SkillMetadata,
        all_skills: List[SkillMetadata]
    ) -> List[SkillMetadata]:
        """
        获取父 Skill 的所有子 Skill
        
        Args:
            parent_skill: 父 Skill
            all_skills: 所有可用的 Skill
        
        Returns:
            子 Skill 列表
        """
        if not parent_skill.structure:
            return []
        
        child_names = parent_skill.structure.child_skills
        skill_dict = {s.name: s for s in all_skills}
        
        children = []
        for name in child_names:
            if name in skill_dict:
                children.append(skill_dict[name])
            else:
                logger.warning(f"子 Skill 不存在: {name} (父: {parent_skill.name})")
        
        return children


def build_prompt_with_skills(
    selected_skills: List[SkillMetadata],
    task_description: str,
    include_full_content: bool = True
) -> str:
    """
    使用选中的 Skill 构建最终的生成 prompt
    
    Args:
        selected_skills: 选中的 Skill 列表
        task_description: 任务描述
        include_full_content: 是否包含完整的 Skill 内容
    
    Returns:
        构建好的 prompt
    """
    skills_content = ""
    
    for skill in selected_skills:
        skills_content += f"\n# {skill.name}\n\n"
        if include_full_content:
            skills_content += skill.content
        else:
            skills_content += skill.description
        skills_content += "\n\n" + "="*60 + "\n"
    
    prompt = f"""# 相关知识和策略

{skills_content}

# 任务

{task_description}

请根据上述知识和策略完成任务。
"""
    
    return prompt
