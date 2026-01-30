# Copyright 2026 Huawei Technologies Co., Ltd
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

"""文档美化 Agent

负责改善文档的格式、排版和可读性。
"""

import json
import logging
import sys
from pathlib import Path
from typing import Tuple, List

# 添加 python 目录到 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))

logger = logging.getLogger(__name__)


BEAUTIFIER_PROMPT = """你是一个专业的文档排版专家，擅长优化文档的格式和可读性。

## 任务
请对以下文档进行美化处理，改善其格式和可读性。

## 文档类型
{document_type}

## 文档语言
{language}

## 待美化的文档内容
```
{content}
```

## 美化要求
1. 优化标题层级和结构
2. 改善段落划分，确保逻辑清晰
3. 统一标点符号使用（如中文使用全角标点）
4. 优化列表格式
5. 添加适当的空行增强可读性
6. 确保 Markdown 语法正确（如果是 Markdown 文档）
7. 保持原有内容的含义不变

## 输出格式
请以 JSON 格式输出，包含以下字段：
```json
{{
    "beautified_content": "美化后的完整文档内容",
    "changes": [
        "具体的修改说明1",
        "具体的修改说明2",
        ...
    ],
    "reasoning": "你的美化思路和推理过程"
}}
```

请直接输出 JSON，不要添加其他说明。
"""


class Beautifier:
    """文档美化 Agent
    
    使用 LLM 优化文档的格式和可读性。
    """
    
    def __init__(self, config: dict = None):
        """初始化 Beautifier
        
        Args:
            config: 配置字典，可包含模型配置等
        """
        self.config = config or {}
        self.model_name = self._get_model_name()
    
    def _get_model_name(self) -> str:
        """获取模型名称"""
        import os
        # 优先使用环境变量
        if os.environ.get("AKG_AGENTS_MODEL_NAME"):
            return os.environ["AKG_AGENTS_MODEL_NAME"]
        # 其次使用配置
        agent_config = self.config.get("agent_model_config", {})
        return agent_config.get("beautifier") or agent_config.get("default") or "deepseek_r1_default"
    
    async def run(
        self, 
        content: str, 
        document_type: str = "markdown",
        language: str = "zh"
    ) -> Tuple[str, List[str], str]:
        """执行文档美化
        
        Args:
            content: 待美化的文档内容
            document_type: 文档类型
            language: 文档语言
            
        Returns:
            Tuple[str, List[str], str]: (美化后的内容, 修改说明列表, 推理过程)
        """
        from akg_agents.core.llm.model_loader import create_model
        
        # 构建 prompt
        prompt = BEAUTIFIER_PROMPT.format(
            content=content,
            document_type=document_type,
            language=language
        )
        
        logger.info(f"[Beautifier] Starting document beautification, content length: {len(content)}")
        
        try:
            # 调用 LLM
            model = create_model(self.model_name)
            result_text = await self._call_llm(model, prompt)
            
            # 解析结果
            return self._parse_result(result_text, content)
            
        except Exception as e:
            logger.error(f"[Beautifier] Failed: {e}")
            import traceback
            traceback.print_exc()
            # 失败时返回原始内容
            return content, [], f"Error: {str(e)}"
    
    async def _call_llm(self, model, prompt: str) -> str:
        """调用 LLM 获取结果
        
        参考 agent_base.py 的实现方式
        """
        # 检查模型类型
        effective_model_name = getattr(model, "model_name", self.model_name)
        
        # 检查是否是 OpenAI AsyncClient
        is_openai_async = False
        try:
            from openai import AsyncOpenAI as OpenAIAsyncClient
            is_openai_async = isinstance(model, OpenAIAsyncClient)
        except ImportError:
            pass
        
        # VLLM 或 OpenAI AsyncClient 模型
        if effective_model_name.startswith("vllm_") or is_openai_async:
            messages = [
                {"role": "system", "content": "你是一个专业的文档排版专家。"},
                {"role": "user", "content": prompt}
            ]
            create_kwargs = {
                "model": effective_model_name,
                "messages": messages,
                "temperature": getattr(model, "temperature", 0.2),
                "top_p": getattr(model, "top_p", 0.9),
            }
            extra_body = getattr(model, "extra_body", None)
            if extra_body:
                create_kwargs["extra_body"] = extra_body
            
            response = await model.chat.completions.create(**create_kwargs)
            return response.choices[0].message.content
        
        # LangChain 模型 (ChatDeepSeek, ChatOllama 等)
        elif hasattr(model, 'ainvoke'):
            response = await model.ainvoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")
    
    def _parse_result(
        self, 
        result_text: str, 
        original_content: str
    ) -> Tuple[str, List[str], str]:
        """解析 LLM 返回的结果
        
        Args:
            result_text: LLM 返回的文本
            original_content: 原始内容（用于回退）
            
        Returns:
            Tuple[str, List[str], str]: (美化后的内容, 修改说明列表, 推理过程)
        """

        try:
            # 尝试提取 JSON
            from akg_agents.utils.common_utils import ParserFactory
            json_str = ParserFactory._extract_json_comprehensive(result_text)
            if not json_str:
                return result_text, [], "No JSON found in the output"
            result = json.loads(json_str)
            
            beautified_content = result.get("beautified_content", original_content)
            changes = result.get("changes", [])
            reasoning = result.get("reasoning", "")
            
            logger.info(f"[Beautifier] Made {len(changes)} beautification changes")
            
            return beautified_content, changes, reasoning
            
        except json.JSONDecodeError as e:
            logger.warning(f"[Beautifier] Failed to parse JSON: {e}")
            # 如果解析失败，尝试直接使用返回的文本
            return result_text, [], "JSON parsing failed, using raw output"
    
    def _extract_json(self, text: str) -> str:
        """从文本中提取 JSON 字符串"""
        import re
        
        # 匹配 ```json ... ``` 或 ``` ... ```
        json_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        match = re.search(json_block_pattern, text)
        if match:
            return match.group(1).strip()
        
        # 尝试找到裸 JSON
        json_pattern = r'\{[\s\S]*\}'
        match = re.search(json_pattern, text)
        if match:
            return match.group(0)
        
        return text

