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

"""错别字修复 Agent

负责检测和修复文档中的错别字、拼写错误和语法问题。
"""

import json
import logging
import sys
from pathlib import Path
from typing import Tuple, List, Dict, Any

# 添加 python 目录到 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))

logger = logging.getLogger(__name__)


TYPO_FIXER_PROMPT = """你是一个专业的文档校对专家，擅长发现和修复中文文档中的错别字、拼写错误和语法问题。

## 任务
请仔细检查以下文档，找出并修正其中的错别字和语法错误。

## 文档类型
{document_type}

## 文档语言
{language}

## 原始文档内容
```
{content}
```

## 要求
1. 识别所有的错别字、拼写错误和明显的语法问题
2. 提供修正后的完整文档
3. 列出所有修改的详细信息

## 输出格式
请以 JSON 格式输出，包含以下字段：
```json
{{
    "corrected_content": "修正后的完整文档内容",
    "corrections": [
        "具体的修改说明1",
        "具体的修改说明2",
        ...
    ],
    "reasoning": "你的分析和推理过程"
}}
```

请直接输出 JSON，不要添加其他说明。
"""


class TypoFixer:
    """错别字修复 Agent
    
    使用 LLM 检测和修复文档中的错别字。
    """
    
    def __init__(self, config: dict = None):
        """初始化 TypoFixer
        
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
        return agent_config.get("typo_fixer") or agent_config.get("default") or "deepseek_r1_default"
    
    async def run(
        self, 
        content: str, 
        document_type: str = "markdown",
        language: str = "zh"
    ) -> Tuple[str, List[Dict[str, Any]], str]:
        """执行错别字修复
        
        Args:
            content: 原始文档内容
            document_type: 文档类型
            language: 文档语言
            
        Returns:
            Tuple[str, List[Dict], str]: (修正后的内容, 修正记录列表, 推理过程)
        """
        from akg_agents.core.llm.model_loader import create_model
        
        # 构建 prompt
        prompt = TYPO_FIXER_PROMPT.format(
            content=content,
            document_type=document_type,
            language=language
        )
        
        logger.info(f"[TypoFixer] Starting typo detection, content length: {len(content)}")
        
        try:
            # 调用 LLM
            model = create_model(self.model_name)
            result_text = await self._call_llm(model, prompt)
            
            # 解析结果
            return self._parse_result(result_text, content)
            
        except Exception as e:
            logger.error(f"[TypoFixer] Failed: {e}")
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
                {"role": "system", "content": "你是一个专业的文档校对专家。"},
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
    ) -> Tuple[str, List[Dict[str, Any]], str]:
        """解析 LLM 返回的结果
        
        Args:
            result_text: LLM 返回的文本
            original_content: 原始内容（用于回退）
            
        Returns:
            Tuple[str, List[Dict], str]: (修正后的内容, 修正记录列表, 推理过程)
        """
        try:
            # 尝试提取 JSON
            from akg_agents.utils.common_utils import ParserFactory
            json_str = ParserFactory._extract_json_comprehensive(result_text)
            if not json_str:
                return result_text, [], "No JSON found in the output"
            result = json.loads(json_str)
            
            corrected_content = result.get("corrected_content", original_content)
            corrections = result.get("corrections", [])
            reasoning = result.get("reasoning", "")
            
            logger.info(f"[TypoFixer] Found {len(corrections)} corrections")
            
            return corrected_content, corrections, reasoning
            
        except json.JSONDecodeError as e:
            logger.warning(f"[TypoFixer] Failed to parse JSON: {e}")
            # 如果解析失败，尝试直接使用返回的文本
            return result_text, [], "JSON parsing failed, using raw output"
    
    def _extract_json(self, text: str) -> str:
        """从文本中提取 JSON 字符串"""
        # 尝试找到 JSON 代码块
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

