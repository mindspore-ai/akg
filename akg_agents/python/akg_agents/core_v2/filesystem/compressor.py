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
Action Compressor - 动作历史压缩服务
"""

import json
import logging
from typing import List, Dict, Any, Optional

from akg_agents.core_v2.llm.client import LLMClient
from akg_agents.core_v2.filesystem.models import ActionRecord

logger = logging.getLogger(__name__)

HISTORY_SUMMARY_PROMPT = """
你是智能历史信息压缩助手。请基于任务需求和当前进度，智能压缩以下历史动作。

## 历史动作数据
(Note: Using XML format to wrap history actions, imitating InfiAgent's approach for better structural parsing)

<历史动作>
{history_content}
</历史动作>

## 压缩要求
1. **目标长度**: 严格控制在 {max_tokens} tokens 以内
2. **智能筛选**: 
   - 保留已完成任务相关的**关键结果**（如生成的文件路径、重要输出）
   - 丢弃无关或失败的尝试信息
3. **必须保留（最高优先级）**:
   - 所有 result.json 文件路径（如 `/path/to/nodes/node_XXX/result.json`）
   - 每个 result.json 中的可用字段列表（result_keys）
   - 每个步骤的 action_id（node_XXX）和 tool_name
   - 这些路径信息是后续数据引用的基础，**绝对不可丢弃**
4. **优先保留**:
   - 成功完成的关键步骤（如创建的文件、执行的代码、获取的数据）
   - 重要的文件路径和位置信息
   - 对后续任务有参考价值的输出
5. **可以丢弃**:
   - 重复的尝试和错误信息
   - 中间的调试过程
   - 与当前任务目标无关的探索性操作
   - 大段的代码内容（保留路径即可）
6. **格式要求**:
   - 按时间顺序总结
   - 突出关键成果和产出
   - 保持信息的连贯性
   - 所有节点路径信息必须以列表形式清晰列出

请直接输出压缩后的总结（中文）：
"""

class ActionCompressor:
    """
    动作压缩器
    
    负责将长动作历史压缩为简短的摘要，以节省 Context Window。
    """
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        
    async def compress_history(
        self, 
        history: List[ActionRecord], 
        max_tokens: int = 2000
    ) -> List[ActionRecord]:
        """
        压缩动作历史
        
        策略：
        1. 保留最近的 N 条动作（不压缩）
        2. 将较早的动作总结为一个 "summary" 类型的虚拟动作
        
        Args:
            history: 完整动作历史
            max_tokens: 目标摘要的最大 token 数
            
        Returns:
            压缩后的动作列表
        """
        if not history:
            return []
            
        # 如果历史很短，不需要压缩
        if len(history) <= 5:
            return history
            
        # 动态计算保留的最近动作数量（保留更多上下文以避免路径丢失）
        # 至少保留 3 个，最多保留一半
        keep_count = max(3, min(len(history) // 2, 5))
        recent_actions = history[-keep_count:]
        actions_to_summarize = history[:-keep_count]
        
        if not actions_to_summarize:
            return history
            
        # 转换要压缩的动作为 XML 格式
        history_xml = self._actions_to_xml(actions_to_summarize)
        
        # 调用 LLM 进行总结
        summary_text = await self._summarize_text(history_xml, max_tokens)
        
        # 创建总结动作
        summary_action = ActionRecord(
            action_id=f"summary_{history[0].action_id}_{history[-keep_count-1].action_id}",
            tool_name="history_summary",
            arguments={"original_actions": len(actions_to_summarize)},
            result={
                "summary": summary_text,
                "_is_summary": True
            },
            compressed=True
        )
        
        # 组合结果
        compressed_history = [summary_action] + recent_actions
        
        logger.info(f"Compressed {len(history)} actions to {len(compressed_history)} actions")
        return compressed_history
        
    def _actions_to_xml(self, actions: List[ActionRecord]) -> str:
        """
        将动作列表转换为 XML 格式
        
        Note: Imitating InfiAgent's approach. XML tags are robust for delimiting 
        complex action structures in LLM context.
        """
        xml_parts = []
        for action in actions:
            tool_name = action.tool_name
            arguments = action.arguments
            result = action.result
            
            action_xml = f"<action>\n  <tool_name>{tool_name}</tool_name>\n"
            
            # 参数
            for k, v in arguments.items():
                v_str = str(v).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                action_xml += f"  <tool_use:{k}>{v_str}</tool_use:{k}>\n"
            
            # 结果
            result_json = json.dumps(result, ensure_ascii=False, indent=2)
            action_xml += f"  <result>\n{result_json}\n  </result>\n</action>"
            
            xml_parts.append(action_xml)
            
        return "\n\n".join(xml_parts)
        
    async def _summarize_text(self, text: str, max_tokens: int) -> str:
        """调用 LLM 总结文本"""
        try:
            # 这里的 prompt 已经包含了 system 指令和 user input
            prompt = HISTORY_SUMMARY_PROMPT.format(
                history_content=text,
                max_tokens=max_tokens
            )
            
            messages = [
                {"role": "system", "content": "你是智能历史信息压缩助手。"},
                {"role": "user", "content": prompt}
            ]
            
            result = await self.llm_client.generate(
                messages, 
                max_tokens=max_tokens,
                temperature=0.3
            )
            
            return result.get("content", "").strip()
            
        except Exception as e:
            logger.error(f"Failed to summarize history: {e}")
            return f"[Error summarizing history: {e}]"
