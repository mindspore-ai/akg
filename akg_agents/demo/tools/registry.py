"""
工具注册中心 - 参考 OpenCode 的 ToolRegistry 设计

所有工具统一注册，供 ReAct Agent 调度。
每个工具包含: id, description, parameters(JSON Schema), execute 函数。
"""
import logging
from typing import Dict, Any, Callable, List

logger = logging.getLogger(__name__)


class ToolRegistry:
    """工具注册中心"""

    _tools: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(cls, tool_id: str, description: str,
                 parameters: Dict[str, Any],
                 execute: Callable):
        """
        注册一个工具

        Args:
            tool_id: 工具唯一标识
            description: 工具描述（给 LLM 看）
            parameters: JSON Schema 格式的参数定义
            execute: 执行函数，签名 (args: dict) -> dict
        """
        cls._tools[tool_id] = {
            "id": tool_id,
            "description": description,
            "parameters": parameters,
            "execute": execute,
        }
        logger.info(f"[ToolRegistry] 注册工具: {tool_id}")

    @classmethod
    def get(cls, tool_id: str) -> Dict[str, Any]:
        tool = cls._tools.get(tool_id)
        if not tool:
            raise KeyError(f"未知工具: {tool_id}")
        return tool

    @classmethod
    def execute(cls, tool_id: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """执行工具并返回标准结果 {status, output, error}"""
        tool = cls.get(tool_id)
        try:
            result = tool["execute"](arguments)
            # 统一格式
            if isinstance(result, dict) and "status" in result:
                return result
            return {"status": "success", "output": str(result), "error": ""}
        except Exception as e:
            logger.error(f"[ToolRegistry] 执行失败 {tool_id}: {e}")
            return {"status": "error", "output": "", "error": str(e)}

    @classmethod
    def list_for_prompt(cls) -> List[Dict[str, Any]]:
        """生成供 LLM system prompt 使用的工具列表"""
        return [
            {
                "name": t["id"],
                "description": t["description"],
                "parameters": t["parameters"],
            }
            for t in cls._tools.values()
        ]

    @classmethod
    def list_ids(cls) -> List[str]:
        return list(cls._tools.keys())
