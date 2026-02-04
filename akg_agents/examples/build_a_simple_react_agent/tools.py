"""
简单工具集合 - 用于演示 ReActAgent 的工具调用能力
"""

import math
import datetime
from typing import Dict, Any


def calculate(expression: str) -> Dict[str, Any]:
    """
    计算数学表达式
    
    Args:
        expression: 数学表达式，如 "2 + 3 * 4"
    
    Returns:
        计算结果
    """
    try:
        # 安全的数学计算（只允许基本运算）
        allowed_names = {
            "abs": abs, "round": round,
            "min": min, "max": max,
            "sum": sum, "pow": pow,
            "sqrt": math.sqrt,
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "log": math.log, "log10": math.log10,
            "pi": math.pi, "e": math.e
        }
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return {
            "status": "success",
            "output": f"计算结果: {expression} = {result}",
            "result": result
        }
    except Exception as e:
        return {
            "status": "error",
            "output": "",
            "error_information": f"计算失败: {str(e)}"
        }


def get_current_time() -> Dict[str, Any]:
    """
    获取当前时间
    
    Returns:
        当前日期和时间
    """
    now = datetime.datetime.now()
    return {
        "status": "success",
        "output": f"当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')} (星期{['一','二','三','四','五','六','日'][now.weekday()]})",
        "datetime": now.isoformat()
    }


def search_knowledge(query: str) -> Dict[str, Any]:
    """
    模拟知识库搜索
    
    Args:
        query: 搜索关键词
    
    Returns:
        搜索结果
    """
    # 模拟的知识库
    knowledge_base = {
        "python": "Python 是一种高级编程语言，以简洁易读著称。常用于数据科学、机器学习、Web 开发等领域。",
        "机器学习": "机器学习是人工智能的一个分支，通过算法使计算机能够从数据中学习并做出预测或决策。",
        "深度学习": "深度学习是机器学习的子领域，使用多层神经网络来学习数据的层次表示。",
        "react": "ReAct（Reasoning + Acting）是一种 Agent 范式，结合推理和行动，让 LLM 能够使用工具解决问题。",
        "transformer": "Transformer 是一种基于注意力机制的神经网络架构，广泛用于 NLP 和 CV 领域。",
    }
    
    # 简单的关键词匹配
    results = []
    query_lower = query.lower()
    for key, value in knowledge_base.items():
        if query_lower in key.lower() or query_lower in value.lower():
            results.append(f"【{key}】: {value}")
    
    if results:
        return {
            "status": "success",
            "output": f"找到 {len(results)} 条相关结果:\n" + "\n".join(results),
            "count": len(results)
        }
    else:
        return {
            "status": "success",
            "output": f"未找到与 '{query}' 相关的信息。",
            "count": 0
        }


def weather(city: str) -> Dict[str, Any]:
    """
    获取天气信息（模拟）
    
    Args:
        city: 城市名称
    
    Returns:
        天气信息
    """
    import random
    
    # 模拟天气数据
    weather_types = ["晴", "多云", "阴", "小雨", "大雨", "雪"]
    temp = random.randint(-5, 35)
    humidity = random.randint(30, 90)
    weather_type = random.choice(weather_types)
    
    return {
        "status": "success",
        "output": f"{city}的天气: {weather_type}，温度 {temp}°C，湿度 {humidity}%",
        "city": city,
        "weather": weather_type,
        "temperature": temp,
        "humidity": humidity
    }


# 工具定义（用于 LLM 理解如何调用）
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算数学表达式。支持基本运算(+,-,*,/,**)和数学函数(sqrt,sin,cos,log等)。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，如 '2 + 3 * 4' 或 'sqrt(16)'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前的日期和时间。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "搜索知识库获取信息。可以查询技术概念、编程语言等知识。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "weather",
            "description": "获取指定城市的天气信息。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如 '北京'、'上海'"
                    }
                },
                "required": ["city"]
            }
        }
    }
]


# 工具执行函数映射
TOOL_FUNCTIONS = {
    "calculate": calculate,
    "get_current_time": get_current_time,
    "search_knowledge": search_knowledge,
    "weather": weather
}
