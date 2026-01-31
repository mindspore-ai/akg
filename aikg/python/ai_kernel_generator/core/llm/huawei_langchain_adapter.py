"""
华为AKSK签名 + LangChain适配器
让华为API能够作为LangChain的LLM使用
"""

import os
import hashlib
from datetime import datetime
import requests
import json
from typing import Any, Dict, Iterator, List, Optional, Union
import asyncio

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.callbacks import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.runnables import Runnable, RunnableBinding
from pydantic import Field


def sign_request(app_id, app_key, service_uri, http_method, request_date, args):
    """
    华为AKSK签名函数
    
    Args:
        app_id: Access Key
        app_key: Secret Key
        service_uri: 服务URI路径
        http_method: HTTP方法
        request_date: 请求日期时间
        args: 查询参数
    
    Returns:
        签名字符串
    """
    # 1. 参数排序并拼接
    if args:
        sorted_params = sorted(args.items(), key=lambda x: x[0])
        params_str = "|".join([f"{k}={v}" for k, v in sorted_params])
    else:
        params_str = ""

    # 2. 构造签名字符串（SHA256 模式）
    sign_string = f"{service_uri}|{http_method}|{request_date}|{app_id}|{app_key}"
    
    # 3. 使用 SHA256 计算签名
    signature = hashlib.sha256(sign_string.encode('utf-8')).hexdigest()
    return signature


class HuaweiChatModel(BaseChatModel):
    """
    华为AKSK认证的LangChain Chat Model
    
    可以直接用于：
    1. LangChain chain: prompt | model
    2. LangChain agent: create_agent(model=...)
    3. 所有LangChain兼容的场景
    """
    
    # 配置参数
    access_key: str = Field(description="华为Access Key")
    secret_key: str = Field(description="华为Secret Key")
    apigw_host: str = Field(description="API网关主机地址")
    model_name: str = Field(default="Qwen3-30B-A3B-Instruct-2507", description="模型名称")
    api_path: str = Field(default="/maas/llm/v1/chat/completions", description="API路径")
    temperature: float = Field(default=0.7, description="温度参数")
    max_tokens: int = Field(default=512, description="最大token数")
    timeout: int = Field(default=600, description="请求超时时间（秒）")
    
    class Config:
        arbitrary_types_allowed = True
    
    @property
    def _llm_type(self) -> str:
        """返回LLM类型标识"""
        return "huawei_chat"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """返回识别参数（用于缓存等）"""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
    
    def _build_headers(self, method: str = "POST", query_params: Dict = None) -> Dict[str, str]:
        """构建请求头（包含华为签名）"""
        # 生成当前时间（GMT格式）
        date = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        
        # 生成签名
        signature = sign_request(
            app_id=self.access_key,
            app_key=self.secret_key,
            service_uri=self.api_path,
            http_method=method,
            request_date=date,
            args=query_params or {}
        )
        
        # 构造请求头
        headers = {
            "X-HW-ACCESS-KEY": self.access_key,
            "X-HW-DATE": date,
            "X-HW-SIGN": signature,
            "Content-Type": "application/json"
        }
        
        return headers
    
    def _convert_messages_to_api_format(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """将LangChain消息格式转换为API格式"""
        api_messages = []
        
        for msg in messages:
            if isinstance(msg, SystemMessage):
                api_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                api_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                msg_dict = {"role": "assistant", "content": msg.content or ""}
                # 处理tool_calls
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    msg_dict["tool_calls"] = msg.tool_calls
                api_messages.append(msg_dict)
            elif isinstance(msg, ToolMessage):
                # 将ToolMessage转换为tool响应格式
                api_messages.append({
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": msg.tool_call_id
                })
            else:
                # 其他类型消息转为用户消息
                api_messages.append({"role": "user", "content": str(msg.content)})
        
        return api_messages
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """同步生成（非流式）"""
        # 构建请求
        api_messages = self._convert_messages_to_api_format(messages)
        
        request_body = {
            "model": self.model_name,
            "messages": api_messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
        }
        
        if stop:
            request_body["stop"] = stop
        
        # 添加tools参数（如果有）
        if "tools" in kwargs and kwargs["tools"]:
            request_body["tools"] = kwargs["tools"]
        
        # 添加tool_choice参数（如果有）
        if "tool_choice" in kwargs and kwargs["tool_choice"]:
            request_body["tool_choice"] = kwargs["tool_choice"]
        
        # 构造请求头
        headers = self._build_headers(method="POST")
        
        # 构造URL
        url = f"https://{self.apigw_host}{self.api_path}"
        
        # 发送请求
        response = requests.post(
            url,
            headers=headers,
            json=request_body,
            timeout=self.timeout
        )
        
        response.encoding = 'utf-8'
        
        if response.status_code != 200:
            raise Exception(f"API请求失败: {response.status_code}, {response.text}")
        
        result = json.loads(response.text)
        
        # 解析响应
        if "choices" not in result or len(result["choices"]) == 0:
            raise Exception(f"API返回格式错误: {result}")
        
        choice = result["choices"][0]
        message_data = choice.get("message", {})
        message_content = message_data.get("content", "") or ""
        
        # 构造LangChain响应
        additional_kwargs = {}
        tool_calls = message_data.get("tool_calls", [])
        
        if tool_calls:
            additional_kwargs["tool_calls"] = tool_calls
        
        message = AIMessage(
            content=message_content,
            additional_kwargs=additional_kwargs
        )
        
        # 如果有tool_calls，将其设置为message的属性
        if tool_calls:
            # LangChain需要的格式
            message.tool_calls = tool_calls
        
        generation = ChatGeneration(message=message)
        
        # 添加token使用信息（如果有）
        usage = result.get("usage", {})
        llm_output = {
            "model_name": self.model_name,
            "usage": usage,
        }
        
        return ChatResult(generations=[generation], llm_output=llm_output)
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """异步生成（非流式）"""
        # 简单实现：在线程池中运行同步版本
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: self._generate(messages, stop, None, **kwargs)
        )
    
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """同步流式生成"""
        # 构建请求
        api_messages = self._convert_messages_to_api_format(messages)
        
        request_body = {
            "model": self.model_name,
            "messages": api_messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "stream": True,  # 开启流式
        }
        
        if stop:
            request_body["stop"] = stop
        
        # 添加tools参数（如果有）
        if "tools" in kwargs and kwargs["tools"]:
            request_body["tools"] = kwargs["tools"]
        
        # 添加tool_choice参数（如果有）
        if "tool_choice" in kwargs and kwargs["tool_choice"]:
            request_body["tool_choice"] = kwargs["tool_choice"]
        
        # 构造请求头
        headers = self._build_headers(method="POST")
        
        # 构造URL
        url = f"https://{self.apigw_host}{self.api_path}"
        
        # 发送流式请求
        response = requests.post(
            url,
            headers=headers,
            json=request_body,
            stream=True,
            timeout=self.timeout
        )
        
        response.encoding = 'utf-8'
        
        if response.status_code != 200:
            raise Exception(f"API请求失败: {response.status_code}, {response.text}")
        
        # 逐行处理流式响应
        for line in response.iter_lines():
            if not line:
                continue
            
            line_str = line.decode('utf-8')
            
            # 跳过空行和注释
            if not line_str.strip() or line_str.startswith(':'):
                continue
            
            # 处理 SSE 格式的数据
            if line_str.startswith('data: '):
                data_str = line_str[6:]  # 去掉 "data: " 前缀
                
                # 检查是否是结束标记
                if data_str.strip() == '[DONE]':
                    break
                
                try:
                    data = json.loads(data_str)
                    
                    # 提取内容
                    if "choices" in data and len(data["choices"]) > 0:
                        delta = data["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        tool_calls = delta.get("tool_calls", [])
                        
                        # 构造additional_kwargs
                        additional_kwargs = {}
                        if tool_calls:
                            additional_kwargs["tool_calls"] = tool_calls
                        
                        if content or tool_calls:
                            # 构造LangChain chunk
                            chunk_message = AIMessageChunk(
                                content=content,
                                additional_kwargs=additional_kwargs
                            )
                            
                            # 如果有tool_calls，设置为属性
                            if tool_calls:
                                chunk_message.tool_calls = tool_calls
                            
                            chunk = ChatGenerationChunk(message=chunk_message)
                            
                            # 回调
                            if run_manager and content:
                                run_manager.on_llm_new_token(content)
                            
                            yield chunk
                
                except json.JSONDecodeError:
                    continue
    
    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ):
        """异步流式生成"""
        # 在线程池中运行同步流式版本
        loop = asyncio.get_event_loop()
        
        for chunk in self._stream(messages, stop, None, **kwargs):
            if run_manager:
                await run_manager.on_llm_new_token(chunk.message.content)
            yield chunk
    
    def bind_tools(
        self,
        tools: List[Union[Dict[str, Any], type, BaseTool]],
        **kwargs: Any,
    ) -> Runnable:
        """
        绑定工具到模型
        
        Args:
            tools: 工具列表，可以是dict、type或BaseTool
            **kwargs: 其他参数，例如tool_choice
        
        Returns:
            一个绑定了工具的Runnable
        """
        # 将工具转换为OpenAI格式（华为API兼容OpenAI格式）
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        
        # 构建bind参数
        bind_kwargs = {"tools": formatted_tools}
        
        # 处理tool_choice参数
        if "tool_choice" in kwargs:
            tool_choice = kwargs.pop("tool_choice")
            if tool_choice:
                bind_kwargs["tool_choice"] = tool_choice
        
        # 添加其他kwargs
        bind_kwargs.update(kwargs)
        
        # 调用父类的bind方法
        return super().bind(**bind_kwargs)


def create_huawei_chat_model(
    access_key: str,
    secret_key: str,
    apigw_host: str,
    model_name: str = "Qwen3-30B-A3B-Instruct-2507",
    **kwargs
) -> HuaweiChatModel:
    """
    便捷函数：创建华为Chat Model
    
    Args:
        access_key: 华为Access Key
        secret_key: 华为Secret Key
        apigw_host: API网关主机地址
        model_name: 模型名称
        **kwargs: 其他参数（temperature, max_tokens等）
    
    Returns:
        HuaweiChatModel实例
    """
    return HuaweiChatModel(
        access_key=access_key,
        secret_key=secret_key,
        apigw_host=apigw_host,
        model_name=model_name,
        **kwargs
    )


def main():
    """示例使用"""
    
    # 从环境变量读取配置
    access_key = os.getenv("AIKG_HW_ACCESS_KEY")
    secret_key = os.getenv("AIKG_HW_SECRET_KEY")
    apigw_host = os.getenv("AIKG_HW_APIGW_HOST")
    
    if not all([access_key, secret_key, apigw_host]):
        print("❌ 请设置环境变量:")
        print("   export AIKG_HW_ACCESS_KEY='your-access-key'")
        print("   export AIKG_HW_SECRET_KEY='your-secret-key'")
        print("   export AIKG_HW_APIGW_HOST='your-api-gateway-host.com'")
        return
    
    # 创建模型
    model = create_huawei_chat_model(
        access_key=access_key,
        secret_key=secret_key,
        apigw_host=apigw_host,
        model_name="Qwen3-30B-A3B-Instruct-2507",
        temperature=0.7,
        max_tokens=512
    )
    
    print("=" * 60)
    print("【示例1】直接调用模型")
    print("=" * 60)
    
    try:
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="你好，请介绍一下你自己。")
        ]
        
        result = model.invoke(messages)
        print(result.content)
    except Exception as e:
        print(f"调用失败: {e}")
    
    print("\n" + "=" * 60)
    print("【示例2】使用LangChain chain (prompt | model)")
    print("=" * 60)
    
    try:
        from langchain_core.prompts import ChatPromptTemplate
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("user", "{input}")
        ])
        
        chain = prompt | model
        
        result = chain.invoke({"input": "什么是机器学习？"})
        print(result.content)
    except Exception as e:
        print(f"调用失败: {e}")
    
    print("\n" + "=" * 60)
    print("【示例3】流式输出")
    print("=" * 60)
    
    try:
        messages = [
            HumanMessage(content="请用三句话介绍Python编程语言。")
        ]
        
        print("流式输出: ", end="", flush=True)
        for chunk in model.stream(messages):
            print(chunk.content, end="", flush=True)
        print()
    except Exception as e:
        print(f"调用失败: {e}")
    
    print("\n" + "=" * 60)
    print("【示例4】在LangChain Agent中使用")
    print("=" * 60)
    
    print("可以直接将此模型传入 create_agent():")
    print("agent = create_agent(model=model, tools=tools, ...)")


if __name__ == "__main__":
    main()
