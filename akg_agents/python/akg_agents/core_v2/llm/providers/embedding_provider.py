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

"""
OpenAI 兼容的 Embedding 模型提供者

支持：
- OpenAI Embeddings
- DeepSeek Embeddings
- 硅流平台
- vLLM 本地部署
- 其他 OpenAI 兼容 API
"""

import logging
from typing import List, Optional

import requests
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class OpenAICompatibleEmbeddings(Embeddings):
    """
    调用 OpenAI 兼容格式的 Embedding API
    
    支持本地部署（如 vLLM）和远程 API（如硅流平台）
    兼容 LangChain Embeddings 接口
    """
    
    def __init__(
        self,
        api_url: str,
        model_name: str,
        api_key: Optional[str] = None,
        verify_ssl: bool = False,
        timeout: int = 60
    ):
        """
        初始化 Embedding 客户端
        
        Args:
            api_url: Embedding API 的完整 URL（如 http://localhost:8001/v1/embeddings）
            model_name: 模型名称
            api_key: API 密钥（可选，远程 API 需要）
            verify_ssl: 是否验证 SSL 证书（默认 False）
            timeout: 超时时间（秒）
        """
        self.api_url = api_url
        self.model_name = model_name
        self.api_key = api_key
        self.verify_ssl = verify_ssl
        self.timeout = timeout
        
        logger.info(f"Initialized OpenAICompatibleEmbeddings: api_url={api_url}, model={model_name}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """为文档生成 Embeddings"""
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        """为查询生成 Embeddings"""
        return self._embed([text])[0]

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """调用 API 生成 Embeddings"""
        payload = {
            "model": self.model_name,
            "input": texts
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                headers=headers,
                verify=self.verify_ssl,
                timeout=self.timeout
            )
            response.raise_for_status()

            # 解析返回的 JSON
            result = response.json()

            # 检查返回格式是否正确
            if 'data' not in result:
                raise ValueError(f"API返回格式错误，缺少'data'字段: {result}")

            embeddings = []
            for item in result['data']:
                if 'embedding' not in item:
                    raise ValueError(f"API返回格式错误，缺少'embedding'字段: {item}")
                embeddings.append(item['embedding'])

            return embeddings

        except requests.exceptions.RequestException as e:
            logger.error(f"Embedding API请求失败: {e}")
            raise RuntimeError(f"Embedding API请求失败: {e}") from e
