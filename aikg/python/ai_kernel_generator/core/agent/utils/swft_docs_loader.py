#!/usr/bin/env python3
# coding: utf-8
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
SWFT后端文档加载工具

提供特殊的swft后端文档查找功能，通过import swft找到其__path__，
再通过路径关系找到实际的docs目录和api目录。
"""

import logging
from typing import List
from pathlib import Path

logger = logging.getLogger(__name__)


def find_swft_docs_and_api_files() -> List[str]:
    """
    特殊的swft后端文档查找函数
    通过import swft找到其__path__，再通过路径关系找到实际的docs目录和api目录
    
    Returns:
        List[str]: 包含docs、api目录全部文件内容的列表
    """
    try:
        # 尝试导入swft模块
        import swft
        
        # 获取swft模块的路径
        if hasattr(swft, '__path__') and swft.__path__:
            swft_path = Path(swft.__path__[0])
        else:
            # 如果__path__不存在，尝试通过__file__获取
            if hasattr(swft, '__file__'):
                swft_path = Path(swft.__file__).parent
            else:
                raise ValueError("无法获取swft模块路径")
        
        logger.info(f"找到swft模块路径: {swft_path}")
        
        # 通过路径关系找到docs目录（swft模块的上级目录的docs）
        docs_dir = swft_path.parent.parent / "docs"
        api_dir = swft_path / "api"
        
        all_files_content = []
        
        # 处理docs目录
        if docs_dir.exists() and docs_dir.is_dir():
            logger.info(f"找到swft docs目录: {docs_dir}")
            # 支持的文件扩展名
            supported_extensions = ['*.py', '*.md', '*.txt']
            
            for extension in supported_extensions:
                for file_path in docs_dir.rglob(extension[1:]):  # 使用rglob递归搜索
                    if file_path.is_file():
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                content = f.read().strip()
                                if content:
                                    # 添加文件标识
                                    relative_path = file_path.relative_to(docs_dir)
                                    all_files_content.append(f"# Docs File: {relative_path}\n{content}\n")
                        except Exception as e:
                            logger.warning(f"读取docs文件 {file_path} 时发生错误: {str(e)}")
                            continue
        else:
            logger.warning(f"swft docs目录不存在: {docs_dir}")
        
        # 处理api目录
        if api_dir.exists() and api_dir.is_dir():
            logger.info(f"找到swft api目录: {api_dir}")
            # 支持的文件扩展名
            supported_extensions = ['*.py', '*.md', '*.txt']
            
            for extension in supported_extensions:
                for file_path in api_dir.rglob(extension[1:]):  # 使用rglob递归搜索
                    if file_path.is_file():
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                content = f.read().strip()
                                if content:
                                    # 添加文件标识
                                    relative_path = file_path.relative_to(api_dir)
                                    all_files_content.append(f"# API File: {relative_path}\n{content}\n")
                        except Exception as e:
                            logger.warning(f"读取api文件 {file_path} 时发生错误: {str(e)}")
                            continue
        else:
            logger.warning(f"swft api目录不存在: {api_dir}")
        
        if not all_files_content:
            logger.warning("未找到任何swft文档或API文件")
            return []
        
        logger.info(f"成功加载了 {len(all_files_content)} 个swft文档/API文件")
        return all_files_content
        
    except ImportError as e:
        logger.error(f"无法导入swft模块: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"查找swft文档时发生错误: {str(e)}")
        return []


def get_swft_docs_content() -> str:
    """
    获取swft文档内容的便捷函数
    
    Returns:
        str: 所有swft文档和API文件的内容拼接字符串
    """
    files_content = find_swft_docs_and_api_files()
    return "\n".join(files_content) if files_content else "" 