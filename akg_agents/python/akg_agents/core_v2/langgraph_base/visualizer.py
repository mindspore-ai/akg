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

"""工作流可视化工具

提供 LangGraph 工作流的可视化功能，支持 Mermaid 和 PNG 格式。
"""

import logging

logger = logging.getLogger(__name__)


class WorkflowVisualizer:
    """工作流可视化工具
    
    提供静态方法用于生成工作流的可视化图表。
    """
    
    @staticmethod
    def generate_mermaid(app) -> str:
        """生成 Mermaid 格式流程图
        
        Args:
            app: 编译后的 LangGraph 应用
            
        Returns:
            Mermaid 图表字符串
        """
        try:
            return app.get_graph().draw_mermaid()
        except Exception as e:
            logger.error(f"Failed to generate Mermaid diagram: {e}")
            return "# Error generating diagram"
    
    @staticmethod
    def save_png(app, output_path: str):
        """保存流程图为 PNG 文件
        
        Args:
            app: 编译后的 LangGraph 应用
            output_path: PNG 文件输出路径
        """
        try:
            png_data = app.get_graph().draw_mermaid_png()
            with open(output_path, "wb") as f:
                f.write(png_data)
            logger.info(f"Workflow visualization saved to {output_path}")
        except Exception as e:
            logger.warning(f"Failed to save PNG visualization: {e}")
            logger.info("You may need to install: pip install grandalf")
    
    @staticmethod
    def print_ascii(app):
        """打印 ASCII 流程图到控制台
        
        Args:
            app: 编译后的 LangGraph 应用
        """
        try:
            print(app.get_graph().draw_ascii())
        except Exception as e:
            logger.warning(f"ASCII visualization not available: {e}")

