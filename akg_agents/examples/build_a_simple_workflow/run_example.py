#!/usr/bin/env python3
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

"""文档修复示例运行脚本

演示如何基于 langgraph_base 构建一个简单的多 Agent 工作流。

本示例展示了如何：
1. 继承 BaseState 定义自定义状态
2. 创建多个 Agent (TypoFixer, Beautifier)
3. 继承 BaseWorkflow 构建工作流
4. 继承 BaseLangGraphTask 创建任务执行器

使用方法:
    # 在项目根目录运行，使用内置示例文档
    python examples/build_a_simple_workflow/run_example.py
    
    # 使用自定义文档
    python examples/build_a_simple_workflow/run_example.py --input /path/to/doc.md
    
    # 保存输出结果
    python examples/build_a_simple_workflow/run_example.py --output /path/to/output.md

环境变量:
    AKG_AGENTS_MODEL_NAME: LLM 模型名称（默认使用 deepseek）
    AKG_AGENTS_API_KEY: API 密钥
    AKG_AGENTS_BASE_URL: API 基础 URL（可选）
"""

import asyncio
import argparse
import logging
import os
import sys
from pathlib import Path

# 添加 python 目录到 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s() - %(message)s'
)
logger = logging.getLogger(__name__)


def get_sample_document() -> str:
    """获取示例文档内容"""
    # 从 resources 目录加载示例文档
    sample_path = Path(__file__).parent / "resources" / "sample_doc_with_typos.md"
    
    if sample_path.exists():
        with open(sample_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    # 如果文件不存在，返回内置示例
    return """# 示例文档

这是一篇有错别子的文章，用于侧试文档修复功能。

## 第一章 介绍

AIKG是一个很好的工俱，它可以帮助我门快速生成代码。
这个项目的目标是实显自动化的内核生成。

## 第二章 使用方法

1. 按照依赖包
2. 配置环竟变量
3. 运行侧试

希望这篇文档对你有帮组！
"""


def print_banner():
    """打印欢迎信息"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║        📝 AIKG 文档修复示例 (Doc Fixer Example)              ║
║                                                              ║
║   基于 LangGraph 的文档错别字修复和美化工作流演示            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_section(title: str, content: str = None):
    """打印带标题的分节"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)
    if content:
        print(content)


async def run_doc_fixer(
    input_content: str,
    output_path: str = None,
    document_type: str = "markdown",
    language: str = "zh"
) -> bool:
    """运行文档修复任务
    
    Args:
        input_content: 输入文档内容
        output_path: 输出文件路径（可选）
        document_type: 文档类型
        language: 文档语言
        
    Returns:
        是否成功
    """
    # 使用 importlib 加载本地模块，避免相对导入问题
    import importlib.util
    task_spec = importlib.util.spec_from_file_location("task", Path(__file__).parent / "task.py")
    task_module = importlib.util.module_from_spec(task_spec)
    task_spec.loader.exec_module(task_module)
    DocFixerTask = task_module.DocFixerTask
    
    # 准备配置
    config = {
        "max_step": 10,
        "agent_model_config": {}
    }
    
    # 模型级别配置（对应 settings.json 中 models 的 key）
    # 实际模型名/API 地址/密钥等由 settings.json 或环境变量统一管理
    model_level = "standard"
    config["agent_model_config"]["default"] = model_level
    config["agent_model_config"]["typo_fixer"] = model_level
    config["agent_model_config"]["beautifier"] = model_level
    
    print_section("配置信息")
    print(f"  模型级别: {model_level}")
    print(f"  文档类型: {document_type}")
    print(f"  语言: {language}")
    print(f"  输入长度: {len(input_content)} 字符")
    
    print_section("原始文档内容")
    print(input_content)
    
    # 创建任务
    task = DocFixerTask(
        task_id="example_001",
        config=config,
        document_content=input_content,
        document_type=document_type,
        language=language
    )
    
    print_section("开始处理...")
    print("  ⏳ 正在调用 LLM 进行文档修复...")
    
    # 执行任务
    success, final_state = await task.run()
    
    if success:
        print_section("✅ 处理完成")
        
        # 打印结果摘要
        print(task.get_result_summary(final_state))
        
        # 打印修复后的文档
        print_section("修复后的文档内容")
        beautified_content = final_state.get("beautified_content", "")
        print(beautified_content)
        
        # 保存输出
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(beautified_content)
            print(f"\n📁 结果已保存到: {output_path}")
        
        # 打印推理过程（如果有）
        typo_reasoning = final_state.get("typo_fixer_reasoning", "")
        beautifier_reasoning = final_state.get("beautifier_reasoning", "")
        
        if typo_reasoning or beautifier_reasoning:
            print_section("Agent 推理过程")
            if typo_reasoning:
                print("\n[TypoFixer 推理]")
                print(typo_reasoning[:500] + "..." if len(typo_reasoning) > 500 else typo_reasoning)
            if beautifier_reasoning:
                print("\n[Beautifier 推理]")
                print(beautifier_reasoning[:500] + "..." if len(beautifier_reasoning) > 500 else beautifier_reasoning)
        
        return True
    else:
        print_section("❌ 处理失败")
        error = final_state.get("error", "Unknown error")
        print(f"  错误信息: {error}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="AIKG 文档修复示例",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python examples/build_a_simple_workflow/run_example.py
  python examples/build_a_simple_workflow/run_example.py --input doc.md
  python examples/build_a_simple_workflow/run_example.py --input doc.md --output fixed.md
        """
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="输入文档路径（默认使用内置示例）"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="输出文档路径（可选）"
    )
    parser.add_argument(
        "--type", "-t",
        type=str,
        default="markdown",
        choices=["markdown", "text", "code"],
        help="文档类型（默认: markdown）"
    )
    parser.add_argument(
        "--lang", "-l",
        type=str,
        default="zh",
        choices=["zh", "en"],
        help="文档语言（默认: zh）"
    )
    
    args = parser.parse_args()
    
    # 打印欢迎信息
    print_banner()
    
    # 检查环境变量
    if not os.environ.get("AKG_AGENTS_API_KEY") and not os.environ.get("AKG_AGENTS_BASE_URL"):
        print("⚠️  警告: 未设置 AKG_AGENTS_API_KEY 或 AKG_AGENTS_BASE_URL 环境变量")
        print("   请确保已正确配置 LLM API 访问")
        print()
    
    # 获取输入内容
    if args.input:
        if not os.path.exists(args.input):
            print(f"❌ 错误: 输入文件不存在: {args.input}")
            sys.exit(1)
        with open(args.input, 'r', encoding='utf-8') as f:
            input_content = f.read()
        print(f"📄 使用输入文件: {args.input}")
    else:
        input_content = get_sample_document()
        print("📄 使用内置示例文档")
    
    # 运行任务
    try:
        success = asyncio.run(run_doc_fixer(
            input_content=input_content,
            output_path=args.output,
            document_type=args.type,
            language=args.lang
        ))
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ 运行错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

