#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Token统计脚本
用于统计目录中所有 .md 和 .j2 文件的 token 数量
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def count_tokens(text: str) -> int:
    """使用tiktoken准确计算字符串的token数量

    Args:
        text: 要统计的文本

    Returns:
        token数量，如果计算失败返回-1
    """
    if not text:
        return 0

    try:
        import tiktoken
    except ImportError:
        logger.error("tiktoken库未安装，请运行: pip install tiktoken")
        return -1

    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        token_count = len(tokens)
        return token_count

    except Exception as e:
        logger.error(f"计算token时发生错误: {e}")
        return -1


def find_files(directory: str, extensions: List[str]) -> List[Path]:
    """递归查找指定扩展名的文件

    Args:
        directory: 要搜索的目录
        extensions: 文件扩展名列表，如 ['.md', '.j2']

    Returns:
        找到的文件路径列表
    """
    files = []
    directory_path = Path(directory)

    if not directory_path.exists():
        logger.error(f"目录不存在: {directory}")
        return files

    for ext in extensions:
        files.extend(directory_path.rglob(f"*{ext}"))

    return sorted(files)


def read_file_content(file_path: Path) -> str:
    """读取文件内容

    Args:
        file_path: 文件路径

    Returns:
        文件内容字符串
    """
    try:
        # 尝试多种编码方式读取文件
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue

        logger.warning(f"无法读取文件 {file_path}，编码问题")
        return ""

    except Exception as e:
        logger.error(f"读取文件 {file_path} 时发生错误: {e}")
        return ""


def analyze_files(directory: str) -> Dict[str, Dict]:
    """分析目录中的文件token数量

    Args:
        directory: 要分析的目录

    Returns:
        包含文件信息的字典
    """
    extensions = ['.md', '.j2']
    files = find_files(directory, extensions)

    if not files:
        logger.info(f"在目录 {directory} 中未找到 {extensions} 文件")
        return {}

    results = {}
    total_tokens = 0
    total_files = 0

    logger.info(f"找到 {len(files)} 个文件，开始分析...")

    for file_path in files:
        relative_path = file_path.relative_to(directory)
        content = read_file_content(file_path)

        if content:
            token_count = count_tokens(content)
            file_size = len(content)

            results[str(relative_path)] = {
                'path': str(relative_path),
                'tokens': token_count,
                'chars': file_size,
                'size_kb': file_size / 1024
            }

            if token_count > 0:
                total_tokens += token_count
                total_files += 1

            logger.info(f"文件: {relative_path} - Tokens: {token_count}, 字符数: {file_size}")
        else:
            results[str(relative_path)] = {
                'path': str(relative_path),
                'tokens': -1,
                'chars': 0,
                'size_kb': 0
            }

    # 添加汇总信息
    results['_summary'] = {
        'total_files': len(files),
        'successful_files': total_files,
        'total_tokens': total_tokens,
        'average_tokens_per_file': total_tokens / total_files if total_files > 0 else 0
    }

    return results


def print_summary(results: Dict[str, Dict]) -> None:
    """打印统计摘要

    Args:
        results: 分析结果
    """
    if not results:
        return

    summary = results.get('_summary', {})

    print("\n" + "="*60)
    print("TOKEN 统计摘要")
    print("="*60)
    print(f"总文件数: {summary.get('total_files', 0)}")
    print(f"成功分析文件数: {summary.get('successful_files', 0)}")
    print(f"总Token数: {summary.get('total_tokens', 0):,}")
    print(f"平均每文件Token数: {summary.get('average_tokens_per_file', 0):.1f}")
    print("="*60)

    # 按token数量排序显示文件
    file_results = {k: v for k, v in results.items() if k != '_summary' and v['tokens'] > 0}
    sorted_files = sorted(file_results.items(), key=lambda x: x[1]['tokens'], reverse=True)

    print("\n文件Token统计 (按数量降序):")
    # 使用更宽的列宽和更大的间距
    print("-" * 150)
    print(f"{'文件路径':<95} {'Tokens':>15} {'字符数':>15}")
    print("-" * 150)

    for file_path, info in sorted_files:  # 显示前20个文件
        # 确保数字右对齐且使用千分位分隔符
        tokens_str = f"{info['tokens']:,}"
        chars_str = f"{info['chars']:,}"
        print(f"{file_path:<100} {tokens_str:>15} {chars_str:>15}")


def main():
    """主函数"""
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = "."  # 默认当前目录

    logger.info(f"开始分析目录: {os.path.abspath(directory)}")

    # 分析文件
    results = analyze_files(directory)
    print_summary(results)


# python tools/count_tokens_script.py python/ai_kernel_generator/resources
if __name__ == "__main__":
    main()
