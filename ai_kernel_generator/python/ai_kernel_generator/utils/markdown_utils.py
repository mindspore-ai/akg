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

import logging
import re
import os
from pathlib import Path
from ai_kernel_generator import get_project_root

logger = logging.getLogger(__name__)

class SWFTDocsProcessor:
    def __init__(self, api_to_use):
        self.api_to_use = api_to_use
        self.output_lines = []

    def add_function_dscb(self, content, header):
        level = 2
        prefix = '#' * level
        target = f"{prefix} {header}"
        end_pattern = re.compile(rf'^#{{1,{level}}}\s+', re.MULTILINE)

        start_idx = content.find(target)
        while start_idx > 0 and content[start_idx + len(target)] == '\n':
            start_idx += 1
        if start_idx == -1:
            return None

        # 找到内容结束位置（下一个同级或更高级标题）
        remaining = content[start_idx + len(target):]
        end_match = end_pattern.search(remaining)
        end_idx = end_match.start() if end_match else len(remaining)
        while remaining[end_idx - 1] == '\n':
            end_idx -= 1
        text = "\n" + remaining[:end_idx] + "\n"
        return text

    def add_function_code(self, py_file, function):
        if os.path.exists(py_file):
            function_code = self.extract_function_code(py_file, function)
            if function_code:
                text = f"### API代码\n```python\n{function_code}\n```\n\n"
                return text

    def extract_function_code(self, filename, function_name):
        """从 Python 文件中提取指定函数的代码"""
        try:
            with open(filename, 'r', encoding="utf-8") as f:
                lines = f.readlines()
        except FileNotFoundError:
            return None

        function_pattern = re.compile(rf'^\s*def\s+{function_name}\s*\(')
        in_function = False
        function_lines = []
        indent_level = None

        for line in lines:
            if function_pattern.match(line):
                in_function = True
                # 获取函数定义行的缩进
                indent_level = len(re.match(r'^\s*', line).group(0))
                function_lines.append(line)
                continue

            if in_function:
                # 检查是否仍在函数内（相同或更深缩进）
                current_indent = len(re.match(r'^\s*', line).group(0))
                if line.strip() == '' or current_indent > indent_level:
                    function_lines.append(line)
                else:
                    break

        return ''.join(function_lines).strip() if function_lines else None

    def run(self):
        root_dir = get_project_root()
        for file_name in self.api_to_use.keys():
            if not self.api_to_use[file_name]:
                continue
            md_file = os.path.join(root_dir, "resources", "docs", "swft_docs", f"{file_name}.md")
            py_file = os.path.join(root_dir, "resources", "docs", "swft_docs", "api", f"{file_name}.py")
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            self.output_lines.append(f"# {file_name}.py\n\n")
            for function in self.api_to_use[file_name]:
                func_name = f"## {function}"
                api_docs = self.add_function_dscb(content, function)
                api_code = self.add_function_code(py_file, function)

                if not func_name or not api_docs or not api_code:
                    print("No matching function found.")
                    continue

                self.output_lines.append(func_name)
                self.output_lines.append(api_docs)
                self.output_lines.append(api_code)
        return "\n".join(self.output_lines)


def generate_available_api(swft_api):
    processor = SWFTDocsProcessor(swft_api)
    result = processor.run()
    return result

def extract_function_details():
    swft_doc_files = [
        "compute.md",
        "composite.md"
    ]

    aul_docs_dir = os.path.join(get_project_root(), "resources", "docs", "swft_docs")
    combined_spec = ""

    for doc_file in swft_doc_files:
        doc_path = os.path.join(aul_docs_dir, doc_file)
        try:
            with open(doc_path, "r", encoding="utf-8") as f:
                content = f.read()
                combined_spec += content
                combined_spec += "\n\n"
        except Exception as e:
            logger.error(f"Failed to load AUL doc {doc_file}: {e}")
            continue


    # 匹配二级标题和对应的函数说明
    pattern = r'^##\s+(.+?)\n.*?^###\s+函数说明\s*?\n([\s\S]*?)(?=^##|\Z)'
    matches = re.findall(pattern, combined_spec, flags=re.MULTILINE | re.DOTALL)

    result = {}
    for title, desc in matches:
        # 清理描述中的多余空行和代码块
        cleaned_desc = re.sub(r'```.*?\n', '', desc, flags=re.DOTALL).strip()
        cleaned_desc = re.sub(r'\n{2,}', '\n', cleaned_desc)
        result[title.strip()] = cleaned_desc
        
    return result
