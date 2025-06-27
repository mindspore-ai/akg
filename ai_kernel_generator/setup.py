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

from setuptools import setup, find_packages
import os

# 读取版本号
def _read_version():
    try:
        with open("version.txt", "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "0.1.0"

# 读取 README.md 作为长描述
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# 读取 requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# 确保包含所有数据文件
def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            if filename.endswith(('.j2', '.md', '.txt', '.json', '.yaml')):
                paths.append(os.path.join('..', path, filename))
    return paths

# 获取所有需要包含的数据文件
extra_files = []
extra_files.extend(package_files('python/ai_kernel_generator'))

setup(
    name="ai_kernel_generator",
    version=_read_version(),
    author="The MindSpore Authors",
    author_email="contact@mindspore.cn",
    description="AI Kernel Generator (AIKG)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://www.mindspore.cn/",
    download_url="https://gitee.com/mindspore/akg/tags",
    project_urls={
        'Sources': 'https://gitee.com/mindspore/akg',
        'Issue Tracker': 'https://gitee.com/mindspore/akg/issues',
    },
    license="Apache 2.0",
    package_dir={"": "python"},
    packages=find_packages(where="python", include=['ai_kernel_generator', 'ai_kernel_generator.*']),
    package_data={
        'ai_kernel_generator': [
            '**/*.j2',
            '**/*.md',
            '**/*.txt',
            '**/*.json',
            '**/*.yaml',
        ],
    },
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
    ],
    python_requires=">=3.8",
) 