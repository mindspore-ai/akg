# Copyright 2025-2026 Huawei Technologies Co., Ltd
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

# setup.py 所在目录，用于定位同级文件
_HERE = os.path.abspath(os.path.dirname(__file__))


# 读取版本号
def _read_version():
    try:
        with open(os.path.join(_HERE, "version.txt"), "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "0.2.0"


# 读取 README.md 作为长描述
with open(os.path.join(_HERE, "README.md"), "r", encoding="utf-8") as fh:
    long_description = fh.read()

# 读取 requirements.txt
with open(os.path.join(_HERE, "requirements.txt"), "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="akg_agents",
    version=_read_version(),
    author="The MindSpore Authors",
    author_email="contact@mindspore.cn",
    description="AKG Agents (AKG Agents)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://www.mindspore.cn/",
    download_url="https://gitcode.com/mindspore/akg/tags",
    project_urls={
        'Sources': 'https://gitcode.com/mindspore/akg/tree/br_agents/akg_agents',
        'Issue Tracker': 'https://gitcode.com/mindspore/akg/issues',
    },
    license="Apache 2.0",
    package_dir={"": "python"},
    packages=find_packages(where="python", include=['akg_agents', 'akg_agents.*']),
    package_data={
        'akg_agents': [
            '**/*.j2',
            '**/*.md',
            '**/*.txt',
            '**/*.json',
            '**/*.yaml',
            '**/*.py',
            '**/*.ans',
        ],
    },
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "akg_cli=akg_agents.cli.cli:app",
        ],
    },
)
