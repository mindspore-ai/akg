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

'''
# 使用示例：
python tests/ut/batch_gen_metadata.py --impl_type "triton" --framework "torch" --backend "ascend" --arch "ascend910b4" --path "temp"
'''

import argparse
import asyncio
from pathlib import Path
from ai_kernel_generator import get_project_root
from ai_kernel_generator.database.database import Database
from ai_kernel_generator.utils.common_utils import get_fixed_suffix_content

DEFAULT_CONFIG_PATH = Path(get_project_root()) / "database" / "database_config.yaml"


async def insert_one_case(impl_type: str, framework: str, backend: str, arch: str, path: str):
    """处理指定架构下的所有算子目录，生成或更新metadata.json"""
    db_system = Database()
    impl_code = get_fixed_suffix_content(suffix=impl_type, path=path)
    framework_code = get_fixed_suffix_content(suffix=framework, path=path)
    await db_system.insert(impl_code, framework_code, backend, arch, impl_type, framework)


async def insert_multi_case(impl_type: str, framework: str, backend: str, arch: str, path: str):
    """处理指定架构下的所有算子目录，生成或更新metadata.json"""
    db_system = Database()
    for path in Path(path).iterdir():
        if path.is_dir():
            impl_code = get_fixed_suffix_content(suffix=impl_type, path=path)
            framework_code = get_fixed_suffix_content(suffix=framework, path=path)
            await db_system.insert(impl_code, framework_code, backend, arch, impl_type, framework)


async def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='批量提取Python文件特征并生成元数据')
    parser.add_argument('--impl_type', required=True, help='实现类型（如triton、swft等）')
    parser.add_argument('--backend', required=True, help='后端名称（如ascend、cuda等）')
    parser.add_argument('--arch', required=True, help='架构名称（如ascend910b4、ascend310p3等）')
    parser.add_argument('--framework', required=True, help='框架名称（如torch、mindspore等）')
    parser.add_argument('--path', required=True, help='待处理文件当前所在的目录')
    parser.add_argument('--config_path', default=DEFAULT_CONFIG_PATH, help='配置文件路径')
    args = parser.parse_args()

    await insert_one_case(
        impl_type=args.impl_type,
        framework=args.framework,
        backend=args.backend,
        arch=args.arch,
        path=args.path
    )

    # await insert_multi_case(
    #     impl_type=args.impl_type,
    #     framework=args.framework,
    #     backend=args.backend,
    #     arch=args.arch,
    #     path=args.path
    # )


if __name__ == "__main__":
    asyncio.run(main())
