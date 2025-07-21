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
python tests/ut/batch_gen_metadata.py --impl_type "triton" --framework "numpy" --backend "ascend" --arch "ascend910b4" --path "database/temp/2_standard_matrix_multiplication_"
'''

import argparse
import logging
from pathlib import Path
from ai_kernel_generator import get_project_root
from ai_kernel_generator.database.database_rag import DatabaseRAG

DEFAULT_CONFIG_PATH = Path(get_project_root()) / "database" / "rag_config.yaml"


def get_code(impl_type: str, framework: str, path: str):
    """将目录下所有Python文件移动到对应的算子目录"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    src_dir = Path(path)

    if not src_dir.is_dir():
        logging.error(f"无效的目录路径: {src_dir}")
        return

    # 查找指定后缀的Python文件
    impl_files = list(src_dir.glob(f'*{impl_type}.py'))
    framework_files = list(src_dir.glob(f'*{framework}.py'))
    if len(impl_files) != 1:
        logging.error(f"找到{len(impl_files)}个{impl_type}.py文件，要求必须且只能有1个")
        raise ValueError(f"必须且只能有1个{impl_type}.py文件")
    if len(framework_files) != 1:
        logging.error(f"找到{len(framework_files)}个{framework}.py文件，要求必须且只能有1个")
        raise ValueError(f"必须且只能有1个{framework}.py文件")

    impl_file = impl_files[0]
    framework_file = framework_files[0]
    # 读取文件内容生成md5_hash
    with open(impl_file, 'r', encoding='utf-8') as f:
        impl_code = f.read()

    with open(framework_file, 'r', encoding='utf-8') as f:
        framework_code = f.read()

    return impl_code, framework_code


def insert_one_case(impl_type: str, framework: str, backend: str, arch: str, path: str):
    """处理指定架构下的所有算子目录，生成或更新metadata.json"""
    db_rag = DatabaseRAG()
    impl_code, framework_code = get_code(impl_type=impl_type, framework=framework, path=path)
    db_rag.insert(impl_code, framework_code, backend, arch, impl_type, framework)


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='批量提取Python文件特征并生成元数据')
    parser.add_argument('--impl_type', required=True, help='实现类型（如triton、swft等）')
    parser.add_argument('--backend', required=True, help='后端名称（如ascend、cuda等）')
    parser.add_argument('--arch', required=True, help='架构名称（如ascend910b4、ascend310p3等）')
    parser.add_argument('--framework', required=True, help='框架名称（如torch、mindspore等）')
    parser.add_argument('--path', required=True, help='待处理文件当前所在的目录')
    parser.add_argument('--config_path', default=DEFAULT_CONFIG_PATH, help='配置文件路径')
    args = parser.parse_args()

    # 第二步：处理metadata生成
    insert_one_case(
        impl_type=args.impl_type,
        framework=args.framework,
        backend=args.backend,
        arch=args.arch,
        path=args.path
    )


if __name__ == "__main__":
    main()
