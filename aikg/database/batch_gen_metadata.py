'''
# 使用示例：
python database/batch_gen_metadata.py --impl_type "triton" --backend "ascend" --arch "ascend910b4" --path ""database/triton/ascend910b4"" --mode overwrite

## 默认模式：跳过已存在的metadata.json
python database/batch_gen_metadata.py --impl_type xxx --backend xxx --arch xxx --path "./your_files_dir" --mode skip

## 覆盖模式：重新生成所有metadata.json
python database/batch_gen_metadata.py --impl_type xxx --backend xxx --arch xxx --path "./your_files_dir" --mode overwrite
'''

import argparse
import json
import asyncio
import yaml
import shutil
import logging
from pathlib import Path
from ai_kernel_generator import get_project_root
from ai_kernel_generator.utils.common_utils import get_md5_hash
from ai_kernel_generator.core.agent.utils.feature_extraction import FeatureExtraction

# 配置路径常量
DEFAULT_DATABASE_PATH = Path(get_project_root()).parent.parent / "database"
DEFAULT_CONFIG_PATH = Path(get_project_root()) / "database" / "rag_config.yaml"


def load_config(config_path: Path) -> dict:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def feature_extractor(task_code, impl_type: str, backend: str, arch: str, model_config: dict):
    """提取特征的封装函数"""
    feature_extractor = FeatureExtraction(
        task_desc=task_code,
        model_config=model_config,
        impl_type=impl_type,
        backend=backend,
        arch=arch
    )
    extracted_features, _, _ = asyncio.run(feature_extractor.run())
    return extracted_features


def process_python_files(impl_type: str, backend: str, arch: str, path: str):
    """将目录下所有Python文件移动到对应的算子目录"""
    # 验证目录
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    path = Path(path).resolve()
    # 处理单个文件或目录中的文件
    if path.is_file():
        if path.name.endswith(f"_{impl_type}.py"):
            py_files = [path]
    else:
        # 递归查找所有子目录中符合条件的文件
        py_files = list(path.glob(f"*.py"))
    
    for py_file in py_files:
        if py_file.is_file():
            try:
                # 读取文件内容生成md5_hash
                with open(py_file, 'r', encoding='utf-8') as f:
                    impl_code = f.read()
                
                # 生成唯一哈希路径
                md5_hash = get_md5_hash(
                    impl_code=impl_code,
                    impl_type=impl_type,
                    backend=backend,
                    arch=arch
                )

                # 创建目标目录
                file_path = DEFAULT_DATABASE_PATH / "operators" / arch / md5_hash
                file_path.mkdir(parents=True, exist_ok=True)

                # 移动文件
                dest_file = file_path / f"{impl_type}.py"
                if dest_file.exists():
                    logging.warning(f"目标文件已存在，跳过复制: {dest_file}")
                    continue
                shutil.copy2(str(py_file), str(dest_file))
                logging.info(f"已复制文件: {py_file} -> {dest_file}")

            except Exception as e:
                logging.error(f"移动文件 {py_file} 时出错: {str(e)}")

def process_operator_directories(impl_type: str, backend: str, arch: str, mode: str, config_path: Path):
    """处理指定架构下的所有算子目录，生成或更新metadata.json"""
    operator_path = DEFAULT_DATABASE_PATH / "operators"
    arch_dir = operator_path / arch
    if not arch_dir.exists():
        arch_dir.mkdir(parents=True, exist_ok=True)

    # 加载配置
    config = load_config(config_path)
    model_config = config.get("model_config")
    if not model_config:
        raise ValueError("配置文件中未找到model_config")

    # 遍历所有md5_hash目录
    for md5_dir in arch_dir.iterdir():
        if md5_dir.is_dir():
            # 查找目录下的.py文件
            py_files = list(md5_dir.glob("*.py"))
            if not py_files:
                logging.warning(f"目录 {md5_dir} 中未找到Python文件，跳过")
                continue

            # 假设每个目录只有一个实现文件
            py_file = py_files[0]
            metadata_file = md5_dir / "metadata.json"

            # 检查metadata文件状态
            if metadata_file.exists():
                if mode == "skip":
                    logging.info(f"metadata已存在，跳过: {md5_dir}")
                    continue
                elif mode == "overwrite":
                    logging.info(f"metadata已存在，将覆盖: {md5_dir}")
                else:
                    logging.warning(f"未知模式 {mode}，跳过: {md5_dir}")
                    continue

            try:
                # 读取文件内容
                with open(py_file, 'r', encoding='utf-8') as f:
                    impl_code = f.read()
                task_code = impl_code

                # 提取特征
                features = feature_extractor(
                    task_code=task_code,
                    impl_type=impl_type,
                    backend=backend,
                    arch=arch,
                    model_config=model_config
                )

                # 写入元数据文件
                with open(metadata_file, "w", encoding="utf-8") as f:
                    json.dump(features, f, ensure_ascii=False, indent=4)

                logging.info(f"成功生成metadata: {metadata_file}")

            except Exception as e:
                print(f"处理目录 {md5_dir} 时出错: {str(e)}")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='批量提取Python文件特征并生成元数据')
    parser.add_argument('--impl_type', required=True, help='实现类型（如triton、swft等）')
    parser.add_argument('--backend', required=True, help='后端名称（如ascend、cuda等）')
    parser.add_argument('--arch', required=True, help='架构名称（如ascend910b4、ascend310p3等）')
    parser.add_argument('--path', required=True, help='待处理文件当前所在的目录')
    parser.add_argument('--config_path', default=DEFAULT_CONFIG_PATH, help='配置文件路径')
    parser.add_argument('--mode', choices=['skip', 'overwrite'], default='skip', 
                        help='如果metadata.json已存在，处理方式：skip（跳过）或overwrite（覆盖）')
    args = parser.parse_args()

    # 第一步：移动文件到目标目录
    process_python_files(
        impl_type=args.impl_type,
        backend=args.backend,
        arch=args.arch,
        path=args.path
    )

    # 第二步：处理metadata生成
    process_operator_directories(
        impl_type=args.impl_type,
        backend=args.backend,
        arch=args.arch,
        mode=args.mode,
        config_path=args.config_path
    )


if __name__ == "__main__":
    main()