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
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger(__name__)


def check_backend_arch(backend: str, arch: str):
    """
    验证后端与架构的匹配关系
    Args:
        backend: 计算后端名称(ascend/cuda/cpu)
        arch: 硬件架构名称
    """
    if backend not in ["ascend", "cuda", "cpu"]:
        raise ValueError("backend must be ascend, cuda or cpu")
    elif backend == "ascend":
        # 支持 ascend910b1, b2, b2c, b3, b4 和 ascend310p3
        supported_ascend_archs = ["ascend910b1", "ascend910b2", "ascend910b2c", "ascend910b3", "ascend910b4", "ascend310p3"]
        if arch not in supported_ascend_archs:
            raise ValueError("ascend backend only support ascend910b1/b2/b2c/b3/b4 and ascend310p3")
    elif backend == "cuda" and arch not in ["a100", "v100", "h20", "l20", "rtx3090"]:
        raise ValueError("cuda backend only support a100, v100, h20, l20, and rtx3090")
    elif backend == "cpu" and arch not in ["x86_64", "aarch64"]:
        raise ValueError("cpu backend only support x86_64 and aarch64")


def normalize_dsl(dsl: str, backend: str = None) -> str:
    """
    规范化DSL类型，将通用的triton根据backend转换为triton_cuda或triton_ascend
    
    Args:
        dsl: 实现类型
        backend: 硬件后端名称(ascend/cuda/cpu)，用于自动转换triton
        
    Returns:
        规范化后的DSL类型
        
    Raises:
        ValueError: 如果dsl为"triton"但backend未提供或无效
    """
    dsl = dsl.lower()
    
    # 如果已经是规范化的类型，直接返回
    if dsl in ["triton_cuda", "triton_ascend", "triton-russia", "swft", "cuda_c", "cpp", "tilelang_npuir", "tilelang_cuda", "ascendc", "torch"]:
        return dsl
    
    # 如果是通用的triton，需要根据backend转换
    if dsl == "triton":
        if backend is None:
            raise ValueError(
                "dsl='triton' is no longer supported. Please use 'triton_cuda' (for CUDA backend) "
                "or 'triton_ascend' (for Ascend backend) explicitly. "
                "Alternatively, provide backend parameter for automatic conversion."
            )
        backend = backend.lower()
        if backend == "cuda":
            return "triton_cuda"
        elif backend == "ascend":
            return "triton_ascend"
        else:
            raise ValueError(
                f"dsl='triton' cannot be used with backend='{backend}'. "
                "Please use 'triton_cuda' (for CUDA) or 'triton_ascend' (for Ascend) explicitly."
            )
    
    # 其他情况直接返回
    return dsl


def check_dsl(dsl: str):
    """
    验证实现类型
    Args:
        dsl: 实现类型(triton_cuda/triton_ascend/triton-russia/swft/torch等)
    """
    valid_dsls = ["triton_cuda", "triton_ascend", "triton-russia", "swft", "cuda_c", "cpp", "tilelang_npuir", "tilelang_cuda", "ascendc", "torch"]
    if dsl not in valid_dsls:
        raise ValueError(
            f"dsl must be one of {valid_dsls}. "
            "Note: 'triton' is no longer supported. Use 'triton_cuda' or 'triton_ascend' instead."
        )


def check_task_type(task_type: str):
    """
    验证任务类型
    Args:
        task_type: 任务类型(precision_only/profile)
    """
    if task_type not in ["precision_only", "profile"]:
        raise ValueError("task_type must be precision_only or profile")


# 配置依赖关系映射表
# 注意：ascend910b1/b2/b2c/b3/b4 使用相同的配置
VALID_CONFIGS = {
    # framework -> backend -> arch -> dsl
    "mindspore": {
        "ascend": {
            "ascend910b1": ["triton_ascend", "triton-russia"],
            "ascend910b2": ["triton_ascend", "triton-russia"],
            "ascend910b2c": ["triton_ascend", "triton-russia"],
            "ascend910b3": ["triton_ascend", "triton-russia"],
            "ascend910b4": ["triton_ascend", "triton-russia"],
            "ascend310p3": ["swft"]
        },
    },
    "torch": {
        "ascend": {
            "ascend910b1": ["triton_ascend", "triton-russia", "tilelang_npuir", "ascendc", "torch"],
            "ascend910b2": ["triton_ascend", "triton-russia", "tilelang_npuir", "ascendc", "torch"],
            "ascend910b2c": ["triton_ascend", "triton-russia", "tilelang_npuir", "ascendc", "torch"],
            "ascend910b3": ["triton_ascend", "triton-russia", "tilelang_npuir", "ascendc", "torch"],
            "ascend910b4": ["triton_ascend", "triton-russia", "tilelang_npuir", "ascendc", "torch"],
            "ascend310p3": ["swft", "ascendc", "torch"]
        },
        "cuda": {
            "a100": ["triton_cuda", "cuda_c", "tilelang_cuda", "torch"],
            "h20": ["triton_cuda", "cuda_c", "tilelang_cuda", "torch"],
            "l20": ["triton_cuda", "cuda_c", "tilelang_cuda", "torch"],
            "rtx3090": ["triton_cuda", "cuda_c", "tilelang_cuda", "torch"],
        },
        "cpu": {
            "x86_64": ["cpp"],
            "aarch64": ["cpp"],
        },
    },
    "numpy": {
        "ascend": {
            "ascend310p3": ["swft"]
        },
    }
}


def _get_config_for_arch(backend_config: dict, arch: str) -> list:
    """
    获取指定架构的配置
    Args:
        backend_config: 后端配置字典
        arch: 架构名称
    Returns:
        DSL 列表
    """
    # 直接匹配
    if arch in backend_config:
        return backend_config[arch]
    
    return None


def check_task_config(framework: str, backend: str, arch: str, dsl: str):
    """
    统一验证配置参数之间的依赖关系
    Args:
        framework: 框架类型
        backend: 硬件后端名称
        arch: 硬件架构名称
        dsl: 实现类型（会自动转换triton为triton_cuda或triton_ascend）
    """
    # 首先规范化DSL（自动转换triton）
    normalized_dsl = normalize_dsl(dsl, backend)
    
    if framework not in VALID_CONFIGS:
        raise ValueError(f"Unsupported framework: {framework}")

    if backend not in VALID_CONFIGS[framework]:
        raise ValueError(f"Framework {framework} does not support backend: {backend}")

    backend_config = VALID_CONFIGS[framework][backend]
    dsl_list = _get_config_for_arch(backend_config, arch)
    
    if dsl_list is None:
        raise ValueError(f"Backend {backend} does not support arch: {arch}")

    if normalized_dsl not in dsl_list:
        raise ValueError(f"Arch {arch} does not support dsl: {normalized_dsl}")
    
    # 返回规范化后的DSL，供调用者使用
    return normalized_dsl


def collect_and_save_all_examples(
    arch: str,
    dsl: str,
    project_root_path: Path,
    source_dirs: Dict[str, Path],
) -> Optional[Path]:
    """
    汇总所有examples并保存到统一目录 database/all_examples/{arch}/{dsl}
    
    此函数将来自不同源目录的示例文件统一复制到目录中，Python文件和其他文件分别保存。
    - Python文件保存到: database/all_examples/{arch}/{dsl}/code/
    - 其他文件保存到: database/all_examples/{arch}/{dsl}/docs/
    
    Args:
        arch: 硬件架构名称
        dsl: DSL类型
        project_root_path: 项目根路径
        source_dirs: 源目录字典，格式为 {"prefix": Path("source_dir")}
                    文件会被复制并重命名为 "{prefix}_{原文件名}"
                    例如: {"user": Path("user_examples/"), "local": Path("local_examples/")}
    
    Returns:
        Path: 统一保存目录的根路径，如果失败则返回None
    """
    if not arch or not dsl:
        logger.warning("arch或dsl为空，无法汇总示例代码")
        return None
    
    # 创建统一的保存目录
    base_dir = project_root_path / "database" / "all_examples" / arch / dsl
    code_dir = base_dir / "code"
    doc_dir = base_dir / "docs"
    
    code_dir.mkdir(parents=True, exist_ok=True)
    doc_dir.mkdir(parents=True, exist_ok=True)
    
    saved_code_count = 0
    saved_doc_count = 0
    
    # 遍历所有源目录
    for prefix, source_dir in source_dirs.items():
        if not source_dir or not isinstance(source_dir, Path):
            logger.warning(f"跳过无效的源目录: {prefix} -> {source_dir}")
            continue
            
        if not source_dir.exists():
            logger.warning(f"源目录不存在，跳过: {source_dir}")
            continue
        
        try:
            # 如果是目录，复制其中的所有文件
            if source_dir.is_dir():
                for file_path in source_dir.glob("*"):
                    if not file_path.is_file():
                        continue
                    
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read().strip()
                        if content:
                            # 根据文件扩展名选择目标目录
                            if file_path.suffix == ".py":
                                target_dir = code_dir
                                saved_code_count += 1
                            else:
                                target_dir = doc_dir
                                saved_doc_count += 1
                            
                            # 目标文件名：前缀_原文件名
                            save_path = target_dir / f"{prefix}_{file_path.name}"
                            with open(save_path, "w", encoding="utf-8") as f:
                                f.write(content)
                    except Exception as e:
                        logger.warning(f"复制文件 {file_path} 失败: {e}")
            
            # 如果是单个文件，直接复制
            elif source_dir.is_file():
                try:
                    with open(source_dir, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                    if content:
                        # 根据文件扩展名选择目标目录
                        if source_dir.suffix == ".py":
                            target_dir = code_dir
                            saved_code_count += 1
                        else:
                            target_dir = doc_dir
                            saved_doc_count += 1
                        
                        save_path = target_dir / f"{prefix}_{source_dir.name}"
                        with open(save_path, "w", encoding="utf-8") as f:
                            f.write(content)
                except Exception as e:
                    logger.warning(f"复制文件 {source_dir} 失败: {e}")
                    
        except Exception as e:
            logger.warning(f"处理源目录 {prefix}:{source_dir} 时发生错误: {e}")
    
    logger.info(f"汇总完成，共保存 {saved_code_count} 个Python文件到: {code_dir}")
    logger.info(f"汇总完成，共保存 {saved_doc_count} 个其他文件到: {doc_dir}")
    return code_dir
