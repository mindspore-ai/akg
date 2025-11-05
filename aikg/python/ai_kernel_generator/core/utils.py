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
    elif backend == "cuda" and arch not in ["a100", "v100"]:
        raise ValueError("cuda backend only support a100 and v100")
    elif backend == "cpu" and arch not in ["x86_64", "aarch64"]:
        raise ValueError("cpu backend only support x86_64 and aarch64")


def check_dsl(dsl: str):
    """
    验证实现类型
    Args:
        dsl: 实现类型(triton/swft)
    """
    if dsl not in ["triton", "triton-russia", "swft"]:
        raise ValueError("dsl must be triton or swft")


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
            "ascend910b1": ["triton", "triton-russia"],
            "ascend910b2": ["triton", "triton-russia"],
            "ascend910b2c": ["triton", "triton-russia"],
            "ascend910b3": ["triton", "triton-russia"],
            "ascend910b4": ["triton", "triton-russia"],
            "ascend310p3": ["swft"]
        },
    },
    "torch": {
        "ascend": {
            "ascend910b1": ["triton", "triton-russia", "tilelang_npuir", "ascendc"],
            "ascend910b2": ["triton", "triton-russia", "tilelang_npuir", "ascendc"],
            "ascend910b2c": ["triton", "triton-russia", "tilelang_npuir", "ascendc"],
            "ascend910b3": ["triton", "triton-russia", "tilelang_npuir", "ascendc"],
            "ascend910b4": ["triton", "triton-russia", "tilelang_npuir", "ascendc"],
            "ascend310p3": ["swft", "ascendc"]
        },
        "cuda": {
            "a100": ["triton", "cuda_c", "tilelang_cuda"],
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
        dsl: 实现类型
    """
    if framework not in VALID_CONFIGS:
        raise ValueError(f"Unsupported framework: {framework}")

    if backend not in VALID_CONFIGS[framework]:
        raise ValueError(f"Framework {framework} does not support backend: {backend}")

    backend_config = VALID_CONFIGS[framework][backend]
    dsl_list = _get_config_for_arch(backend_config, arch)
    
    if dsl_list is None:
        raise ValueError(f"Backend {backend} does not support arch: {arch}")

    if dsl not in dsl_list:
        raise ValueError(f"Arch {arch} does not support dsl: {dsl}")
