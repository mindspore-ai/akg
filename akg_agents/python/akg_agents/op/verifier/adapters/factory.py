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

"""Factory for creating adapters."""


def get_framework_adapter(framework: str):
    """Get framework adapter by name.
    
    Args:
        framework: Framework name (torch, mindspore, numpy)
        
    Returns:
        FrameworkAdapter instance
    """
    framework_lower = framework.lower()
    
    if framework_lower == "torch":
        from .framework.torch import FrameworkAdapterTorch
        return FrameworkAdapterTorch()
    elif framework_lower == "mindspore":
        from .framework.mindspore import FrameworkAdapterMindSpore
        return FrameworkAdapterMindSpore()
    elif framework_lower == "numpy":
        from .framework.numpy import FrameworkAdapterNumpy
        return FrameworkAdapterNumpy()
    else:
        raise ValueError(f"Unsupported framework: {framework}")


def get_dsl_adapter(dsl: str):
    """Get DSL adapter by name.
    
    Args:
        dsl: DSL name (triton_cuda, triton_ascend, swft, ascendc, etc.)
        
    Returns:
        DSLAdapter instance
    """
    dsl_lower = dsl.lower()
    
    if dsl_lower == "triton_cuda":
        from .dsl.triton_cuda import DSLAdapterTritonCuda
        return DSLAdapterTritonCuda()
    elif dsl_lower in ["triton_ascend", "triton-russia"]:
        from .dsl.triton_ascend import DSLAdapterTritonAscend
        return DSLAdapterTritonAscend()
    elif dsl_lower == "swft":
        from .dsl.swft import DSLAdapterSwft
        return DSLAdapterSwft()
    elif dsl_lower == "ascendc":
        from .dsl.ascendc import DSLAdapterAscendC
        return DSLAdapterAscendC()
    elif dsl_lower == "cpp":
        from .dsl.cpp import DSLAdapterCpp
        return DSLAdapterCpp()
    elif dsl_lower == "cuda_c":
        from .dsl.cuda_c import DSLAdapterCudaC
        return DSLAdapterCudaC()
    elif dsl_lower == "tilelang_npuir":
        from .dsl.tilelang_npuir import DSLAdapterTilelangNpuir
        return DSLAdapterTilelangNpuir()
    elif dsl_lower == "tilelang_cuda":
        from .dsl.tilelang_cuda import DSLAdapterTilelangCuda
        return DSLAdapterTilelangCuda()
    elif dsl_lower == "torch":
        from .dsl.torch import DSLAdapterTorch
        return DSLAdapterTorch()
    else:
        raise ValueError(f"Unsupported DSL: {dsl}")


def get_backend_adapter(backend: str):
    """Get backend adapter by name.
    
    Args:
        backend: Backend name (cuda, ascend, cpu)
        
    Returns:
        BackendAdapter instance
    """
    backend_lower = backend.lower()
    
    if backend_lower == "cuda":
        from .backend.cuda import BackendAdapterCuda
        return BackendAdapterCuda()
    elif backend_lower == "ascend":
        from .backend.ascend import BackendAdapterAscend
        return BackendAdapterAscend()
    elif backend_lower == "cpu":
        from .backend.cpu import BackendAdapterCpu
        return BackendAdapterCpu()
    else:
        raise ValueError(f"Unsupported backend: {backend}")

