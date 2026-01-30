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

"""CUDA backend adapter."""

import torch
from typing import Optional, Any

from .base import BackendAdapter


class BackendAdapterCuda(BackendAdapter):
    """Adapter for CUDA backend."""
    
    def setup_environment(self, device_id: int, arch: str) -> None:
        """Setup CUDA environment variables."""
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
    
    def synchronize(self) -> None:
        """Synchronize CUDA device."""
        torch.cuda.synchronize()
    
    def get_profiler(self) -> Optional[Any]:
        """Get CUDA profiler (nsys)."""
        # Profiler is handled in kernel_verifier, not here
        return None
    
    def get_device_string(self, device_id: int) -> str:
        """Get CUDA device string."""
        return f"cuda:{device_id}"
    
    def validate_arch(self, arch: str) -> bool:
        """Validate CUDA architecture."""
        supported_archs = ["a100", "v100", "h20", "l20", "rtx3090"]
        return arch in supported_archs

