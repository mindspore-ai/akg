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

"""Ascend backend adapter."""

import torch
from typing import Optional, Any

from .base import BackendAdapter


class BackendAdapterAscend(BackendAdapter):
    """Adapter for Ascend backend."""
    
    def setup_environment(self, device_id: int, arch: str) -> None:
        """Setup Ascend environment variables."""
        import os
        os.environ['DEVICE_ID'] = str(device_id)
    
    def synchronize(self) -> None:
        """Synchronize Ascend device."""
        try:
            torch.npu.synchronize()
        except AttributeError:
            # If torch_npu is not available, skip synchronization
            pass
    
    def get_profiler(self) -> Optional[Any]:
        """Get Ascend profiler (msprof)."""
        # Profiler is handled in kernel_verifier, not here
        return None
    
    def get_device_string(self, device_id: int) -> str:
        """Get Ascend device string."""
        return f"npu:{device_id}"
    
    def validate_arch(self, arch: str) -> bool:
        """Validate Ascend architecture."""
        supported_archs = ["ascend910b1", "ascend910b2", "ascend910b2c", 
                          "ascend910b3", "ascend910b4", "ascend310p3"]
        return arch in supported_archs

