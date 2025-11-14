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

"""CPU backend adapter."""

from typing import Optional, Any

from .base import BackendAdapter


class BackendAdapterCpu(BackendAdapter):
    """Adapter for CPU backend."""
    
    def setup_environment(self, device_id: int, arch: str) -> None:
        """Setup CPU environment (no-op)."""
        pass
    
    def synchronize(self) -> None:
        """Synchronize CPU (no-op)."""
        pass
    
    def get_profiler(self) -> Optional[Any]:
        """Get CPU profiler (time-based)."""
        return None
    
    def get_device_string(self, device_id: int) -> str:
        """Get CPU device string."""
        return "cpu"
    
    def validate_arch(self, arch: str) -> bool:
        """Validate CPU architecture."""
        supported_archs = ["x86_64", "aarch64"]
        return arch in supported_archs

