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

"""Base class for backend adapters."""

from abc import ABC, abstractmethod
from typing import Optional, Any


class BackendAdapter(ABC):
    """Abstract base class for backend adapters.
    
    Backend adapters provide a unified interface for different hardware backends
    (CUDA, Ascend, CPU) to handle environment setup, synchronization, and profiling.
    """
    
    @abstractmethod
    def setup_environment(self, device_id: int, arch: str) -> None:
        """Setup environment variables.
        
        Args:
            device_id: Device ID
            arch: Architecture
        """
        pass
    
    @abstractmethod
    def synchronize(self) -> None:
        """Synchronize device (wait for computation to complete)."""
        pass
    
    @abstractmethod
    def get_profiler(self) -> Optional[Any]:
        """Get profiler object for performance analysis.
        
        Returns:
            Profiler object or None
        """
        pass
    
    @abstractmethod
    def get_device_string(self, device_id: int) -> str:
        """Get device string for logging.
        
        Args:
            device_id: Device ID
            
        Returns:
            str: Device string (e.g., "cuda:0", "npu:0", "cpu")
        """
        pass
    
    @abstractmethod
    def validate_arch(self, arch: str) -> bool:
        """Validate if architecture is supported.
        
        Args:
            arch: Architecture string
            
        Returns:
            bool: True if architecture is supported
        """
        pass

