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

"""Base class for framework adapters."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Union


class FrameworkAdapter(ABC):
    """Abstract base class for framework adapters.
    
    Framework adapters provide a unified interface for different deep learning
    frameworks (PyTorch, MindSpore, NumPy) to handle device setup, input processing,
    output conversion, and other framework-specific operations.
    """
    
    @abstractmethod
    def get_import_statements(self) -> str:
        """Return import statements for the framework.
        
        Returns:
            str: Import statements as a string (e.g., "import torch\n")
        """
        pass
    
    @abstractmethod
    def get_framework_import(self, op_name: str, is_dynamic_shape: bool) -> str:
        """Return import statement for framework model and input functions.
        
        Args:
            op_name: Operator name
            is_dynamic_shape: Whether dynamic shape is used
            
        Returns:
            str: Import statement (e.g., "from {op_name}_torch import Model as FrameworkModel, get_inputs\n")
        """
        pass
    
    @abstractmethod
    def setup_device(self, backend: str, arch: str, device_id: int) -> Any:
        """Setup and return device object.
        
        Args:
            backend: Backend type (cuda, ascend, cpu)
            arch: Architecture (a100, ascend910b4, etc.)
            device_id: Device ID
            
        Returns:
            Device object (torch.device, str, or None)
        """
        pass
    
    @abstractmethod
    def process_input(self, x: Any, device: Any) -> Any:
        """Process input data and move to device if needed.
        
        Args:
            x: Input data (tensor, array, or other)
            device: Device object
            
        Returns:
            Processed input data
        """
        pass
    
    @abstractmethod
    def convert_to_numpy(self, tensor: Any) -> Any:
        """Convert framework tensor to numpy array.
        
        Args:
            tensor: Framework tensor
            
        Returns:
            numpy.ndarray
        """
        pass
    
    @abstractmethod
    def get_limit(self, dtype: Any) -> float:
        """Get precision limit for the given dtype.
        
        Args:
            dtype: Data type
            
        Returns:
            float: Precision limit
        """
        pass
    
    @abstractmethod
    def save_tensor(self, tensor: Any, bin_path: str) -> None:
        """Save tensor to binary file.
        
        Args:
            tensor: Framework tensor
            bin_path: Path to save binary file
        """
        pass
    
    @abstractmethod
    def load_tensor(self, bin_path: str, reference_tensor: Any) -> Any:
        """Load tensor from binary file.
        
        Args:
            bin_path: Path to binary file
            reference_tensor: Reference tensor for dtype and shape
            
        Returns:
            Framework tensor
        """
        pass
    
    @abstractmethod
    def set_seed(self, backend: Optional[str] = None) -> None:
        """Set random seed.
        
        Args:
            backend: Backend type (for framework-specific seed setting)
        """
        pass
    
    @abstractmethod
    def move_model_to_device(self, model: Any, device: Any) -> Any:
        """Move model to device.
        
        Args:
            model: Framework model
            device: Device object
            
        Returns:
            Model on device (may be same object if no-op)
        """
        pass
    
    @abstractmethod
    def get_tensor_type(self) -> type:
        """Get tensor type for the framework.
        
        Returns:
            type: Tensor type class
        """
        pass
    
    @abstractmethod
    def get_tensor_type_name(self) -> str:
        """Get tensor type name as string (full path).
        
        Returns:
            str: Full path to tensor type (e.g., "torch.Tensor", "np.ndarray", "ms.Tensor")
        """
        pass
    
    def get_dtype_mapping(self) -> Optional[dict]:
        """Get dtype mapping (only needed for MindSpore).
        
        Returns:
            dict or None: Dtype mapping dictionary
        """
        return None
    
    def get_binary_io_functions(self, op_name: str) -> str:
        """Get binary I/O functions code (save_tensor, load_tensor, gen_binary_data, load_binary_data).
        
        Args:
            op_name: Operator name
            
        Returns:
            str: Function definitions as string
        """
        tensor_type = self.get_tensor_type().__name__
        save_code = self._get_save_tensor_code(tensor_type)
        load_code = self._get_load_tensor_code(tensor_type)
        gen_binary_code = self._get_gen_binary_data_code(tensor_type, op_name)
        load_binary_code = self._get_load_binary_data_code(tensor_type, op_name)
        return save_code + load_code + gen_binary_code + load_binary_code
    
    def _get_save_tensor_code(self, tensor_type: str) -> str:
        """Get save_tensor function code."""
        pass
    
    def _get_load_tensor_code(self, tensor_type: str) -> str:
        """Get load_tensor function code."""
        pass
    
    def _get_gen_binary_data_code(self, tensor_type: str, op_name: str) -> str:
        """Get gen_binary_data function code."""
        pass
    
    def _get_load_binary_data_code(self, tensor_type: str, op_name: str) -> str:
        """Get load_binary_data function code."""
        pass
    
    def get_device_setup_code(self, backend: str, arch: str, device_id: int) -> str:
        """Get device setup code.
        
        Args:
            backend: Backend type
            arch: Architecture
            device_id: Device ID
            
        Returns:
            str: Device setup code
        """
        pass
    
    def get_process_input_code(self, backend: str, dsl: str) -> str:
        """Get process_input function code.
        
        Args:
            backend: Backend type
            dsl: DSL type
            
        Returns:
            str: process_input function code
        """
        pass
    
    def get_set_seed_code(self, backend: str) -> str:
        """Get set seed code.
        
        Args:
            backend: Backend type
            
        Returns:
            str: Set seed code
        """
        pass
    
    @abstractmethod
    def get_compare_code(self) -> str:
        """Get compare function code for this framework.
        
        Each framework should implement its own compare logic using native operations.
        
        Returns:
            str: Compare function code
        """
        pass
    
    @abstractmethod
    def get_compare_outputs_code(self) -> str:
        """Get code for comparing framework output and impl output.
        
        This code will be used in the template to compare outputs.
        
        Returns:
            str: Code for comparing outputs
        """
        pass

