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

"""Unit tests for Backend Adapters."""

import os
import pytest
from ai_kernel_generator.core.verifier.adapters.factory import get_backend_adapter


class TestBackendAdapterCuda:
    """Test CUDA Backend Adapter."""
    
    def test_setup_environment(self):
        """Test environment setup."""
        adapter = get_backend_adapter("cuda")
        original_value = os.environ.get('CUDA_VISIBLE_DEVICES')
        try:
            adapter.setup_environment(0, "a100")
            assert os.environ.get('CUDA_VISIBLE_DEVICES') == "0"
        finally:
            if original_value is None:
                os.environ.pop('CUDA_VISIBLE_DEVICES', None)
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = original_value
    
    def test_get_device_string(self):
        """Test device string generation."""
        adapter = get_backend_adapter("cuda")
        device_str = adapter.get_device_string(0)
        assert device_str == "cuda:0"
    
    def test_validate_arch(self):
        """Test architecture validation."""
        adapter = get_backend_adapter("cuda")
        assert adapter.validate_arch("a100") is True
        assert adapter.validate_arch("v100") is True
        assert adapter.validate_arch("rtx3090") is True
        assert adapter.validate_arch("invalid") is False


class TestBackendAdapterAscend:
    """Test Ascend Backend Adapter."""
    
    def test_setup_environment(self):
        """Test environment setup."""
        adapter = get_backend_adapter("ascend")
        original_value = os.environ.get('DEVICE_ID')
        try:
            adapter.setup_environment(0, "ascend910b4")
            assert os.environ.get('DEVICE_ID') == "0"
        finally:
            if original_value is None:
                os.environ.pop('DEVICE_ID', None)
            else:
                os.environ['DEVICE_ID'] = original_value
    
    def test_get_device_string(self):
        """Test device string generation."""
        adapter = get_backend_adapter("ascend")
        device_str = adapter.get_device_string(0)
        assert device_str == "npu:0"
    
    def test_validate_arch(self):
        """Test architecture validation."""
        adapter = get_backend_adapter("ascend")
        assert adapter.validate_arch("ascend910b4") is True
        assert adapter.validate_arch("ascend910b1") is True
        assert adapter.validate_arch("ascend310p3") is True
        assert adapter.validate_arch("invalid") is False


class TestBackendAdapterCpu:
    """Test CPU Backend Adapter."""
    
    def test_setup_environment(self):
        """Test environment setup (should be no-op)."""
        adapter = get_backend_adapter("cpu")
        # Should not raise any exception
        adapter.setup_environment(0, "x86_64")
    
    def test_get_device_string(self):
        """Test device string generation."""
        adapter = get_backend_adapter("cpu")
        device_str = adapter.get_device_string(0)
        assert device_str == "cpu"
    
    def test_validate_arch(self):
        """Test architecture validation."""
        adapter = get_backend_adapter("cpu")
        assert adapter.validate_arch("x86_64") is True
        assert adapter.validate_arch("aarch64") is True
        assert adapter.validate_arch("invalid") is False


class TestBackendAdapterFactory:
    """Test Backend Adapter Factory."""
    
    def test_get_backend_adapter_cuda(self):
        """Test getting CUDA adapter."""
        adapter = get_backend_adapter("cuda")
        assert adapter is not None
        assert adapter.__class__.__name__ == "BackendAdapterCuda"
    
    def test_get_backend_adapter_ascend(self):
        """Test getting Ascend adapter."""
        adapter = get_backend_adapter("ascend")
        assert adapter is not None
        assert adapter.__class__.__name__ == "BackendAdapterAscend"
    
    def test_get_backend_adapter_cpu(self):
        """Test getting CPU adapter."""
        adapter = get_backend_adapter("cpu")
        assert adapter is not None
        assert adapter.__class__.__name__ == "BackendAdapterCpu"
    
    def test_get_backend_adapter_invalid(self):
        """Test getting invalid backend adapter."""
        with pytest.raises(ValueError, match="Unsupported backend"):
            get_backend_adapter("invalid")

