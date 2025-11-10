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

import pytest
from ai_kernel_generator.core.utils import check_task_config


class TestTaskConfig:
    """测试任务配置验证函数"""

    @pytest.mark.level0
    def test_valid_configs(self):
        """测试有效的配置组合"""
        valid_configs = [
            ("mindspore", "triton_ascend", "ascend", "ascend910b4"),
            ("mindspore", "swft", "ascend", "ascend310p3"),
            ("torch", "triton_ascend", "ascend", "ascend910b4"),
            ("torch", "swft", "ascend", "ascend310p3"),
            ("torch", "triton_cuda", "cuda", "a100"),
            ("numpy", "swft", "ascend", "ascend310p3"),
        ]

        for framework, dsl, backend, arch in valid_configs:
            try:
                check_task_config(framework, backend, arch, dsl)
                print(f"有效配置: {framework} + {backend} + {arch} + {dsl}")
            except ValueError as e:
                pytest.fail(f"有效配置验证未通过: {framework} + {backend} + {arch} + {dsl}, 错误: {e}")

    @pytest.mark.level0
    def test_invalid_framework(self):
        """测试无效的框架"""
        invalid_configs = [
            ("invalid_framework", "triton_ascend", "ascend", "ascend910b4"),
            ("pytorch", "triton_cuda", "cuda", "a100"),  # 应该是torch
            ("mindspore_old", "swft", "ascend", "ascend310p3"),
        ]

        for framework, dsl, backend, arch in invalid_configs:
            with pytest.raises(ValueError, match="Unsupported framework"):
                check_task_config(framework, backend, arch, dsl)
                print(f"正确捕获无效框架错误: {framework}")

    @pytest.mark.level0
    def test_invalid_backend(self):
        """测试无效的后端"""
        invalid_configs = [
            ("mindspore", "triton_ascend", "invalid_backend", "ascend910b4"),
            ("torch", "triton_ascend", "invalid_backend", "ascend910b4"),
            ("numpy", "swft", "invalid_backend", "ascend310p3"),
        ]

        for framework, dsl, backend, arch in invalid_configs:
            with pytest.raises(ValueError, match="does not support backend"):
                check_task_config(framework, backend, arch, dsl)
                print(f"正确捕获无效后端错误: {framework} + {backend}")

    @pytest.mark.level0
    def test_invalid_arch(self):
        """测试无效的架构"""
        invalid_configs = [
            ("mindspore", "triton_ascend", "ascend", "invalid_arch"),
            ("torch", "triton_cuda", "cuda", "invalid_arch"),
            ("numpy", "swft", "ascend", "invalid_arch"),
        ]

        for framework, dsl, backend, arch in invalid_configs:
            with pytest.raises(ValueError, match="does not support arch"):
                check_task_config(framework, backend, arch, dsl)
                print(f"正确捕获无效架构错误: {backend} + {arch}")

    @pytest.mark.level0
    def test_invalid_dsl(self):
        """测试无效的实现类型"""
        invalid_configs = [
            ("mindspore", "invalid_impl", "ascend", "ascend910b4"),
            ("torch", "invalid_impl", "cuda", "a100"),
            ("numpy", "invalid_impl", "ascend", "ascend310p3"),
        ]

        for framework, dsl, backend, arch in invalid_configs:
            with pytest.raises(ValueError, match="does not support dsl"):
                check_task_config(framework, backend, arch, dsl)
                print(f"正确捕获无效实现类型错误: {dsl}")

    @pytest.mark.level0
    def test_mismatched_combinations(self):
        """测试不匹配的组合"""
        mismatched_configs = [
            # ascend910b4只支持triton_ascend，但使用了swft
            ("mindspore", "swft", "ascend", "ascend910b4"),
            ("torch", "swft", "ascend", "ascend910b4"),
            # cuda只支持triton_cuda，但使用了swft
            ("torch", "swft", "cuda", "a100"),
            # numpy只支持swft，但使用了triton_ascend
            ("numpy", "triton_ascend", "ascend", "ascend310p3"),
        ]

        for framework, dsl, backend, arch in mismatched_configs:
            with pytest.raises(ValueError, match="does not support dsl"):
                check_task_config(framework, backend, arch, dsl)
                print(f"正确捕获不匹配组合错误: {framework} + {backend} + {arch} + {dsl}")

    @pytest.mark.level0
    def test_nonexistent_combinations(self):
        """测试不存在的组合"""
        nonexistent_configs = [
            # mindspore不支持cuda
            ("mindspore", "triton_cuda", "cuda", "a100"),
            # numpy不支持cuda
            ("numpy", "swft", "cuda", "a100"),
            # torch不支持某些不存在的组合
            ("torch", "triton_ascend", "ascend", "ascend310p3"),  # torch的ascend310p3只支持swft
        ]

        for framework, dsl, backend, arch in nonexistent_configs:
            with pytest.raises(ValueError):
                check_task_config(framework, backend, arch, dsl)
                print(f"正确捕获不存在组合错误: {framework} + {backend} + {arch} + {dsl}")

    @pytest.mark.level0
    def test_edge_cases(self):
        """测试边界情况"""
        # 空字符串
        with pytest.raises(ValueError):
            check_task_config("", "ascend", "ascend910b4", "triton_ascend")

        # None值
        with pytest.raises(ValueError):
            check_task_config(None, "ascend", "ascend910b4", "triton_ascend")

        # 大小写混合（应该不区分大小写，但这里测试一下）
        with pytest.raises(ValueError):
            check_task_config("MindSpore", "ascend", "ascend910b4", "triton_ascend")

        print("正确捕获边界情况错误")

    @pytest.mark.level0
    def test_all_valid_combinations(self):
        """测试所有有效的组合"""
        # 根据VALID_CONFIGS映射表测试所有有效组合
        all_valid_combinations = [
            # mindspore
            ("mindspore", "triton_ascend", "ascend", "ascend910b4"),
            ("mindspore", "swft", "ascend", "ascend310p3"),
            # torch
            ("torch", "triton_ascend", "ascend", "ascend910b4"),
            ("torch", "swft", "ascend", "ascend310p3"),
            ("torch", "triton_cuda", "cuda", "a100"),
            # numpy
            ("numpy", "swft", "ascend", "ascend310p3"),
        ]

        for framework, dsl, backend, arch in all_valid_combinations:
            try:
                check_task_config(framework, backend, arch, dsl)
                print(f"有效组合: {framework} + {backend} + {arch} + {dsl}")
            except ValueError as e:

                pytest.fail(f"有效组合验证未通过: {framework} + {backend} + {arch} + {dsl}, 错误: {e}")
