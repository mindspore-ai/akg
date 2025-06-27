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

from typing import Union

from ai_kernel_generator.core.agent.swft_coder import SWFTCoder
from ai_kernel_generator.core.agent.triton_coder import TritonCoder


class CoderFactory:
    """
    Coder工厂类，根据impl_type创建相应的coder实例
    """

    # 支持的coder类型映射
    CODER_TYPES = {
        "swft": SWFTCoder,
        "triton": TritonCoder,
    }

    @classmethod
    def create_coder(cls, op_name: str, task_desc: str, model_config: dict,
                     impl_type: str, framework: str) -> Union[SWFTCoder, TritonCoder]:
        """
        根据impl_type创建相应的coder实例

        Args:
            op_name (str): 算子名称
            task_desc (str): 算子功能描述
            model_config (dict): 模型配置
            impl_type (str): 实现类型，支持 "swft" 和 "triton"
            framework (str): 框架类型，支持 "torch" 和 "mindspore"
        Returns:
            Union[SWFTCoder, TritonCoder]: 对应的coder实例
  
        Raises:
            ValueError: 当impl_type不受支持时
        """
        if impl_type not in cls.CODER_TYPES:
            supported_types = ", ".join(cls.CODER_TYPES.keys())
            raise ValueError(f"不支持的impl_type: {impl_type}。支持的类型有: {supported_types}")

        coder_class = cls.CODER_TYPES[impl_type]

        # 创建并返回相应的coder实例
        return coder_class(
            op_name=op_name,
            task_desc=task_desc,
            model_config=model_config,
            impl_type=impl_type,
            framework=framework
        )
