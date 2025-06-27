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

"""
大模型推理痕迹追踪模块
"""
import os
from ai_kernel_generator.core.utils import Record, ActionType
from ai_kernel_generator.utils.common_utils import ParserFactory
        
class Trace:
    """
    大模型推理痕迹追踪类，用于记录大模型的推理过程。
    """

    def __init__(self, op_name, task_id, log_dir:str):
        self.op_name = op_name
        self.task_id = task_id
        self.log_dir = log_dir
        self.code_parser = ParserFactory.get_code_parser()

        self.trace_list = []  # 存储追踪记录的列表
        self.base_doc = {} # 存储基础文档

    def save_parameters_to_files(self, action_type: ActionType, params: list):
        """统一保存参数到文件的私有方法"""
        expanded_log_dir = os.path.expanduser(self.log_dir)
        target_dir = os.path.join(expanded_log_dir, self.op_name)
        os.makedirs(target_dir, exist_ok=True)

        base_name = f"I{self.task_id}_S{len(self.trace_list):02d}_{self.op_name}_{action_type.value}_"

        for param_name, content in params:
            file_path = os.path.join(target_dir, f"{base_name}{param_name}.txt")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(str(content))

    def insert_designer_or_coder_record(self, res:str, prompt:str, reasoning:str, action_type:ActionType) -> None:
        """
        插入设计器或编码器的记录。

        Args:
            res (str): 结果。
            prompt (str): 提示。
            reasoning (str): 推理过程。
            action_type (ActionType): 操作类型。
        """
        if action_type not in [ActionType.DO_DESIGNER, ActionType.DO_CODER, ActionType.FIX_DESIGNER, ActionType.FIX_CODER]:
            raise ValueError("action_type must be Designer or Coder")
        
        record = Record(action_type=action_type, result=res, prompt=prompt, reasoning=reasoning)
        self.trace_list.append(record)

        try:
            parsed_result = ParserFactory.robust_parse(res, self.code_parser)
            res_code = parsed_result.code
            res_description = parsed_result.description
            res_to_save = res_code + "\n\n\n" + "'''" + "\n" + res_description + "\n" + "'''"
        except:
            res_to_save = res

        self.save_parameters_to_files(action_type, [
            ('result', res_to_save),
            ('prompt', prompt),
            ('reasoning', reasoning)
        ])
        

    def insert_tester_record(self, verify_res:str, verify_log:str, profile:str, action_type=ActionType.DO_TESTER) -> None:
        """
        插入测试器的记录。

        Args:
            verify_log (str): 验证日志。
            profile (str): 性能数据。
            action_type (ActionType, optional): 操作类型。默认为ActionType.DO_TESTER。
        """
        if action_type != ActionType.DO_TESTER:
            raise ValueError("action_type must be Tester")

        record = Record(action_type=action_type, result=verify_res, error_log=verify_log, profile=profile)
        self.trace_list.append(record)
        self.save_parameters_to_files(action_type, [
            ('error_log', verify_log)
        ])

    def insert_conductor_record(self, res:str, prompt:str, reasoning:str, action_type:ActionType) -> None:
        expanded_log_dir = os.path.expanduser(self.log_dir)
        target_dir = os.path.join(expanded_log_dir, self.op_name, "conductor")
        os.makedirs(target_dir, exist_ok=True)

        if action_type in [ActionType.DO_DESIGNER, ActionType.FIX_DESIGNER]:
            base_name = f"I{self.task_id}_S{len(self.trace_list):02d}_{self.op_name}_CheckDesigner_"
        elif action_type in [ActionType.DO_CODER, ActionType.FIX_CODER]:
            base_name = f"I{self.task_id}_S{len(self.trace_list):02d}_{self.op_name}_CheckCoder_"
        elif action_type == ActionType.DO_TESTER:
            base_name = f"I{self.task_id}_S{len(self.trace_list):02d}_{self.op_name}_AnalyzeError_"
        params = [
            ('result', res),
            ('prompt', prompt),
            ('reasoning', reasoning)
        ]
        for param_name, content in params:
            if not content:
                continue
            file_path = os.path.join(target_dir, f"{base_name}{param_name}.txt")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(str(content))