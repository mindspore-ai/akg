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
集成测试模块

测试 FileSystemState 和 TraceSystem 在各算子生成模式下的集成：
- 单次生成模式 (coder_only, default, verifier_only, connect_all)
- TreeSearch 模式 (evolve, adaptive_search)
- 对话模式 (MainOpAgent)
"""
