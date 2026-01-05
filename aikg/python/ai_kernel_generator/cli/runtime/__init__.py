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

from .message_sender import (
    register_message_sender,
    send_message,
    unregister_message_sender,
)

__all__ = [
    "LocalExecutor",
    "register_message_sender",
    "send_message",
    "unregister_message_sender",
]

def __getattr__(name):
    if name == "LocalExecutor":
        from .local_executor import LocalExecutor
        return LocalExecutor
    raise AttributeError(name)
