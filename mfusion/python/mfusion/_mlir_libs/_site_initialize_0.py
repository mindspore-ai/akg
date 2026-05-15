# Copyright 2026 Huawei Technologies Co., Ltd
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
MLIR context initialization for mfusion package.

This module is automatically imported when the mfusion._mlir_libs package is loaded.
It provides a context_init_hook function that is called whenever a new MLIR
context is created, ensuring all necessary dialects are registered.
"""

# pylint: disable=import-outside-toplevel
# Reason: Must import here to ensure proper initialization order

def context_init_hook(context):
    """
    Initialize MLIR context with all required dialects.

    This function is called automatically by the MLIR Python bindings
    when creating a new context.

    Args:
        context: MlirContext object to initialize
    """

    from ._mfusion import register_mfuse_dialect, register_dvm_dialect

    register_mfuse_dialect(context)
    register_dvm_dialect(context)

    # Allow unregistered dialects for flexibility
    context.allow_unregistered_dialects = True
