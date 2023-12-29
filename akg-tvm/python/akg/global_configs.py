# Copyright 2021 Huawei Technologies Co., Ltd
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

"""global_configs"""
import akg.tvm
import os


@akg.tvm.register_func
def get_kernel_meta_path():
    kernel_meta_dir = os.getenv("KERNEL_META_DIR", default="akg_kernel_meta")
    return os.path.realpath(os.getenv('MS_COMPILER_CACHE_PATH', './')) + '/' + kernel_meta_dir + '/'


@akg.tvm.register_func
def get_dump_ir_flag():
    return 'MS_DEV_DUMP_IR'


@akg.tvm.register_func
def get_dump_code_flag():
    return 'MS_DEV_DUMP_CODE'
