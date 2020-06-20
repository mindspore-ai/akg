#!/usr/bin/env python3
# coding: utf-8
# Copyright 2019 Huawei Technologies Co., Ltd
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

"""Runtime function related hooks"""
from __future__ import absolute_import as _abs
import os
import json
import hashlib
import akg.tvm
from akg.backend import cce_conf as cceconf

def write_code(js_dict, fname):
    if os.path.exists(fname):
        os.remove(fname)
    with os.fdopen(os.open(fname, os.O_WRONLY | os.O_CREAT, 0o400), 'w') as f:
        json.dump(js_dict, f,
                  sort_keys=True, indent=4, separators=(',', ':'))
# block_dim: cpu num,default value is 1.
@akg.tvm.register_func
def tvm_callback_cce_postproc(code, block_dim=1):
    """
    Function for dumping json datas from cce code.

    Args:
        code: cce code.
        block_dim: Default: 1.

    Returns:
        code.
    """
    kernel_name = code.split("_kernel")[0].split(" ")[-1]
    is_aicpu = False
    if "__aicore__" in code:
        title_dict = {"magic": "RT_DEV_BINARY_MAGIC_ELF"}
    elif "__aicpu__" in code:
        title_dict = {"magic": "RT_DEV_BINARY_MAGIC_ELF_AICPU"}
        is_aicpu = True
    elif "aarch64-hisilicon-cce" in code:
        title_dict = {"magic": "RT_DEV_BINARY_MAGIC_ELF_AICPU"}
        is_aicpu = True
        file_name = "kernel_meta/" + kernel_name[1:] + ".json"
    else:
        title_dict = dict()
    title_dict["blockDim"] = block_dim

    # bin file without suffix
    bin_file_name = ""
    bin_file_suffix = ".o"
    # for aicpu support os only
    cce_product_params = cceconf.CceProductParams()
    aicpu_support_os = cce_product_params.get_params_("Compiler_aicpu_support_os")
    bin_file_name = kernel_name
    if is_aicpu and aicpu_support_os:
        bin_file_name = "lib" + bin_file_name
        bin_file_suffix = ".so"
    if cce_product_params.enable_aicpuos:
        # new parameters in aicpuos feature
        title_dict["kernelName"] = kernel_name + "_kernel0"
        title_dict["binFileSuffix"] = bin_file_suffix
        title_dict["binFileName"] = bin_file_name

    # the op json file used by domi
    file_name = "kernel_meta/" + kernel_name + ".json"

    kernel_file_name = "kernel_meta/" + bin_file_name + bin_file_suffix
    buf_size = 64 * 1024  # once read 64kb
    sha256 = hashlib.sha256()
    with open(kernel_file_name, 'rb') as kf:
        while True:
            data = kf.read(buf_size)
            if not data:
                break
            sha256.update(data)
    title_dict["sha256"] = sha256.hexdigest()

    load_dict = {}
    if not os.path.exists("kernel_meta"):
        try:
            os.mkdir("kernel_meta")
        except OSError as err:
            # 17, OSError: [Errno 17] File exists
            if err.errno == 17:
                pass
            else:
                raise err
    else:
        fname = "kernel_meta/" + kernel_name + "wk.json"
        if os.path.exists(fname):
            with open(fname, "r") as f:
                load_dict = json.load(f)
            os.remove(fname)

    final_dict = title_dict.copy()
    final_dict.update(load_dict)
    write_code(final_dict, file_name)

    return code
