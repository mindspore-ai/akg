#!/usr/bin/env python3
# coding: utf-8
# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

"""save ascend param"""
from __future__ import absolute_import as _abs
import os
import json
import hashlib
import logging
import akg.tvm
from akg.global_configs import get_kernel_meta_path
from akg.utils.util import parse_workspace, write_code


@akg.tvm.register_func
def tvm_callback_cce_postproc(code, block_dim=1, workspace=None):
    """Function for dumping ascend meta."""
    if "__aicore__" in code:
        title_dict = {"magic": "RT_DEV_BINARY_MAGIC_ELF"}
    else:
        logging.warning("__aicore__ not found in code.")
        title_dict = dict()

    # kernel name
    kernel_name = code.split("_kernel")[0].split(" ")[-1]
    title_dict["kernelName"] = kernel_name + "_kernel0"

    # thread info
    title_dict["blockDim"] = block_dim

    # bin file info
    bin_file_suffix = ".o"
    title_dict["binFileSuffix"] = bin_file_suffix

    bin_file_name = kernel_name
    title_dict["binFileName"] = bin_file_name

    # sha256
    buf_size = 64 * 1024  # once read 64kb
    root_path = get_kernel_meta_path()
    kernel_file_name = root_path + bin_file_name + bin_file_suffix
    sha256 = hashlib.sha256()
    with open(kernel_file_name, 'rb') as kf:
        while True:
            data = kf.read(buf_size)
            if not data:
                break
            sha256.update(data)
    title_dict["sha256"] = sha256.hexdigest()

    # workspace
    workspace_dict = parse_workspace(workspace)
    if workspace_dict is not None:
        title_dict["workspace"] = workspace_dict

    load_dict = {}
    if not os.path.exists(get_kernel_meta_path()):
        try:
            os.mkdir(root_path)
        except OSError as err:
            # 17, OSError: [Errno 17] File exists
            if err.errno == 17:
                pass
            else:
                raise err
    else:
        fname = root_path + kernel_name + "wk.json"
        if os.path.exists(fname):
            with open(fname, "r") as f:
                load_dict = json.load(f)
            os.remove(fname)

    final_dict = title_dict.copy()
    final_dict.update(load_dict)

    json_file = root_path + kernel_name + ".json"
    write_code(final_dict, json_file)

    return code
