#!/usr/bin/env python3
# coding: utf-8
# Copyright 2020 Huawei Technologies Co., Ltd
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

"""save gpu param"""
import os
import fcntl
import hashlib
import akg.tvm


@akg.tvm.register_func
def dump_cuda_meta(code, ptx, thread_info):
    """
    Function for dumping cuda meta.

    Args:
        code: gpu code.
        ptx: ptx code.
        thread_info: thread info, written to json file.
    """
    # kernel name
    kernel_name = code.split("_kernel")[0].split(" ")[-1]

    # sha256 of ptx
    sha256 = hashlib.sha256()
    sha256.update(ptx.encode("utf-8"))
    hash_str = sha256.hexdigest()

    # thread info
    thread_info_dict = {
        "blockIdx.x": "1",
        "blockIdx.y": "1",
        "blockIdx.z": "1",
        "threadIdx.x": "1",
        "threadIdx.y": "1",
        "threadIdx.z": "1"
    }
    for thread_tag in thread_info_dict.keys():
        if thread_tag in thread_info:
            if isinstance(thread_info[thread_tag], int):
                thread_info_dict[thread_tag] = str(thread_info[thread_tag])
            elif isinstance(thread_info[thread_tag], akg.tvm.expr.IntImm):
                thread_info_dict[thread_tag] = str(thread_info[thread_tag].value)

    meta_path = "./cuda_meta_" + str(os.getpid()) + "/"
    cuda_path = os.path.realpath(meta_path)
    if not os.path.isdir(cuda_path):
        os.makedirs(cuda_path)

    # save ptx file to cuda meta
    ptx_file = os.path.realpath(meta_path + kernel_name + ".ptx")
    if os.path.exists(ptx_file):
        os.remove(ptx_file)
    with open(ptx_file, "at") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.seek(0, 2)
        if f.tell() == 0:
            f.write(ptx)

    # modify the file permisson to 400
    os.chmod(ptx_file, 0o400)

    # save json file to cuda meta
    json_file = os.path.realpath(meta_path + kernel_name + ".json")
    if os.path.exists(json_file):
        os.remove(json_file)
    with os.fdopen(os.open(json_file, os.O_WRONLY | os.O_CREAT, 0o400), 'w') as fo:
        fo.write("{\n")
        fo.write('"kernelName" : ' + '"' + kernel_name + "_kernel0" + '",\n')
        fo.write('"blockIdx.x" : ' + thread_info_dict["blockIdx.x"] + ',\n')
        fo.write('"blockIdx.y" : ' + thread_info_dict["blockIdx.y"] + ',\n')
        fo.write('"blockIdx.z" : ' + thread_info_dict["blockIdx.z"] + ',\n')
        fo.write('"threadIdx.x" : ' + thread_info_dict["threadIdx.x"] + ',\n')
        fo.write('"threadIdx.y" : ' + thread_info_dict["threadIdx.y"] + ',\n')
        fo.write('"threadIdx.z" : ' + thread_info_dict["threadIdx.z"] + ',\n')
        fo.write('"sha256" : ' + '"' + hash_str + '"\n')
        fo.write("}\n")
