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

"""save cpu param"""
import os
import hashlib
import akg.tvm
from akg.global_configs import get_kernel_meta_path
from akg.utils.util import write_code


@akg.tvm.register_func
def dump_cpu_meta(mod, kernel_name):
    """
    Function for dumping cpu meta.

    Args:
        mod: the module code of cpu.
    """

    title_dict = dict()

    # kernel name
    code = mod.get_source()
    title_dict["kernelName"] = kernel_name + "_kernel"

    #thread number
    thread_num = "null"
    title_dict["threadNumber"] = thread_num

    #meta path
    path_name = get_kernel_meta_path()
    meta_path = os.path.realpath(path_name)
    if not os.path.isdir(meta_path):
        os.makedirs(meta_path, exist_ok=True)

    # save libraries to kernel meta
    obj_file = os.path.join(meta_path, kernel_name + '.o')
    lib_file = os.path.join(meta_path, kernel_name + '.so')
    mod.save(obj_file, 'k')
    mod.export_library(lib_file)

    # sha256 of files
    obj_sha256 = hashlib.sha256()
    lib_sha256 = hashlib.sha256()
    with open(obj_file, 'rb') as f:
        obj_sha256.update(f.read())
    with open(lib_file, 'rb') as f:
        lib_sha256.update(f.read())
    obj_hash_str = obj_sha256.hexdigest()
    lib_hash_str = lib_sha256.hexdigest()
    title_dict["objSha256"] = obj_hash_str
    title_dict["sha256"] = lib_hash_str

    # save json file to kernel meta
    json_file = os.path.join(meta_path, kernel_name + ".json")
    write_code(title_dict, json_file)
