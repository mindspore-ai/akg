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
""" akg launch and compile utils """
import ctypes
import logging
import subprocess
import numpy as np

from akg.message import get_npucompiler_path
from akg import akgAscendLaunch
from akg.utils.dynamic_utils import get_device_shape


def ascend_compile(input_file, output_so_path):
    """ using bisheng-compile """
    bishengir_compile_path = get_npucompiler_path()
    compile_cmd = [
        bishengir_compile_path,
        input_file,
        "-enable-hfusion-compile=true",
        "-enable-hivm-compile=true",
        "-enable-bin-relocation=false",
        "-block-dim=40",
        "-enable-auto-multi-buffer=true",
        "-o",
        output_so_path
    ]
    logging.info("exec command: %s", compile_cmd)
    subprocess.run(
        compile_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )


def transform_data_to_ascend(
    data,
    kernel_name,
    output_indexes,
    is_dyn_shape=False,
    backend="ascend",
    is_profile_params=False
):
    """ transform tensor input data to ctypes for ascend """
    data_ctypes = []
    if len(data) == 0:
        # dynamic shape info cannot generate inputs while compilation
        return data_ctypes

    device_shape, _, _ = get_device_shape(
        data, kernel_name, is_dyn_shape and not is_profile_params
    )

    output_idx_set = []
    for output_idx in output_indexes:
        if output_idx >= 0:
            output_idx_set.append(output_idx)
        else:
            output_idx_set.append(output_idx + len(data))
    output_idx_set = set(output_idx_set)
    for data_idx, d in enumerate(data):
        data_shape = np.array(device_shape[data_idx])
        data_bytes = d.nbytes
        is_numpy_bf16 = False
        is_numpy_output = False
        if isinstance(d, int):
            data_ctypes.append(ctypes.c_int(d))
        elif isinstance(d, np.ndarray):
            if data_idx in output_idx_set:
                is_numpy_output = True
            if d.dtype.name == "bfloat16":
                d = d.astype(np.float32)
                data[data_idx] = d
                is_numpy_bf16 = True

        ascend_tensor_obj = akgAscendLaunch.AscendTensorObjStructPyTorch()
        ascend_tensor_obj.tensor_info = d
        ascend_tensor_obj.shape_info = data_shape
        ascend_tensor_obj.nbytes = data_bytes
        ascend_tensor_obj.is_output = is_numpy_output
        ascend_tensor_obj.is_bf16 = is_numpy_bf16
        data_ctypes.append(ascend_tensor_obj)

    return data_ctypes


def launch(
    output_so_dir,
    kernel_name,
    device_id,
    is_dyn_shape,
    *input_for_mod_ctypes
):
    """ launch .so file by akg_ascend_backend """
    akgAscendLaunch.akg_ascend_run(
        output_so_dir,
        kernel_name,
        device_id,
        is_dyn_shape,
        *input_for_mod_ctypes
    )
