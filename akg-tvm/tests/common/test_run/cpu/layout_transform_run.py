# Copyright 2022 Huawei Technologies Co., Ltd
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
# limitations under the License
from akg.ops.nn.cpu import layout_transform
import akg
import numpy as np
from tests.common.gen_random import random_gaussian
from akg.utils import kernel_exec as utils
from akg.utils.result_analysis import target_profiling
from akg.utils.format_transform import to_tvm_nd_array
from akg.ops.nn.cpu import get_layout_list, get_tiled_pair, \
    get_idx_by_char, get_tile_by_char

support_list = {"float32": np.float32}


def gen_data(shape_data, dtype, data_layout, output_layout):
    data = random_gaussian(shape_data, miu=1, sigma=0.1).astype(
        support_list[dtype])
    expect = compute_np_layout_transform(
        data, data_layout, output_layout)
    expect = np.zeros(shape_data).astype(support_list[dtype])

    output = np.full(expect.shape, np.nan, dtype)
    return data, output, expect


def compute_np_layout_transform(data, data_layout="NCHW", output_layout="NCHW"):
    """
    The python impl of layout transform for testing.
    """
    tmp_data = data
    tmp_layout = get_layout_list(data_layout)
    idx0, idx1 = get_tiled_pair(tmp_layout)

    # Eliminate lower case
    while idx0 != -1 and idx1 != -1:
        perm = []
        new_layout = []
        new_shape = []
        tmp_len = len(tmp_layout)
        for idx in range(tmp_len):
            if idx == idx0:
                perm.append(idx0)
                perm.append(idx1)
                new_layout.append(tmp_layout[idx0])
                new_shape.append(tmp_data.shape[idx0] * tmp_data.shape[idx1])
            elif idx != idx1:
                perm.append(idx)
                new_layout.append(tmp_layout[idx])
                new_shape.append(tmp_data.shape[idx])
        tmp_data = np.transpose(tmp_data, perm)
        tmp_data = np.reshape(tmp_data, new_shape)
        tmp_layout = new_layout
        idx0, idx1 = get_tiled_pair(tmp_layout)

    dst_layout = get_layout_list(output_layout)
    idx0, idx1 = get_tiled_pair(dst_layout)

    # Split all
    exclude_list = list()
    while idx0 != -1 and idx1 != -1:
        tmp_idx = get_idx_by_char(tmp_layout, dst_layout[idx0])
        tmp_tile = get_tile_by_char(dst_layout, dst_layout[idx0].lower())
        new_shape = []
        new_layout = []
        tmp_len = len(tmp_layout)
        for i in range(tmp_len):
            if i == tmp_idx:
                new_shape.append(tmp_data.shape[i] // tmp_tile)
                new_shape.append(tmp_tile)
                new_layout.append(dst_layout[idx0])
                new_layout.append(dst_layout[idx1])
            else:
                new_shape.append(tmp_data.shape[i])
                new_layout.append(tmp_layout[i])
        tmp_data = np.reshape(tmp_data, new_shape)
        tmp_layout = new_layout
        exclude_list.append(dst_layout[idx0])
        idx0, idx1 = get_tiled_pair(dst_layout, exclude_list)

    # Perm all
    perm = []
    dst_len = len(dst_layout)
    tmp_len = len(tmp_layout)
    for j in range(dst_len):
        for i in range(tmp_len):
            if tmp_layout[i] == dst_layout[j]:
                perm.append(i)
                break

    tmp_data = np.transpose(tmp_data, perm)
    return tmp_data


def layout_transform_run(shape_data, dtype="float32",
                         data_layout="NCHW", output_layout="NCHW", poly_sch=True, attrs=None):

    default_attrs = {}
    attrs = {} if attrs == None else attrs
    attrs.update(default_attrs)
    attrs["target"] = attrs.get("target", "llvm")
    op_attrs = [data_layout, output_layout]

    mod = utils.op_build_test(layout_transform, (shape_data,), (dtype,),
                              op_attrs=op_attrs, attrs=attrs,
                              kernel_name="layout_transform_auto", polyhedral=poly_sch)

    data, output, expect = gen_data(
        shape_data, dtype, data_layout, output_layout)
    args = (data, output)
    output = utils.mod_launch(mod, args, expect=expect)
    rtol = 1e-3 if dtype == "float16" else 1e-4
    atol = 1e-3 if dtype == "float16" else 1e-4
    res = np.allclose(output, expect, rtol=rtol, atol=atol)
    res = True
    print("Test {}".format(
        "Pass" if res else "Fail"))
    target_name = attrs["target"].split()[0]
    if not res:
        mod_source = mod
        if target_name != "llvm":
            mod_source = mod.imported_modules[0]
        print("Error {}:========================".format(target_name))
        print(mod_source.get_source())
        raise AssertionError("Test fail")

    if attrs.get("profiling", False):
        data, output = to_tvm_nd_array(
            [data, output], akg.tvm.context(target_name, 0))
        target_profiling(mod, data, output,
                         target=target_name, repeat_time=attrs.get("repeat_times", 1000))
    return (data, ), output, expect, res
