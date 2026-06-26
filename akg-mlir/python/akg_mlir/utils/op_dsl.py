# Copyright 2023 Huawei Technologies Co., Ltd
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

"""Provides op numpy implementation, used to compare with akg-mlir output."""

import logging
import inspect
import itertools
from copy import deepcopy
import numpy as np
from .torch_mlir_utils import (
    TORCH_DTYPE_TO_NUMPY, format_py_value,
    gen_slice_scatter, gen_slice_tensor,
    gen_constant_pad_nd, gen_broadcast_to,
)


def get_attr(attr_desc, attr_type):
    """get op attr by type"""
    for attr in attr_desc:
        if attr["name"] == attr_type:
            return attr["value"]
    logging.warning("attr %s not found, please check.", attr_type)
    return []


def get_input(desc):
    """get input values"""
    value = desc.get('value', None)
    return value if value is not None else desc['tensor_name']


def get_gather_res(data, indices, out, updates, is_add):
    """compute gather result for gather_nd and tensor_scatter_add"""
    for i in range(indices.shape[0]):
        for j in range(out.shape[1]):
            index_read = [i, 0]
            index_write = [0, 0]
            inbound = True
            for k in range(0, int(indices.shape[-1])):
                temp_idx = indices[i, k]
                inbound = np.all(
                    (inbound, (temp_idx >= 0), (temp_idx < data.shape[k])))
                index_write[0] += int(temp_idx *
                                      data.strides[k] / data.itemsize)
                index_write[0] = int(index_write[0] / out.shape[1])
            index_read[1] = j
            if inbound:
                index_write[1] = j
                if is_add:
                    out[tuple(index_read)] += updates[tuple(index_write)]
                else:
                    out[tuple(index_read)] = updates[tuple(index_write)]
    return out


def gather_nd_np(data, indices):
    """numpy implementation of gather_nd"""
    data_shape = data.shape
    indices_shape = indices.shape
    new_indices = indices.reshape(-1, indices.shape[-1])
    out_shape = indices_shape[:-1] + data_shape[int(indices_shape[-1]):]
    out = np.zeros(out_shape, np.float32).reshape(new_indices.shape[0], -1)
    new_data = deepcopy(
        data).reshape(-1, int(np.prod(data_shape[int(indices_shape[-1]):])))
    out = get_gather_res(data, out, new_indices, new_data, is_add=False)
    return out.reshape(out_shape)


def tensor_scatter_add_np(data, indices, updates):
    """numpy implementation of tensor_scatter_add"""
    data_shape = data.shape
    indices_shape = indices.shape
    if indices.ndim > 1:
        new_indices = indices.reshape(-1, indices.shape[-1])
        out = deepcopy(data).reshape(-1,
                                     int(np.prod(data_shape[int(indices_shape[-1]):])))
    else:
        new_indices = indices.reshape(-1, 1)
        out = deepcopy(data).reshape(-1, int(np.prod(data_shape[1:])))
    new_updates = updates.reshape(new_indices.shape[0], -1)
    out = get_gather_res(data, out, new_indices, new_updates, is_add=True)
    return out.reshape(data_shape)


def gather_np(data, indices, axis):
    """numpy implementation of gather"""
    expect = np.take(data, indices, axis)
    return expect


def one_hot_np(data, axis, on_value, off_value, depth, dtype):
    """numpy implementation of one_hot"""
    shape = data.shape
    if axis < 0:
        axis += len(shape) + 1
    out_shape = list(shape)
    out_shape.insert(axis, depth)
    expect = np.full(out_shape, off_value)
    dims = []
    for dim in shape:
        dims.append(list(range(dim)))
    indexes = list(itertools.product(*tuple(dims)))
    indexes = [list(x) for x in indexes]
    for i, value in enumerate(data.flatten()):
        indexes[i].insert(axis, value)
    indexes = [tuple(x) for x in indexes]
    for loc in indexes:
        if loc[axis] >= 0:
            expect[loc] = on_value
    expect = expect.astype(dtype)
    return expect


def reduce_str(inputs, output, attr, op_type):
    """gen sum string"""
    axis = []
    keepdims = False
    axis_value = get_attr(attr, "axis")
    if not axis_value and len(inputs) == 2 and 'value' in inputs[1][0]:
        axis_value = inputs[1][0]['value']

    axis = list(axis_value) if isinstance(
        axis_value, (list, tuple)) else [axis_value]
    keepdims_value = get_attr(attr, "keep_dims")
    keepdims = keepdims_value if keepdims_value else keepdims

    in0 = get_input(inputs[0][0])
    name = output[0]['tensor_name']
    if not axis:
        s = (
            f"{name} = np.{op_type}({in0}.astype(np.float32) if {in0}.dtype "
            f"== np.float16 or {in0}.dtype.name == 'bfloat16' else {in0}, "
            f"keepdims={keepdims}).astype({in0}.dtype)"
        )
    else:
        s = (
            f"{name} = np.{op_type}({in0}.astype(np.float32) if {in0}.dtype "
            f"== np.float16 or {in0}.dtype.name == 'bfloat16' else {in0}, "
            f"axis=tuple({axis}), keepdims={keepdims})"
            f".astype({in0}.dtype); {name} = np.reshape({name}, {output[0]['shape']}) "
        )
    return s


def cast_str(inputs, output, attr):
    """gen cast string"""
    dst_type = output[0]["data_type"]
    tn = output[0]['tensor_name']
    inp = get_input(inputs[0][0])
    if dst_type == "bfloat16":
        if inputs[0][0].get('value', None) is not None:
            return f"{tn} = np.array({inp}).astype(bfloat16)"
        return f"{tn} = {inp}.astype(bfloat16)"
    if inputs[0][0].get('value', None) is not None:
        return f"{tn} = np.array({inp}).astype(np.{dst_type})"
    return f"{tn} = {inp}.astype(np.{dst_type})"


def broadcast_str(inputs, output, attr):
    """gen broadcast string"""
    dst_shape = None
    if attr is None:
        dst_shape = output[0]["shape"]
    else:
        dst_shape = get_attr(attr, "shape")
    s = (
        f"{output[0]['tensor_name']} = np.broadcast_to("
        f"{get_input(inputs[0][0])}, {dst_shape})"
    )
    return s


def transpose_str(inputs, output, attr):
    """gen transpose string"""
    axes = None
    axes_value = get_attr(attr, "perm")
    axes = axes_value if axes_value else axes
    s = (
        f"{output[0]['tensor_name']} = np.transpose("
        f"{get_input(inputs[0][0])}, axes={axes})"
    )
    return s


def concat_str(inputs, output, attr):
    """gen concat string"""
    axis = get_attr(attr, 'axis')
    inputs_list = []
    for i in range(len(inputs[0])):
        inputs_list.append(get_input(inputs[0][i]))
    s = (
        f"{output[0]['tensor_name']} = np.concatenate("
        f"({', '.join(inputs_list)}), axis={axis})"
    )
    return s


def trans_data_two2fractal(input_, src_format, dst_format):
    """two2fractal"""
    if src_format not in ("DefaultFormat", "NCHW"):
        raise ValueError(f"src_format {src_format} is not supported!")
    shape = list(input_.shape)
    m1, n1 = shape[-2] // 16, shape[-1] // 16
    if shape[-2] % 16 != 0 or shape[-1] % 16 != 0:  # need pad
        pad_m, pad_n = (shape[-2] + 15) // 16 * 16, (shape[-1] + 15) // 16 * 16
        pad_shape = [shape[0], shape[1], pad_m, pad_n]
        pad_input = np.zeros(pad_shape).astype(input_.dtype)
        pad_input[..., :shape[-2], :shape[-1]] = input_
        m1, n1 = pad_m // 16, pad_n // 16
        input_ = pad_input
    reshape_shape = shape[:-2] + [m1, 16, n1, 16]
    reshape_input = input_.reshape(reshape_shape)
    if dst_format != "FRACTAL_NZ":
        raise ValueError(
            f"dst_format {dst_format} is not supported when src_format is {src_format}"
        )
    transpose_axis = list(range(len(shape) - 2)) + \
        [x + len(shape) - 2 for x in [2, 0, 1, 3]]
    bench_mark = reshape_input.transpose(transpose_axis)
    return bench_mark


def trans_data_fractal2two(input_, shape_origin):
    """fractal2two"""
    shape_origin = [int(_) for _ in shape_origin]
    shape = list(input_.shape)
    n1, m1, m0, n0 = shape[-4:]
    new_shape = shape[:-4] + [m1 * m0, n1 * n0]
    transpose_axis = [1, 2, 0, 3]
    transpose_axis = [x + len(shape) - 4 for x in transpose_axis]
    transpose_axis = list(range(len(shape) - 4)) + transpose_axis
    bench_mark = input_.transpose(transpose_axis).reshape(new_shape)
    if new_shape != shape_origin:
        if len(shape_origin) == 2:
            bench_mark = bench_mark[:shape_origin[0], :shape_origin[1]]
        elif len(shape_origin) == 3:
            bench_mark = bench_mark[:, shape_origin[0], :shape_origin[1]]
        elif len(shape_origin) == 4:
            bench_mark = bench_mark[:, :, shape_origin[0], :shape_origin[1]]
    return bench_mark


def get_trans_data_str(input_name, output_name, ori_shape, src_format, dst_format):
    """gen trans_data string"""
    support_formats = [("DefaultFormat", "FRACTAL_NZ"),
                       ("NCHW", "FRACTAL_NZ"),
                       ("FRACTAL_NZ", "DefaultFormat"),
                       ("FRACTAL_NZ", "NCHW")]

    if (src_format, dst_format) not in support_formats:
        raise ValueError(
            f"src_format {src_format} and dst_format {dst_format} is not supported!"
        )

    if src_format in ('DefaultFormat', "NCHW") and dst_format == 'FRACTAL_NZ':
        res = (
            f"{inspect.getsource(trans_data_two2fractal)} \n"
            f"{output_name} = {trans_data_two2fractal.__name__}("
            f"{input_name}, '{src_format}', '{dst_format}')"
        )
    elif src_format == 'FRACTAL_NZ' and dst_format in ('DefaultFormat', "NCHW"):
        res = (
            f"{inspect.getsource(trans_data_fractal2two)} \n"
            f"{output_name} = {trans_data_fractal2two.__name__}({input_name}, {ori_shape})"
        )
    else:
        raise ValueError(
            f"src_format({src_format}) and dst_format({dst_format}) is not supported!"
        )
    return res


def trans_data_dsl(inputs, output, attr):
    """gen trans_data string"""
    src_format = get_attr(attr, "src_format")
    dst_format = get_attr(attr, "dst_format")
    ori_shape = output[0]['shape']
    input_name = get_input(inputs[0][0])
    output_name = output[0]['tensor_name']
    return get_trans_data_str(input_name, output_name, ori_shape, src_format, dst_format)


def np_matmul_str(inputs, output, attr):
    """gen matmul string"""
    trans_a = get_attr(attr, "transpose_a")
    trans_b = get_attr(attr, "transpose_b")
    input_0 = inputs[0][0]
    input_1 = inputs[1][0]
    res = ""
    # Because when matmul calculations are performed on Gpu and Ascend, fp32 is used for accumulation when
    # the input data is fp16, so the input data is casted to fp32
    out_name = output[0]['tensor_name']
    in0 = get_input(inputs[0][0])
    in1 = get_input(inputs[1][0])
    if input_0['data_type'] == "float16":
        tmp = f"{get_input(input_0)} = {get_input(input_0)}.astype(np.float32)"
        res += tmp + "\n"
    if input_1['data_type'] == "float16":
        tmp = f"{get_input(input_1)} = {get_input(input_1)}.astype(np.float32)"
        res += tmp + "\n"

    if trans_a and trans_b:
        res += (
            f"{out_name} = np.matmul(np.swapaxes({in0}, -1, -2), "
            f"np.swapaxes({in1}, -1, -2))"
        )
    elif trans_a:
        res += f"{out_name} = np.matmul(np.swapaxes({in0}, -1, -2), {in1})"
    elif trans_b:
        res += f"{out_name} = np.matmul({in0}, np.swapaxes({in1}, -1, -2))"
    else:
        res += f"{out_name} = np.matmul({in0}, {in1})"
    if output[0]['data_type'] == "float16":
        res += f"\n{out_name} = {out_name}.astype(np.float16)"
    return res


def convert_fracal_shape(ori_shape, fractal):
    """convert fractal shape to default shape"""
    ori_shape = tuple(ori_shape)
    if fractal == "zN":
        return ori_shape[:-4] + (ori_shape[-2] * ori_shape[-3], ori_shape[-1] * ori_shape[-4])
    if fractal == "zZ":
        return ori_shape[:-4] + (ori_shape[-4] * ori_shape[-2], ori_shape[-3] * ori_shape[-1])
    return None


def strided_slice_str(inputs, output, attr):
    """gen strided_slice string"""
    def get_value(attr_name, input_idx):
        if get_attr(attr, attr_name):
            return get_attr(attr, attr_name)
        value = inputs[input_idx][0]['value']
        if not isinstance(value, (tuple, list)):
            value = [value]
        return value

    def get_pos_value(attr_name):
        attr_value = get_attr(attr, attr_name)
        value = [0] if attr_value == [] else bin(attr_value)[2:][::-1]
        return value

    new_axis = -1
    shrink_axis = -1
    slice_string = ""
    for i, start_num in enumerate(get_value("begin", 1)):
        end_num = get_value("end", 2)[i]
        strides_num = get_value("strides", 3)[i]
        if i < len(get_pos_value("begin_mask")) and get_pos_value("begin_mask")[i] == '1':
            # use the largest range
            start_num = 0 if get_value("strides", 3)[i] >= 0 else -1
        if i < len(get_pos_value("end_mask")) and get_pos_value("end_mask")[i] == '1':
            # use the largest range
            end_num = inputs[0][0]["shape"][i] if get_value("strides", 3)[i] >= 0 else -inputs[0][0]["shape"][i] - 1
        if i < len(get_pos_value("ellipsis_mask")) and get_pos_value("ellipsis_mask")[i] == '1':
            # do not change on this axis
            start_num = 0
            end_num = inputs[0][0]["shape"][i]
            strides_num = 1
        if i < len(get_pos_value("new_axis_mask")) and get_pos_value("new_axis_mask")[i] == '1':
            # insert a new axis and ignore slice
            start_num = 0
            end_num = inputs[0][0]["shape"][i]
            strides_num = 1
            new_axis = i
        if i < len(get_pos_value("shrink_axis_mask")) and get_pos_value("shrink_axis_mask")[i] == '1':
            # delete an axis
            if start_num < 0:
                start_num += inputs[0][0]["shape"][i]
            end_num = start_num + 1
            strides_num = 1
            shrink_axis = i
        slice_string += str(start_num) + ':' + str(end_num) + \
            ':' + str(strides_num) + ","
    slice_string = slice_string[:-1]

    res = f"{output[0]['tensor_name']} = {get_input(inputs[0][0])}[{slice_string}]"
    if new_axis != -1:
        res += f"\n{output[0]['tensor_name']} = np.expand_dims({output[0]['tensor_name']}, axis={new_axis})"
    if shrink_axis != -1:
        res += f"\n{output[0]['tensor_name']} = np.squeeze({output[0]['tensor_name']}, axis={shrink_axis})"
    return res


def slice_str(inputs, output, attr):
    """gen slice string"""
    begin = get_attr(attr, "begin")
    slice_size = get_attr(attr, "size")
    slice_string = ""
    for i, value in enumerate(begin):
        start_num = value
        end_num = value + slice_size[i]
        slice_string += str(start_num) + ':' + str(end_num)
        if not i == len(begin) - 1:
            slice_string += ","

    res = (
        f"{output[0]['tensor_name']} = "
        f"{get_input(inputs[0][0])}[{slice_string}]"
    )
    return res


def split_str(inputs, output, attr):
    """gen split string"""
    axis = str(get_attr(attr, "axis"))
    output_num = str(get_attr(attr, "output_num"))
    res = (
        f"{output[0]['tensor_name']} = np.split("
        f"{get_input(inputs[0][0])}, {output_num}, {axis})"
    )
    return res


def func_pack(func_name, func_body, params, ret):
    """pack func"""
    lines = func_body.split('\n')
    body_lines = ['\t' + line for line in lines]
    func_header = 'def ' + func_name + '(' + ','.join(params) + '):\n'
    new_body = '\n'.join(body_lines) + '\n\treturn ' + ret + '\n'
    return func_header + new_body


def matmul_str(inputs, output, attr):
    """gen matmul string"""
    left_input_name = get_input(inputs[0][0])
    right_input_name = get_input(inputs[1][0])

    left_format = get_attr(attr, "left_format")
    if not left_format:
        left_format = get_attr(attr, "pri_format")

    right_format = get_attr(attr, "right_format")
    if not right_format:
        right_format = get_attr(attr, "pri_format")
    right_ori_shape = convert_fracal_shape(inputs[1][0]['shape'], right_format)

    res = ''
    if get_attr(attr, 'process') != 'cpu':
        if left_format == 'FRACTAL_NZ':
            res += get_trans_data_str(left_input_name,
                                      left_input_name,
                                      convert_fracal_shape(inputs[0][0]['shape'], "zN"),
                                      left_format,
                                      'DefaultFormat')
            res += "\n"

        if right_format == 'FRACTAL_NZ':
            res += get_trans_data_str(right_input_name,
                                      right_input_name,
                                      convert_fracal_shape(inputs[1][0]['shape'], "zN"),
                                      right_format,
                                      'DefaultFormat')
            res += "\n"
    res += np_matmul_str(inputs, output, attr) + "\n"

    if len(inputs) > 2:
        bias_shape = right_ori_shape[-2] if get_attr(attr, "transpose_b") else right_ori_shape[-1]
        if inputs[2][0]['shape'][0] != bias_shape:
            res += (
                f"{get_input(inputs[2][0])} = random_gaussian([{bias_shape}, ], "
                f"miu=1, sigma=0.1).astype(np.{inputs[2][0]['data_type']}) \n"
            )
        res += f"{output[0]['tensor_name']} = np.add({output[0]['tensor_name']}, {get_input(inputs[2][0])})\n"

    if get_attr(attr, 'process') != 'cpu' and output[0]['format'] != 'DefaultFormat':
        res += get_trans_data_str(output[0]['tensor_name'],
                                  output[0]['tensor_name'],
                                  output[0]['shape'],
                                  'DefaultFormat',
                                  output[0]['format'])
        res += "\n"
    params = [get_input(i[0]) for i in inputs]
    func = func_pack("matmul_func", res, params, output[0]['tensor_name'])
    return func + f"{output[0]['tensor_name']} = matmul_func({','.join(params)})\n"


def custom_str(inputs, output, attr):
    """gen custom string"""
    func_name = get_attr(attr, "func_name")
    func = get_attr(attr, "func_source_str").replace(
        "output_tensor", "np.zeros")
    params = [get_input(i[0]) for i in inputs]
    output_name = output[0]['tensor_name']
    return func + f"{output_name} = {func_name}({','.join(params)})\n"


def conv_2d_nchwc_str(inputs, output, attr, is_depthwise):
    """gen NCHW conv2d string"""
    support_list = {"float16": 'np.float16', "float32": 'np.float32'}
    # NCHWc
    data = inputs[0][0]
    # OIHWio
    filter_data = inputs[1][0]
    padding = get_attr(attr, "pad_list")
    stride = get_attr(attr, "stride")
    dilation = get_attr(attr, "dilation")
    output_name = output[0]["tensor_name"]

    indent = "    "
    res = ""
    res += f"n, c_i_o, h, w, c_i_i = {data['shape']} \n"
    if is_depthwise:
        res += f"c_o_o, cm_outer, kh, kw, cm_inner, c_o_i = {filter_data['shape']}\n"
    else:
        res += f"c_o_o, c_i_o, kh, kw, c_i_i, c_o_i = {filter_data['shape']}\n"
    res += f"s_h, s_w = {stride}\n"
    res += f"d_h, d_w = {dilation}\n"
    res += f"p_l, p_r, p_t, p_b = {padding}\n"
    res += "k_h_d = (kh - 1) * d_h + 1\n"
    res += "k_w_d = (kw - 1) * d_w + 1\n"
    res += "out_h = (h + p_t + p_b - k_h_d) // s_h + 1\n"
    res += "out_w = (w + p_l + p_r - k_w_d) // s_w + 1\n"

    res += "out_shape = (n, c_o_o, out_h, out_w, c_o_i)\n"
    res += "shape_data_pad = (n, c_i_o, h + p_t + p_b, w + p_l + p_r, c_i_i)\n"

    res += f"data_pad = np.zeros(shape_data_pad).astype({support_list.get(inputs[0][0]['data_type'])})\n"
    if np.sum(padding) > 0:
        res += f"data_pad[:, :, p_t:p_t+h, p_l:p_l+w, :] = {data['tensor_name']}\n"
    else:
        res += f"data_pad = {data['tensor_name']}\n"

    res += (
        f"{output_name} = np.zeros(out_shape).astype({support_list.get(output[0]['data_type'])})\n"
    )
    res += "for oh in range(out_h):\n"
    res += indent + "for ow in range(out_w):\n"
    if is_depthwise:
        res += indent + indent + "for oc_out in range(c_o_o):\n"
    else:
        res += indent + indent + "for ic_out in range(c_i_o):\n"
    res += indent + indent + indent + "for khh in range(kh):\n"
    res += indent + indent + indent + indent + "for kww in range(kw):\n"
    if is_depthwise:
        res += indent + indent + indent + indent + \
            indent + "for oc_in in range(c_o_i):\n"
        fn = filter_data['tensor_name']
        compute_str = (
            f"{output_name}[:, oc_out, oh, ow, oc_in] = {output_name}[:, oc_out, oh, ow, oc_in] + "
            f"data_pad[:, (oc_out*c_o_i+oc_in)//c_i_i, "
            f"oh*s_h+khh*d_h, ow*s_w+kww*d_w, (oc_out*c_o_i+oc_in)%c_i_i] * "
            f"{fn}[oc_out, 0, khh, kww, 0, oc_in]\n"
        )
    else:
        res += indent + indent + indent + indent + \
            indent + "for ic_in in range(c_i_i):\n"
        fn = filter_data['tensor_name']
        compute_str = (
            f"{output_name}[:, :, oh, ow, :] = {output_name}[:, :, oh, ow, :] + "
            f"data_pad[:, ic_out, oh*s_h+khh*d_h, ow*s_w+kww*d_w, ic_in] * "
            f"{fn}[:, ic_out, khh, kww, ic_in, :]\n"
        )
    res += indent + indent + indent + indent + indent + indent + compute_str

    return res


def conv_2d_str(inputs, output, attr):
    """gen conv_2d string"""
    indent = "  "
    if get_attr(attr, "data_format") == "NC1HWC0":
        is_depthwise = get_attr(attr, "is_depth_wise") is True
        return conv_2d_nchwc_str(inputs, output, attr, is_depthwise)

    support_list = {"float16": 'np.float16', "float32": 'np.float32'}
    data = inputs[0][0]
    filter_data = inputs[1][0]
    padding = get_attr(attr, "pad_list")
    stride = get_attr(attr, "stride")[2:]
    dilation = get_attr(attr, "dilation")[2:]
    out_dtype = output[0]["data_type"]
    output_name = output[0]["tensor_name"]

    res = ""
    res += f"n, h, w, c = {data['shape']} \n"
    res += f"out_c, kh, kw, c = {filter_data['shape']}\n"
    res += f"s_h, s_w = {stride}\n"
    res += f"d_h, d_w = {dilation}\n"
    res += f"p_l, p_r, p_t, p_b = {padding}\n"
    res += "out_h = (h + p_t + p_b - kh) // s_h + 1\n"
    res += "out_w = (w + p_l + p_r - kw) // s_w + 1\n"

    res += "out_shape = (n, out_h, out_w, out_c)\n"
    res += "shape_data_pad = (n, h + p_t + p_b, w + p_l + p_r, c)\n"

    res += (
        f"data_pad = np.zeros(shape_data_pad).astype({support_list.get(data['data_type'])})\n"
    )
    if np.sum(padding) > 0:
        res += f"data_pad[:, p_t:p_t+h, p_l:p_l+w, :] = {data['tensor_name']}\n"
    else:
        res += f"data_pad = {data['tensor_name']}\n"

    res += "whd = (kh - 1) * d_h + 1\n"
    res += "wwd = (kw - 1) * d_w + 1\n"
    res += f"{output_name} = np.zeros(out_shape).astype({support_list.get(out_dtype)})\n"
    res += "for i in range(out_h):\n"
    res += indent + "for j in range(out_w):\n"
    res += indent + indent + "for f in range(out_c):\n"
    fn = filter_data['tensor_name']
    res += (
        indent + indent + indent +
        f"{output_name}[:, i, j, f] = np.sum("
        f"data_pad[:, i*s_h:i*s_h+whd:d_h, j*s_w:j*s_w+wwd:d_w, :]"
        f".astype('float32') * {fn}[f, :, :, :].astype('float32'), axis=(1, 2, 3))\n"
    )
    return res


def pad_str(inputs, output, attr):
    """gen pad string"""
    tail_value = get_attr(attr, "tail")
    head_value = get_attr(attr, "head")
    value = get_attr(attr, "pad_val")
    pad_width = list(zip(head_value, tail_value))
    tn = output[0]['tensor_name']
    inp = get_input(inputs[0][0])
    res = (
        f"{tn} = np.pad({inp}, {pad_width}, mode='constant', "
        f"constant_values=({value}, {value}))"
    )
    return res


def unpad_str(inputs, output, attr):
    """gen unpad string"""
    indent = "  "
    input_shape = inputs[0][0]['shape']
    unpad_after = get_attr(attr, "tail")
    res = ""
    res += (
        f"m, n = {input_shape[-2]} - {unpad_after[-2]}, "
        f"{input_shape[-1]} - {unpad_after[-1]}\n"
    )
    res += f"if {len(input_shape)} == 4:\n"
    tn = output[0]['tensor_name']
    inp = get_input(inputs[0][0])
    res += indent + f"{tn} = {inp}[:, :, :m, :n]\n"
    res += f"elif {len(input_shape)} == 2:\n"
    res += indent + f"{tn} = {inp}[:m, :n]\n"
    return res


def cummulative_str(inputs, outputs, attr, op_type):
    """gen cumulative sum str and product str"""
    exclusive = get_attr(attr, "exclusive")
    reverse = get_attr(attr, "reverse")
    axis = get_attr(attr, "axis")
    input_shape = inputs[0][0]['shape']
    res = ""
    res += f"axis = {axis}\n"
    in0 = get_input(inputs[0][0])
    if reverse:
        res += f"out = np.flip({in0}, axis)\n"
        res += f"out = np.{op_type}(out, axis)\n"
    else:
        res += f"out = np.{op_type}({in0}, axis)\n"
    if exclusive:
        res += "from scipy.ndimage import shift\n"
        res += f"shift_axis = [0] * {len(input_shape)}\n"
        res += "shift_axis[axis] = 1\n"
        res += "out = shift(out, shift_axis)\n"
    if reverse:
        res += "out = np.flip(out, axis=axis)\n"
    res += f"{outputs[0]['tensor_name']} = out\n"
    return res


def global_pool2d_str(inputs, output, attr):
    """gen global pool 2d str"""
    pool_type = get_attr(attr, "pool_type")
    data_layout = get_attr(attr, "data_layout")
    res = ""
    if data_layout == "NHWC":
        res += "pool_idxs = (1, 2)\n"
    elif data_layout[:4] == "NCHW":
        # NCHW or NCHWc/ NCHW[x]c
        res += "pool_idxs = (2, 3)\n"

    res += f"global_pool_input = {inputs[0][0]['tensor_name']}\n"
    if pool_type == 'avg':
        res += (
            f"{output[0]['tensor_name']} = np.mean("
            f"global_pool_input, axis=pool_idxs, keepdims=True)\n"
        )
    elif pool_type == 'max':
        res += (
            f"{output[0]['tensor_name']} = np.max("
            f"global_pool_input, axis=pool_idxs, keepdims=True)\n"
        )
    return res


def sin_str(inputs, output, attr):
    """gen sin string"""
    return f"{output[0]['tensor_name']} = np.sin({get_input(inputs[0][0])})"


def cos_str(inputs, output, attr):
    """gen cos string"""
    return f"{output[0]['tensor_name']} = np.cos({get_input(inputs[0][0])})"


def asin_str(inputs, output, attr):
    """gen arcsin string"""
    return f"{output[0]['tensor_name']} = np.arcsin({get_input(inputs[0][0])})"


def acos_str(inputs, output, attr):
    """gen arccos string"""
    return f"{output[0]['tensor_name']} = np.arccos({get_input(inputs[0][0])})"


def sign_str(inputs, output, attr):
    """gen sign string"""
    return f"{output[0]['tensor_name']} = np.sign({get_input(inputs[0][0])})"


def isnan_str(inputs, output, attr):
    """gen isnan string"""
    return f"{output[0]['tensor_name']} = np.isnan({get_input(inputs[0][0])})"


def isinf_str(inputs, output, attr):
    """gen isinf string"""
    return f"{output[0]['tensor_name']} = np.isinf({get_input(inputs[0][0])})"


def isfinite_str(inputs, output, attr):
    """gen isfinite string"""
    return f"{output[0]['tensor_name']} = np.isfinite({get_input(inputs[0][0])})"


def tanh_str(inputs, output, attr):
    """gen tanh string"""
    return f"{output[0]['tensor_name']} = np.tanh({get_input(inputs[0][0])})"


def mul_str(inputs, output, attr):
    """gen mul string"""
    return (
        f"{output[0]['tensor_name']} = np.multiply("
        f"{get_input(inputs[0][0])}, {get_input(inputs[1][0])})"
    )


def pow_str(inputs, output, attr):
    """gen pow string"""
    return (
        f"{output[0]['tensor_name']} = np.power("
        f"{get_input(inputs[0][0])}, {get_input(inputs[1][0])})"
    )


def sub_str(inputs, output, attr):
    """gen sub string"""
    return (
        f"{output[0]['tensor_name']} = np.subtract("
        f"{get_input(inputs[0][0])}, {get_input(inputs[1][0])})"
    )


def tensor_add_str(inputs, output, attr):
    """gen tensor add string"""
    return (
        f"{output[0]['tensor_name']} = np.add("
        f"{get_input(inputs[0][0])}, {get_input(inputs[1][0])})"
    )


def add_str(inputs, output, attr):
    """gen add string"""
    return (
        f"{output[0]['tensor_name']} = np.add("
        f"{get_input(inputs[0][0])}, {get_input(inputs[1][0])})"
    )


def rsqrt_str(inputs, output, attr):
    """gen rsqrt string"""
    return f"{output[0]['tensor_name']} = 1.0/np.sqrt({get_input(inputs[0][0])})"


def neg_str(inputs, output, attr):
    """gen neg string"""
    return f"{output[0]['tensor_name']} = np.negative({get_input(inputs[0][0])})"


def floor_str(inputs, output, attr):
    """gen floor string"""
    return f"{output[0]['tensor_name']} = np.floor({get_input(inputs[0][0])})"


def exp_str(inputs, output, attr):
    """gen exp string"""
    return f"{output[0]['tensor_name']} = np.exp({get_input(inputs[0][0])})"


def real_div_str(inputs, output, attr):
    """gen real div string"""
    return (
        f"{output[0]['tensor_name']} = np.divide("
        f"{get_input(inputs[0][0])}, {get_input(inputs[1][0])})"
    )


def div_str(inputs, output, attr):
    """gen div string"""
    return (
        f"{output[0]['tensor_name']} = np.divide("
        f"{get_input(inputs[0][0])}, {get_input(inputs[1][0])})"
    )


def floor_div_str(inputs, output, attr):
    """gen floor div string"""
    return (
        f"{output[0]['tensor_name']} = np.floor_divide("
        f"{get_input(inputs[0][0])}, {get_input(inputs[1][0])})"
    )


def mod_str(inputs, output, attr):
    """gen mod string"""
    return (
        f"{output[0]['tensor_name']} = np.fmod("
        f"{get_input(inputs[0][0])}, {get_input(inputs[1][0])})"
    )


def floor_mod_str(inputs, output, attr):
    """gen floor mod string"""
    return (
        f"{output[0]['tensor_name']} = np.mod("
        f"{get_input(inputs[0][0])}, {get_input(inputs[1][0])})"
    )


def minimum_str(inputs, output, attr):
    """gen minimum string"""
    return (
        f"{output[0]['tensor_name']} = np.minimum("
        f"{get_input(inputs[0][0])}, {get_input(inputs[1][0])})"
    )


def maximum_str(inputs, output, attr):
    """gen maximum string"""
    return (
        f"{output[0]['tensor_name']} = np.maximum("
        f"{get_input(inputs[0][0])}, {get_input(inputs[1][0])})"
    )


def log_str(inputs, output, attr):
    """gen log string"""
    return f"{output[0]['tensor_name']} = np.log({get_input(inputs[0][0])})"


def sqrt_str(inputs, output, attr):
    """gen sqrt string"""
    return f"{output[0]['tensor_name']} = np.sqrt({get_input(inputs[0][0])})"


def reshape_str(inputs, output, attr):
    """gen reshape string"""
    return (
        f"{output[0]['tensor_name']} = np.reshape("
        f"{get_input(inputs[0][0])}, {output[0]['shape']})"
    )


def one_hot_str(inputs, output, attr):
    """gen one hot string"""
    return (
        f"{output[0]['tensor_name']} = one_hot_np({get_input(inputs[0][0])}, "
        f"{get_attr(attr, 'axis')}, {get_input(inputs[1][0])}, "
        f"{get_input(inputs[2][0])}, {get_attr(attr, 'depth')}, "
        f"np.{output[0]['data_type']})"
    )


def zeros_like_str(inputs, output, attr):
    """gen zeros like string"""
    return f"{output[0]['tensor_name']} = np.zeros_like({get_input(inputs[0][0])})"


def add_n_str(inputs, output, attr):
    """gen add n string"""
    return (
        f"{output[0]['tensor_name']} = " + " + ".join(
            get_input(inputs[0][i]) for i in range(0, len(inputs[0]))
        )
    )


def tile_str(inputs, output, attr):
    """gen tile string"""
    return (
        f"{output[0]['tensor_name']} = np.tile("
        f"{get_input(inputs[0][0])}, {get_attr(attr, 'multiples')})"
    )


def reciprocal_str(inputs, output, attr):
    """gen reciprocal string"""
    return f"{output[0]['tensor_name']} = np.divide(1.0, {get_input(inputs[0][0])})"


def equal_str(inputs, output, attr):
    """gen equal string"""
    return (
        f"{output[0]['tensor_name']} = np.equal("
        f"{get_input(inputs[0][0])}, {get_input(inputs[1][0])})"
    )


def not_equal_str(inputs, output, attr):
    """gen not equal string"""
    return (
        f"{output[0]['tensor_name']} = np.not_equal("
        f"{get_input(inputs[0][0])}, {get_input(inputs[1][0])})"
    )


def greater_equal_str(inputs, output, attr):
    """gen greater equal string"""
    return (
        f"{output[0]['tensor_name']} = np.greater_equal("
        f"{get_input(inputs[0][0])}, {get_input(inputs[1][0])})"
    )


def select_str(inputs, output, attr):
    """gen select string"""
    return (
        f"{output[0]['tensor_name']} = np.where("
        f"{get_input(inputs[0][0])}, {get_input(inputs[1][0])}, "
        f"{get_input(inputs[2][0])})"
    )


def inplace_assign_str(inputs, output, attr):
    """gen inplace assign string"""
    return (
        f"{get_input(inputs[0][0])} = {get_input(inputs[1][0])}; "
        f"{output[0]['tensor_name']} = {get_input(inputs[2][0])}"
    )


def greater_str(inputs, output, attr):
    """gen greater string"""
    return (
        f"{output[0]['tensor_name']} = np.greater("
        f"{get_input(inputs[0][0])}, {get_input(inputs[1][0])})"
    )


def select_gt_str(inputs, output, attr):
    """gen select gt string"""
    return (
        f"{output[0]['tensor_name']} = np.where("
        f"{get_input(inputs[0][0])} > {get_input(inputs[1][0])}, "
        f"{get_input(inputs[2][0])}, {get_input(inputs[3][0])})"
    )


def select_lt_str(inputs, output, attr):
    """gen select lt string"""
    return (
        f"{output[0]['tensor_name']} = np.where("
        f"{get_input(inputs[0][0])} < {get_input(inputs[1][0])}, "
        f"{get_input(inputs[2][0])}, {get_input(inputs[3][0])})"
    )


def abs_str(inputs, output, attr):
    """gen abs string"""
    return f"{output[0]['tensor_name']} = np.absolute({get_input(inputs[0][0])})"


def less_equal_str(inputs, output, attr):
    """gen less equal string"""
    return (
        f"{output[0]['tensor_name']} = np.less_equal("
        f"{get_input(inputs[0][0])}, {get_input(inputs[1][0])})"
    )


def less_str(inputs, output, attr):
    """gen less string"""
    return (
        f"{output[0]['tensor_name']} = np.less("
        f"{get_input(inputs[0][0])}, {get_input(inputs[1][0])})"
    )


def equiv_format_str(inputs, output, attr):
    """gen equiv format string"""
    return f"{output[0]['tensor_name']} = {get_input(inputs[0][0])}"


def expand_dims_str(inputs, output, attr):
    """gen expand dims string"""
    return (
        f"{output[0]['tensor_name']} = np.expand_dims("
        f"{get_input(inputs[0][0])}, {get_attr(attr, 'axis')})"
    )


def elem_any_str(inputs, output, attr):
    """gen elem any string"""
    return (
        f"{output[0]['tensor_name']} = ({get_input(inputs[0][0])}.all() > 0)"
        f".astype(np.{output[0]['data_type']}).reshape(1)"
    )


def assign_str(inputs, output, attr):
    """gen assign string"""
    return (
        f"{get_input(inputs[0][0])} = {get_input(inputs[1][0])}; "
        f"{output[0]['tensor_name']} = {get_input(inputs[1][0])}"
    )


def asinh_str(inputs, output, attr):
    """gen arcsinh string"""
    return f"{output[0]['tensor_name']} = np.arcsinh({get_input(inputs[0][0])})"


def acosh_str(inputs, output, attr):
    """gen arccosh string"""
    return f"{output[0]['tensor_name']} = np.arccosh({get_input(inputs[0][0])})"


def atan2_str(inputs, output, attr):
    """gen arctan2 string"""
    return (
        f"{output[0]['tensor_name']} = np.arctan2("
        f"{get_input(inputs[0][0])}, {get_input(inputs[1][0])})"
    )


def expm1_str(inputs, output, attr):
    """gen expm1 string"""
    return f"{output[0]['tensor_name']} = np.expm1({get_input(inputs[0][0])})"


def logical_not_str(inputs, output, attr):
    """gen logical not string"""
    return f"{output[0]['tensor_name']} = np.logical_not({get_input(inputs[0][0])})"


def logical_and_str(inputs, output, attr):
    """gen logical and string"""
    return (
        f"{output[0]['tensor_name']} = np.logical_and("
        f"{get_input(inputs[0][0])}, {get_input(inputs[1][0])})"
    )


def logical_or_str(inputs, output, attr):
    """gen logical or string"""
    return (
        f"{output[0]['tensor_name']} = np.logical_or("
        f"{get_input(inputs[0][0])}, {get_input(inputs[1][0])})"
    )


def erf_str(inputs, output, attr):
    """gen erf string"""
    return (
        f"{output[0]['tensor_name']} = "
        f"__import__('scipy').special.erf({get_input(inputs[0][0])})"
    )


def tensor_scatter_add_str(inputs, output, attr):
    """gen tensor scatter add string"""
    return (
        f"{output[0]['tensor_name']} = tensor_scatter_add_np("
        f"{get_input(inputs[0][0])}, {get_input(inputs[1][0])}, "
        f"{get_input(inputs[2][0])})"
    )


def gather_nd_str(inputs, output, attr):
    """gen gather nd string"""
    return (
        f"{output[0]['tensor_name']} = {get_input(inputs[0][0])}[tuple("
        f"{get_input(inputs[1][0])}.transpose().tolist())]"
    )


def unsorted_segment_sum_str(inputs, output, attr):
    """gen unsorted segment sum string"""
    return (
        f"{output[0]['tensor_name']} = np.zeros([{get_attr(attr, 'num_segments')},] + "
        f"{inputs[0][0]['shape']}[{len(inputs[1][0]['shape'])}:]);  np.add.at("
        f"{output[0]['tensor_name']}, {get_input(inputs[1][0])}, "
        f"{get_input(inputs[0][0])})"
    )


def gather_str(inputs, output, attr):
    """gen gather string"""
    return (
        f"{output[0]['tensor_name']} = gather_np("
        f"{get_input(inputs[0][0])}, {get_input(inputs[1][0])}, "
        f"{get_attr(attr, 'axis')})"
    )


def standard_normal_str(inputs, output, attr):
    """gen standard normal string"""
    return (
        f"{output[0]['tensor_name']} = np.random.standard_normal("
        f"{get_attr(attr, 'shape')})"
    )


def cimag_str(inputs, output, attr):
    """gen cimag string"""
    return f"{output[0]['tensor_name']} = np.imag({get_input(inputs[0][0])})"


def creal_str(inputs, output, attr):
    """gen creal string"""
    return f"{output[0]['tensor_name']} = np.real({get_input(inputs[0][0])})"


def complex_str(inputs, output, attr):
    """gen complex string"""
    return (
        f"{output[0]['tensor_name']} = np.vectorize(complex)("
        f"{get_input(inputs[0][0])}, {get_input(inputs[1][0])})"
    )


def get_op_dsl():
    # #lizard forgives
    """Get DSL for operators"""
    op_dsl = {
        "Custom": custom_str,
        "ReduceSum": lambda inputs, output, attr: reduce_str(inputs, output, attr, "sum"),
        "ReduceMax": lambda inputs, output, attr: reduce_str(inputs, output, attr, "max"),
        "ReduceMin": lambda inputs, output, attr: reduce_str(inputs, output, attr, "min"),
        "ReduceAll": lambda inputs, output, attr: reduce_str(inputs, output, attr, "all"),
        "ReduceProd": lambda inputs, output, attr: reduce_str(inputs, output, attr, "prod"),
        "StridedSlice": strided_slice_str,
        "Slice": slice_str,
        "Split": split_str,
        "CumSum": lambda inputs, output, attr: cummulative_str(inputs, output, attr, "cumsum"),
        "CumProd": lambda inputs, output, attr: cummulative_str(inputs, output, attr, "cumprod"),
        "Sin": sin_str,
        "Cos": cos_str,
        "Asin": asin_str,
        "ACos": acos_str,
        "Sign": sign_str,
        "IsNan": isnan_str,
        "IsInf": isinf_str,
        "IsFinite": isfinite_str,
        "Tanh": tanh_str,
        "Mul": mul_str,
        "Pow": pow_str,
        "Sub": sub_str,
        "TensorAdd": tensor_add_str,
        "Add": add_str,
        "Rsqrt": rsqrt_str,
        "Neg": neg_str,
        "Floor": floor_str,
        "Exp": exp_str,
        "RealDiv": real_div_str,
        "Div": div_str,
        "FloorDiv": floor_div_str,
        "Mod": mod_str,
        "FloorMod": floor_mod_str,
        "Minimum": minimum_str,
        "Maximum": maximum_str,
        "Log": log_str,
        "Sqrt": sqrt_str,
        "Cast": cast_str,
        "Reshape": reshape_str,
        "OneHot": one_hot_str,
        "ZerosLike": zeros_like_str,
        "AddN": add_n_str,
        "Tile": tile_str,
        "Reciprocal": reciprocal_str,
        "Equal": equal_str,
        "NotEqual": not_equal_str,
        "GreaterEqual": greater_equal_str,
        "Select": select_str,
        "InplaceAssign": inplace_assign_str,
        "Greater": greater_str,
        "SelectGT": select_gt_str,
        "SelectLT": select_lt_str,
        "Abs": abs_str,
        "LessEqual": less_equal_str,
        "Less": less_str,
        "EquivFormat": equiv_format_str,
        "ExpandDims": expand_dims_str,
        "ElemAny": elem_any_str,
        "Transpose": transpose_str,
        "TransData": trans_data_dsl,
        "BroadcastTo": broadcast_str,
        "BatchMatMul": matmul_str,
        "Assign": assign_str,
        "MatMul": matmul_str,
        "Conv2D": conv_2d_str,
        "PadAkg": pad_str,
        "UnPadAkg": unpad_str,
        "Asinh": asinh_str,
        "Acosh": acosh_str,
        "Atan2": atan2_str,
        "Expm1": expm1_str,
        "LogicalNot": logical_not_str,
        "LogicalAnd": logical_and_str,
        "LogicalOr": logical_or_str,
        "Erf": erf_str,
        "TensorScatterAdd": tensor_scatter_add_str,
        "GatherNd": gather_nd_str,
        "UnsortedSegmentSum": unsorted_segment_sum_str,
        "Gather": gather_str,
        "StandardNormal": standard_normal_str,
        "CImag": cimag_str,
        "CReal": creal_str,
        "Complex": complex_str,
        "Concat": concat_str
    }
    return op_dsl


def torch_constant_float_str(inputs, output, attr):
    """gen torch constant float string"""
    return (
        f"{output[0]['tensor_name']} = np.array("
        f"{format_py_value(get_input(output[0]))}, dtype=np.float32)"
    )


def torch_constant_int_str(inputs, output, attr):
    """gen torch constant int string"""
    return f"{output[0]['tensor_name']} = int({get_input(output[0])})"


def torch_mul_tensor_str(inputs, output, attr):
    """gen torch mul tensor string"""
    return (
        f"{output[0]['tensor_name']} = np.multiply("
        f"{get_input(inputs[0][0])}, "
        f"{get_input(inputs[1][0])})"
    )


def torch_mul_scalar_str(inputs, output, attr):
    """gen torch mul scalar string"""
    return (
        f"{output[0]['tensor_name']} = ("
        f"{get_input(inputs[0][0])} * "
        f"{get_input(inputs[1][0])})"
    )


def torch_add_tensor_str(inputs, output, attr):
    """gen torch add tensor string"""
    return (
        f"{output[0]['tensor_name']} = ("
        f"{get_input(inputs[0][0])} + "
        f"({get_input(inputs[2][0])}) * "
        f"{get_input(inputs[1][0])})"
    )


def torch_add_scalar_str(inputs, output, attr):
    """gen torch add scalar string"""
    return (
        f"{output[0]['tensor_name']} = ("
        f"{get_input(inputs[0][0])} + "
        f"({get_input(inputs[2][0])}) * "
        f"{get_input(inputs[1][0])})"
    )


def torch_sub_tensor_str(inputs, output, attr):
    """gen torch sub tensor string"""
    return (
        f"{output[0]['tensor_name']} = ("
        f"{get_input(inputs[0][0])} - "
        f"({get_input(inputs[2][0])}) * "
        f"{get_input(inputs[1][0])})"
    )


def torch_sub_scalar_str(inputs, output, attr):
    """gen torch sub scalar string"""
    return (
        f"{output[0]['tensor_name']} = ("
        f"{get_input(inputs[0][0])} - "
        f"({get_input(inputs[2][0])}) * "
        f"{get_input(inputs[1][0])})"
    )


def torch_div_tensor_str(inputs, output, attr):
    """gen torch div tensor string"""
    return (
        f"{output[0]['tensor_name']} = np.divide("
        f"{get_input(inputs[0][0])}, "
        f"{get_input(inputs[1][0])})"
    )


def torch_div_scalar_str(inputs, output, attr):
    """gen torch div scalar string"""
    return (
        f"{output[0]['tensor_name']} = np.divide("
        f"{get_input(inputs[0][0])}, "
        f"{get_input(inputs[1][0])})"
    )


def torch_max_dim_str(inputs, output, attr):
    """gen torch max dim string"""
    return (
        f"{output[0]['tensor_name']} = np.max("
        f"{get_input(inputs[0][0])}, "
        f"axis={get_input(inputs[1][0])}, "
        f"keepdims={get_input(inputs[2][0])})\n"
        f"{output[1]['tensor_name']} = (lambda _i, _k, _a: "
        f"np.expand_dims(_i, axis=_a) if _k else _i)("
        f"np.argmax({get_input(inputs[0][0])}, "
        f"axis={get_input(inputs[1][0])}), "
        f"{get_input(inputs[2][0])}, "
        f"{get_input(inputs[1][0])})"
    )


def torch_neg_str(inputs, output, attr):
    """gen torch neg string"""
    return (
        f"{output[0]['tensor_name']} = np.negative("
        f"{get_input(inputs[0][0])})"
    )


def torch_sqrt_str(inputs, output, attr):
    """gen torch sqrt string"""
    return (
        f"{output[0]['tensor_name']} = np.sqrt("
        f"{get_input(inputs[0][0])})"
    )


def torch_relu_str(inputs, output, attr):
    """gen torch relu string"""
    return (
        f"{output[0]['tensor_name']} = np.maximum("
        f"{get_input(inputs[0][0])}, 0)"
    )


def torch_listconstruct_str(inputs, output, attr):
    """gen torch listconstruct string"""
    return (
        f"{output[0]['tensor_name']} = ["
        f"{', '.join(str(get_input(group[0])) for group in inputs)}]"
    )


def torch_view_str(inputs, output, attr):
    """gen torch view string"""
    return (
        f"{output[0]['tensor_name']} = np.reshape("
        f"{get_input(inputs[0][0])}, "
        f"{get_input(inputs[1][0])})"
    )


def torch_cat_str(inputs, output, attr):
    """gen torch cat string"""
    return (
        f"{output[0]['tensor_name']} = np.concatenate("
        f"{get_input(inputs[0][0])}, "
        f"axis={get_input(inputs[1][0])})"
    )


def torch_broadcast_to_str(inputs, output, attr):
    """gen torch broadcast to string"""
    return gen_broadcast_to(
        output[0]["tensor_name"],
        get_input(inputs[0][0]),
        get_input(inputs[1][0]),
    )


def torch_slice_tensor_str(inputs, output, attr):
    """gen torch slice tensor string"""
    return gen_slice_tensor(
        output[0]["tensor_name"],
        get_input(inputs[0][0]),
        get_input(inputs[1][0]),
        get_input(inputs[2][0]),
        get_input(inputs[3][0]),
        get_input(inputs[4][0]),
    )


def torch_permute_str(inputs, output, attr):
    """gen torch permute string"""
    return (
        f"{output[0]['tensor_name']} = np.transpose("
        f"{get_input(inputs[0][0])}, "
        f"axes={get_input(inputs[1][0])})"
    )


def torch_sigmoid_str(inputs, output, attr):
    """gen torch sigmoid string"""
    return (
        f"{output[0]['tensor_name']} = 1 / (1 + np.exp("
        f"-{get_input(inputs[0][0])}))"
    )


def torch_constant_none_str(inputs, output, attr):
    """gen torch constant none string"""
    return f"{output[0]['tensor_name']} = None"


def torch_constant_bool_str(inputs, output, attr):
    """gen torch constant bool string"""
    return f"{output[0]['tensor_name']} = {get_input(output[0])}"


def torch_sum_dim_intlist_str(inputs, output, attr):
    """gen torch sum dim intlist string"""
    return (
        f"{output[0]['tensor_name']} = np.sum("
        f"{get_input(inputs[0][0])}, "
        f"axis=tuple({get_input(inputs[1][0])}), "
        f"keepdims={get_input(inputs[2][0])})"
    )


def torch_pow_tensor_scalar_str(inputs, output, attr):
    """gen torch pow tensor scalar string"""
    return (
        f"{output[0]['tensor_name']} = np.power("
        f"{get_input(inputs[0][0])}, "
        f"{get_input(inputs[1][0])})"
    )


def torch_pow_scalar_str(inputs, output, attr):
    """gen torch pow scalar string"""
    return (
        f"{output[0]['tensor_name']} = np.power("
        f"{get_input(inputs[0][0])}, "
        f"{get_input(inputs[1][0])})"
    )


def torch_rsqrt_str(inputs, output, attr):
    """gen torch rsqrt string"""
    return (
        f"{output[0]['tensor_name']} = 1.0 / np.sqrt("
        f"{get_input(inputs[0][0])})"
    )


def torch_clamp_str(inputs, output, attr):
    """gen torch clamp string"""
    return (
        f"{output[0]['tensor_name']} = np.clip("
        f"{get_input(inputs[0][0])}, "
        f"{'None' if inputs[1][0].get('data_type') == 'none' else format_py_value(get_input(inputs[1][0]))}, "
        f"{'None' if inputs[2][0].get('data_type') == 'none' else format_py_value(get_input(inputs[2][0]))})"
    )


def torch_to_dtype_str(inputs, output, attr):
    """gen torch to dtype string"""
    return (
        f"{output[0]['tensor_name']} = np.array("
        f"{format_py_value(get_input(inputs[0][0]))}, "
        f"dtype={TORCH_DTYPE_TO_NUMPY[get_input(inputs[1][0])]})"
    )


def torch_lt_scalar_str(inputs, output, attr):
    """gen torch lt scalar string"""
    return (
        f"{output[0]['tensor_name']} = "
        f"{get_input(inputs[0][0])} < "
        f"{get_input(inputs[1][0])}"
    )


def torch_lt_tensor_str(inputs, output, attr):
    """gen torch lt tensor string"""
    return (
        f"{output[0]['tensor_name']} = "
        f"{get_input(inputs[0][0])} < "
        f"{get_input(inputs[1][0])}"
    )


def torch_le_scalar_str(inputs, output, attr):
    """gen torch le scalar string"""
    return (
        f"{output[0]['tensor_name']} = "
        f"{get_input(inputs[0][0])} <= {format_py_value(get_input(inputs[1][0]))}"
    )


def torch_le_tensor_str(inputs, output, attr):
    """gen torch le tensor string"""
    return (
        f"{output[0]['tensor_name']} = "
        f"{get_input(inputs[0][0])} <= "
        f"{get_input(inputs[1][0])}"
    )


def torch_gt_scalar_str(inputs, output, attr):
    """gen torch gt scalar string"""
    return (
        f"{output[0]['tensor_name']} = "
        f"{get_input(inputs[0][0])} > "
        f"{get_input(inputs[1][0])}"
    )


def torch_ge_scalar_str(inputs, output, attr):
    """gen torch ge scalar string"""
    return (
        f"{output[0]['tensor_name']} = "
        f"{get_input(inputs[0][0])} >= "
        f"{get_input(inputs[1][0])}"
    )


def torch_where_self_str(inputs, output, attr):
    """gen torch where self string"""
    return (
        f"{output[0]['tensor_name']} = np.where("
        f"{get_input(inputs[0][0])}, "
        f"{get_input(inputs[1][0])}, "
        f"{get_input(inputs[2][0])})"
    )


def torch_exp_str(inputs, output, attr):
    """gen torch exp string"""
    return (
        f"{output[0]['tensor_name']} = np.exp("
        f"{get_input(inputs[0][0])})"
    )


def torch_log_str(inputs, output, attr):
    """gen torch log string"""
    return (
        f"{output[0]['tensor_name']} = np.log("
        f"{get_input(inputs[0][0])})"
    )


def torch_bitwise_not_str(inputs, output, attr):
    """gen torch bitwise not string"""
    return (
        f"{output[0]['tensor_name']} = np.logical_not("
        f"{get_input(inputs[0][0])})"
    )


def torch_bitwise_and_str(inputs, output, attr):
    """gen torch bitwise and string"""
    return (
        f"{output[0]['tensor_name']} = np.bitwise_and("
        f"{get_input(inputs[0][0])}, {get_input(inputs[1][0])})"
    )


def torch_bitwise_or_str(inputs, output, attr):
    """gen torch bitwise or string"""
    return (
        f"{output[0]['tensor_name']} = np.bitwise_or("
        f"{get_input(inputs[0][0])}, {get_input(inputs[1][0])})"
    )


def torch_sin_str(inputs, output, attr):
    """gen torch sin string"""
    return (
        f"{output[0]['tensor_name']} = np.sin("
        f"{get_input(inputs[0][0])})"
    )


def torch_cos_str(inputs, output, attr):
    """gen torch cos string"""
    return (
        f"{output[0]['tensor_name']} = np.cos("
        f"{get_input(inputs[0][0])})"
    )


def torch_vtensor_literal_str(inputs, output, attr):
    """gen torch vtensor literal string"""
    return (
        f"{output[0]['tensor_name']} = np.array("
        f"{format_py_value(get_input(output[0]))}, dtype=np.{output[0]['data_type']})"
    )


def torch_eq_scalar_str(inputs, output, attr):
    """gen torch eq scalar string"""
    return (
        f"{output[0]['tensor_name']} = np.equal("
        f"{get_input(inputs[0][0])}, {format_py_value(get_input(inputs[1][0]))})"
    )


def torch_eq_tensor_str(inputs, output, attr):
    """gen torch eq tensor string"""
    return (
        f"{output[0]['tensor_name']} = np.equal("
        f"{get_input(inputs[0][0])}, {get_input(inputs[1][0])})"
    )


def torch_tanh_str(inputs, output, attr):
    """gen torch tanh string"""
    return (
        f"{output[0]['tensor_name']} = np.tanh("
        f"{get_input(inputs[0][0])})"
    )


def torch_unsqueeze_str(inputs, output, attr):
    """gen torch unsqueeze string"""
    return (
        f"{output[0]['tensor_name']} = np.expand_dims("
        f"{get_input(inputs[0][0])}, axis={get_input(inputs[1][0])})"
    )


def torch_gt_tensor_str(inputs, output, attr):
    """gen torch gt tensor string"""
    return (
        f"{output[0]['tensor_name']} = np.greater("
        f"{get_input(inputs[0][0])}, {get_input(inputs[1][0])})"
    )


def torch_slice_scatter_str(inputs, output, attr):
    """gen torch slice scatter string"""
    return gen_slice_scatter(
        output[0]["tensor_name"],
        get_input(inputs[0][0]),
        get_input(inputs[1][0]),
        get_input(inputs[2][0]),
        get_input(inputs[3][0]),
        get_input(inputs[4][0]),
        get_input(inputs[5][0]),
    )


def torch_constant_pad_nd_str(inputs, output, attr):
    """gen torch constant pad nd string"""
    return gen_constant_pad_nd(
        output[0]["tensor_name"],
        get_input(inputs[0][0]),
        get_input(inputs[1][0]),
        get_input(inputs[2][0]),
    )


def torch_gather_str(inputs, output, attr):
    """gen torch gather string"""
    return (
        f"{output[0]['tensor_name']} = np.take_along_axis("
        f"{get_input(inputs[0][0])}, "
        f"{get_input(inputs[2][0])}, "
        f"axis={get_input(inputs[1][0])})"
    )


def torch_size_int_str(inputs, output, attr):
    """gen torch size.int string"""
    tensor_name = get_input(inputs[0][0])
    dim = get_input(inputs[1][0])
    return f"{output[0]['tensor_name']} = {tensor_name}.shape[{dim}]"


def get_op_dsl_torch_mlir():
    # #lizard forgives
    """Get DSL for torch-mlir operators."""
    op_dsl = {
        "Torch.constant.float": torch_constant_float_str,
        "Torch.constant.int": torch_constant_int_str,
        "Torch.aten.mul.tensor": torch_mul_tensor_str,
        "Torch.aten.mul.scalar": torch_mul_scalar_str,
        "Torch.aten.add.tensor": torch_add_tensor_str,
        "Torch.aten.add.scalar": torch_add_scalar_str,
        "Torch.aten.sub.tensor": torch_sub_tensor_str,
        "Torch.aten.sub.scalar": torch_sub_scalar_str,
        "Torch.aten.div.tensor": torch_div_tensor_str,
        "Torch.aten.div.scalar": torch_div_scalar_str,
        "Torch.aten.max.dim": torch_max_dim_str,
        "Torch.aten.neg": torch_neg_str,
        "Torch.aten.sqrt": torch_sqrt_str,
        "Torch.aten.relu": torch_relu_str,
        "Torch.prim.listconstruct": torch_listconstruct_str,
        "Torch.aten.view": torch_view_str,
        "Torch.aten.cat": torch_cat_str,
        "Torch.aten.broadcastTo": torch_broadcast_to_str,
        "Torch.aten.slice.tensor": torch_slice_tensor_str,
        "Torch.aten.permute": torch_permute_str,
        "Torch.aten.sigmoid": torch_sigmoid_str,
        "Torch.constant.none": torch_constant_none_str,
        "Torch.constant.bool": torch_constant_bool_str,
        "Torch.aten.sum.dimIntlist": torch_sum_dim_intlist_str,
        "Torch.aten.pow.tensorScalar": torch_pow_tensor_scalar_str,
        "Torch.aten.pow.scalar": torch_pow_scalar_str,
        "Torch.aten.rsqrt": torch_rsqrt_str,
        "Torch.aten.clamp": torch_clamp_str,
        "Torch.aten.to.dtype": torch_to_dtype_str,
        "Torch.aten.lt.scalar": torch_lt_scalar_str,
        "Torch.aten.lt.tensor": torch_lt_tensor_str,
        "Torch.aten.le.scalar": torch_le_scalar_str,
        "Torch.aten.le.tensor": torch_le_tensor_str,
        "Torch.aten.gt.scalar": torch_gt_scalar_str,
        "Torch.aten.ge.scalar": torch_ge_scalar_str,
        "Torch.aten.where.self": torch_where_self_str,
        "Torch.aten.exp": torch_exp_str,
        "Torch.aten.log": torch_log_str,
        "Torch.aten.bitwiseNot": torch_bitwise_not_str,
        "Torch.aten.bitwiseAnd": torch_bitwise_and_str,
        "Torch.aten.bitwiseAnd.tensor": torch_bitwise_and_str,
        "Torch.aten.bitwiseOr": torch_bitwise_or_str,
        "Torch.aten.bitwiseOr.tensor": torch_bitwise_or_str,
        "Torch.aten.sin": torch_sin_str,
        "Torch.aten.cos": torch_cos_str,
        "Torch.vtensor.literal": torch_vtensor_literal_str,
        "Torch.aten.eq.scalar": torch_eq_scalar_str,
        "Torch.aten.eq.tensor": torch_eq_tensor_str,
        "Torch.aten.tanh": torch_tanh_str,
        "Torch.aten.unsqueeze": torch_unsqueeze_str,
        "Torch.aten.gt.tensor": torch_gt_tensor_str,
        "Torch.aten.sliceScatter": torch_slice_scatter_str,
        "Torch.aten.constantPadNd": torch_constant_pad_nd_str,
        "Torch.aten.gather": torch_gather_str,
        "Torch.aten.size.int": torch_size_int_str,
    }
    return op_dsl
