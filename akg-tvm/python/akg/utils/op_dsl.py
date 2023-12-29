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
# limitations under the License.

"""Provides op numpy implementation, used to compare with akg output."""

import logging
import inspect
from copy import deepcopy
import numpy as np


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


def gather_nd_np(data, indices):
    """numpy implementation of gather_nd"""
    data_shape = data.shape
    indices_shape = indices.shape
    new_indices = indices.reshape(-1, indices.shape[-1])
    left_shape = indices_shape[:-1]
    right_shape = data_shape[int(indices_shape[-1]):]
    out_shape = left_shape + right_shape
    out = np.zeros(out_shape, np.float32).reshape(new_indices.shape[0], -1)
    new_data = deepcopy(data).reshape(-1, int(np.prod(data_shape[int(indices_shape[-1]):])))
    for i in range(new_indices.shape[0]):
        for j in range(out.shape[1]):
            index_read = [i, 0]
            index_write = [0, 0]
            inbound = True
            for k in range(0, int(indices_shape[-1])):
                temp_idx = new_indices[i, k]
                inbound = np.all((inbound, (temp_idx >= 0), (temp_idx < data_shape[k])))
                index_write[0] += int(temp_idx * data.strides[k] / data.itemsize)
                index_write[0] = int(index_write[0] / out.shape[1])
            index_read[1] = j
            if inbound:
                index_write[1] = j
                out[tuple(index_read)] = new_data[tuple(index_write)]
    return out.reshape(out_shape)


def tensor_scatter_add_np(data, indices, updates):
    """numpy implementation of tensor_scatter_add"""
    data_shape = data.shape
    indices_shape = indices.shape
    if indices.ndim > 1:
        new_indices = indices.reshape(-1, indices.shape[-1])
        out = deepcopy(data).reshape(-1, int(np.prod(data_shape[int(indices_shape[-1]):])))
    else:
        new_indices = indices.reshape(-1, 1)
        out = deepcopy(data).reshape(-1, int(np.prod(data_shape[1:])))
    new_updates = updates.reshape(new_indices.shape[0], -1)
    for i in range(new_indices.shape[0]):
        for j in range(out.shape[1]):
            index_read = [i, 0]
            index_write = [0, 0]
            inbound = True
            for k in range(0, int(indices_shape[-1])):
                temp_idx = new_indices[i, k]
                inbound = np.all((inbound, (temp_idx >= 0), (temp_idx < data_shape[k])))
                index_write[0] += int(temp_idx * data.strides[k] / data.itemsize)
                index_write[0] = int(index_write[0] / out.shape[1])
            index_read[1] = j
            if inbound:
                index_write[1] = j
                temp = new_updates[tuple(index_read)] + out[tuple(index_write)]
                out[tuple(index_write)] = temp
    return out.reshape(data_shape)


def gather_np(data, indices, axis):
    """numpy implementation of gather"""
    expect = np.take(data, indices, axis)
    return expect


def csrmv_np(indptr, indices, data, weight, shape):
    """numpy implementation of csrmv"""
    import scipy.sparse
    sparse_data = scipy.sparse.csr_matrix((data, indices, indptr), shape)
    expect = sparse_data * weight
    return np.asarray(expect)


def csrmm_np(indptr, indices, data, weight, shape):
    """numpy implementation of csrmm"""
    import scipy.sparse
    sparse_data = scipy.sparse.csr_matrix((data, indices, indptr), shape)
    expect = sparse_data * weight
    return np.asarray(expect)


def csr_reduce_sum_np(indptr, indices, data, shape, axis):
    """numpy implementation of csr_reduce_sum"""
    import scipy.sparse
    axis = axis % len(shape)
    x = data.reshape(data.shape[0], -1)
    expect = []
    for i in range(x.shape[-1]):
        sparse = scipy.sparse.csr_matrix((x[..., i], indices, indptr), shape=shape[:2])
        out = np.array(sparse.sum(axis))
        expect.append(out.data)
    expect = np.moveaxis(np.stack(expect, 0).reshape(shape[2:] + [shape[1 - axis]]), -1, 0)
    return expect.reshape([shape[0], 1] + shape[2:])


def csr_mul_np(indptr, indices, sparse_data, dense, shape):
    """numpy implementation of csr_mul"""
    import scipy.sparse
    x = sparse_data.reshape(sparse_data.shape[0], -1)
    y = np.broadcast_to(dense, shape).reshape(shape[0], shape[1], -1)
    expect = []
    for i in range(x.shape[-1]):
        sparse = scipy.sparse.csr_matrix((x[..., i], indices, indptr), shape=shape[:2])
        out = sparse.multiply(y[..., i])
        expect.append(out.data)
    expect = np.moveaxis(np.stack(expect, 0).reshape(shape[2:] + [sparse_data.shape[0]]), -1, 0)
    return expect


def csr_div_np(indptr, indices, sparse_data, dense, shape):
    """numpy implementation of csr_div"""
    import scipy.sparse
    x = sparse_data.reshape(sparse_data.shape[0], -1)
    y = np.broadcast_to(dense, shape).reshape(shape[0], shape[1], -1)
    expect = []
    for i in range(x.shape[-1]):
        sparse = scipy.sparse.csr_matrix((x[..., i], indices, indptr), shape=shape[:2])
        out = sparse.multiply(np.divide(1, y[..., i]))
        expect.append(out.data)
    expect = np.moveaxis(np.stack(expect, 0).reshape(shape[2:] + [sparse_data.shape[0]]), -1, 0)
    return expect


def csr_gather_np(indptr, indices, dense, shape):
    """numpy implementation of csr_gather"""
    import scipy.sparse
    sparse_data = scipy.sparse.csr_matrix((np.zeros(indices.shape), indices, indptr), shape[:2])
    coo = sparse_data.tocoo()
    coo_idx = np.stack((coo.row, coo.col))
    expect = dense[tuple(coo_idx.tolist())]
    return expect


def one_hot_np(data, axis, on_value, off_value, depth, dtype):
    """numpy implementation of one_hot"""
    shape = data.shape
    if axis < 0:
        axis = axis + len(shape) + 1
    out_shape = [i for i in shape]
    out_shape.insert(axis, depth)
    expect = np.full(out_shape, off_value)
    dims = []
    for dim in shape:
        dims.append(list(range(dim)))
    import itertools
    indexs = [x for x in itertools.product(*tuple(dims))]
    indexs = [list(x) for x in indexs]
    temp = 0
    flatinput = data.flatten()
    for value in flatinput:
        indexs[temp].insert(axis, value)
        temp = temp + 1
    indexs = [tuple(x) for x in indexs]
    for loc in indexs:
        if loc[axis] >= 0:
            expect[loc] = on_value
    expect = expect.astype(dtype)
    return expect


def reduce_str(inputs, output, attr, op_type):
    """gen sum string"""
    axis = []
    keepdims = False
    axis_value = get_attr(attr, "axis")
    if axis_value:
        axis = list(axis_value) if isinstance(axis_value, (list, tuple)) else [axis_value]
    keepdims_value = get_attr(attr, "keep_dims")
    keepdims = keepdims_value if keepdims_value else keepdims

    if not axis:
        s = "%s = np.%s(%s.astype(np.float32) if %s.dtype == np.float16 else %s, keepdims=%s).astype(%s.dtype)" % (
            output[0]['tensor_name'], op_type, get_input(inputs[0][0]), get_input(inputs[0][0]),
            get_input(inputs[0][0]), keepdims, get_input(inputs[0][0]))
    else:
        s = "%s = np.%s(%s.astype(np.float32) if %s.dtype == np.float16 else %s, axis=tuple(%s), keepdims=%s)" \
            ".astype(%s.dtype); %s = np.reshape(%s, %s) " % \
            (output[0]['tensor_name'], op_type, get_input(inputs[0][0]), get_input(inputs[0][0]),
             get_input(inputs[0][0]), axis, keepdims, get_input(inputs[0][0]),
             output[0]['tensor_name'], output[0]['tensor_name'], output[0]['shape'])
    return s


def cast_str(inputs, output, attr):
    """gen cast string"""
    dst_type = get_attr(attr, "dst_type")
    if inputs[0][0].get('value', None) is not None:
        s = "%s = np.array(%s).astype(np.%s)" % (output[0]['tensor_name'], get_input(inputs[0][0]), dst_type)
    else:
        s = "%s = %s.astype(np.%s)" % (output[0]['tensor_name'], get_input(inputs[0][0]), dst_type)
    return s


def broadcast_str(inputs, output, attr):
    """gen broadcast string"""
    dst_shape = get_attr(attr, "shape")
    s = "%s = np.broadcast_to(%s, %s)" % (
        output[0]["tensor_name"], get_input(inputs[0][0]), dst_shape)
    return s


def transpose_str(inputs, output, attr):
    """gen transpose string"""
    axes = None
    axes_value = get_attr(attr, "perm")
    axes = axes_value if axes_value else axes
    s = "%s = np.transpose(%s, axes=%s)" % (
        output[0]['tensor_name'], get_input(inputs[0][0]), axes)
    return s

def concat_str(inputs, output, attr):
    axis = get_attr(attr, 'axis')
    inputs_list = []
    for i in range(len(inputs[0])):
        inputs_list.append(get_input(inputs[0][i]))
    s = '{} = np.concatenate(({}), axis={})'.format(output[0]['tensor_name'], ', '.join(inputs_list), axis)
    return s
    

def trans_data_two2fractal(input_, src_format, dst_format):
    """two2fractal"""
    shape = list(input_.shape)
    dtype = input_.dtype
    if src_format == "DefaultFormat" or src_format == "NCHW":
        m, n = shape[-2], shape[-1]
        m1, n1 = m // 16, n // 16
        m0, n0 = 16, 16
        need_pad = m % 16 != 0 or n % 16 != 0
        if need_pad:
            pad_m, pad_n = (m + 15) // 16 * 16, (n + 15) // 16 * 16
            pad_shape = [x for x in shape]
            pad_shape[-1] = pad_n
            pad_shape[-2] = pad_m
            pad_input = np.zeros(pad_shape).astype(dtype)
            if len(shape) == 2:
                pad_input[:m, :n] = input_
            elif len(shape) == 3:
                pad_input[:, :m, :n] = input_
            elif len(shape) == 4:
                pad_input[:, :, :m, :n] = input_
            m1, n1 = pad_m // 16, pad_n // 16
            reshape_shape = shape[:-2] + [m1, m0, n1, n0]
            reshape_input = pad_input.reshape(reshape_shape)
        else:
            reshape_shape = shape[:-2] + [m1, m0, n1, n0]
            reshape_input = input_.reshape(reshape_shape)
        if dst_format == "FRACTAL_NZ":
            transpose_axis = [2, 0, 1, 3]
        else:
            raise ValueError("dst_format %s is not supported when src_format is %s" % (
                dst_format, src_format))
        transpose_axis = [x + len(shape) - 2 for x in transpose_axis]
        transpose_axis = [x for x in range(len(shape) - 2)] + transpose_axis
        bench_mark = reshape_input.transpose(transpose_axis)
        return bench_mark
    raise ValueError("src_format %s is not supported!" % src_format)


def trans_data_fractal2two(input_, shape_origin):
    """fractal2two"""
    shape_origin = [int(_) for _ in shape_origin]
    shape = list(input_.shape)
    n1, m1, m0, n0 = shape[-4:]
    new_shape = shape[:-4] + [m1 * m0, n1 * n0]
    transpose_axis = [1, 2, 0, 3]
    transpose_axis = [x + len(shape) - 4 for x in transpose_axis]
    transpose_axis = [i for i in range(len(shape) - 4)] + transpose_axis
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
        raise ValueError("src_format %s and dst_format %s is not supported!" %
                         (src_format, dst_format))

    if (src_format == 'DefaultFormat' or src_format == "NCHW") and dst_format == 'FRACTAL_NZ':
        res = "%s \n%s = %s(%s, '%s', '%s')" % (inspect.getsource(trans_data_two2fractal),
                                                output_name, trans_data_two2fractal.__name__, input_name,
                                                src_format, dst_format)
    elif src_format == 'FRACTAL_NZ' and (dst_format == 'DefaultFormat' or dst_format == "NCHW"):
        res = "%s \n%s = %s(%s, %s)" % (inspect.getsource(trans_data_fractal2two),
                                        output_name, trans_data_fractal2two.__name__, input_name, ori_shape)
    else:
        raise ValueError("src_format(%s) and dst_format(%s) is not supported!" % (src_format, dst_format))
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
    if input_0['data_type'] == "float16":
        tmp = "%s = %s.astype(np.float32)" % (get_input(input_0), get_input(input_0))
        res = res + tmp + "\n"
    if input_1['data_type'] == "float16":
        tmp = "%s = %s.astype(np.float32)" % (get_input(input_1), get_input(input_1))
        res = res + tmp + "\n"

    if trans_a and trans_b:
        res += "%s = np.matmul(np.swapaxes(%s, -1, -2), np.swapaxes(%s, -1, -2))" % \
               (output[0]['tensor_name'], get_input(inputs[0][0]), get_input(inputs[1][0]))
    elif trans_a:
        res += "%s = np.matmul(np.swapaxes(%s, -1, -2), %s)" % \
               (output[0]['tensor_name'], get_input(inputs[0][0]), get_input(inputs[1][0]))
    elif trans_b:
        res += "%s = np.matmul(%s, np.swapaxes(%s, -1, -2))" % \
               (output[0]['tensor_name'], get_input(inputs[0][0]), get_input(inputs[1][0]))
    else:
        res += "%s = np.matmul(%s, %s)" % \
               (output[0]['tensor_name'], get_input(inputs[0][0]), get_input(inputs[1][0]))
    if output[0]['data_type'] == "float16":
        res += "\n" + "%s = %s.astype(np.float16)" % (output[0]['tensor_name'], output[0]['tensor_name'])
    return res


def convert_fracal_shape(ori_shape, fractal):
    """convert fractal shape to default shape"""
    ori_shape = tuple(ori_shape)
    if fractal == "zN":
        return ori_shape[:-4] + (ori_shape[-2] * ori_shape[-3], ori_shape[-1] * ori_shape[-4])
    if fractal == "zZ":
        return ori_shape[:-4] + (ori_shape[-4] * ori_shape[-2], ori_shape[-3] * ori_shape[-1])


def strided_slice_str(inputs, output, attr):
    """gen strided_slice string"""
    begin = get_attr(attr, "begin")
    end = get_attr(attr, "end")
    strides = get_attr(attr, "strides")
    shape = inputs[0][0]["shape"]
    begin_pos = [0] if get_attr(attr, "begin_mask") == [] else bin(get_attr(attr, "begin_mask"))[-1:1:-1]
    end_pos = [0] if get_attr(attr, "end_mask") == [] else bin(get_attr(attr, "end_mask"))[-1:1:-1]
    ellipsis_pos = [0] if get_attr(attr, "ellipsis_mask") == [] else bin(get_attr(attr, "ellipsis_mask"))[-1:1:-1]
    new_axis_pos = [0] if get_attr(attr, "new_axis_mask") == [] else bin(get_attr(attr, "new_axis_mask"))[-1:1:-1]
    shrink_axis_pos = [0] if get_attr(attr, "shrink_axis_mask") == [] else \
        bin(get_attr(attr, "shrink_axis_mask"))[-1:1:-1]
    new_axis = -1
    shrink_axis = -1
    slice_str = ""
    for i, bg in enumerate(begin):
        if i < len(begin_pos) and begin_pos[i] == '1':
            # use the largest range
            start_num = 0 if strides[i] >= 0 else -1
        else:
            start_num = bg
        if i < len(end_pos) and end_pos[i] == '1':
            # use the largest range
            end_num = shape[i] if strides[i] >= 0 else -shape[i] - 1
        else:
            end_num = end[i]
        if i < len(ellipsis_pos) and ellipsis_pos[i] == '1':
            # do not change on this axis
            start_num = 0
            end_num = shape[i]
            strides_num = 1
        else:
            strides_num = strides[i]
        if i < len(new_axis_pos) and new_axis_pos[i] == '1':
            # insert a new axis and ignore slice
            start_num = 0
            end_num = shape[i]
            strides_num = 1
            new_axis = i
        else:
            strides_num = strides[i]
        if i < len(shrink_axis_pos) and shrink_axis_pos[i] == '1':
            # delete an axis
            if start_num < 0:
                start_num += shape[i]
            end_num = start_num + 1
            strides_num = 1
            shrink_axis = i
        else:
            strides_num = strides[i]
        slice_str += str(start_num) + ':' + str(end_num) + ':' + str(strides_num)
        if not i == len(begin) - 1:
            slice_str += ","
    res = "%s = %s[%s]" % (output[0]['tensor_name'], get_input(inputs[0][0]), slice_str)
    if new_axis != -1:
        res += "\n%s = np.expand_dims(%s, axis=%s)" % (
            output[0]['tensor_name'], output[0]['tensor_name'], str(new_axis))
    if shrink_axis != -1:
        res += "\n%s = np.squeeze(%s, axis=%s)" % (output[0]['tensor_name'], output[0]['tensor_name'], str(shrink_axis))
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
    left_format = get_attr(attr, "left_format")
    if not left_format:
        left_format = get_attr(attr, "pri_format")
    right_format = get_attr(attr, "right_format")
    if not right_format:
        right_format = get_attr(attr, "pri_format")
    trans_a = get_attr(attr, "transpose_a")
    trans_b = get_attr(attr, "transpose_b")
    left_input = inputs[0][0]
    right_input = inputs[1][0]
    output_name = output[0]['tensor_name']
    output_format = output[0]['format']
    output_shape = output[0]['shape']
    right_ori_shape = convert_fracal_shape(right_input['shape'], right_format)

    left_input_name = get_input(left_input)
    right_input_name = get_input(right_input)
    res = ''
    if get_attr(attr, 'process') != 'cpu':
        if left_format == 'FRACTAL_NZ':
            left_ori_shape = convert_fracal_shape(left_input['shape'], "zN")
            left_trans_str = get_trans_data_str(left_input_name, left_input_name, left_ori_shape, left_format,
                                                'DefaultFormat')
            res = res + left_trans_str + "\n"
        if right_format == 'FRACTAL_NZ':
            right_ori_shape = convert_fracal_shape(right_input['shape'], "zN")
            right_trans_str = get_trans_data_str(right_input_name, right_input_name, right_ori_shape, right_format,
                                                'DefaultFormat')
            res = res + right_trans_str + "\n"
    mm_str = np_matmul_str(inputs, output, attr)
    res = res + mm_str + "\n"

    has_bias = (len(inputs) > 2)
    if has_bias:
        bias = inputs[2][0]
        bias_shape = right_ori_shape[-2] if trans_b else right_ori_shape[-1]
        if bias['shape'][0] != bias_shape:
            res += "%s = random_gaussian([%s, ], miu=1, sigma=0.1).astype(np.%s) \n" % (
                get_input(bias), str(bias_shape), bias['data_type'])

        res += "%s = np.add(%s, %s)\n" % (output_name, output_name, get_input(bias))
    if get_attr(attr, 'process') != 'cpu' and output_format != 'DefaultFormat':
        output_trans_str = get_trans_data_str(output_name, output_name, output_shape, 'DefaultFormat', output_format)
        res = res + output_trans_str + "\n"
    func_name = "matmul_func"
    params = [get_input(i[0]) for i in inputs]
    func = func_pack(func_name, res, params, output_name)
    return func + "%s = %s(%s)\n" % (output_name, func_name, ','.join(params))


def custom_str(inputs, output, attr):
    """gen custom string"""
    func_name = get_attr(attr, "func_name")
    func = get_attr(attr, "func_source_str").replace("output_tensor", "np.zeros")
    params = [get_input(i[0]) for i in inputs]
    output_name = output[0]['tensor_name']
    return func + "%s = %s(%s)\n" % (output_name, func_name, ','.join(params))

def conv_2d_nchwc_str(inputs, output, attr):
    
    support_list = {"float32": 'np.float32'}
    # NCHWc
    shape_data = inputs[0][0]['shape']
    shape_data_name = inputs[0][0]['tensor_name']
    # OIHWio
    shape_filter = inputs[1][0]['shape']
    shape_filter_name = inputs[1][0]['tensor_name']
    dtype = inputs[0][0]['data_type']
    padding = get_attr(attr, "pad_list")
    has_pad = np.sum(padding) > 0
    stride = get_attr(attr, "stride")
    dilation = get_attr(attr, "dilation")
    out_dtype = output[0]["data_type"]
    output_name = output[0]["tensor_name"]

    res = ""
    res += "n, c_i_o, h, w, c_i_i = {} \n".format(shape_data)
    res += "c_o_o, c_i_o, kh, kw, c_i_i, c_o_i = {}\n".format(shape_filter)
    res += "s_h, s_w = {}\n".format(stride)
    res += "d_h, d_w = {}\n".format(dilation)
    res += "p_l, p_r, p_t, p_b = {}\n".format(padding)
    res += "k_h_d = (kh - 1) * d_h + 1\n"
    res += "k_w_d = (kw - 1) * d_w + 1\n"
    res += "out_h = (h + p_t + p_b - k_h_d) // s_h + 1\n"
    res += "out_w = (w + p_l + p_r - k_w_d) // s_w + 1\n"

    res += "out_shape = (n, c_o_o, out_h, out_w, c_o_i)\n"
    res += "shape_data_pad = (n, c_i_o, h + p_t + p_b, w + p_l + p_r, c_i_i)\n"

    res += "data_pad = np.zeros(shape_data_pad).astype({})\n".format(support_list[dtype])
    if has_pad:
        res += ("data_pad[:, :, p_t:p_t+h, p_l:p_l+w, :] = {}\n".format(shape_data_name))
    else:
        res += ("data_pad = {}\n".format(shape_data_name))

    res += "{} = np.zeros(out_shape).astype({})\n".format(output_name, support_list[out_dtype])
    res += "for oh in range(out_h):\n"
    res += "    for ow in range(out_w):\n"
    res += "        for ic_out in range(c_i_o):\n"
    res += "            for khh in range(kh):\n"
    res += "                for kww in range(kw):\n"
    res += "                    for ic_in in range(c_i_i):\n"
    compute_str = "{}[:, :, oh, ow, :] = {}[:, :, oh, ow, :] + \
        data_pad[:, ic_out, oh*s_h+khh*d_h, ow*s_w+kww*d_w, ic_in] * {}[:, ic_out, khh, kww, ic_in, :]\n".format(
        output_name, output_name, shape_filter_name
    )
    res += "                        " + compute_str

    return res

def depthwise_conv_2d_nchwc_str(inputs, output, attr):
    
    support_list = {"float32": 'np.float32'}
    # NCHWc
    shape_data = inputs[0][0]['shape']
    shape_data_name = inputs[0][0]['tensor_name']
    # OIHWio
    shape_filter = inputs[1][0]['shape']
    shape_filter_name = inputs[1][0]['tensor_name']
    dtype = inputs[0][0]['data_type']
    padding = get_attr(attr, "pad_list")
    has_pad = np.sum(padding) > 0
    stride = get_attr(attr, "stride")
    dilation = get_attr(attr, "dilation")
    out_dtype = output[0]["data_type"]
    output_name = output[0]["tensor_name"]

    res = ""
    res += "n, c_i_o, h, w, c_i_i = {} \n".format(shape_data)
    res += "c_o_o, cm_outer, kh, kw, cm_inner, c_o_i = {}\n".format(shape_filter)
    res += "s_h, s_w = {}\n".format(stride)
    res += "d_h, d_w = {}\n".format(dilation)
    res += "p_l, p_r, p_t, p_b = {}\n".format(padding)
    res += "k_h_d = (kh - 1) * d_h + 1\n"
    res += "k_w_d = (kw - 1) * d_w + 1\n"
    res += "out_h = (h + p_t + p_b - k_h_d) // s_h + 1\n"
    res += "out_w = (w + p_l + p_r - k_w_d) // s_w + 1\n"

    res += "out_shape = (n, c_o_o, out_h, out_w, c_o_i)\n"
    res += "shape_data_pad = (n, c_i_o, h + p_t + p_b, w + p_l + p_r, c_i_i)\n"

    res += "data_pad = np.zeros(shape_data_pad).astype({})\n".format(support_list[dtype])
    if has_pad:
        res += ("data_pad[:, :, p_t:p_t+h, p_l:p_l+w, :] = {}\n".format(shape_data_name))
    else:
        res += ("data_pad = {}\n".format(shape_data_name))

    res += "{} = np.zeros(out_shape).astype({})\n".format(output_name, support_list[out_dtype])
    res += "for oh in range(out_h):\n"
    res += "    for ow in range(out_w):\n"
    res += "        for oc_out in range(c_o_o):\n"
    res += "            for khh in range(kh):\n"
    res += "                for kww in range(kw):\n"
    res += "                    for oc_in in range(c_o_i):\n"
    compute_str = "{}[:, oc_out, oh, ow, oc_in] = {}[:, oc_out, oh, ow, oc_in] + \
        data_pad[:, (oc_out*c_o_i+oc_in)//c_i_i, oh*s_h+khh*d_h, ow*s_w+kww*d_w, (oc_out*c_o_i+oc_in)%c_i_i] * {}[oc_out, 0, khh, kww, 0, oc_in]\n".format(
        output_name, output_name, shape_filter_name
    )
    res += "                        " + compute_str

    return res

def conv_2d_str(inputs, output, attr):
    """gen conv_2d string"""
    is_conv2d_nchwc = get_attr(attr, "data_format") == "NC1HWC0"
    
    if is_conv2d_nchwc:
        is_depth_wise = get_attr(attr, "is_depth_wise") == True
        if is_depth_wise:
            return depthwise_conv_2d_nchwc_str(inputs, output, attr)
        return conv_2d_nchwc_str(inputs, output, attr)

    support_list = {"float16": 'np.float16', "float32": 'np.float32'}
    shape_data = inputs[0][0]['shape']
    shape_data_name = inputs[0][0]['tensor_name']
    shape_filter = inputs[1][0]['shape']
    shape_filter_name = inputs[1][0]['tensor_name']
    dtype = inputs[0][0]['data_type']
    padding = get_attr(attr, "pad_list")
    has_pad = np.sum(padding) > 0
    stride = get_attr(attr, "stride")[2:]
    dilation = get_attr(attr, "dilation")[2:]
    out_dtype = output[0]["data_type"]
    output_name = output[0]["tensor_name"]

    res = ""
    res += "n, h, w, c = {} \n".format(shape_data)
    res += "out_c, kh, kw, c = {}\n".format(shape_filter)
    res += "s_h, s_w = {}\n".format(stride)
    res += "d_h, d_w = {}\n".format(dilation)
    res += "p_l, p_r, p_t, p_b = {}\n".format(padding)
    res += "out_h = (h + p_t + p_b - kh) // s_h + 1\n"
    res += "out_w = (w + p_l + p_r - kw) // s_w + 1\n"

    res += "out_shape = (n, out_h, out_w, out_c)\n"
    res += "shape_data_pad = (n, h + p_t + p_b, w + p_l + p_r, c)\n"

    res += "data_pad = np.zeros(shape_data_pad).astype({})\n".format(support_list.get(dtype))
    if has_pad:
        res += ("data_pad[:, p_t:p_t+h, p_l:p_l+w, :] = {}\n".format(shape_data_name))
    else:
        res += ("data_pad = {}\n".format(shape_data_name))

    res += "whd = (kh - 1) * d_h + 1\n"
    res += "wwd = (kw - 1) * d_w + 1\n"
    res += "{} = np.zeros(out_shape).astype({})\n".format(output_name, support_list.get(out_dtype))
    res += "for i in range(out_h):\n"
    res += "    for j in range(out_w):\n"
    res += "        for f in range(out_c):\n"
    res += ("            {}[:, i, j, f] = np.sum(data_pad[:, i*s_h:i*s_h+whd:d_h, j*s_w:j*s_w+wwd:d_w, :]"
            ".astype('float32') *{}[f, :, :, :].astype('float32'),axis=(1, 2, 3))\n"
            .format(output_name, shape_filter_name))
    return res


def pad_str(inputs, output, attr):
    """gen pad string"""
    tail_value = get_attr(attr, "tail")
    head_value = get_attr(attr, "head")
    value = get_attr(attr, "pad_val")
    pad_width = list(zip(head_value, tail_value))
    res = "%s = np.pad(%s, %s, mode='constant', constant_values=(%s, %s))" % (
        output[0]['tensor_name'], get_input(inputs[0][0]), pad_width, value, value)
    return res


def unpad_str(inputs, output, attr):
    """gen unpad string"""
    input_shape = inputs[0][0]['shape']
    unpad_after = get_attr(attr, "tail")
    res = ""
    res += "m, n = {} - {}, {} - {}\n".format(input_shape[-2], unpad_after[-2], input_shape[-1], unpad_after[-1])
    res += "if {} == 4:\n".format(len(input_shape))
    res += "    %s = %s[:, :, :m, :n]\n" % (
        output[0]['tensor_name'], get_input(inputs[0][0]))
    res += "elif {} == 2:\n".format(len(input_shape))
    res += "    %s = %s[:m, :n]\n" % (
        output[0]['tensor_name'], get_input(inputs[0][0]))
    return res


def cummulative_str(inputs, outputs, attr, op_type):
    """gen cummulative sum str and product str"""
    exclusive = get_attr(attr, "exclusive")
    reverse = get_attr(attr, "reverse")
    axis = get_attr(attr, "axis")
    input_shape = inputs[0][0]['shape']
    res = ""
    res += "axis = {}\n".format(axis)
    if reverse:
        res += "out = np.flip({}, axis)\n".format(get_input(inputs[0][0]))
        res += "out = np.{}(out, axis)\n".format(op_type)
    else:
        res += "out = np.{}({}, axis)\n".format(op_type, get_input(inputs[0][0]))
    if exclusive:
        res += "from scipy.ndimage.interpolation import shift\n"
        res += "shift_axis = [0] * {}\n".format(len(input_shape))
        res += "shift_axis[axis] = 1\n"
        res += "out = shift(out, shift_axis)\n"
    if reverse:
        res += "out = np.flip(out, axis=axis)\n"
    res += "{} = out\n".format(outputs[0]['tensor_name'])
    return res

def pool2d_str(inputs, output, attr):

    import tvm
    import akg.topi as akg_topi
    from akg.topi.util import get_const_tuple

    if get_attr(attr, "global"):
        # global_pool2d python impl
        return global_pool2d_str(inputs, output, attr)

    kh, kw = get_attr(attr, "kernel_size")
    sh, sw = get_attr(attr, "strides")
    paddings = get_attr(attr, "pad")
    pt, pl, pb, pr = paddings if len(paddings) == 4 else [0, 0, 0, 0]
    pool_type = get_attr(attr, "pool_type")
    ceil_mode = get_attr(attr, "round_mode")
    count_include_pad = True
    data_layout = get_attr(attr, "data_layout")

    res = ""

    # We temporarily reshape all format as NCHWc
    if data_layout == "NCHW":
        n, ic_out, ih, iw = inputs[0][0]['shape']
        ic_in = 1

    elif data_layout == "NHWC":
        n, ih, iw, ic_in = inputs[0][0]['shape']
        ic_out = 1
    else:
        # NCHWc
        import re
        pattern = re.compile(r'NCHW\d*c')
        if pattern.match(data_layout) == None:
            raise ValueError("Invalid data_layout = {}".format(data_layout))
        n, ic_out, ih, iw, ic_in = inputs[0][0]['shape']

    tmp_shape = "({},{},{},{},{})".format(n, ic_out, ih, iw, ic_in)
    res += "tmp_data = np.reshape({}, {})\n".format(inputs[0][0]["tensor_name"], tmp_shape)
    res += "pt, pl, pb, pr = {}, {}, {}, {}\n".format(pt, pl, pb, pr)
    res += "kh ,kw = {}, {}\n".format(kh, kw)
    res += "sh, sw = {}, {}\n".format(sh, sw)
    input0 = tvm.placeholder((n, ic_out, ih, iw, ic_in), name='a')
    output0 = akg_topi.nn.pool(input0, kernel=[kh, kw], stride=[sh, sw], padding=(pt, pl, pb, pr),
                         pool_type=pool_type, ceil_mode=ceil_mode,
                         layout="NCHWc", count_include_pad=count_include_pad)
    _, oc_out, oh, ow, oc_in = get_const_tuple(output0.shape)
    res += "n, ic_out, ih, iw, ic_in = {}, {}, {}, {}, {}\n".format(
        n, ic_out, ih, iw, ic_in)
    res += "oc_out, oh, ow, oc_in = {}, {}, {}, {}\n".format(
        oc_out, oh, ow, oc_in)
    res += "dtype = np.{}\n".format(inputs[0][0]["data_type"])
    res += "pad_np = np.zeros(shape=(n, ic_out, ih+pt+pb, iw+pl+pr, ic_in)).astype(dtype)\n"
    res += "no_zero = (range(n), range(ic_out), (range(pt, ih+pt)), (range(pl, iw+pl)), range(ic_in))\n"
    res += "pad_np[np.ix_(*no_zero)] = tmp_data\n"
    res += "b_np = np.zeros(shape=(n, oc_out, oh, ow, oc_in)).astype(dtype)\n"

    if pool_type == "avg":
        res += "for i in range(oh):\n"
        res += "    for j in range(ow):\n"
        res += "        if count_include_pad:\n"
        res += "            b_np[:, :, i, j, :] = np.mean(\n"
        res += "                pad_np[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw, :], axis=(2, 3))\n"
        res += "        else:\n"
        res += "            pad_count = np.sum(\n"
        res += "                pad_np[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw, :] > 0, axis=(2, 3))\n"
        res += "            b_np[:, :, i, j, :] = np.sum(\n"
        res += "                pad_np[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw, :], axis=(2, 3)) / np.maximum(pad_count, 1)\n"
    elif pool_type == "max":
        res += "for i in range(oh):\n"
        res += "    for j in range(ow):\n"
        res += "        b_np[:, :, i, j, :] = np.max(\n"
        res += "            pad_np[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw, :], axis=(2, 3))\n"
    else:
        raise ValueError("Invalid pool_type=={}".format(pool_type))

    if data_layout == "NCHW" or data_layout == "NHWC":
        res += "{} = np.squeeze(b_np)\n".format(output[0]["tensor_name"])
    else:
        res += "{} = b_np\n".format(output[0]["tensor_name"])

    return res


def global_pool2d_str(inputs, output, attr):
    pool_type = get_attr(attr, "pool_type")
    data_layout = get_attr(attr, "data_layout")
    res = ""
    if data_layout == "NHWC":
        res += "pool_idxs = (1, 2)\n"
    elif data_layout[:4] == "NCHW":
        # NCHW or NCHWc/ NCHW[x]c
        res += "pool_idxs = (2, 3)\n"

    res += "global_pool_input = {}\n".format(inputs[0][0]["tensor_name"])
    if pool_type == 'avg':
        res += "{} = np.mean(global_pool_input, axis=pool_idxs, keepdims=True)\n".format(
            output[0]["tensor_name"])
    elif pool_type == 'max':
        res += "{} = np.max(global_pool_input, axis=pool_idxs, keepdims=True)\n".format(
            output[0]["tensor_name"])
    return res


def layout_transform_str(inputs, output, attr, op_type):
    """gen layout_transform string"""

    from akg.ops.nn.cpu import get_layout_list, get_tiled_pair, \
                                get_idx_by_char, get_tile_by_char

    res = "tmp_data = {}\n".format(get_input(inputs[0][0]))

    tmp_data = np.zeros(inputs[0][0]['shape'])
    data_layout = get_attr(attr, "src_format")
    output_layout = get_attr(attr, "dst_format")
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
        res += "tmp_data = np.transpose(tmp_data, {})\n".format(str(perm))
        tmp_data = np.reshape(tmp_data, new_shape)
        res += "tmp_data = np.reshape(tmp_data, {})\n".format(str(new_shape))
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
        res += "tmp_data = np.reshape(tmp_data, {})\n".format(str(new_shape))
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
    res += "tmp_data = np.transpose(tmp_data, {})\n".format(str(perm))

    res += "{} = tmp_data\n".format(output[0]['tensor_name'])
    return res


def fast_gelu_str(inputs, output, attr):
    x = inputs[0][0]["tensor_name"]
    x_type = inputs[0][0]["data_type"]
    res = "abs_x = np.absolute(%s)\n" % x
    res += "sub1 = np.subtract(%s, abs_x)\n" % x
    res += "mul1 = np.multiply(sub1, np.%s(0.851))\n" % x_type
    res += "exp1 = np.exp(mul1)\n"
    res += "tmp1 = np.multiply(%s, exp1)\n" % x
    res += "mul2 = np.multiply(abs_x, np.%s(-1.702))\n" % x_type
    res += "exp2 = np.exp(mul2)\n"
    res += "tmp2 = np.add(exp2, np.%s(1))\n" % x_type
    res += "%s = np.divide(tmp1, tmp2)\n" % output[0]["tensor_name"]
    return res


op_dsl = {
    "Custom": lambda inputs, output, attr: custom_str(inputs, output, attr),
    "ReduceSum": lambda inputs, output, attr: reduce_str(inputs, output, attr, "sum"),
    "ReduceMax": lambda inputs, output, attr: reduce_str(inputs, output, attr, "max"),
    "ReduceMin": lambda inputs, output, attr: reduce_str(inputs, output, attr, "min"),
    "ReduceProd": lambda inputs, output, attr: reduce_str(inputs, output, attr, "prod"),
    "StridedSlice": lambda inputs, output, attr: strided_slice_str(inputs, output, attr),
    "CumSum": lambda inputs, output, attr: cummulative_str(inputs, output, attr, "cumsum"),
    "CumProd": lambda inputs, output, attr: cummulative_str(inputs, output, attr, "cumprod"),
    "Sin": lambda inputs, output, attr: "%s = np.sin(%s)" %
                                        (output[0]['tensor_name'], get_input(inputs[0][0])),
    "Cos": lambda inputs, output, attr: "%s = np.cos(%s)" %
                                        (output[0]['tensor_name'], get_input(inputs[0][0])),
    "Asin": lambda inputs, output, attr: "%s = np.arcsin(%s)" %
                                         (output[0]['tensor_name'], get_input(inputs[0][0])),
    "ACos": lambda inputs, output, attr: "%s = np.arccos(%s)" %
                                         (output[0]['tensor_name'], get_input(inputs[0][0])),
    "Sign": lambda inputs, output, attr: "%s = np.sign(%s)" %
                                         (output[0]['tensor_name'], get_input(inputs[0][0])),
    "IsNan": lambda inputs, output, attr: "%s = np.isnan(%s)" %
                                          (output[0]['tensor_name'], get_input(inputs[0][0])),
    "IsInf": lambda inputs, output, attr: "%s = np.isinf(%s)" %
                                          (output[0]['tensor_name'], get_input(inputs[0][0])),
    "IsFinite": lambda inputs, output, attr: "%s = np.isfinite(%s)" %
                                             (output[0]['tensor_name'], get_input(inputs[0][0])),
    "Tanh": lambda inputs, output, attr: "%s = np.tanh(%s)" %
                                         (output[0]['tensor_name'], get_input(inputs[0][0])),
    "Mul": lambda inputs, output, attr: "%s = np.multiply(%s, %s)" %
                                        (output[0]['tensor_name'], get_input(
                                            inputs[0][0]), get_input(inputs[1][0])),
    "Pow": lambda inputs, output, attr: "%s = np.power(%s, %s)" %
                                        (output[0]['tensor_name'], get_input(
                                            inputs[0][0]), get_input(inputs[1][0])),
    "Sub": lambda inputs, output, attr: "%s = np.subtract(%s, %s)" %
                                        (output[0]['tensor_name'], get_input(
                                            inputs[0][0]), get_input(inputs[1][0])),
    "TensorAdd": lambda inputs, output, attr: "%s = np.add(%s, %s)" %
                                              (output[0]['tensor_name'], get_input(
                                                  inputs[0][0]), get_input(inputs[1][0])),
    "Add": lambda inputs, output, attr: "%s = np.add(%s, %s)" %
                                        (output[0]['tensor_name'], get_input(
                                            inputs[0][0]), get_input(inputs[1][0])),
    "Rsqrt": lambda inputs, output, attr: "%s = 1.0/np.sqrt(%s)" %
                                          (output[0]['tensor_name'], get_input(inputs[0][0])),
    "Neg": lambda inputs, output, attr: "%s = np.negative(%s)" %
                                        (output[0]['tensor_name'], get_input(inputs[0][0])),
    "Floor": lambda inputs, output, attr: "%s = np.floor(%s)" %
                                        (output[0]['tensor_name'], get_input(inputs[0][0])),
    "Exp": lambda inputs, output, attr: "%s = np.exp(%s)" %
                                        (output[0]['tensor_name'], get_input(inputs[0][0])),
    "RealDiv": lambda inputs, output, attr: "%s = np.divide(%s, %s)" %
                                            (output[0]['tensor_name'], get_input(
                                                inputs[0][0]), get_input(inputs[1][0])),
    "Div": lambda inputs, output, attr: "%s = np.divide(%s, %s)" %
                                        (output[0]['tensor_name'], get_input(
                                            inputs[0][0]), get_input(inputs[1][0])),
    "FloorDiv": lambda inputs, output, attr: "%s = np.floor_divide(%s, %s)" %
                                             (output[0]['tensor_name'], get_input(
                                                 inputs[0][0]), get_input(inputs[1][0])),
    "Mod": lambda inputs, output, attr: "%s = np.fmod(%s, %s)" %
                                        (output[0]['tensor_name'], get_input(
                                            inputs[0][0]), get_input(inputs[1][0])),
    "FloorMod": lambda inputs, output, attr: "%s = np.mod(%s, %s)" %
                                             (output[0]['tensor_name'], get_input(
                                                 inputs[0][0]), get_input(inputs[1][0])),
    "Minimum": lambda inputs, output, attr: "%s = np.minimum(%s, %s)" %
                                            (output[0]['tensor_name'], get_input(
                                                inputs[0][0]), get_input(inputs[1][0])),
    "Maximum": lambda inputs, output, attr: "%s = np.maximum(%s, %s)" %
                                            (output[0]['tensor_name'], get_input(
                                                inputs[0][0]), get_input(inputs[1][0])),
    "Log": lambda inputs, output, attr: "%s = np.log(%s)" %
                                        (output[0]['tensor_name'], get_input(inputs[0][0])),
    "Sqrt": lambda inputs, output, attr: "%s = np.sqrt(%s)" %
                                         (output[0]['tensor_name'], get_input(inputs[0][0])),
    "Cast": lambda inputs, output, attr: cast_str(inputs, output, attr),
    "Reshape": lambda inputs, output, attr: "%s = np.reshape(%s, %s)" %
                                            (output[0]['tensor_name'], get_input(inputs[0][0]), output[0]['shape']),
    "OneHot": lambda inputs, output, attr: "%s = one_hot_np(%s, %s, %s, %s, %s, np.%s)" %
                                           (output[0]['tensor_name'], get_input(inputs[0][0]), get_attr(attr, "axis"),
                                            get_input(inputs[1][0]),
                                            get_input(inputs[2][0]), get_attr(attr, "depth"), output[0]['data_type']),
    "ZerosLike": lambda inputs, output, attr: "%s = np.zeros_like(%s)" %
                                              (output[0]['tensor_name'], get_input(inputs[0][0])),
    "AddN": lambda inputs, output, attr: "%s = %s" %
                                         (output[0]['tensor_name'], ' + '.join([get_input(inputs[0][i])
                                                                                for i in range(0, len(inputs[0]))])),
    "Tile": lambda inputs, output, attr: "%s = np.tile(%s, %s)" %
                                         (output[0]['tensor_name'], get_input(
                                             inputs[0][0]), get_attr(attr, "multiples")),
    "Reciprocal": lambda inputs, output, attr: "%s = np.divide(1.0, %s)" %
                                               (output[0]['tensor_name'], get_input(inputs[0][0])),
    "Equal": lambda inputs, output, attr: "%s = np.equal(%s, %s)" %
                                          (output[0]['tensor_name'], get_input(
                                              inputs[0][0]), get_input(inputs[1][0])),
    "NotEqual": lambda inputs, output, attr: "%s = np.not_equal(%s, %s)" %
                                             (output[0]['tensor_name'], get_input(
                                                 inputs[0][0]), get_input(inputs[1][0])),
    "GreaterEqual": lambda inputs, output, attr: "%s = np.greater_equal(%s, %s)" %
                                                 (output[0]['tensor_name'], get_input(
                                                     inputs[0][0]), get_input(inputs[1][0])),
    "Select": lambda inputs, output, attr: "%s = np.where(%s, %s, %s)" %
                                           (output[0]['tensor_name'], get_input(inputs[0][0]),
                                            get_input(inputs[1][0]), get_input(inputs[2][0])),
    "InplaceAssign": lambda inputs, output, attr: "%s = %s; %s = %s" %
                                                  (get_input(inputs[0][0]), get_input(inputs[1][0]),
                                                   output[0]['tensor_name'], get_input(inputs[2][0])),
    "Greater": lambda inputs, output, attr: "%s = np.greater(%s, %s)" %
                                            (output[0]['tensor_name'], get_input(
                                                inputs[0][0]), get_input(inputs[1][0])),
    "SelectGT": lambda inputs, output, attr: "%s = np.where(%s > %s, %s, %s)" %
                                             (
                                                 output[0]['tensor_name'], get_input(inputs[0][0]),
                                                 get_input(inputs[1][0]),
                                                 get_input(inputs[2][0]), get_input(inputs[3][0])),
    "SelectLT": lambda inputs, output, attr: "%s = np.where(%s < %s, %s, %s)" %
                                             (
                                                 output[0]['tensor_name'], get_input(inputs[0][0]),
                                                 get_input(inputs[1][0]),
                                                 get_input(inputs[2][0]), get_input(inputs[3][0])),
    "Abs": lambda inputs, output, attr: "%s = np.absolute(%s)" %
                                        (output[0]['tensor_name'], get_input(inputs[0][0])),
    "LessEqual": lambda inputs, output, attr: "%s = np.less_equal(%s, %s)" %
                                              (output[0]['tensor_name'], get_input(inputs[0][0]),
                                               get_input(inputs[1][0])),
    "Less": lambda inputs, output, attr: "%s = np.less(%s, %s)" %
                                         (output[0]['tensor_name'], get_input(inputs[0][0]), get_input(inputs[1][0])),
    "EquivFormat": lambda inputs, output, attr: "%s = %s" %
                                                (output[0]['tensor_name'], get_input(inputs[0][0])),
    "ExpandDims": lambda inputs, output, attr: "%s = np.expand_dims(%s, %s)" %
                                               (output[0]['tensor_name'], get_input(inputs[0][0]),
                                                get_attr(attr, "axis")),
    "ElemAny": lambda inputs, output, attr: "%s = (%s.all() > 0).astype(np.%s).reshape(1)" %
                                               (output[0]['tensor_name'], get_input(inputs[0][0]),
                                                output[0]['data_type']),
    "Transpose": lambda inputs, output, attr: transpose_str(inputs, output, attr),
    "TransData": trans_data_dsl,
    "BroadcastTo": lambda inputs, output, attr: broadcast_str(inputs, output, attr),
    "BatchMatMul": lambda inputs, output, attr: matmul_str(inputs, output, attr),
    "Assign": lambda inputs, output, attr: "%s = %s; %s = %s" %
                                           (get_input(inputs[0][0]), get_input(inputs[1][0]), output[0]['tensor_name'],
                                            get_input(inputs[1][0])),
    "MatMul": lambda inputs, output, attr: matmul_str(inputs, output, attr),
    "Conv2D": lambda inputs, output, attr: conv_2d_str(inputs, output, attr),
    "PadAkg": lambda inputs, output, attr: pad_str(inputs, output, attr),
    "UnPadAkg": lambda inputs, output, attr: unpad_str(inputs, output, attr),
    "Asinh": lambda inputs, output, attr: "%s = np.arcsinh(%s)" %
                                          (output[0]['tensor_name'], get_input(inputs[0][0])),
    "Acosh": lambda inputs, output, attr: "%s = np.arccosh(%s)" %
                                          (output[0]['tensor_name'], get_input(inputs[0][0])),
    "Atan2": lambda inputs, output, attr: "%s = np.arctan2(%s, %s)" %
                                          (output[0]['tensor_name'], get_input(inputs[0][0]), get_input(inputs[1][0])),
    "Expm1": lambda inputs, output, attr: "%s = np.expm1(%s)" %
                                          (output[0]['tensor_name'], get_input(inputs[0][0])),
    "LogicalNot": lambda inputs, output, attr: "%s = np.logical_not(%s)" %
                                               (output[0]['tensor_name'], get_input(inputs[0][0])),
    "LogicalAnd": lambda inputs, output, attr: "%s = np.logical_and(%s, %s)" %
                                               (output[0]['tensor_name'], get_input(inputs[0][0]),
                                                get_input(inputs[1][0])),
    "LogicalOr": lambda inputs, output, attr: "%s = np.logical_or(%s, %s)" %
                                              (output[0]['tensor_name'], get_input(inputs[0][0]),
                                               get_input(inputs[1][0])),
    "Erf": lambda inputs, output, attr: "%s = sp.special.erf(%s)" %
                                        (output[0]['tensor_name'], get_input(inputs[0][0])),
    "TensorScatterAdd": lambda inputs, output, attr: "%s = tensor_scatter_add_np(%s, %s, %s)" %
                                                     (output[0]['tensor_name'], get_input(inputs[0][0]),
                                                      get_input(inputs[1][0]),
                                                      get_input(inputs[2][0])),
    "GatherNd": lambda inputs, output, attr: "%s = %s[tuple(%s.transpose().tolist())]" %
                                             (output[0]['tensor_name'], get_input(inputs[0][0]),
                                              get_input(inputs[1][0])),
    "UnsortedSegmentSum": lambda inputs, output, attr: "%s = np.zeros([%s,] + %s[%s:]);  np.add.at(%s, %s, %s)" %
                                                       (output[0]['tensor_name'], get_attr(attr, 'num_segments'),
                                                        inputs[0][0]['shape'], len(inputs[1][0]['shape']),
                                                        output[0]['tensor_name'], get_input(inputs[1][0]),
                                                        get_input(inputs[0][0])),
    "Gather": lambda inputs, output, attr: "%s = gather_np(%s, %s, %s)" %
                                           (output[0]['tensor_name'], get_input(inputs[0][0]), get_input(inputs[1][0]),
                                            get_attr(attr, "axis")),
    "StandardNormal": lambda inputs, output, attr: "%s = np.random.standard_normal(%s)" %
                                                   (output[0]['tensor_name'], get_attr(attr, "shape")),
    "CSRMV": lambda inputs, output, attr: "{} = csrmv_np({}, {}, {}, {}, {})".format(
                                           output[0]['tensor_name'], get_input(inputs[0][0]), get_input(inputs[1][0]),
                                           get_input(inputs[2][0]),
                                           get_input(inputs[3][0]), get_attr(attr, "dense_shape")),
    "CSRReduceSum": lambda inputs, output, attr: "{} = csr_reduce_sum_np({}, {}, {}, {}, {})".format(
                                                  output[0]['tensor_name'], get_input(inputs[0][0]),
                                                  get_input(inputs[1][0]), get_input(inputs[2][0]),
                                                  get_attr(attr, "dense_shape"), get_attr(attr, "axis")),
    "CSRMul": lambda inputs, output, attr: "{} = csr_mul_np({}, {}, {}, {}, {})".format(
                                                  output[0]['tensor_name'], get_input(inputs[0][0]),
                                                  get_input(inputs[1][0]), get_input(inputs[2][0]),
                                                  get_input(inputs[3][0]), get_attr(attr, "dense_shape")),
    "CSRDiv": lambda inputs, output, attr: "{} = csr_div_np({}, {}, {}, {}, {})".format(
                                                  output[0]['tensor_name'], get_input(inputs[0][0]),
                                                  get_input(inputs[1][0]), get_input(inputs[2][0]),
                                                  get_input(inputs[3][0]), get_attr(attr, "dense_shape")),
    "CSRGather": lambda inputs, output, attr: "{} = csr_gather_np({}, {}, {}, {})".format(
                                                  output[0]['tensor_name'], get_input(inputs[0][0]),
                                                  get_input(inputs[1][0]), get_input(inputs[2][0]),
                                                  get_attr(attr, "dense_shape")),
    "CSRMM": lambda inputs, output, attr: "{} = csrmm_np({}, {}, {}, {}, {})".format(
                                           output[0]['tensor_name'], get_input(inputs[0][0]), get_input(inputs[1][0]),
                                           get_input(inputs[2][0]),
                                           get_input(inputs[3][0]), get_attr(attr, "dense_shape")),
    "CImag": lambda inputs, output, attr: "%s = np.imag(%s)" %
    (output[0]['tensor_name'], get_input(inputs[0][0])),

    "CReal": lambda inputs, output, attr: "%s = np.real(%s)" %
    (output[0]['tensor_name'], get_input(inputs[0][0])),

    "Complex": lambda inputs, output, attr: "%s = np.vectorize(complex)(%s, %s)" %
    (output[0]['tensor_name'], get_input(inputs[0][0]),
                                               get_input(inputs[1][0])),
    "Pool2D": lambda inputs, output, attr: pool2d_str(inputs, output, attr),  
    "LayoutTransform": lambda inputs, output, attr: layout_transform_str(inputs, output, attr, "layout_transform"),
    "Concat": lambda inputs, output, attr: concat_str(inputs, output, attr),
    "FastGeLU": lambda inputs, output, attr: fast_gelu_str(inputs, output, attr)
}
