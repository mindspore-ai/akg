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

"""generate numpy data for composite json"""

import json
import logging
import numpy as np
from gen_random import random_gaussian
import inspect

class CodePrinter(object):
    """print numpy file"""
    def __init__(self, out_file):
        self.fout_ = open(out_file, 'w')
        self.indent_ = 0
    def __del__(self):
        self.fout_.close()
    def out(self, data, new_line=False):
        """write data"""
        if new_line:
            self.fout_.write("\n")
            for i in range(0, self.indent_):
                self.fout_.write('    ')
        if isinstance(data, str):
            self.fout_.write(data)
        else:
            self.fout_.write(str(data))
    def null_line(self):
        """add null line"""
        self.fout_.write("\n")
    def close(self):
        """close file"""
        self.fout_.close()

def get_input(desc):
    """get input values"""
    value = desc.get('value', None)
    return value if value is not None else desc['tensor_name']

def sum_str(inputs, output, attr):
    """gen sum string"""
    if attr[0]['value'] == []:
        s = "%s = np.sum(%s, keepdims=%s)" % (output[0]['tensor_name'], get_input(inputs[0][0]), attr[1]['value'])
    else:
        s = "%s = np.sum(%s, axis=tuple(%s), keepdims=%s); %s = np.reshape(%s, %s)" %\
            (output[0]['tensor_name'], get_input(inputs[0][0]), attr[0]['value'], attr[1]['value'],
             output[0]['tensor_name'], output[0]['tensor_name'], output[0]['shape'])
    return s


def trans_data_two2fractal(input_, src_format, dst_format):
    shape = list(input_.shape)
    dtype = input_.dtype
    if dtype == "float32":
        input_ = input_.astype(np.float16)
    if src_format == "DefaultFormat":
        m, n = shape[-2], shape[-1]
        m1, n1 = m // 16, n // 16
        m0, n0 = 16, 16
        needPad = m % 16 != 0 or n % 16 != 0
        if needPad:
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
            new_shape = [n1, m1, m0, n0]
        else:
            raise ValueError("dst_fromat %s is not suppored when src_format is %s"  %(
                            dst_format, src_format))
        transpose_axis = [x + len(shape) - 2 for x in transpose_axis]
        transpose_axis = [x for x in range(len(shape) - 2)] + transpose_axis
        new_shape = shape[:-2] + new_shape
        bench_mark = reshape_input.transpose(transpose_axis).astype('float16')
        return bench_mark


def trans_data_fractal2two(input_, src_format, dst_format, shape_origin):
    shape_origin = [int(_) for _ in shape_origin]
    shape = list(input_.shape)
    n1, m1, m0, n0 = shape[-4:]
    new_shape = shape[:-4] + [m1 * m0, n1 * n0]
    tranpose_axis = [1, 2, 0, 3]
    tranpose_axis = [x + len(shape) - 4 for x in tranpose_axis]
    tranpose_axis = [i for i in range(len(shape) - 4)] + tranpose_axis
    bench_mark = input_.transpose(tranpose_axis).reshape(new_shape)
    if new_shape != shape_origin:
        if len(shape_origin) == 2:
            bench_mark = bench_mark[:shape_origin[0], :shape_origin[1]]
        elif len(shape_origin) == 3:
            bench_mark = bench_mark[:, shape_origin[0], :shape_origin[1]]
        elif len(shape_origin) == 4:
            bench_mark = bench_mark[:, :, shape_origin[0], :shape_origin[1]]
    return bench_mark


def trans_data_dsl(inputs, output, attr):
    src_format = attr[0]['value']
    dst_format = attr[1]['value']

    support_formats = [("DefaultFormat", "FRACTAL_NZ"),
                       ("FRACTAL_NZ", "DefaultFormat")]

    if (src_format, dst_format) not in support_formats:
        raise ValueError("src_format %s and dst_format %s is not supported!" %
                         (src_format, dst_format))

    if src_format == 'DefaultFormat' and dst_format == 'FRACTAL_NZ':
        res = "%s \n%s = %s(%s, '%s', '%s')" % (inspect.getsource(trans_data_two2fractal),
              output[0]['tensor_name'], trans_data_two2fractal.__name__, get_input(inputs[0][0]),
              attr[0]['value'], attr[1]['value'])
    elif src_format == 'FRACTAL_NZ' and dst_format == 'DefaultFormat':
        res = "%s \n%s = %s(%s, '%s', '%s', %s)" % (inspect.getsource(trans_data_fractal2two),
              output[0]['tensor_name'], trans_data_fractal2two.__name__, get_input(inputs[0][0]),
              attr[0]['value'], attr[1]['value'], attr[2]['value'])
    return res


op_dsl = {
    "ReduceSum" : lambda inputs, output, attr: sum_str(inputs, output, attr),
    "Mul" : lambda inputs, output, attr: "%s = np.multiply(%s, %s)" %
            (output[0]['tensor_name'], get_input(inputs[0][0]), get_input(inputs[1][0])),
    "Pow" : lambda inputs, output, attr: "%s = np.power(%s, %s)" %
            (output[0]['tensor_name'], get_input(inputs[0][0]), get_input(inputs[1][0])),
    "Sub" : lambda inputs, output, attr: "%s = np.subtract(%s, %s)" %
            (output[0]['tensor_name'], get_input(inputs[0][0]), get_input(inputs[1][0])),
    "TensorAdd" : lambda inputs, output, attr: "%s = np.add(%s, %s)" %
                  (output[0]['tensor_name'], get_input(inputs[0][0]), get_input(inputs[1][0])),
    "Rsqrt" : lambda inputs, output, attr: "%s = 1.0/np.sqrt(%s)" %
              (output[0]['tensor_name'], get_input(inputs[0][0])),
    "Neg" : lambda inputs, output, attr: "%s = np.negative(%s)" %
            (output[0]['tensor_name'], get_input(inputs[0][0])),
    "Exp" : lambda inputs, output, attr: "%s = np.exp(%s)" %
            (output[0]['tensor_name'], get_input(inputs[0][0])),
    "RealDiv" : lambda inputs, output, attr: "%s = np.divide(%s, %s)" %
                (output[0]['tensor_name'], get_input(inputs[0][0]), get_input(inputs[1][0])),
    "Minimum" : lambda inputs, output, attr: "%s = np.minimum(%s, %s)" %
                (output[0]['tensor_name'], get_input(inputs[0][0]), get_input(inputs[1][0])),
    "Maximum" : lambda inputs, output, attr: "%s = np.maximum(%s, %s)" %
                (output[0]['tensor_name'], get_input(inputs[0][0]), get_input(inputs[1][0])),
    "Log" : lambda inputs, output, attr: "%s = np.log(%s)" %
            (output[0]['tensor_name'], get_input(inputs[0][0])),
    "Sqrt" : lambda inputs, output, attr: "%s = np.sqrt(%s)" %
             (output[0]['tensor_name'], get_input(inputs[0][0])),
    "Cast" : lambda inputs, output, attr: "%s = %s.astype(np.%s)" %
             (output[0]['tensor_name'], get_input(inputs[0][0]), attr[0]['value']),
    "Reshape" : lambda inputs, output, attr: "%s = np.reshape(%s, %s)" %
                (output[0]['tensor_name'], get_input(inputs[0][0]), attr[0]['value']),
    "ReduceMax" : lambda inputs, output, attr: "%s = np.max(%s, %s[0], keepdims=%s)" %
                  (output[0]['tensor_name'], get_input(inputs[0][0]), attr[0]['value'], attr[1]['value']),
    "ReduceMin" : lambda inputs, output, attr: "%s = np.min(%s, %s[0], keepdims=%s)" %
                  (output[0]['tensor_name'], get_input(inputs[0][0]), attr[0]['value'], attr[1]['value']),
    "OneHot" : lambda inputs, output, attr: "%s = np.one_hot(%s, %s, %s, %s, %s, %s)" %
               (output[0]['tensor_name'], get_input(inputs[0][0]), get_input(inputs[1][0]), get_input(inputs[2][0]),
                attr[0]['value'], attr[1]['value'], inputs[0][0]['data_type']),
    "ZerosLike" : lambda inputs, output, attr: "%s = np.zeros_like(%s)" %
                  (output[0]['tensor_name'], get_input(inputs[0][0])),
    "AddN" : lambda inputs, output, attr: "%s = %s" %
             (output[0]['tensor_name'], ' + '.join([get_input(inputs[0][i]) for i in range(0, len(inputs[0]))])),
    "Tile" : lambda inputs, output, attr: "%s = np.tile(%s, %s)" %
             (output[0]['tensor_name'], get_input(inputs[0][0]), attr[0]['value']),
    "Reciprocal" : lambda inputs, output, attr: "%s = np.divide(1.0, %s)" %
                   (output[0]['tensor_name'], get_input(inputs[0][0])),
    "Equal" : lambda inputs, output, attr: "%s = np.equal(%s, %s)" %
              (output[0]['tensor_name'], get_input(inputs[0][0]), get_input(inputs[1][0])),
    "GreaterEqual" : lambda inputs, output, attr: "%s = np.greater_equal(%s, %s)" %
                     (output[0]['tensor_name'], get_input(inputs[0][0]), get_input(inputs[1][0])),
    "Select" : lambda inputs, output, attr: "%s = np.where(%s, %s, %s)" %
               (output[0]['tensor_name'], get_input(inputs[0][0]), get_input(inputs[1][0]), get_input(inputs[2][0])),
    "InplaceAssign" : lambda inputs, output, attr: "%s = %s; %s = %s" %
                      (get_input(inputs[0][0]), get_input(inputs[1][0]),
                       output[0]['tensor_name'], get_input(inputs[2][0])),
    "Greater" : lambda inputs, output, attr: "%s = np.greater(%s, %s)" %
                (output[0]['tensor_name'], get_input(inputs[0][0]), get_input(inputs[1][0])),
    "SelectGT" : lambda inputs, output, attr: "%s = np.where(%s > %s, %s, %s)" %
                 (output[0]['tensor_name'], get_input(inputs[0][0]), get_input(inputs[1][0]),
                  get_input(inputs[2][0]), get_input(inputs[3][0])),
    "SelectLT" : lambda inputs, output, attr: "%s = np.where(%s < %s, %s, %s)" %
                 (output[0]['tensor_name'], get_input(inputs[0][0]), get_input(inputs[1][0]),
                  get_input(inputs[2][0]), get_input(inputs[3][0])),
    "Abs" : lambda inputs, output, attr: "%s = np.absolute(%s)" %
            (output[0]['tensor_name'], get_input(inputs[0][0])),
    "LessEqual" : lambda inputs, output, attr: "%s = np.less_equal(%s, %s)" %
                     (output[0]['tensor_name'], get_input(inputs[0][0]), get_input(inputs[1][0])),
    "EquivFormat" : lambda inputs, output, attr: "%s = %s" %
                    (output[0]['tensor_name'], get_input(inputs[0][0])),
    "ExpandDims" : lambda inputs, output, attr: "%s = np.expand_dims(%s, %s)" %
                   (output[0]['tensor_name'], get_input(inputs[0][0]), attr[0]['value']),
    "TransData" : trans_data_dsl,
}

def gen_json_data(op_desc):
    """Generating test data for composite json"""
    desc = json.loads(op_desc)
    input_for_mod = []
    input_dict = {}
    input_order = {}
    inplace_assign_dict = {}
    output_indexes = []
    expect = []
    with_inplace_assign = False
    if isinstance(desc["op"], str) and desc["op"].startswith("Fused_LambUpdateWithLR"):
        with_inplace_assign = True

    p = CodePrinter('json_data.py')
    idx = 0
    for input_desc in desc["input_desc"]:
        shape = [1] if not input_desc[0]["shape"] else input_desc[0]["shape"]
        dtype = input_desc[0]["data_type"]
        item = random_gaussian(shape, miu=1, sigma=0.1).astype(dtype)
        input_for_mod.append(item)
        tensor_name = input_desc[0]["tensor_name"]
        input_order[tensor_name] = idx
        idx += 1
        input_dict[tensor_name] = item
        p.out(tensor_name)
        p.out(" = np.array(input_dict.get(\"")
        p.out(tensor_name)
        p.out("\"))")
        p.null_line()

    for op in desc["op_desc"]:
        dsl_fun = op_dsl.get(op["name"], None)
        if op["name"] == "InplaceAssign" and with_inplace_assign:
            out_name = op["output_desc"][0]["tensor_name"]
            in_name = op["input_desc"][0][0]["tensor_name"]
            inplace_assign_dict[out_name] = in_name
        if dsl_fun is None:
            logging.info("[%s] is not support for %s", op["name"], op)
            continue
        sent = dsl_fun(op['input_desc'], op['output_desc'], op['attr'])
        logging.debug(sent)
        p.out(sent, True)

    idx = 0
    inplace_assign_num = 0
    inplace_assign_idx = -1
    out_nums = len(desc["output_desc"])
    for output_desc in desc["output_desc"]:
        shape = [1] if not output_desc["shape"] else output_desc["shape"]
        dtype = output_desc["data_type"]
        item = np.full(shape, 0, dtype)
        input_for_mod.append(item)
        tensor_name = output_desc["tensor_name"]
        if tensor_name in inplace_assign_dict:
            real_idx = input_order[inplace_assign_dict[tensor_name]]
            inplace_assign_num += 1
            if inplace_assign_idx == -1:
                inplace_assign_idx = idx
        else:
            real_idx = idx - out_nums
        output_indexes.append(real_idx)
        idx += 1
        p.out("expect.append(", True)
        p.out(tensor_name)
        p.out(")")
    p.close()
    # offset of inplace assign index
    if inplace_assign_num > 0:
        for i in range(len(output_indexes)):
            if i > inplace_assign_idx and output_indexes[i] < 0:
                output_indexes[i] -= inplace_assign_num

    with open("json_data.py", 'r') as f:
        sent = f.read()
    exec(sent)

    return input_for_mod, expect, output_indexes
