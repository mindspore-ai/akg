# Copyright 2021-2022 Huawei Technologies Co., Ltd
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

"""common utils for composite operation test"""
import os
import tempfile
import json
import logging
import random
import sys
import time
import math
import traceback
import string
from logging.handlers import TimedRotatingFileHandler
from collections import namedtuple

import numpy as np

from akg.global_configs import get_kernel_meta_path
from akg.utils.gen_random import random_gaussian, gen_indices, gen_csr_indices
from akg.utils.op_dsl import get_attr, op_dsl

RANDOM_SEED_NUM = 20
MakeIndices = namedtuple("MakeIndices", "name data_shape indices_shape indices_dtype attrs")
CpuPackBBlockSize = {
    "neon": 12,
    "sse": 12,
    "avx": 24,
    "avx2": 24,
    "avx512": 48
}

class Log(logging.Logger):
    def __init__(self, case_name, case_path):
        super(Log, self).__init__(case_name)
        self.log = logging.getLogger(
            case_name + ''.join([random.choice(string.digits + string.ascii_letters) for _ in range(8)]))
        self.log.setLevel(logging.DEBUG)
        fmt = '%(levelname)s %(asctime)s - %(filename)s:%(funcName)s:%(lineno)s - %(message)s'
        datefmt = '%Y-%m-%d %H:%M:%S'
        formatter = logging.Formatter(fmt, datefmt)
        logfile = os.path.join(case_path, '{0}.log'.format(case_name))
        fh = TimedRotatingFileHandler(
            logfile, when='D', interval=1, backupCount=10)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        self.log.removeHandler(fh)
        self.log.addHandler(fh)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.log.removeHandler(ch)
        self.log.addHandler(ch)

    def traceback(self):
        """
        The traceback module prints out the details of the case execution failure.
        """
        self.log.error("There are something error appear.")
        traceback.print_exc()


def compare_tensor(acu_output, exp_output, rtol=1.e-5, atol=1.e-8, equal_nan=False):
    """
    Output and expected result comparison method
    :param acu_output: array_like Input arrays to compare.
    :param exp_output: array_like Input arrays to compare.
    :param rtol: float The relative tolerance parameter (see Notes).
    :param atol: float The absolute tolerance parameter (see Notes).
    :param equal_nan: bool
            Whether to compare NaN's as equal.  If True, NaN's in `a` will be
            considered equal to NaN's in `b` in the output array.
    :return: True / False
    """
    res = np.allclose(acu_output, exp_output, rtol, atol, equal_nan)
    if not res:
        pandora_logger_ = Log(case_name=os.path.dirname(
            __file__), case_path=os.getcwd())
        pandora_logger_.log.error(
            "This shape precision is not up to standard, compare failed.")
    return res


def get_rtol_atol(op_name, dtype, rtol=5e-03, atol=5e-03):
    run_mode = os.environ.get('RUNTIME_MODE')
    if run_mode in ("air_cloud",):
        if dtype == "float16":
            rtol = atol = 1e-03
        else:
            rtol = atol = 1e-04
    return rtol, atol


def precheck(desc):
    """
    This utils is used to:
        1. Run a precheck for those testing cases that have only element-wise computations and then
        2. Return a reasonable mean value for generating random Gaussian input data.
    to avoid the precision error caused by computing division by zero, the reciprocal of zero or the root squared of
    zero.
    """
    elemwise_op_func_map = {
        "Neg": lambda a: -a, "Abs": lambda a: abs(a), "Cast": lambda a: a, "Log": lambda a: math.log(a),
        "Exp": lambda a: math.exp(a), "Sqrt": lambda a: math.sqrt(a), "Rsqrt": lambda a: 1 / math.sqrt(a),
        "Reciprocal": lambda a: 1 / a, "Square": lambda a: a ** 2,
        "Add": lambda a, b: a + b, "Sub": lambda a, b: a - b, "Mul": lambda a, b: a * b, "RealDiv": lambda a, b: a / b,
        "Minimum": lambda a, b: min(a, b), "Maximum": lambda a, b: max(a, b), "Pow": lambda a, b: pow(a, b)
    }
    stop_forward = set()
    variable = dict()

    def update_stop_forward(out_desc):
        for out_tensor in out_desc:
            stop_forward.add(out_tensor['tensor_name'])

    def need_jump(op_desc):
        for in_desc in op_desc['input_desc']:
            for in_tensor in in_desc:
                if in_tensor['tensor_name'] in stop_forward:
                    update_stop_forward(op_desc['output_desc'])
                    return True
        return False

    def fill_input_value(input_desc, input_value):
        inputs = []
        for in_desc in input_desc:
            for in_tensor in in_desc:
                if "value" in in_tensor:
                    val = in_tensor["value"]
                elif in_tensor['tensor_name'] in variable:
                    val = variable[in_tensor['tensor_name']]
                else:
                    val = input_value
                inputs.append(val)
        return inputs

    def compute_math(op_name, inputs, input_value):
        if op_name == "Rsqrt" and abs(inputs[0]) <= 0.01:
            logging.info(
                "The input with mean value {} fails the precheck because zero has no square root".format(input_value))
            return None
        elif op_name == "Reciprocal" and abs(inputs[0]) <= 0.01:
            logging.info(
                "The input with mean value {} fails the precheck because zero has no reciprocal".format(input_value))
            return None
        elif op_name == "RealDiv" and abs(inputs[1]) <= 0.01:
            logging.info(
                "The input with mean value {} fails the precheck because zero cannot be a divisor".format(input_value))
            return None
        else:
            return elemwise_op_func_map[op_name](*inputs)

    def check_pass(input_value):
        for op_desc in desc['op_desc']:
            if op_desc['name'] not in elemwise_op_func_map:
                update_stop_forward(op_desc['output_desc'])
            elif not need_jump(op_desc):
                inputs = fill_input_value(op_desc['input_desc'], input_value)
                output = op_desc['output_desc'][0]['tensor_name']
                if compute_math(op_desc['name'], inputs, input_value) is None:
                    return False
                variable[output] = compute_math(
                    op_desc['name'], inputs, input_value)
        return True

    initial_input = 1
    while not check_pass(initial_input):
        initial_input += 1
        if initial_input > 20:
            logging.info(
                "Input mean value check failed! Just use mean value 1. Precision error may occur! ")
            return 1
    logging.info(
        "Input data with mean value {} is generated".format(initial_input))
    return initial_input


def random_data_to_disk(size, miu=None, sigma=None, seed=None, random_data_disk_path=None):
    """
    Generate local disk data
    :param size:  Generate disk data size
    :param miu:   Average value
    :param sigma: Standard deviation
    :param seed:  Seed of random number
    :param random_data_disk_path: Specify the disk data save path
    :return:
    """
    if miu is None or sigma is None:
        miu_sigma_list = [[1, 0.1]]
    else:
        miu_sigma_list = []
        for i in miu:
            for j in sigma:
                miu_sigma_list.append([i, j])

    for miu_sigma in miu_sigma_list:
        random_data = size // 8
        random_data = random_gaussian(tuple([random_data]), miu=miu_sigma[0], sigma=miu_sigma[1], seed=seed)
        if random_data_disk_path is None:
            random_data_disk_path = os.environ.get("RANDOM_DATA_DISK_PATH")
            if random_data_disk_path is None:
                raise ValueError("Environment variable is missing from the current environment RANDOM_DATA_DISK_PATH "
                                 ": {0}".format(random_data_disk_path))
        data_path = random_data_disk_path + "/random_data_%s_%s.bin" % (str(miu_sigma[0]), str(miu_sigma[1]))
        with open(data_path, "w+") as file:
            random_data.tofile(file)
            file.close()


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


def _get_attr_dict(attr_desc):
    """get op attr dict"""
    attr_dict = {}
    for attr in attr_desc:
        attr_dict[attr["name"]] = attr["value"]
    return attr_dict


def _gen_uniq_file_name(op_name):
    """Generate uniq file name."""
    if len(op_name.split("_")) > 0:
        op_hash = op_name.split("_")[-1]
    else:
        op_hash = str(time.time())

    uni_file_name_suffix = ".json_data_" + op_hash + ".py"
    fd, uni_file_name = tempfile.mkstemp(suffix=uni_file_name_suffix)
    os.close(fd)
    return uni_file_name


def _collect_inplace_assign_infos(op, infos, sum_out):
    """Collect inplace assign infos."""
    infos["fake_output_tensors"].append(op["output_desc"][0]["tensor_name"])
    input0, input1 = op["input_desc"][0][0], op["input_desc"][1][0]
    if input1["tensor_name"] in sum_out:
        infos["clean_input"].append(input0["tensor_name"])


def _collect_infos(desc, infos):
    """Collect infos."""
    sum_out = []
    target = desc["process"]
    if target  == "cpu":
        target_info = desc.get("target_info", {})
        infos["feature"] = target_info.get("feature", "avx")
    for op in desc["op_desc"]:
        if (op["name"] in ["ReduceSum", "UnsortedSegmentSum", "CSRReduceSum"] and
                "enable_atomic_add" in _get_attr_dict(op["attr"])) or op["name"] in ["ElemAny"]:
            sum_out.append(op["output_desc"][0]["tensor_name"])

        if op["name"] == "UnsortedSegmentSum":
            input0, input1 = op["input_desc"][0][0], op["input_desc"][1][0]
            assert input1["data_type"] == "int32", "Default indices type should be int32"
            infos["indices_input"][input1["tensor_name"]] = MakeIndices(name=op["name"],
                                                                        data_shape=input0["shape"],
                                                                        indices_shape=input1["shape"],
                                                                        indices_dtype=input1["data_type"],
                                                                        attrs=get_attr(op["attr"], "num_segments"))
        elif op["name"] == "Assign":
            _collect_inplace_assign_infos(op, infos, sum_out)
            infos["inplace_assign_write"].append(op["input_desc"][0][0]["tensor_name"])
        elif op["name"] in ["TensorScatterAdd", "Gather", "GatherNd"]:
            input0, input1 = op["input_desc"][0][0], op["input_desc"][1][0]
            assert input1["data_type"] == "int32", "Default indices type should be int32"
            infos["indices_input"][input1["tensor_name"]] = MakeIndices(name=op["name"], data_shape=input0["shape"],
                                                                        indices_shape=input1["shape"],
                                                                        indices_dtype=input1["data_type"],
                                                                        attrs=None)
            if op["name"] == "Gather":
                assert op["attr"][0]["name"] == "axis", "Gather only accepts axis attribute."
                infos["indices_input"][input1["tensor_name"]] = MakeIndices(name=op["name"],
                                                                            data_shape=input0["shape"],
                                                                            indices_shape=input1["shape"],
                                                                            indices_dtype=input1["data_type"],
                                                                            attrs=op["attr"][0]["value"])
        elif op["name"].startswith("CSR"):
            input0, input1, input2 = op["input_desc"][0][0], op["input_desc"][1][0], op["input_desc"][2][0]
            if op["name"] != "CSRGather":
                assert op["input_desc"][1][0]["shape"][0] == op["input_desc"][2][0]["shape"][0], \
                    "indices and data should have the same shape"
            infos["csr_indptr"][input0["tensor_name"]] = MakeIndices(name=input1["tensor_name"],
                                                                     data_shape=get_attr(op["attr"], "dense_shape"),
                                                                     indices_shape=input1["shape"],
                                                                     indices_dtype=input0["data_type"],
                                                                     attrs=None)
            infos["csr_indices"][input1["tensor_name"]] = MakeIndices(name=input0["tensor_name"],
                                                                      data_shape=get_attr(op["attr"], "dense_shape"),
                                                                      indices_shape=input1["shape"],
                                                                      indices_dtype=input1["data_type"],
                                                                      attrs=None)
        elif target == "cpu" and op["name"] in ["MatMul", "BatchMatMul"]:
            input1 = op["input_desc"][1][0]
            infos["need_pack_b"][input1["tensor_name"]] = get_attr(op["attr"], "pack_b")
            infos["need_transpose"][input1["tensor_name"]] = get_attr(op["attr"], "transpose_b")

def _pack_matrix(data, feature):
    def _get_size(shape):
        res = 1
        for i in shape:
            res *= i
        return res
    block_size = CpuPackBBlockSize.get(feature, "avx")
    shape = data.shape
    if shape[-1] <= block_size:
        return data
    block_num = shape[-1] // block_size
    split_pos = block_num * block_size
    new_data = np.split(data, indices_or_sections=[split_pos,], axis=-1)
    body_data = np.split(new_data[0], block_num, -1)
    dim_size = (int)(_get_size(shape) / shape[-1])
    data_list = []
    for block in body_data:
        data_list.append(block.reshape((dim_size * block_size,)))
    data_list.append(new_data[1].reshape((dim_size * (shape[-1] % block_size)),))
    packed_data = np.concatenate(data_list, axis=-1)
    packed_data = packed_data.reshape(shape)
    return packed_data

def _gen_input_data(desc, infos, input_for_mod, commands):
    """Generate input data."""
    idx = 0
    csr_idx_pair = {}
    input_mean_value = precheck(desc)
    target = desc["process"]
    for input_desc in desc["input_desc"] if desc.get("input_desc") is not None else []:
        tensor_name = input_desc[0]["tensor_name"]
        infos["input_order"][tensor_name] = idx
        commands.append("%s = np.array(input_dict.get('%s'))" % (tensor_name, tensor_name))
        if not infos["gen_data"] and idx < len(input_for_mod):
            infos["input_dict"][tensor_name] = input_for_mod[idx]
            idx += 1
            continue
        shape = [1] if not input_desc[0]["shape"] else input_desc[0]["shape"]
        dtype = input_desc[0]["data_type"]
        if tensor_name in infos["clean_input"]:
            item = np.zeros(shape).astype(dtype)
        elif tensor_name in infos["csr_indptr"]:
            if tensor_name in csr_idx_pair:
                item = csr_idx_pair[tensor_name]
            else:
                indptr, indices = gen_csr_indices(infos["csr_indptr"][tensor_name])
                item = indptr
                csr_idx_pair[infos["csr_indptr"][tensor_name].name] = indices
        elif tensor_name in infos["csr_indices"]:
            if tensor_name in csr_idx_pair:
                item = csr_idx_pair[tensor_name]
            else:
                indptr, indices = gen_csr_indices(infos["csr_indices"][tensor_name])
                item = indices
                csr_idx_pair[infos["csr_indices"][tensor_name].name] = indptr
        elif tensor_name in infos["indices_input"].keys():
            item = gen_indices(infos["indices_input"][tensor_name])
        else:
            item = random_gaussian(shape, miu=input_mean_value, sigma=0.1).astype(dtype)
        if target == "cpu" and tensor_name in infos["need_pack_b"].keys() and \
            infos["need_pack_b"][tensor_name]:
            if infos["need_transpose"][tensor_name]:
                axis = [x - len(shape) for x in range(len(shape))]
                axis[-1] = -2
                axis[-2] = -1
                item = item.transpose(axis)
            input_for_mod.append(_pack_matrix(item, infos["feature"]))
        else:
            input_for_mod.append(item)
        infos["input_dict"][tensor_name] = item
        idx += 1


def _gen_output_data(desc, infos, input_for_mod, output_indexes, commands):
    """Generate output data."""
    idx = 0
    fake_output_tensors = infos["fake_output_tensors"]
    out_nums = len(desc["output_desc"])
    for output_desc in desc["output_desc"]:
        tensor_name = output_desc["tensor_name"]
        if infos["gen_data"]:
            shape = [1] if not output_desc["shape"] else output_desc["shape"]
            dtype = output_desc["data_type"]
            item = np.full(shape, np.nan, dtype)
            input_for_mod.append(item)
        if tensor_name not in fake_output_tensors:
            real_idx = idx - out_nums
            output_indexes.append(real_idx)
            commands.append("expect.append(%s)" % tensor_name)
        idx += 1


def _check_need_reshape(input_desc):
    """Check if input shape needs reshape."""
    if len(input_desc) != 2:
        return False, None, None
    inputs_format = [input_desc[0][0]["format"], input_desc[1][0]["format"]]
    if inputs_format == ["DefaultFormat", "FRACTAL_NZ"]:
        fractal_tensor = input_desc[1][0]
        default_tensor = input_desc[0][0]
        return True, fractal_tensor, default_tensor
    elif inputs_format == ["FRACTAL_NZ", "DefaultFormat"]:
        fractal_tensor = input_desc[0][0]
        default_tensor = input_desc[1][0]
        return True, fractal_tensor, default_tensor
    return False, None, None


def _emit_reshape(fractal_tensor, default_tensor):
    """Emit reshape."""
    shape_fractal = fractal_tensor["shape"]
    shape_default = default_tensor["shape"]
    shape_tmp = []
    shape_new = []
    for i in range(len(shape_default) - 2):
        shape_new.append(shape_default[i])
    for i in range(len(shape_default), 2):
        shape_tmp.append(1)
    for _, sh in enumerate(shape_default):
        shape_tmp.append(sh)
    if shape_tmp[-2] == 1 and shape_tmp[-1] == 1:
        shape_new.extend([1, 1, 1, 1])
    elif shape_tmp[-2] == 1 and shape_tmp[-1] == shape_default[-1]:
        shape_new.extend(
            [shape_fractal[-4], 1, 1, shape_fractal[-1]])
    elif shape_tmp[-2] == shape_default[-2] and shape_tmp[-1] == 1:
        shape_new.extend(
            [1, shape_fractal[-3], shape_fractal[-2], 1])
    if "value" in default_tensor:
        sent_reshape_tensor = "%s = np.full(%s, %s, np.%s)" \
                              % (default_tensor["tensor_name"], shape_new, default_tensor["value"],
                                 default_tensor["data_type"])
    else:
        if np.zeros(shape_default).size != np.zeros(shape_new).size:
            raise ValueError("It is error to reshape %s to %s!" % (shape_default, shape_new))
        sent_reshape_tensor = "%s = np.reshape(%s, %s)" \
                              % (default_tensor["tensor_name"], default_tensor["tensor_name"], tuple(shape_new))
    return sent_reshape_tensor


def _gen_op_compute(desc, commands):
    """Generate op compute."""
    elemwise_op_list = ["TensorAdd", "Add", "RealDiv", "Mul", "Minimum", "Maximum", "Sub"]
    for op in desc["op_desc"]:
        dsl_fun = op_dsl.get(op["name"], None)
        if dsl_fun is None:
            raise ValueError("op [%s] is not supported!" % op["name"])
        if op["name"] in elemwise_op_list and op["output_desc"][0].get("format") == "FRACTAL_NZ":
            need_reshape, fractal_tensor, default_tensor = _check_need_reshape(op["input_desc"])
            if need_reshape:
                commands.append(_emit_reshape(fractal_tensor, default_tensor))
        if op.get('attr', None):
            op['attr'].append({'name': 'process', 'value': desc['process']})
        sent = dsl_fun(op['input_desc'], op['output_desc'], op['attr'])
        commands.append(sent)


def _update_inplace_tensors(infos, output_indexes, commands):
    """Update inplace tensors."""
    inplace_assign_write = infos["inplace_assign_write"]
    input_order = infos["input_order"]
    if inplace_assign_write:
        inplace_tensors = "["
        inplace_tensors_index = []

        for tensor_name in inplace_assign_write:
            inplace_tensors_index.append(input_order[tensor_name])
            inplace_tensors += "{}, ".format(tensor_name)
        inplace_tensors += "]"
        commands.append("inplace_tensors = {}".format(inplace_tensors))
        commands.append("expect.extend(inplace_tensors)")
        output_indexes.extend(inplace_tensors_index)


def _update_workspace_data(kernel_name, input_for_mod, output_indexes):
    """Update workspace tensors."""
    workspace_tensors = []
    json_file = get_kernel_meta_path() + kernel_name + ".json"
    if os.path.isfile(json_file):
        with open(json_file, 'r') as f:
            kernel_json = f.read()
            kernel_desc = json.loads(kernel_json)
            if "workspace" in kernel_desc:
                workspace_bytes = kernel_desc["workspace"]["size"]
                item = np.full(workspace_bytes, np.nan, np.int8)
                workspace_tensors.append(item)
    else:
        logging.warning("Kernel json file %s not found", json_file)

    # Add workspace tensors to input_for_mod
    if len(workspace_tensors) > 0:
        # workspace tensors are placed after inputs and outputs, so index in output_indexes should
        # be converted to positive number first, otherwise -1 will point to the last workspace tensor
        # instead of the last output tensor.
        output_indexes = [i if i > 0 else i + len(input_for_mod) for i in output_indexes]
        input_for_mod.extend(workspace_tensors)

    return output_indexes


def gen_json_data(op_desc, with_compute=True, input_for_mod=None):
    """Generating test data for composite json"""
    desc = json.loads(op_desc)
    from akg.ms.info_version_adapt import InfoVersionAdapt
    info_adapter = InfoVersionAdapt(desc)
    ret = info_adapter.run()
    if not ret:
        raise RuntimeError(info_adapter.msg)
    output_indexes = []
    expect = []
    infos = {"gen_data": False,
             "clean_input": [],
             "inplace_assign_write": [],
             "fake_output_tensors": [],
             "indices_input": {},
             "csr_indptr": {},
             "csr_indices": {},
             "input_dict": {},
             "input_order": {},
             "feature": "avx",
             "need_pack_b": {},
             "need_transpose": {},
             }
    if input_for_mod is None:
        input_for_mod = []
        infos["gen_data"] = True

    # Collect necessary information
    _collect_infos(desc, infos)

    commands = []
    # Parse input_desc
    _gen_input_data(desc, infos, input_for_mod, commands)
    # Parse op_desc
    _gen_op_compute(desc, commands)
    # Parse output_desc
    _gen_output_data(desc, infos, input_for_mod, output_indexes, commands)

    # Update inplace tensors
    _update_inplace_tensors(infos, output_indexes, commands)
    # Update workspace
    output_indexes = _update_workspace_data(desc["op"], input_for_mod, output_indexes)

    uni_file_name = _gen_uniq_file_name(desc.get("op"))
    p = CodePrinter(uni_file_name)
    p.out("from akg.utils.op_dsl import *", False)
    p.out("def get_expect(input_dict, expect):", True)
    for command in commands:
        single_commands = command.split("\n")
        for single_command in single_commands:
            if not single_command.strip():
                continue
            single_command = "    " + single_command
            p.out(single_command, True)
    p.close()

    # compute the expect data
    if with_compute:
        import importlib.util
        tmp_mod_spec = importlib.util.spec_from_file_location("tmp_mod", uni_file_name)
        tmp_mod = importlib.util.module_from_spec(tmp_mod_spec)
        tmp_mod_spec.loader.exec_module(tmp_mod)
        tmp_mod.get_expect(infos["input_dict"], expect)
        os.remove(uni_file_name)
        return input_for_mod, expect, output_indexes
    else:
        return input_for_mod, None, output_indexes
