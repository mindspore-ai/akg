# Copyright 2023-2026 Huawei Technologies Co., Ltd
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
import importlib.util
from logging.handlers import TimedRotatingFileHandler
from collections import namedtuple
import numpy as np
from .gen_random import random_gaussian, gen_indices, gen_csr_indices
from .op_dsl import get_attr, get_op_dsl, get_op_dsl_torch_mlir
from .torch_mlir_utils import torch_normalize_dtype


def get_cpptype_from_pytype(pytype):
    """convert cpp type to python type"""
    pytype_to_cpptype_str = {
        "float16": "half",
        "float32": "float",
        "float64": "double",
        "int32": "int",
        "int64": "int64_t",
        "bool": "bool"
    }
    return pytype_to_cpptype_str.get(pytype, None)


MakeIndices = namedtuple(
    "MakeIndices", "name data_shape indices_shape indices_dtype attrs")
CpuPackBBlockSize = {
    "neon": 12,
    "sse": 12,
    "avx": 24,
    "avx2": 24,
    "avx512": 48
}


class Log(logging.Logger):
    """Log class for print"""

    def __init__(self, case_name, case_path):
        super().__init__(case_name)
        self.log = logging.getLogger(
            case_name + ''.join([random.choice(string.digits + string.ascii_letters) for _ in range(8)]))
        self.log.setLevel(logging.DEBUG)
        fmt = '%(levelname)s %(asctime)s - %(filename)s:%(funcName)s:%(lineno)s - %(message)s'
        datefmt = '%Y-%m-%d %H:%M:%S'
        formatter = logging.Formatter(fmt, datefmt)
        logfile = os.path.join(case_path, f'{case_name}.log')
        file_handler = TimedRotatingFileHandler(logfile, when='D', interval=1, backupCount=10)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.log.removeHandler(file_handler)
        self.log.addHandler(file_handler)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        self.log.removeHandler(stream_handler)
        self.log.addHandler(stream_handler)

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
        if exp_output.dtype == np.bool_:
            acu_output = acu_output.astype(np.int32)
            exp_output = exp_output.astype(np.int32)
        absolute_err = np.abs(acu_output - exp_output)
        max_err = np.max(absolute_err)
        index = np.argmax(absolute_err)
        eps = 1e-8
        relative_err = max_err / (abs(exp_output.reshape(-1)[index]) or eps)
        logging.error("This shape precision is not up to standard, compare failed. at %s expect %s but got %s",
                      index, exp_output.reshape(-1)[index], acu_output.reshape(-1)[index])
        logging.error("Max absolute error is %s and Max absolute error's relative error is %s", max_err, relative_err)
    return res


def get_rtol_atol(op_name, dtype, rtol=5e-03, atol=5e-03):
    """return rtol and atol for precision comparison"""
    run_mode = os.environ.get('RUNTIME_MODE')
    if run_mode in ("rpc_cloud", "air_cloud"):
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

    # pylint: disable=unnecessary-lambda
    elemwise_op_func_map = {
        "Neg": lambda a: -a, "Abs": lambda a: abs(a), "Cast": lambda a: a, "Log": lambda a: math.log(a),
        "Exp": lambda a: math.exp(a), "Sqrt": lambda a: math.sqrt(a), "Rsqrt": lambda a: 1 / math.sqrt(a),
        "Reciprocal": lambda a: 1 / a, "Square": lambda a: a ** 2,
        "Add": lambda a, b: a + b, "Sub": lambda a, b: a - b, "Mul": lambda a, b: a * b, "RealDiv": lambda a, b: a / b,
        "Minimum": lambda a, b: min(a, b), "Maximum": lambda a, b: max(a, b), "Pow": lambda a, b: pow(a, b),
        "Cos": lambda a: math.cos(a), "Sin": lambda a: math.sin(a),
        "ACos": lambda a: math.acos(a), "ASin": lambda a: math.asin(a)
    }

    stop_forward = set()
    variable = {}

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
        res = False
        if op_name == "Rsqrt" and inputs[0] <= 0.01:
            logging.info(
                "The input with mean value %s fails the precheck because non-positive"
                " number has no reciprocal of the square root",
                input_value
            )
        elif op_name == "Sqrt" and inputs[0] <= 0.01:
            logging.info(
                "The input with mean value %s fails the precheck because non-positive number has no square root",
                input_value
            )
        elif op_name == "Reciprocal" and abs(inputs[0]) <= 0.01:
            logging.info(
                "The input with mean value %s fails the precheck because zero has no reciprocal",
                input_value
            )
        elif op_name == "RealDiv" and abs(inputs[1]) <= 0.01:
            logging.info(
                "The input with mean value %s fails the precheck because zero cannot be a divisor",
                input_value
            )
        elif op_name == "ACos" and abs(inputs[0]) >= 1:
            logging.info(
                "The input with mean value %s fails the precheck because the value cannot be a input",
                input_value
            )
        elif op_name == "ASin" and abs(inputs[0]) >= 1:
            logging.info(
                "The input with mean value %s fails the precheck because the value cannot be a input",
                input_value
            )
        elif op_name == "Cast":
            logging.info(
                "The input with mean value %s fails the precheck because the value cannot be a input",
                input_value
            )
        elif op_name == "Pow" and (abs(inputs[0]) >= 1e4 or abs(inputs[1]) >= 10) :
            logging.info(
                "The input with mean value %s fails the precheck because base or exponent is too large",
                input_value
            )
        else:
            res = elemwise_op_func_map[op_name](*inputs)
        return res

    def check_pass(input_value):
        for op_desc in desc['op_desc']:
            if op_desc['name'] not in elemwise_op_func_map:
                update_stop_forward(op_desc['output_desc'])
            elif not need_jump(op_desc):
                inputs = fill_input_value(op_desc['input_desc'], input_value)
                output = op_desc['output_desc'][0]['tensor_name']
                if not compute_math(op_desc['name'], inputs, input_value):
                    return False
                variable[output] = compute_math(
                    op_desc['name'], inputs, input_value)
        return True

    initial_input = 1
    positive_fail = False
    while not check_pass(initial_input):
        if not positive_fail:
            initial_input += 1
        if initial_input > 20:
            positive_fail = True
        if positive_fail:
            if initial_input > 0:
                initial_input = 0
            else:
                initial_input -= 1
        if initial_input < -20:
            logging.info(
                "Input mean value check failed! Just use mean value 1. Precision error may occur! ")
            return 1
    logging.info(
        "Input data with mean value %s is generated", initial_input)
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
        random_data = random_gaussian(
            tuple([random_data]), miu=miu_sigma[0], sigma=miu_sigma[1], seed=seed)
        if random_data_disk_path is None:
            random_data_disk_path = os.environ.get("RANDOM_DATA_DISK_PATH")
            if random_data_disk_path is None:
                raise ValueError(f"Environment variable is missing from the current environment RANDOM_DATA_DISK_PATH "
                                 f": {random_data_disk_path}")
        data_path = random_data_disk_path + f"/random_data_{str(miu_sigma[0])}_{str(miu_sigma[1])}.bin"
        with os.fdopen(os.open(data_path, os.O_WRONLY | os.O_CREAT, 0o644), 'w') as file:
            random_data.tofile(file)


class CodePrinter():
    """print numpy file"""

    def __init__(self, out_file):
        self.fout_ = os.fdopen(
            os.open(out_file, os.O_WRONLY | os.O_CREAT, 0o644), 'w')
        self.indent_ = 0

    def __del__(self):
        self.fout_.close()

    def out(self, data, new_line=False):
        """write data"""
        if new_line:
            self.fout_.write("\n")
            for _ in range(0, self.indent_):
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
    """Convert attribute list to dictionary."""
    return {attr["name"]: attr["value"] for attr in attr_desc}


def _gen_uniq_file_name(op_name):
    """Generate uniq file name."""
    if len(op_name.split("_")) > 0:
        op_hash = op_name.split("_")[-1]
    else:
        op_hash = str(time.time())

    uni_file_name_suffix = ".json_data_" + op_hash + ".py"
    file_descriptor, uni_file_name = tempfile.mkstemp(
        suffix=uni_file_name_suffix)
    os.close(file_descriptor)
    return uni_file_name


def _validate_indices_int32(input_tensor):
    """Validate indices tensor is int32 type."""
    if input_tensor["data_type"] != "int32":
        raise ValueError("Default indices type should be int32")


def _create_make_indices(name, data_shape, indices_shape, indices_dtype, attrs):
    """Create MakeIndices instance."""
    return MakeIndices(
        name=name,
        data_shape=data_shape,
        indices_shape=indices_shape,
        indices_dtype=indices_dtype,
        attrs=attrs
    )


def _get_all_tensors(operation):
    """Get all input and output tensors from operation."""
    inputs = [item for op_inputs in operation["input_desc"] for item in op_inputs]
    return inputs + operation["output_desc"]


def _check_bfloat16(operation, infos):
    """Check if operation contains bfloat16 data type."""
    if infos.get("bfloat16", False):
        return

    all_tensors = _get_all_tensors(operation)
    for tensor in all_tensors:
        if tensor.get("data_type") == "bfloat16":
            infos["bfloat16"] = True
            return


def _is_sum_operation(operation):
    """Check if operation is a sum-type operation."""
    sum_ops = ["ReduceSum", "UnsortedSegmentSum", "CSRReduceSum"]
    if operation["name"] in sum_ops:
        attr_dict = _get_attr_dict(operation["attr"])
        return "enable_atomic_add" in attr_dict
    return operation["name"] == "ElemAny"


def _handle_sum_output(operation, infos, sum_out):
    """Handle sum output operations."""
    if _is_sum_operation(operation):
        output_name = operation["output_desc"][0]["tensor_name"]
        sum_out.append(output_name)


def _handle_unsorted_segment_sum(operation, infos):
    """Handle UnsortedSegmentSum operation."""
    input0 = operation["input_desc"][0][0]
    input1 = operation["input_desc"][1][0]

    _validate_indices_int32(input1)

    num_segments = get_attr(operation["attr"], "num_segments")
    indices_info = _create_make_indices(
        name=operation["name"],
        data_shape=input0["shape"],
        indices_shape=input1["shape"],
        indices_dtype=input1["data_type"],
        attrs=num_segments
    )

    infos["indices_input"][input1["tensor_name"]] = indices_info


def _handle_inplace_assign(operation, infos, sum_out):
    """Handle InplaceAssign and Assign operations."""
    _collect_inplace_assign_infos(operation, infos, sum_out)

    write_tensor = operation["input_desc"][0][0]["tensor_name"]
    infos["inplace_assign_write"].append(write_tensor)


def _collect_inplace_assign_infos(operation, infos, sum_out):
    """Collect inplace assign infos."""
    if operation["name"] not in ["InplaceAssign", "Assign"]:
        return

    fake_output = None
    if operation["name"] == "InplaceAssign":
        fake_output = get_attr(operation["attr"], "fake_output")

    if fake_output or operation["name"] == "Assign":
        output_name = operation["output_desc"][0]["tensor_name"]
        infos["fake_output_tensors"].append(output_name)

    input0 = operation["input_desc"][0][0]
    input1 = operation["input_desc"][1][0]

    if input1["tensor_name"] in sum_out:
        infos["clean_input"].append(input0["tensor_name"])


def _handle_indices_operation(operation, infos):
    """Handle TensorScatterAdd and GatherNd operations."""
    input0 = operation["input_desc"][0][0]
    input1 = operation["input_desc"][1][0]

    _validate_indices_int32(input1)

    indices_info = _create_make_indices(
        name=operation["name"],
        data_shape=input0["shape"],
        indices_shape=input1["shape"],
        indices_dtype=input1["data_type"],
        attrs=None
    )

    infos["indices_input"][input1["tensor_name"]] = indices_info


def _handle_gather(operation, infos):
    """Handle Gather operation."""
    input0 = operation["input_desc"][0][0]
    input1 = operation["input_desc"][1][0]

    _validate_indices_int32(input1)

    if operation["attr"][0]["name"] != "axis":
        raise ValueError("Gather only accepts axis attribute.")

    axis_value = operation["attr"][0]["value"]
    indices_info = _create_make_indices(
        name=operation["name"],
        data_shape=input0["shape"],
        indices_shape=input1["shape"],
        indices_dtype=input1["data_type"],
        attrs=axis_value
    )

    infos["indices_input"][input1["tensor_name"]] = indices_info


def _handle_csr_operation(operation, infos):
    """Handle CSR operations."""
    input0 = operation["input_desc"][0][0]
    input1 = operation["input_desc"][1][0]

    _validate_csr_shapes(operation)

    dense_shape = get_attr(operation["attr"], "dense_shape")

    csr_indptr_info = _create_make_indices(
        name=input1["tensor_name"],
        data_shape=dense_shape,
        indices_shape=input1["shape"],
        indices_dtype=input0["data_type"],
        attrs=None
    )
    infos["csr_indptr"][input0["tensor_name"]] = csr_indptr_info

    csr_indices_info = _create_make_indices(
        name=input0["tensor_name"],
        data_shape=dense_shape,
        indices_shape=input1["shape"],
        indices_dtype=input1["data_type"],
        attrs=None
    )
    infos["csr_indices"][input1["tensor_name"]] = csr_indices_info


def _validate_csr_shapes(operation):
    """Validate CSR operation shapes."""
    if operation["name"] != "CSRGather":
        shape0 = operation["input_desc"][1][0]["shape"][0]
        shape1 = operation["input_desc"][2][0]["shape"][0]
        if shape0 != shape1:
            raise ValueError("indices and data should have the same shape")


def _handle_matmul_cpu(operation, infos):
    """Handle MatMul and BatchMatMul for CPU target."""
    input1 = operation["input_desc"][1][0]

    pack_b = get_attr(operation["attr"], "pack_b")
    transpose_b = get_attr(operation["attr"], "transpose_b")

    infos["need_pack_b"][input1["tensor_name"]] = pack_b
    infos["need_transpose"][input1["tensor_name"]] = transpose_b


def _init_target_info(desc, infos, target):
    """Initialize target-specific information."""
    if target == "cpu":
        target_info = desc.get("target_info", {})
        infos["feature"] = target_info.get("feature", "avx")


OPERATION_HANDLERS = {
    "UnsortedSegmentSum": _handle_unsorted_segment_sum,
    "InplaceAssign": _handle_inplace_assign,
    "Assign": _handle_inplace_assign,
    "TensorScatterAdd": _handle_indices_operation,
    "GatherNd": _handle_indices_operation,
    "Gather": _handle_gather,
}


def _dispatch_operation(operation, infos, target, sum_out):
    """Dispatch operation to appropriate handler based on strategy pattern."""
    op_name = operation["name"]

    handler = OPERATION_HANDLERS.get(op_name)
    if handler:
        handler(operation, infos, sum_out)
        return

    if op_name.startswith("CSR"):
        _handle_csr_operation(operation, infos)
    elif target == "cpu" and op_name in ["MatMul", "BatchMatMul"]:
        _handle_matmul_cpu(operation, infos)


def _collect_infos(desc, infos):
    """Collect infos - refactored with strategy pattern.

    This function collects various information from operation descriptions,
    including data types, indices, CSR operations, and target-specific info.

    Args:
        desc: Operation description dictionary
        infos: Information collection dictionary to be populated

    Complexity: O(n) where n is the number of operations
    Cyclomatic complexity: ~8 (reduced from 20+)
    """
    sum_out = []
    target = desc["process"]

    _init_target_info(desc, infos, target)

    infos["bfloat16"] = False

    for operation in desc["op_desc"]:
        _check_bfloat16(operation, infos)
        _handle_sum_output(operation, infos, sum_out)
        _dispatch_operation(operation, infos, target, sum_out)


def _pack_matrix(data, feature):
    """Pack matrix."""
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
    new_data = np.split(data, indices_or_sections=[split_pos, ], axis=-1)
    body_data = np.split(new_data[0], block_num, -1)
    dim_size = (int)(_get_size(shape) / shape[-1])
    data_list = []
    for block in body_data:
        data_list.append(block.reshape((dim_size * block_size,)))
    data_list.append(new_data[1].reshape(
        (dim_size * (shape[-1] % block_size)),))
    packed_data = np.concatenate(data_list, axis=-1)
    packed_data = packed_data.reshape(shape)
    return packed_data


def _gen_input_item(tensor_name, infos, shape, dtype, csr_idx_pair, input_mean_value):
    """Gen input item."""
    dtype = torch_normalize_dtype(dtype)
    if dtype == "bfloat16":
        try:
            from bfloat16 import bfloat16  # pylint: disable=import-outside-toplevel
        except ImportError as err:
            raise ImportError("bfloat16 is not installed, install it first.") from err
        dtype = bfloat16
    item = None
    if tensor_name in infos["clean_input"]:
        item = np.zeros(shape).astype(dtype)
    elif tensor_name in infos["csr_indptr"]:
        if tensor_name in csr_idx_pair:
            item = csr_idx_pair[tensor_name]
        else:
            indptr, indices = gen_csr_indices(
                infos["csr_indptr"][tensor_name])
            item = indptr
            csr_idx_pair[infos["csr_indptr"][tensor_name].name] = indices
    elif tensor_name in infos["csr_indices"]:
        if tensor_name in csr_idx_pair:
            item = csr_idx_pair[tensor_name]
        else:
            indptr, indices = gen_csr_indices(
                infos["csr_indices"][tensor_name])
            item = indices
            csr_idx_pair[infos["csr_indices"][tensor_name].name] = indptr
    elif tensor_name in infos["indices_input"].keys():
        item = gen_indices(infos["indices_input"][tensor_name])
    else:
        item = random_gaussian(
            shape, miu=input_mean_value, sigma=0.1).astype(dtype)
    return item


def _gen_input_data(desc, infos, input_for_mod, commands):
    """Generate input data."""
    idx = 0
    csr_idx_pair = {}
    input_mean_value = precheck(desc)
    target = desc["process"]

    for input_desc in desc["input_desc"] if desc.get("input_desc") is not None else []:
        tensor_name = input_desc[0]["tensor_name"]
        infos["input_order"][tensor_name] = idx
        if not input_desc[0]["shape"] and "int" in input_desc[0].get("data_type", ""):
            commands.append(f"{tensor_name} = int(np.array(input_dict.get('{tensor_name}')).item())")
        else:
            commands.append(f"{tensor_name} = np.array(input_dict.get('{tensor_name}'))")

        if not infos["gen_data"] and idx < len(input_for_mod):
            infos["input_dict"][tensor_name] = input_for_mod[idx]
            idx += 1
            continue

        shape = [1] if not input_desc[0]["shape"] else input_desc[0]["shape"]
        dtype = input_desc[0]["data_type"]
        if "value" in input_desc[0]:
            item = np.full(shape, input_desc[0]["value"],
                           dtype=torch_normalize_dtype(dtype))
        else:
            item = _gen_input_item(tensor_name, infos, shape,
                                   dtype, csr_idx_pair, input_mean_value)

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
    for output_desc in desc["output_desc"]:
        if infos["gen_data"]:
            shape = [1] if not output_desc["shape"] else output_desc["shape"]
            dtype = torch_normalize_dtype(output_desc["data_type"])
            if dtype == "bfloat16":
                try:
                    from bfloat16 import bfloat16  # pylint: disable=import-outside-toplevel
                except ImportError as err:
                    raise ImportError("bfloat16 is not installed, install it first.") from err
                dtype = bfloat16
            item = np.full(shape, np.nan, dtype)
            input_for_mod.append(item)
        if output_desc["tensor_name"] not in infos["fake_output_tensors"]:
            real_idx = idx - len(desc["output_desc"])
            output_indexes.append(real_idx)
            commands.append(f"expect.append({output_desc['tensor_name']})")
        idx += 1


def _check_need_reshape(input_desc):
    """Check if input shape needs reshape."""
    if len(input_desc) != 2:
        return False, -1, -1
    inputs_format = [input_desc[0][0]["format"], input_desc[1][0]["format"]]
    if inputs_format == ["DefaultFormat", "FRACTAL_NZ"]:
        fractal_tensor = input_desc[1][0]
        default_tensor = input_desc[0][0]
        return True, fractal_tensor, default_tensor
    if inputs_format == ["FRACTAL_NZ", "DefaultFormat"]:
        fractal_tensor = input_desc[0][0]
        default_tensor = input_desc[1][0]
        return True, fractal_tensor, default_tensor
    return False, -1, -1


def _emit_reshape(fractal_tensor, default_tensor):
    """Emit reshape."""
    shape_fractal = fractal_tensor["shape"]
    shape_default = default_tensor["shape"]
    shape_tmp = []
    shape_new = []
    shape_new = list(shape_default[:len(shape_default) - 2])
    for _ in range(len(shape_default), 2):
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
        sent_reshape_tensor = (f'{default_tensor["tensor_name"]} = '
                               f'np.full({shape_new}, {default_tensor["value"]}, np.{default_tensor["data_type"]})')
    else:
        if np.zeros(shape_default).size != np.zeros(shape_new).size:
            raise ValueError(f"It is error to reshape {shape_default} to {shape_new}!")
        sent_reshape_tensor = (f'{default_tensor["tensor_name"]} = '
                               f'np.reshape({default_tensor["tensor_name"]}, {tuple(shape_new)})')
    return sent_reshape_tensor


def _gen_op_compute(desc, commands):
    """Generate op compute."""
    elemwise_op_list = ["TensorAdd", "Add", "RealDiv", "Mul", "Minimum", "Maximum", "Sub"]
    for operation in desc["op_desc"]:
        dsl_fun = get_op_dsl().get(operation["name"], None)
        if dsl_fun is None:
            raise ValueError(f'op [{operation["name"]}] is not supported!')
        if operation["name"] in elemwise_op_list and operation["output_desc"][0].get("format") == "FRACTAL_NZ":
            need_reshape, fractal_tensor, default_tensor = _check_need_reshape(
                operation["input_desc"])
            if need_reshape:
                commands.append(_emit_reshape(fractal_tensor, default_tensor))
        if operation.get('attr', None):
            operation['attr'].append(
                {'name': 'process', 'value': desc['process']})
        sent = dsl_fun(operation['input_desc'],
                       operation['output_desc'], operation['attr'])
        commands.append(sent)


def _gen_torch_op_compute(desc, commands):
    """Generate torch op compute."""
    for operation in desc["op_desc"]:
        dsl_fun = get_op_dsl_torch_mlir().get(operation["name"], None)
        if dsl_fun is None:
            raise ValueError(f'op [{operation["name"]}] is not supported!')
        attrs = operation.get("attr", [])
        attrs = list(attrs)
        attrs.append({"name": "process", "value": desc["process"]})
        sent = dsl_fun(operation['input_desc'],
                       operation['output_desc'], attrs)
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
            inplace_tensors += f"{tensor_name}, "
        inplace_tensors += "]"
        commands.append("inplace_tensors = " + inplace_tensors)
        commands.append("expect.extend(inplace_tensors)")
        output_indexes.extend(inplace_tensors_index)


def _update_workspace_data(kernel_name, input_for_mod, output_indexes):
    """Update workspace tensors."""
    workspace_tensors = []
    json_file = os.path.join(os.path.realpath('./'), 'akg_kernel_meta', kernel_name + "_split.json")
    if os.path.isfile(json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            kernel_json = f.read()
            kernel_desc = json.loads(kernel_json)
            if "workspace" in kernel_desc:
                workspace_bytes = kernel_desc["workspace"]["size"]
                item = np.full(workspace_bytes, np.nan, np.int8)
                workspace_tensors.append(item)

    # Add workspace tensors to input_for_mod
    if len(workspace_tensors) > 0:
        # workspace tensors are placed after inputs and outputs, so index in output_indexes should
        # be converted to positive number first, otherwise -1 will point to the last workspace tensor
        # instead of the last output tensor.
        output_indexes = [i if i > 0 else i +
                          len(input_for_mod) for i in output_indexes]
        input_for_mod.extend(workspace_tensors)

    return output_indexes


def gen_json_data(op_desc, with_compute=True, input_for_mod=None):
    """Generating test data for composite json"""
    desc = json.loads(op_desc)

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
    if "ir_type" in desc and desc.get("ir_type") == "torch-mlir":
        _gen_torch_op_compute(desc, commands)
    else:
        _gen_op_compute(desc, commands)
    # Parse output_desc
    _gen_output_data(desc, infos, input_for_mod, output_indexes, commands)

    # Update inplace tensors
    _update_inplace_tensors(infos, output_indexes, commands)
    # Update workspace
    output_indexes = _update_workspace_data(
        desc.get("op"), input_for_mod, output_indexes)

    uni_file_name = _gen_uniq_file_name(desc.get("op"))
    printer = CodePrinter(uni_file_name)
    printer.out("from akg.utils.op_dsl import *", False)
    if infos.get('bfloat16', False):
        printer.out("try:", True)
        printer.out("    from bfloat16 import bfloat16", True)
        printer.out("except ImportError as err:", True)
        printer.out("    raise ImportError(\"bfloat16 is not installed, install it first.\") from err", True)
    printer.out("def get_expect(input_dict, expect):", True)
    for command in commands:
        single_commands = command.split("\n")
        for single_command in single_commands:
            if not single_command.strip():
                continue
            single_command = "    " + single_command
            printer.out(single_command, True)
    printer.close()

    # compute the expect data
    if with_compute:
        tmp_mod_spec = importlib.util.spec_from_file_location(
            "tmp_mod", uni_file_name)
        tmp_mod = importlib.util.module_from_spec(tmp_mod_spec)
        tmp_mod_spec.loader.exec_module(tmp_mod)
        tmp_mod.get_expect(infos["input_dict"], expect)
        os.remove(uni_file_name)
        return input_for_mod, expect, output_indexes
    return input_for_mod, -1, output_indexes
