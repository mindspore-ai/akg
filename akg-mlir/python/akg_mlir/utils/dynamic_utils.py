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

"""Utils Functions for Dynamic Shape Inputs"""
import copy
import os
import sys
import json
import logging
import numpy as np

HOST_SHAPES = "hostShapes"
DEVICE_SHAPES = "deviceShapes"
RUNTIME_VARS = "runtimeVars"
SUPPORT_INFO = "SupportInfo"
RUNTIME_VARS_PRIME = "prime"
RUNTIME_VARS_ARGINDEX = "argIndex"
RUNTIME_VARS_EXPR = "expr"
MAX_GPU_BLOCKDIMS = (2147483647, 65535, 65535)
MAX_GPU_THREADDIMS = (1024, 1024, 64)
MAX_GPU_THREADNUM = 1024
SP_SHAPE_RANGE = (33, 63)
SP_BLOCK_X = 32
SP_BLOCK_Y = 8


def _get_symbol_expr_value(symbol_map, sym_expr):
    accum_value = 1
    for sym in sym_expr.split("*"):
        if sym.isdigit():
            accum_value *= int(sym)
        elif sym in symbol_map:
            accum_value *= symbol_map[sym]
        else:
            raise RuntimeError("Symbol {} not found in symbol map {}!".format(
                sym, symbol_map))
    return accum_value


def get_device_shape(input_for_mod, kernel_name, is_dyn_shape, cur_dir=""):
    """
    Generate tensor shapes on the devivce

    Args:
        input_for_mod: the input tensors
        kernel_name (str): the name of the kernel
        is_dyn_shape (bool): whether the inputs contain dynamic shapes
        cur_dir: the path of current dir. Defaults to "".



    Returns:
        device shapes and related info
    """
    host_shape = []
    support_info = dict()
    for each_in in input_for_mod:
        if isinstance(each_in, int):
            host_shape.append(each_in)
        else:
            host_shape.append(each_in.shape)
    symbol_map = dict()

    if not is_dyn_shape:
        return host_shape, symbol_map, support_info

    shape_info_file = os.path.join(
        cur_dir, "akg_kernel_meta", kernel_name + "_shape_info.json")
    if not os.path.exists(shape_info_file):
        logging.warning(
            "Dynamic shape needs file %s to get the device shape. Please use "
            "`--dump-shape-info` to generate. Otherwise, the result may be incorrect.",
            str(shape_info_file))
        return host_shape, symbol_map, support_info

    device_shape = []
    with open(shape_info_file, "r") as f:
        shape_map = json.loads(f.read())
        host_sym_shape = shape_map.get(HOST_SHAPES)
        device_sym_shape = shape_map.get(DEVICE_SHAPES)
        runtime_vars = shape_map.get(RUNTIME_VARS, dict())
        support_info = shape_map.get(SUPPORT_INFO, dict())
        for var in runtime_vars:
            symbol_map[var.get(RUNTIME_VARS_PRIME)] = var
        if host_sym_shape is None:
            raise RuntimeError(
                "host_shape not found in {}".format(shape_info_file))
        if device_sym_shape is None:
            raise RuntimeError(
                "device_shape not found in {}".format(shape_info_file))
        if len(host_sym_shape) != len(host_shape):
            raise ValueError(
                "Host' real shape and symbolic shape ranks not equal: {} vs {}"
                .format(len(host_shape), len(host_sym_shape)))
        # 1. map the symbol of host to real static shape
        for real_sh, sym_host in zip(host_shape, host_sym_shape):
            for idx, sym_h in enumerate(sym_host):
                symbol_map[sym_h] = real_sh[idx]

        # 2. generate new device shapes based on the symbol_map
        for sym_device in device_sym_shape:
            new_shape = []
            gen = (sym_d for sym_d in sym_device if "*" in sym_d)
            for sym_d in gen:
                # map the symbol expr of device to real static shape like s0x1024xs1
                symbol_map[sym_d] = _get_symbol_expr_value(
                    symbol_map, sym_d)
            new_shape = [
                int(sym_d) if sym_d.isdigit() else symbol_map[sym_d]
                for sym_d in sym_device
            ]
            device_shape.append(tuple(new_shape))
    logging.info("Host shape: %s Device shape %s, symbol_map = %s",
                 str(host_shape), str(device_shape), str(symbol_map))
    return device_shape, symbol_map, support_info


def dump_shape_arg_list(data, kernel_name, cur_dir):
    """
    dump the list of shape args as txt file

    Args:
        data: original inputs
        kernel_name: the name of the kernel
        cur_dir: local file path
    """
    device_shape, _, _ = get_device_shape(data,
                                          kernel_name,
                                          is_dyn_shape=True,
                                          cur_dir=cur_dir)
    shape_arg_list = list()
    for data_idx, d in enumerate(data):
        # NOTE(yanzhi): If input is a memeref<f32>, its shape_arg is [0];
        # for the case of memref<?xf32>, its shape_args is [offset, sizes, strides].
        # Need more discuss here.
        shape_list = [0]
        data_shape = device_shape[data_idx]
        if isinstance(d, np.ndarray):
            shape_list += list(data_shape)
            # for tensor (m, n, k), the strides are [n*k, k, 1]
            stride_list = [1] * len(data_shape)
            for i, _ in enumerate(data_shape[1:]):
                stride_list[-i - 2] = stride_list[-i - 1] * data_shape[-i - 1]
            shape_list += stride_list
        else:
            raise TypeError("wrong data to cytpes, current type is '", type(d),
                            "'")
        shape_arg_list.append(shape_list)
    with os.fdopen(os.open(os.path.join(
            cur_dir, "akg_kernel_meta", kernel_name + "_shape_arg.txt"),
            os.O_WRONLY | os.O_CREAT, 0o400), 'wt') as f:
        f.write(str(shape_arg_list))


def _get_proper_reduce_y_config(red_size, use_atomic=False):
    block_num = 1
    acc_seq = red_size
    if use_atomic:
        if red_size < 256:
            acc_seq = 16
        elif red_size < 1024:
            acc_seq = 32
        else:
            acc_seq = 64
        block_num = (red_size - 1) // acc_seq + 1

    return block_num, acc_seq


def _get_proper_reduce_x_config(red_size, use_atomic=False):
    block_num = 1
    if use_atomic:
        block_num = (red_size - 1) // 1024 + 1
        red_size = (red_size - 1) // block_num + 1
    thread_num = red_size if 32 > red_size else 32
    while (thread_num * 4 < red_size and thread_num < 1024):
        thread_num *= 2
    return block_num, thread_num, (red_size - 1) // (thread_num * block_num) + 1


def _get_proper_parallel_thread_config(length, is_reduce_y):
    if is_reduce_y:
        return 32
    if length < 64:
        base = 1
        thread = 512
        while base < length:
            base *= 2
            thread //= 2
        return thread
    return 32


def _get_reduce_tile_size(map_info, upper_bound, proper_block, proper_thread, proper_seq):
    if "thread" in map_info["mark"]:
        return min(upper_bound, proper_thread)
    elif "reduce-small-seq" in map_info["mark"]:
        return upper_bound
    elif "seq" in map_info["mark"]:
        return min(upper_bound, proper_seq)
    elif "1" == map_info["mark"]:
        return 1


def _decode_expr(orig_expr, upper_bound):
    orig_expr = orig_expr.replace("(", "").replace(")",
                                                   "").replace(" ", "")
    all_expr = orig_expr.split("OR")
    tile_candidates = []
    for expr in all_expr:
        if expr == "-1":
            tile_candidates.append(upper_bound)
        elif expr.isdigit():
            if int(expr) <= upper_bound:
                tile_candidates.append(int(expr))
        elif "min" in expr:
            min_tile = float("inf")
            gen = (n for n in expr.replace("min", "").split(
                ",") if n.isdigit() or n == "-1")
            for n in gen:
                min_tile = min(min_tile, int(
                    n) if n.isdigit() else upper_bound)
            tile_candidates.append(int(min_tile))
        elif "mod" in expr:
            mod_size = int(expr.replace("mod", ""))
            gen = (n for n in range(upper_bound, -1, -1)
                   if n >= mod_size and n % mod_size)
            for n in gen:
                tile_candidates.append(n)
            if len(tile_candidates) == 0:
                tile_candidates.append(1)
        elif "lessequal" in expr:
            num = int(expr.split("lessequal")[-1])
            for i in range(1, min(num + 1, upper_bound)):
                tile_candidates.append(i)
    return tile_candidates


def _need_solve(curr_list):
    return -1 in curr_list


def _get_tile_size(map_info, defined_upper=None):
    expr = map_info.get(RUNTIME_VARS_EXPR)
    upper_bound = defined_upper if defined_upper else map_info.get(
        "upper_bound")
    if expr:
        # tiling strategy already set in c++ code, we only decode here
        tile_candidates = _decode_expr(expr, upper_bound)
    else:
        tile_candidates = [upper_bound]
    return tile_candidates


class DynamicTilingSolver:
    """
    Solve tiling size for dynamic case,
    and create corresponding runtime args
    """

    def __init__(self, symbol_map, dyn_map, runtime_arg_file, support_info):
        self.symbol_map = copy.deepcopy(symbol_map)
        self.dyn_map = copy.deepcopy(dyn_map)
        self.runtime_arg_file = runtime_arg_file
        self.support_info = support_info
        self.runtime_args = dict()
        self.curr_grid = [
            1,
        ] * len(MAX_GPU_BLOCKDIMS)
        self.curr_block = [
            1,
        ] * len(MAX_GPU_THREADDIMS)
        self.total_alloc_blocks = 1
        self.total_alloc_threads = 1

        self.lookup_map = {
            "blockIdx.x": (self.curr_grid, 0),
            "blockIdx.y": (self.curr_grid, 1),
            "blockIdx.z": (self.curr_grid, 2),
            "threadIdx.x": (self.curr_block, 0),
            "threadIdx.y": (self.curr_block, 1),
            "threadIdx.z": (self.curr_block, 2)
        }

    def _update_static_map(self, map_key, map_size):
        if map_key in self.lookup_map:
            write_map = self.lookup_map[map_key][0]
            write_index = self.lookup_map[map_key][1]
            write_map[write_index] = map_size
        if map_size != -1 and isinstance(map_key, str):
            if "block" in map_key:
                self.total_alloc_blocks *= map_size
            elif "thread" in map_key:
                self.total_alloc_threads *= map_size

    def _record_solved(self, map_info, map_key, map_res, tile_size):
        if map_key in self.dyn_map.keys():
            self.dyn_map[map_key] = tile_size
            self._update_static_map(map_key, tile_size)
        outer_map = map_info.get("outer_map", None)
        if outer_map is not None:
            self.dyn_map[outer_map][1] = tile_size

        arg_index = map_info.get(RUNTIME_VARS_ARGINDEX)
        self.runtime_args[arg_index] = tile_size
        if -map_res in self.symbol_map:
            neg_map_info = self.symbol_map.get(-map_res)
            self.runtime_args[neg_map_info.get(
                RUNTIME_VARS_ARGINDEX)] = -tile_size

    def _get_match_key(self, key):
        for k, v in self.dyn_map.items():
            if isinstance(v, int) and v == key:
                return k
        return key

    def _gen_values(self):
        axis_length_left = {}  # record the use of this axis
        product_var = {}  # point out key to bound
        related_values = {}  # record tile = tile0xtile1
        for map_key, map_res in self.dyn_map.items():
            if isinstance(map_res, list) and len(tuple(map_res)) == 2:
                symbol_part = tuple(map_res)[0]
                const_part = tuple(map_res)[1]
                if not (isinstance(symbol_part, str) and isinstance(const_part, int)):
                    continue
                if const_part in self.symbol_map and symbol_part in self.symbol_map:
                    axis_length_left[symbol_part] = self.symbol_map[symbol_part]
                    product_var[const_part] = symbol_part
                else:
                    print("{} or {} not in map".format(
                        const_part, symbol_part))
                    print(const_part in self.symbol_map)
                    print(symbol_part in self.symbol_map)

        keys = self.symbol_map.keys()
        for map_key, map_res in self.dyn_map.items():
            if not ("Idx" in map_key or "Seq" in map_key):
                continue
            if isinstance(map_res, int):
                if map_res <= 1:
                    continue
                if map_res in self.symbol_map:
                    curr_map = -1
                    for k in keys:
                        if not isinstance(k, int) or k <= 1 or k == map_res:
                            continue
                        if k % map_res == 0:
                            product_var[map_res] = product_var[k]
                            product_var[k // map_res] = product_var[k]
                            related_values[k] = [map_res, k // map_res]

                else:
                    curr_map = map_res

            elif len(tuple(map_res)) == 2:
                symbol_part = tuple(map_res)[0]
                const_part = tuple(map_res)[1]
                if not (isinstance(symbol_part, str) and isinstance(const_part, int)):
                    continue
                if const_part in self.symbol_map and symbol_part in self.symbol_map:
                    self.symbol_map[const_part].update({
                        "upper_bound":
                        self.symbol_map[symbol_part],
                        "outer_map":
                        map_key
                    })
                curr_map = -1
            self._update_static_map(map_key, curr_map)
        return axis_length_left, product_var, related_values

    def _solve_reduce(self, axis_length_left, product_var, related_values):
        # solve reduce
        keys = self.symbol_map.keys()
        orders = [
            "reduce-thread-last",
            "reduce-thread",
            "parallel-thread-last",
            "parallel-thread",
            "reduce-y-seq",
            "reduce-x-seq",
            "reduce-small-seq",
            "parallel-seq",
            "1",
            "product"]
        total_red_size = self.support_info["ReduceSizeStatic"]
        proper_block, proper_thread, proper_seq = None, None, None
        if self.support_info["DynAlgorithm"] == "reduce-x":
            for order in ["reduce-thread-last", "reduce-thread", "thread-last", "thread", ]:
                for key in keys:
                    if not isinstance(key, int) or key <= 0:
                        continue
                    if isinstance(self.symbol_map[key], dict) and self.symbol_map[key].get(
                            "mark", "unknown") == order:
                        total_red_size *= axis_length_left[product_var[key]]
            proper_block, proper_thread, proper_seq = _get_proper_reduce_x_config(
                total_red_size, self.support_info["EnableAtomic"])
        elif self.support_info["DynAlgorithm"] == "reduce-y":
            for order in ["reduce-y-seq", ]:
                for key in keys:
                    if not isinstance(key, int) or key <= 0:
                        continue
                    if isinstance(self.symbol_map[key], dict) and self.symbol_map[key].get(
                            "mark", "unknown") == order:
                        total_red_size *= axis_length_left[product_var[key]]
            proper_block, proper_seq = _get_proper_reduce_y_config(
                total_red_size, self.support_info["EnableAtomic"])
        elif self.support_info["DynAlgorithm"] == "reduce-small":
            proper_seq = 64
        if self.support_info["DynAlgorithm"] != "reduce-x":
            # since we don't know real parallel-length here, we set a default value(32) currently.
            # update `information structure` / `auto-tiling` later
            proper_thread = 32
        proper_thread = (proper_thread - 1) // self.total_alloc_threads + 1
        for order in orders:
            for key in keys:
                if not isinstance(key, int) or key <= 0:
                    continue
                if isinstance(self.symbol_map[key], dict) and self.symbol_map[key].get(
                        "mark", "unknown") == order:
                    if "thread" in order:
                        upper_bound = min(
                            axis_length_left[product_var[key]], 1024 // self.total_alloc_threads)
                    else:
                        upper_bound = axis_length_left[product_var[key]]
                    map_info = self.symbol_map.get(key)
                    if key in related_values:
                        tile_size = self.symbol_map[
                            related_values[key][0]]["tile_size"] * self.symbol_map[
                                related_values[key][1]]["tile_size"]
                    else:
                        tile_size = _get_reduce_tile_size(
                            map_info, upper_bound, proper_block, proper_thread, proper_seq)
                        if "seq" in map_info["mark"]:
                            proper_seq = max(proper_seq // tile_size, 1)
                    match_key = self._get_match_key(key)
                    self._record_solved(
                        map_info, match_key, key, tile_size)
                    self.symbol_map[key].update({"tile_size": tile_size})
                    axis_length_left[product_var[key]] = (axis_length_left[product_var[key]] -
                                                          1) // tile_size + 1

    def _solve_elemwise(self):
        # solve elemwise
        block_solve_order = ["threadIdx.x", "threadIdx.y", "threadIdx.z"]
        grid_solve_order = ["blockIdx.x", "blockIdx.y", "blockIdx.z"]
        solve_order = [block_solve_order, grid_solve_order]
        for each in solve_order:
            is_block = each == block_solve_order
            curr_list = self.curr_block if is_block else self.curr_grid
            if not _need_solve(curr_list):
                continue
            for map_key in each:
                map_res = self.dyn_map.get(map_key)
                if not isinstance(map_res, int):
                    continue
                map_info = self.symbol_map.get(map_res)
                if not map_info:
                    continue
                write_index = self.lookup_map[map_key][1]
                map_limit = min(
                    MAX_GPU_THREADNUM //
                    self.total_alloc_threads, MAX_GPU_THREADDIMS[write_index]
                ) if is_block else MAX_GPU_BLOCKDIMS[write_index]
                tile_candidates = _get_tile_size(map_info)
                if len(tile_candidates) > 1:
                    if map_key == "threadIdx.x":
                        tile_size = max(tile_candidates)
                    elif map_key == "threadIdx.y":
                        tile_size = min(tile_candidates)
                    else:
                        logging.warning(
                            "[Warning] Cannot solve multiple tilesize candidates "
                            "for %s now, use the top one from : %s", str(map_key), str(tile_candidates))
                        tile_size = tile_candidates[0]
                else:
                    tile_size = tile_candidates[0]
                # upper_bound can be changed during solving, update code is needed.
                upper_bound = map_info.get("upper_bound")
                if upper_bound in range(SP_SHAPE_RANGE[0], SP_SHAPE_RANGE[1]):
                    if map_key == "threadIdx.x":
                        tile_size = SP_BLOCK_X
                    elif map_key == "threadIdx.y" and self.total_alloc_threads >= SP_BLOCK_X:
                        tile_size = SP_BLOCK_Y
                tile_size = max(min(tile_size, upper_bound, map_limit), 1)
                self._record_solved(map_info, map_key, map_res, tile_size)

    def solve(self):
        """solve information for dynamic inputs"""
        axis_length_left, product_var, related_values = self._gen_values()
        if self.support_info["OperatorType"] == "Reduce":
            self._solve_reduce(axis_length_left, product_var, related_values)
        else:
            self._solve_elemwise()

        logging.info(
            "Dynamic tiling solve result: %s", str(self.runtime_args))
        with os.fdopen(os.open(self.runtime_arg_file, os.O_WRONLY | os.O_CREAT, 0o400), 'w') as f:
            f.write(json.dumps(self.runtime_args))

        return self.symbol_map, self.dyn_map


def get_gpu_setting_dynamic(symbol_map, dyn_map, dyn_map_file, support_info):
    """Set the config for dynamic cases"""
    undef = -sys.maxsize
    dim = {
        "blockIdx.x": undef,
        "blockIdx.y": undef,
        "blockIdx.z": undef,
        "threadIdx.x": undef,
        "threadIdx.y": undef,
        "threadIdx.z": undef
    }

    runtime_arg_file = dyn_map_file.replace(".json", "_runtime_arg.txt")

    solver = DynamicTilingSolver(
        symbol_map, dyn_map, runtime_arg_file, support_info)
    symbol_map, dyn_map = solver.solve()

    for map_key, map_res in dyn_map.items():
        if isinstance(map_res, int) and map_key in dim:
            dim[map_key] = map_res
        elif isinstance(map_res, list) and len(tuple(map_res)) == 2:
            symbol_part = tuple(map_res)[0]
            const_part = tuple(map_res)[1]
            if not (isinstance(symbol_part, str) and isinstance(const_part, int)):
                continue
            if symbol_part not in symbol_map:
                raise ValueError("Cannot find {} in symbol map: {}".format(
                    symbol_part, symbol_map))
            shape = symbol_map[symbol_part]
            if const_part == 0:
                raise ValueError("Const part is zero, please check")
            dim[map_key] = (shape - 1) // const_part + 1

    return dim


def get_gpu_setting_by_input(symbol_map, dyn_map_file: str, support_info):
    """From inputs check whether dynamic or not"""
    # file validation check
    if not os.path.isfile(dyn_map_file):
        raise ValueError(f"{dyn_map_file} is not a valid file")
    if os.path.splitext(dyn_map_file)[1].lower() != ".json":
        raise ValueError(f"{dyn_map_file} is not a .json file")

    with open(dyn_map_file) as f:
        dyn_map = json.load(f)
    if dyn_map.get("is_dynamic", False):
        out = get_gpu_setting_dynamic(symbol_map, dyn_map, dyn_map_file,
                                      support_info)
        return out
    else:
        return dyn_map
