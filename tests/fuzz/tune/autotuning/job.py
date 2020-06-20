# Copyright 2019 Huawei Technologies Co., Ltd
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

"""AutoTuning job"""
import os
import json
import datetime
import importlib
import logging
import numpy as np
from collections import namedtuple
from akg import composite
from akg.utils import kernel_exec as utils
from autotuning.runner import KernelRunner, error_time_list, error_time_string
from autotuning.tuner import ModelBasedTuner
from autotuning.type_definitions import ConvDesc, ConvBackpropDesc, MatmulCubeDesc
from autotuning.space_generators import get_space
from autotuning.space import ListConfigSpace
from autotuning.test_data_generators import gen_data

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

logger = logging.getLogger('fuzz.tune.autotuning.job')

json_file = './res/' + "{0}" + ".json"
json_load = './autotuning/shapes/' + "{0}"

def launch_json(debug_mode: bool = True, save_res: bool = False, json_input_dir=""):
    """composite json tuning launch"""
    iter_times = [3, 3, 3] if debug_mode else [80, 160, 320]
    json_dir = json_load.format(json_input_dir)
    files = os.listdir(json_dir)
    for input_file in files:
        with open(json_dir + '/' + input_file, 'r') as f:
            json_input = f.read()
        json_content = json.loads(json_input)
        for input_desc in json_content["input_desc"]:
            if input_desc[0]["shape"] == []:
                input_desc[0]["shape"] = [1]
        json_input = json.dumps(json_content)
        space_res = composite.get_tiling_space(json_input, 2)
        index_table = space_res['index']
        tiling_spaces = space_res['tuning_space']
        if not tiling_spaces:
            raise RuntimeError('empty tiling spaces')
        dim_names = ['tiling_' + str(i) for i in range(len(tiling_spaces[0]))]
        input_type = namedtuple("json", dim_names)
        space = ListConfigSpace(input_type)
        for tiling_space in tiling_spaces:
            config = input_type(*tiling_space)
            space.add(config)
        key = json_content["op"]
        input_for_mod, expect = gen_data(op_type="json", op_desc=json_input)

        print('space size:', space.length)
        print('index table:', index_table)

        output_para = None  # this is for multi-output
        if len(json_content["output_desc"]) > 1:
            output_para = []
            for i in range(len(json_content["output_desc"])):
                output_para.append(i - len(json_content["output_desc"]))
        runner = KernelRunner(op_type="json", op_desc=json_input, index_table=index_table, input_data=input_for_mod,
                            expect=expect, mod_output_param=output_para, timeout=180, repeat_times=1)

        # we can only get a valid tiling, or accurate get cycles
        is_truly_profiling = utils.get_profiling_mode()

        # available device numbers, normally is 8 or 1
        available_device_numbers = utils.get_available_devices_num()

        tuner = ModelBasedTuner(runner, index_table, space,
                                n_parallel=available_device_numbers if is_truly_profiling else 1,
                                plan_size=64, pre_model=None)
        least_try_times = iter_times[0 if space.length < 10 ** 4 else 1 if space.length < 10 ** 5 else 2]
        tuner.tune(least_try_times, output_file="json.log")

        print_tuning_result("json", space, index_table, tuner, key)

        if save_res:
            save_tuning_result(key, "json", None, index_table, tuner)

def jobs(op_type: str = 'add', desc=None, debug_mode: bool = True,
         save_res: bool = False, insert_key='', conf_of_set_dim=""):
    """AutoTuning jobs"""
    iter_times = [3, 3, 3] if debug_mode else [80, 160, 320]
    index_table, space, key, expect, input_for_mod = get_space(op_type, desc)
    print('space size:', space.length)
    print('index table:', index_table)
    key = key if insert_key == '' else insert_key

    # filter already tuned shape
    if isinstance(conf_of_set_dim, dict) and key in conf_of_set_dim.keys():
        if isinstance(conf_of_set_dim[key], (list, tuple)) and conf_of_set_dim[key]:
            return

        if isinstance(conf_of_set_dim[key], dict):
            return

    output_para = None  # this is for multi-output
    if isinstance(input_for_mod, dict):
        input_for_mod, output_para = input_for_mod['args'], input_for_mod['outputs']
    runner = KernelRunner(op_type, desc, index_table, input_data=input_for_mod,
                          expect=expect, mod_output_param=output_para, timeout=180, repeat_times=1)

    # we can only get a valid tiling, or accurate get cycles
    is_truly_profiling = utils.get_profiling_mode()

    # available device numbers, normally is 8 or 1
    available_device_numbers = utils.get_available_devices_num()

    tuner = ModelBasedTuner(runner, index_table, space,
                            n_parallel=available_device_numbers if is_truly_profiling else 1,
                            plan_size=64, pre_model=None)
    least_try_times = iter_times[0 if space.length < 10 ** 4 else 1 if space.length < 10 ** 5 else 2]
    tuner.tune(least_try_times, output_file=op_type + ".log")

    print_tuning_result(op_type, space, index_table, tuner, key)

    if save_res:
        save_tuning_result(key, op_type, desc, index_table, tuner)


def print_tuning_result(op_type, space, index_table, tuner, key):
    """print tuning result"""
    print(op_type + " shape is:", key)
    print('space size:', space.length)
    print('index table:', index_table)
    print('best config:', tuner.best_config)
    print('best time:',
          tuner.best_time if tuner.best_time not in error_time_string.keys() else error_time_string[tuner.best_time])
    print('original time:', tuner.original_time)
    print('optimal result is ', tuner.original_time / tuner.best_time, "faster then auto set dim.")
    print("total try times", len(tuner.xs))
    for x, y in zip(tuner.xs, tuner.ys):
        print(space.get(x), y if y not in error_time_string.keys() else error_time_string[y])


def save_tuning_result(key, op_type, desc, index_table, tuner):
    """save tuning result"""
    if tuner.best_config is not None and tuner.best_time not in error_time_list:
        set_dim_configs = tuner.best_config.input
        if op_type == "matmul":
            param = []
            for _ in range(len(desc.x_shape) - 2):
                param.append((1, 1))
            if set_dim_configs.n_l1 > 0:
                param.append((set_dim_configs.n_l1, set_dim_configs.n_l0))
            if set_dim_configs.m_l1 > 0:
                param.append((set_dim_configs.m_l1, set_dim_configs.m_l0))
            param.extend([(16, 16), (16, 16), (set_dim_configs.k_l1, set_dim_configs.k_l0)])
            tiling_param = (param, {"bypass": set_dim_configs.bypass})

        # special case with different tiling parameter format
        elif op_type in ("conv", "conv_bn1"):
            param = []
            tile_hh = set_dim_configs.tile_h
            tile_coco = set_dim_configs.tile_co
            tile_mm = set_dim_configs.tile_m
            tile_kk = set_dim_configs.tile_k
            tile_nn = set_dim_configs.tile_n
            tile_ww = set_dim_configs.tile_w
            param = [tile_hh, tile_coco, tile_mm, tile_kk, tile_nn, tile_ww]
            tiling_param = (param, {"bypass": set_dim_configs.bypass})
        elif op_type == "conv_backprop_input":
            param = []
            tile_hh = set_dim_configs.tile_h
            tile_coco = set_dim_configs.tile_co
            tile_mm = set_dim_configs.tile_m
            tile_kk = set_dim_configs.tile_k
            tile_nn = set_dim_configs.tile_n
            tile_ww = set_dim_configs.tile_w
            param = [tile_hh, tile_coco, tile_mm, tile_kk, tile_nn, tile_ww]
            tiling_param = (param)
        elif op_type == "conv_backprop_filter":
            param = []
            tile_cici = set_dim_configs.tile_ci
            tile_khkh = set_dim_configs.tile_kh
            tile_kwkw = set_dim_configs.tile_kw
            tile_coco = set_dim_configs.tile_co
            tile_bb = set_dim_configs.tile_batch
            tile_hh = set_dim_configs.tile_h
            tile_ww = set_dim_configs.tile_w
            tile_mm = set_dim_configs.tile_m
            tile_kk = set_dim_configs.tile_k
            tile_nn = set_dim_configs.tile_n
            param = [tile_cici, tile_khkh, tile_kwkw, tile_coco, tile_bb, tile_hh, tile_ww, tile_mm, tile_kk, tile_nn]
            tiling_param = (param)
        else:
            tiling = [[getattr(set_dim_configs, name), 1]
                      for name in getattr(set_dim_configs, '_fields') if name.startswith('tiling')]
            tiling_param = []
            for i, tile_v in enumerate(tiling):
                tiling_param.append(index_table[i] + tile_v)
    else:
        tiling_param = []

    # when there is a valid result, save the result
    if tuner.best_time not in error_time_list:
        config = {'tiling': tiling_param,
                  'best_cycles': tuner.best_time,
                  'original_cycles': tuner.original_time,
                  "date": str(datetime.datetime.now()),
                  }
        tuner.export_dim_configs(config, json_file.format(op_type), False, str(key))


def load_json_configs(op_type):
    """load json configs"""
    dim_file = json_file.format(op_type)
    file_path = os.path.realpath(dim_file)
    if os.path.isfile(file_path):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                return data
        except IOError as e:
            logger.debug(e)
            return {}
    return {}

def read_shapes_from_file(debug_mode, save_res, conf_of_set_dim, op_type):
    """read tuning shapes from file"""
    file = importlib.import_module('autotuning.shapes.' + op_type)
    shapes = file.shapes
    for _, shp in enumerate(shapes):
        do_profiling(shp, debug_mode, save_res, op_type, conf_of_set_dim)

def do_profiling(shp, debug_mode, save_res, op_type, conf_of_set_dim=None):
    """do profiling"""
    if op_type == 'matmul':
        key = shp[2][0:-1]
        logger.debug("start profiling: [%s]", str(key))
        desc = MatmulCubeDesc(*key)
        jobs(op_type, desc, debug_mode, save_res, key.__str__(), conf_of_set_dim)
        logger.debug("end profiling: [%s]", str(key))
    elif op_type.startswith('conv_backprop'):
        key = shp[2]
        logger.debug("start profiling: [%s]", str(key))
        desc = ConvBackpropDesc(*key)
        jobs(op_type, desc, debug_mode, save_res, key.__str__(), conf_of_set_dim)
        logger.debug("end profiling: [%s]", str(key))
    elif op_type.startswith('conv'):
        key = shp[2]
        logger.debug("start profiling: [%s]", str(key))
        desc = ConvDesc(*key)
        jobs(op_type, desc, debug_mode, save_res, key.__str__(), conf_of_set_dim)
        logger.debug("end profiling: [%s]", str(key))
    else:
        key = shp
        logger.debug("start profiling: [%s]", str(key))
        desc = key
        jobs(op_type, desc, debug_mode, save_res, conf_of_set_dim=conf_of_set_dim)
        logger.debug("end profiling: [%s]", str(key))

def launch(op_type, debug_mode, save_res=False, desc=None):
    # get the existed tiling
    conf_of_set_dim = load_json_configs(op_type)

    if desc is None:
        read_shapes_from_file(debug_mode, save_res, conf_of_set_dim, op_type)
    else:
        shp = desc
        do_profiling(shp, debug_mode, save_res, op_type)
