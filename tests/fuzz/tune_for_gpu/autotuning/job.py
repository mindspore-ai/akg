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
import time
import datetime
import importlib
import logging
import pandas as pd
import subprocess
import numpy as np
from collections import namedtuple
from multiprocessing import Process, Manager
from akg import composite
from akg.utils import kernel_exec as utils
from akg.composite.build_module import generate_trait
from autotuning.runner import KernelRunner, error_time_list, error_time_string
from autotuning.tuner import ModelBasedTuner, Tuner
from autotuning.type_definitions import ConvDesc, ConvBackpropDesc, MatmulCubeDesc
from autotuning.space_generators import get_space
from autotuning.space import ListConfigSpace
from autotuning.data_generators import gen_data
from autotuning.space_generators import gen_bool_list
from autotuning.tuning_utils import *

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger('fuzz.tune.autotuning.job')

storage_dir = './res/'

if not os.path.exists(storage_dir):
    os.makedirs(storage_dir, exist_ok=True)

json_file = './res/' + "{0}" + ".json"
json_load = './autotuning/shapes/' + "{0}"


def get_repo(repo, keys, default=None):
    for key in keys:
        repo = repo.get(key)
        if not repo:
            return default
    return repo


def get_json_space(json_input, space_dict):
    space_res = composite.get_tiling_space(json_input, 2)
    space_dict['res'] = space_res


def launch_json(debug_mode: bool = True, save_res: bool = False, json_dir="", repo_path="", all_space=False,
                skip_exist=True, extra_tune=False, self_attrs=[], tuning_attrs=[]):
    """composite json tuning launch"""
    subprocess.run("mkdir -p res/", shell=True)
    iter_times = [3, 3, 3] if debug_mode else [80, 160, 320]
    files = os.listdir(json_dir)
    with open(repo_path, 'r') as f:
        repo = json.loads(f.read())
    for input_file in files:
        print("----Start tuning for ", input_file)
        with open(json_dir + '/' + input_file, 'r') as f:
            json_input = f.read()
        json_content = json.loads(json_input)
        for input_desc in json_content["input_desc"]:
            if input_desc[0]["shape"] == []:
                input_desc[0]["shape"] = [1]
        json_input = json.dumps(json_content)

        # skip tuning for info in repo
        if skip_exist:
            compute, shape, dtype = generate_trait(json_content)
            if get_repo(repo, [compute, shape, dtype]):
                print("Info for %s already exists" % input_file)
                print("ops are ", str(compute))
                print("shape is ", str(shape))
                print("dtype is ", str(dtype))
                with open('res/skip_file.txt', 'a') as fe:
                    fe.write(input_file)
                    fe.write("\n")
                continue

        # generate tuning space
        if not extra_tune:
            time_start_get_space = time.time()
            with Manager() as manager:
                space_dict = manager.dict()
                p = Process(target=get_json_space,
                            args=(json_input, space_dict))
                p.start()
                p.join(600)
                if 'res' not in space_dict:
                    with open('res/error_space_list.txt', 'a') as fe:
                        fe.write(input_file)
                        fe.write("\n")
                    continue
                space_res = space_dict['res']
            time_end_get_space = time.time()
            print("get space time: ", time_end_get_space - time_start_get_space)
            index_table = space_res['index']
            tiling_spaces = space_res['tuning_space']
            if not isinstance(tiling_spaces, list):
                with open('res/empty_space_list.txt', 'a') as fe:
                    fe.write(input_file)
                    fe.write("\n")
                continue
            dim_names = ['tiling_' + str(i)
                         for i in range(len(tiling_spaces[0]))]
            use_tuning_attrs = len(tiling_spaces) < 10 ** 5
            if tuning_attrs and use_tuning_attrs:
                dim_names.extend(tuning_attrs)
            input_type = namedtuple("json", dim_names)
            space = ListConfigSpace(input_type)
            if tuning_attrs and use_tuning_attrs:
                attr_options = gen_bool_list(tuning_attrs)
                for tiling_space in tiling_spaces:
                    for attr_option in attr_options:
                        tmp = tiling_space[:]
                        tmp.extend(attr_option)
                        config = input_type(*tmp)
                        space.add(config)
            else:
                for tiling_space in tiling_spaces:
                    config = input_type(*tiling_space)
                    space.add(config)
        else:
            index_table = []
            pre_lists = gen_bool_list(self_attrs)
            pre_input_type = namedtuple("extra_tune", self_attrs)
            space = ListConfigSpace(pre_input_type)
            for item in pre_lists:
                config = pre_input_type(*item)
                space.add(config)

        key = json_content["op"]
        try:
            input_for_mod, expect = gen_data(
                op_type="json", op_desc=json_input)
        except BaseException as e:
            logger.debug(
                "gen numpy data from [%s] failed: %s", input_file, str(e))
            with open('res/error_gen_data_list.txt', 'a') as fe:
                fe.write(input_file)
                fe.write(": ")
                fe.write(str(e))
                fe.write("\n")
            continue
        print('space size:', space.length)
        print('index table:', index_table)

        output_para = None  # this is for multi-output
        if len(json_content["output_desc"]) > 1:
            output_para = []
            for i in range(len(json_content["output_desc"])):
                output_para.append(i - len(json_content["output_desc"]))
        runner = KernelRunner(op_type="json", op_desc=json_input, index_table=index_table, self_attrs=self_attrs,
                              input_data=input_for_mod, expect=expect, mod_output_param=output_para, timeout=180,
                              repeat_times=1)

        # we can only get a valid tiling, or accurate get cycles
        is_truly_profiling = utils.get_profiling_mode(
        ) or os.environ['RUNTIME_MODE'] == "gpu"

        # available device numbers, normally is 8 or 1
        available_device_numbers = utils.get_available_devices_num()

        if all_space:
            tuner = Tuner(runner, index_table, space,
                          n_parallel=available_device_numbers)
            least_try_times = 3  # space.length
        else:
            tuner = ModelBasedTuner(runner, index_table, space,
                                    n_parallel=available_device_numbers if is_truly_profiling else 1,
                                    plan_size=64, pre_model=None)
            least_try_times = iter_times[0 if space.length <
                                         10 ** 4 else 1 if space.length < 10 ** 5 else 2]
        tuner.tune(least_try_times, output_file="json.log")

        print_tuning_result("json", space, index_table, tuner, key)

        if save_res:
            if extra_tune:
                save_tuning_result(key, "extra_tune",
                                   json_content, index_table, tuner, repo_path)
            else:
                save_tuning_result(key, "json", json_content,
                                   index_table, tuner, repo_path)


def jobs(op_type: str = 'add', desc=None, debug_mode: bool = True, save_res: bool = False,
         all_space: bool = True, insert_key='', conf_of_set_dim="", tuning_attrs=[], skip_config_set=None, tuning_attrs_info=None):
    """AutoTuning jobs"""
    iter_times = [3, 3, 3] if debug_mode else [80, 160, 320]
    time_start_get_space = time.time()
    index_table, space, key, expect, input_for_mod = get_space(
        op_type, desc, tuning_attrs=tuning_attrs, tuning_attrs_info=tuning_attrs_info)
    time_end_get_space = time.time()
    print("get space time: ", time_end_get_space - time_start_get_space)
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
    runner = KernelRunner(op_type, desc, index_table,
                          self_attrs=None, input_data=input_for_mod,
                          expect=expect, mod_output_param=output_para,
                          timeout=30, repeat_times=1,
                          is_all_space=all_space,
                          skip_config_set=skip_config_set, 
                          need_tune_json=tuning_attrs_info[2])

    # we can only get a valid tiling, or accurate get cycles
    is_truly_profiling = utils.get_profiling_mode()

    # number of multi-processing for build kernels
    available_device_numbers = get_parallel_build_num()

    time_start_tuning = time.time()
    if all_space:
        tuner = Tuner(runner, index_table, space,
                      n_parallel=available_device_numbers)
        least_try_times = space.length
    else:
        tuner = ModelBasedTuner(runner, index_table, space,
                                n_parallel=available_device_numbers if is_truly_profiling else 1,
                                plan_size=100, pre_model=None)
        least_try_times = space.length
    tuner.tune(least_try_times, output_file=op_type + ".log")

    time_end_tuning = time.time()
    print("tuning time: ", time_end_tuning - time_start_tuning)
    print_tuning_result(op_type, space, index_table, tuner, key)
    # save_results_to_csv(op_type, space, index_table, tuner, key)

    # if save_res:
    #     save_tuning_result(key, op_type, desc, index_table, tuner)


def print_tuning_result(op_type, space, index_table, tuner, key):
    """print tuning result"""
    print(op_type + " shape is:", key)
    print('space size:', space.length)
    print('index table:', index_table)
    print('best config:', tuner.best_config)
    print('best time:',
          tuner.best_time if tuner.best_time not in error_time_string.keys() else error_time_string[tuner.best_time])
    print('original time:', tuner.original_time)
    print('optimal result is ', tuner.original_time /
          tuner.best_time, "faster then auto set dim.")
    print("total try times", len(tuner.xs))
    for x, y in zip(tuner.xs, tuner.ys):
        print(space.get(x), y if y not in error_time_string.keys()
              else error_time_string[y])


def save_results_to_csv(op_type, space, index_table, tuner, key):
    """save all results to csv"""
    data = []
    for x, y in zip(tuner.xs, tuner.ys):
        data.append([space.get(x), y if y not in error_time_string.keys()
                     else 9999999])
    df = pd.DataFrame(data, columns=["config", "time"])
    df.to_csv(op_type + "_" + key + ".csv")


def save_tuning_result(key, op_type, desc, index_table, tuner, repo_path="", extra_tune=False, platform="gpu"):
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
            param.extend(
                [(16, 16), (16, 16), (set_dim_configs.k_l1, set_dim_configs.k_l0)])
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
            param = [tile_cici, tile_khkh, tile_kwkw, tile_coco,
                     tile_bb, tile_hh, tile_ww, tile_mm, tile_kk, tile_nn]
            tiling_param = (param)
        elif ("batch_matmul" in op_type) and (platform == "gpu"):
            tiling = [str(getattr(set_dim_configs, name)) for name in getattr(
                set_dim_configs, "_fields") if name.startswith("tiling")]
            tiling_param = ""
            for i, tile_v in enumerate(tiling):
                if i % 2 == 0:
                    tiling_param += "0 " + str(i) + " "
                tiling_param += tile_v + " "

            block_param = get_block_str_from_config(set_dim_configs)
            thread_param = get_thread_str_from_config(set_dim_configs)
            config = {
                'attrs': {
                    'dim': tiling_param,
                    'bind_block': block_param,
                    'bind_thread': thread_param
                },
                'best_cycles': tuner.best_time,
                'original_cycles': tuner.original_time,
                'date': str(datetime.datetime.now()),
                'tuning_time': tuner.tuning_time,
            }
        elif op_type == "json":
            from autotuning.runner import get_attr_from_config
            tiling_param = get_attr_from_config(set_dim_configs, index_table)
        elif op_type == "reduce_sum_gpu":
            print(set_dim_configs)
            tiling = [str(getattr(set_dim_configs, name))
                      for name in getattr(set_dim_configs, '_fields') if name.startswith('tiling')]
            tiling_param = ""
            for i, tile_v in enumerate(tiling):
                tiling_param += "0 " + str(i) + " "
                tiling_param += tile_v + " 1 "

            block_param = get_block_str_from_config(set_dim_configs)
            thread_param = get_thread_str_from_config(set_dim_configs)
            config = {
                'attrs': {
                    'dim': tiling_param,
                    'bind_block': block_param,
                    'bind_thread': thread_param
                },
                'best_cycles': tuner.best_time,
                'original_cycles': tuner.original_time,
                'date': str(datetime.datetime.now()),
                'tuning_time': tuner.tuning_time,
            }
        else:
            print(set_dim_configs)
            tiling = [[getattr(set_dim_configs, name), 1]
                      for name in getattr(set_dim_configs, '_fields') if name.startswith('tiling')]
            tiling_param = []
            for i, tile_v in enumerate(tiling):
                tiling_param.append(index_table[i] + tile_v)
            config = []
    else:
        tiling_param = []

    # when there is a valid result, save the result
    if op_type in ("json", "extra_tune") and tuner.best_time not in error_time_list:
        config = {'attrs': tiling_param,
                  'best_cycles': tuner.best_time,
                  'original_cycles': tuner.original_time,
                  "date": str(datetime.datetime.now()),
                  "tuning time": tuner.tuning_time,
                  }
        if op_type == "json":
            config["file_name"] = str(key)
        compute, shape, dtype = generate_trait(desc)
        tuner.export_dim_configs(
            config, json_file.format(op_type), False, str(key))
        save_file = "autotuning/extra_tune.json" if extra_tune else repo_path
        with open(save_file, 'r') as f:
            repo = json.loads(f.read())
            if len(tiling_param) != 0 and (get_repo(repo, [compute, shape, dtype]) is None or
                                           int(tuner.best_time) < int(repo[compute][shape][dtype]["metadata"]["best_cycles"])):
                tuner.export_dim_configs_for_keys(config, save_file, False, [
                                                  compute, shape, dtype, "metadata"])
    else:
        try:
            tuner.export_dim_configs(
                config, json_file.format(op_type), False, str(key))
        except UnboundLocalError as e:
            logger.warning(e)
            print("[save_tuning_result]: ", "no result is saved.")


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


def read_shapes_from_file(debug_mode, save_res, all_space, conf_of_set_dim, op_type):
    """read tuning shapes from file"""
    file = importlib.import_module('autotuning.shapes.' + op_type)
    shapes = file.shapes
    for _, shp in enumerate(shapes):
        do_profiling(shp, debug_mode, save_res,
                     all_space, op_type, conf_of_set_dim)


def do_profiling(shp, debug_mode, save_res, all_space, op_type, conf_of_set_dim=None, tuning_attrs=None, skip_config_set=None, tuning_attrs_info=None):
    """do profiling"""
    # remove undeleted JOB files for previous shapes
    subprocess.run("rm -rf /var/log/npu/profiling/JOB*", shell=True)
    if op_type == 'matmul':
        key = shp[2][0:-1]
        logger.debug("start profiling: [%s]", str(key))
        desc = MatmulCubeDesc(*key)
        jobs(op_type, desc, debug_mode, save_res,
             all_space, key.__str__(), conf_of_set_dim)
        logger.debug("end profiling: [%s]", str(key))
    elif op_type.startswith('conv_backprop'):
        key = shp[2]
        logger.debug("start profiling: [%s]", str(key))
        desc = ConvBackpropDesc(*key)
        jobs(op_type, desc, debug_mode, save_res,
             all_space, key.__str__(), conf_of_set_dim)
        logger.debug("end profiling: [%s]", str(key))
    elif op_type.startswith('conv') and "gpu" not in op_type:
        key = shp[2]
        logger.debug("start profiling: [%s]", str(key))
        desc = ConvDesc(*key)
        jobs(op_type, desc, debug_mode, save_res,
             all_space, key.__str__(), conf_of_set_dim)
        logger.debug("end profiling: [%s]", str(key))
    elif op_type in ["batch_matmul_gpu", "conv_image2col_gemm_gpu", "reduce_sum_gpu"]:
        logger.debug("start profiling: [%s]", str(shp))
        jobs(op_type, shp, debug_mode, save_res,
             all_space, conf_of_set_dim=conf_of_set_dim, tuning_attrs=tuning_attrs, skip_config_set=skip_config_set, tuning_attrs_info=tuning_attrs_info)
    else:
        key = shp
        logger.debug("start profiling: [%s]", str(key))
        desc = key
        jobs(op_type, desc, debug_mode, save_res,
             all_space, conf_of_set_dim=conf_of_set_dim, skip_config_set=skip_config_set)
        logger.debug("end profiling: [%s]", str(key))


def launch(op_type, debug_mode, save_res=False, desc=None, all_space=False,
           from_json=False, tuning_attrs=None, skip_config_set=None, tuning_attrs_info=None):
    # get the existed tiling
    conf_of_set_dim = load_json_configs(op_type) if from_json else None

    if desc is None:
        read_shapes_from_file(debug_mode, save_res,
                              all_space, conf_of_set_dim, op_type)
    else:
        shp = desc
        do_profiling(shp, debug_mode, save_res, all_space, op_type,
                     tuning_attrs=tuning_attrs, skip_config_set=skip_config_set, tuning_attrs_info=tuning_attrs_info)
