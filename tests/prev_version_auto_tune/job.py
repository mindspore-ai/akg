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
import subprocess
from collections import namedtuple
from multiprocessing import Process, Manager
from akg import composite
from akg.utils import kernel_exec as utils
from akg.composite.build_module import generate_trait
from tests.prev_version_auto_tune.runner import KernelRunner, error_time_list, error_time_string
from tests.prev_version_auto_tune.tuner import ModelBasedTuner, Tuner
from tests.prev_version_auto_tune.type_definitions import ConvDesc, ConvBackpropDesc, MatmulCubeDesc
from tests.prev_version_auto_tune.space_generators import get_space
from tests.prev_version_auto_tune.space import ListConfigSpace
from tests.prev_version_auto_tune.data_generators import gen_data
from tests.prev_version_auto_tune.kernel_compiler import get_matmul_cube_attrs


logger = logging.getLogger('tests.prev_version_auto_tune.job')

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


def gen_bool_list(attr_list):
    bool_list = []
    for _ in attr_list:
        if len(bool_list) == 0:
            bool_list = [[True], [False]]
        else:
            tmp_list = []
            for attr_option in bool_list:
                tmp = attr_option[:]
                tmp.append(True)
                tmp1 = tmp[:]
                tmp.pop()
                tmp.append(False)
                tmp2 = tmp[:]
                tmp_list.append(tmp1)
                tmp_list.append(tmp2)
            bool_list = tmp_list
    return bool_list


def get_matmul_op_desc(json_input):
    for op_desc in json_input["op_desc"]:
        if op_desc["name"] == "MatMul" or op_desc["name"] == "BatchMatMul":
            return op_desc
    return None

def convert_fracal_shape(ori_shape, fractal):
    ori_shape = tuple(ori_shape)
    if fractal == "zN":
        return ori_shape[:-4] + (ori_shape[-2] * ori_shape[-3], ori_shape[-1] * ori_shape[-4])
    if fractal == "zZ":
        return ori_shape[:-4] + (ori_shape[-4] * ori_shape[-2], ori_shape[-3] * ori_shape[-1])

def get_matmul_frac(s):
    if s[-2:] == "NZ": 
        return "zN"
    if s[-2:] == "ZZ":
        return "zZ"

def get_attr(attrs, name):
    for attr in attrs:
        if attr["name"] == name:
            return attr["value"]
    return None    

def get_matmul_desc(op_desc):
    
    input_desc = op_desc["input_desc"]
    output_desc = op_desc["output_desc"]
    op_attrs = op_desc["attr"]
    if len(input_desc) == 3 :
        bias = 1
        dtype_bias = input_desc[2][0]["data_type"]
    else:
        bias = 0
        dtype_bias = None 

    dtype_a = input_desc[0][0]["data_type"]
    dtype_c = output_desc[0]["data_type"]
    fractal_a = get_matmul_frac(input_desc[0][0]["format"])
    fractal_b = get_matmul_frac(input_desc[1][0]["format"])
    fractal_c = get_matmul_frac(output_desc[0]["format"])
    trans_a = get_attr(op_attrs, "transpose_a") 
    trans_b = get_attr(op_attrs, "transpose_b")
    shape_a = convert_fracal_shape(input_desc[0][0]["shape"], fractal_a)
    shape_b = convert_fracal_shape(input_desc[1][0]["shape"], fractal_b)
    res = [tuple(shape_a), tuple(shape_b), bias, fractal_a, fractal_b, fractal_c, trans_a, trans_b, dtype_a, dtype_bias, dtype_c]    
    print(res)
    return MatmulCubeDesc(*res)

def tune_json_file(input_path, input_file, iter_times, save_res, repo_path, all_space, skip_exist, extra_tune,
                   self_attrs, tuning_attrs, tuned_file):
    """tune for single json file"""
    with open(repo_path, 'r') as f:
        repo = json.loads(f.read())
    whole_path = input_path + '/' + input_file
    if os.path.exists(whole_path):
        with open(whole_path, 'r') as f:
            json_input = f.read()
    else:
        json_input = input_file

    try:
        json_content = json.loads(json_input)
    except BaseException as e:
        logger.warning("load json [%s] failed: %s", input_file, str(e))
        with open('res/wrong_json.txt', 'a') as fe:
            fe.write(input_file)
            fe.write("\n")
        return
    if "input_desc" not in json_content:
        with open('res/wrong_json.txt', 'a') as fe:
            logger.warning("wrong json format [%s]", input_file)
            fe.write(input_file)
            fe.write("\n")
        return

    # specialize tunning attrs for RealDiv
    for op_desc in json_content["op_desc"]:
        if op_desc["name"] == "RealDiv" and 'enable_rewrite_scalar_compute' not in tuning_attrs:
            tuning_attrs.append('enable_rewrite_scalar_compute')
            break

    for input_desc in json_content["input_desc"]:
        if input_desc[0]["shape"] == []:
            input_desc[0]["shape"] = [1]
    json_input = json.dumps(json_content)

    # skip tuning for info in repo
    if skip_exist:
        compute, shape, dtype = generate_trait(json_content)
        bst = get_repo(repo, [compute, shape, dtype])
        if bst:
            print("Info for %s already exists" % input_file)
            print("ops are ", str(compute))
            print("shape is ", str(shape))
            print("dtype is ", str(dtype))
            with open('res/skip_file.txt', 'a') as fe:
                fe.write(input_file)
                fe.write("\n")
            return bst
    
    matmul_op_desc = get_matmul_op_desc(json_content)
    has_matmul = (matmul_op_desc != None)
    # generate tuning space

    if has_matmul:
        matmul_desc = get_matmul_desc(matmul_op_desc)
        index_table, space, _, _, _ = get_space("matmul", matmul_desc, tuning_attrs)
    else:
        if not extra_tune:
            time_start_get_space = time.time()
            with Manager() as manager:
                space_dict = manager.dict()
                p = Process(target=get_json_space, args=(json_input, space_dict))
                p.daemon = True
                p.start()
                p.join(600)
                if 'res' not in space_dict:
                    with open('res/error_space_list.txt', 'a') as fe:
                        fe.write(input_file)
                        fe.write("\n")
                    return
                space_res = space_dict['res']
            time_end_get_space = time.time()
            logger.debug("get space time: %f", time_end_get_space - time_start_get_space)
            index_table = space_res['index']
            tiling_spaces = space_res['tuning_space']
            if not isinstance(tiling_spaces, list) or len(tiling_spaces) == 0:
                with open('res/empty_space_list.txt', 'a') as fe:
                    fe.write(input_file)
                    fe.write("\n")
                return
            dim_names = ['tiling_' + str(i) for i in range(len(tiling_spaces[0]))]
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
        input_for_mod, expect, output_indexes = gen_data(op_type="json", op_desc=json_input)

    except BaseException as e:
        logger.warning("gen numpy data from [%s] failed: %s", input_file, str(e))
        with open('res/error_gen_data_list.txt', 'a') as fe:
            fe.write(input_file)
            fe.write(": ")
            fe.write(str(e))
            fe.write("\n")
        return
    print('space size:', space.length)
    print('index table:', index_table)
    runner_op_type = "matmul_json" if has_matmul else "json"   
    op_desc = matmul_desc if has_matmul else json_input
    runner = KernelRunner(op_type=runner_op_type, op_desc=op_desc, json_desc=json_input, index_table=index_table, input_data=input_for_mod,
                          expect=expect, mod_output_param=output_indexes, timeout=180, repeat_times=1)

    # we can only get a valid tiling, or accurate get cycles
    is_truly_profiling = utils.get_profiling_mode() or os.environ['RUNTIME_MODE'] == "gpu"

    # available device numbers, normally is 8 or 1
    available_device_numbers = utils.get_available_devices_num()

    if all_space:
        tuner = Tuner(runner, index_table, space, n_parallel=available_device_numbers)
        least_try_times = space.length
    else:
        tuner = ModelBasedTuner(runner, index_table, space,
                                n_parallel=available_device_numbers if is_truly_profiling else 1,
                                plan_size=64, pre_model=None)
        least_try_times = iter_times[0 if space.length < 10 ** 4 else 1 if space.length < 10 ** 5 else 2]
    tuner.tune(least_try_times, output_file="json.log")
    tuner.index_table = index_table
    if tuned_file:
        with open(tuned_file, 'a') as fe:
            fe.write(input_file)
            fe.write("\n")

    print_tuning_result("json", space, index_table, tuner, key)

    if save_res:
        if extra_tune:
            save_tuning_result(key, "extra_tune", None, json_content, index_table, tuner, repo_path)
        else:
            if has_matmul:
                save_tuning_result(key, runner_op_type, op_desc, json_content, index_table, tuner, repo_path)
            else:
                save_tuning_result(key, "json", None, json_content, index_table, tuner, repo_path)
    
    return tuner

def launch_json(debug_mode: bool = True, save_res: bool = False, input_str="", repo_path="", all_space=False,
                skip_exist=True, skip_file=True, extra_tune=False, self_attrs=[], tuning_attrs=[], iter_times=None):
    """launch tuning for composite json files"""
    best_tuners = list()
    subprocess.run("mkdir -p res/", shell=True)
    if not os.path.exists(repo_path):
        with open(repo_path, 'w') as f:
            f.write(json.dumps({}))
    if not iter_times:
        iter_times = [3, 3, 3] if debug_mode else [80, 160, 320]
    if os.path.isdir(input_str):
        files = os.listdir(input_str)
        tuned_file = "res/tuned_file.txt"
        tuned_list = []
        if os.path.exists(tuned_file):
            with open(tuned_file, 'r') as f:
                for line in f:
                    tuned_list.append(line.rstrip())
        for idx, input_file in enumerate(files):
            print("[%d/%d]Start tuning for %s" % (idx + 1, len(files), input_file))
            if skip_file and input_file in tuned_list:
                continue
            bst = tune_json_file(input_str, input_file, iter_times, save_res, repo_path, all_space, skip_exist, extra_tune,
                                 self_attrs, tuning_attrs, tuned_file)
            best_tuners.append(bst)
    else:
        bst = tune_json_file(".", input_str, iter_times, save_res, repo_path, all_space, skip_exist, extra_tune, self_attrs,
                             tuning_attrs, None)
        best_tuners.append(bst)
    return best_tuners

def jobs(op_type: str = 'add', desc=None, debug_mode: bool = True, save_res: bool = False,
         all_space: bool = True, insert_key='', conf_of_set_dim=""):
    """AutoTuning jobs"""
    iter_times = [3, 3, 3] if debug_mode else [80, 160, 320]
    time_start_get_space = time.time()
    index_table, space, key, expect, input_for_mod = get_space(op_type, desc)
    time_end_get_space = time.time()
    logger.debug("get space time: %f", time_end_get_space - time_start_get_space)
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
    runner = KernelRunner(op_type, desc, None, index_table, input_data=input_for_mod,
                          expect=expect, mod_output_param=output_para, timeout=180, repeat_times=1)

    # we can only get a valid tiling, or accurate get cycles
    is_truly_profiling = utils.get_profiling_mode()

    # available device numbers, normally is 8 or 1
    available_device_numbers = utils.get_available_devices_num()

    time_start_tuning = time.time()
    if all_space:
        tuner = Tuner(runner, index_table, space, n_parallel=available_device_numbers)
        least_try_times = space.length
    else:
        tuner = ModelBasedTuner(runner, index_table, space,
                                n_parallel=available_device_numbers if is_truly_profiling else 1,
                                plan_size=64, pre_model=None)
        least_try_times = iter_times[0 if space.length < 10 ** 4 else 1 if space.length < 10 ** 5 else 2]
    tuner.tune(least_try_times, output_file=op_type + ".log")

    time_end_tuning = time.time()
    print("tuning time: ", time_end_tuning - time_start_tuning)
    print_tuning_result(op_type, space, index_table, tuner, key)

    if save_res:
        save_tuning_result(key, op_type, desc, None, index_table, tuner)


def print_tuning_result(op_type, space, index_table, tuner, key):
    """print tuning result"""
    print(op_type + " input is:", key)
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


def save_tuning_result(key, op_type, op_desc, json_desc, index_table, tuner, repo_path=""):
    """save tuning result"""
    if tuner.best_config is not None and tuner.best_time not in error_time_list:
        set_dim_configs = tuner.best_config.input
        if op_type == "matmul":
            param = []
            for _ in range(len(op_desc.x_shape) - 2):
                param.append((1, 1))
            if set_dim_configs.n_l1 > 0:
                param.append((set_dim_configs.n_l1, set_dim_configs.n_l0))
            if set_dim_configs.m_l1 > 0:
                param.append((set_dim_configs.m_l1, set_dim_configs.m_l0))
            param.extend([(16, 16), (16, 16), (set_dim_configs.k_l1, set_dim_configs.k_l0)])
            tiling_param = (param, {"bypass": set_dim_configs.bypass})
        elif op_type == "matmul_json":
            tiling_param = get_matmul_cube_attrs(op_desc, set_dim_configs)
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
        elif op_type == "json":
            from tests.prev_version_auto_tune.runner import get_attr_from_config
            tiling_param = get_attr_from_config(set_dim_configs, index_table)
        else:
            tiling = [[getattr(set_dim_configs, name), 1]
                      for name in getattr(set_dim_configs, '_fields') if name.startswith('tiling')]
            tiling_param = []
            for i, tile_v in enumerate(tiling):
                tiling_param.append(index_table[i] + tile_v)
    else:
        tiling_param = []

    config = {'attrs': tiling_param,
              'best_cycles': tuner.best_time,
              'original_cycles': tuner.original_time,
              "date": str(datetime.datetime.now()),
              "tuning time": tuner.tuning_time,
              }
    # save results to repo file when tuning time for composite is valid
    if op_type in ("json", "extra_tune", "matmul_json") and tuner.best_time not in error_time_list:
        config["file_name"] = str(key)
        compute, shape, dtype = generate_trait(json_desc)
        save_file = "autotuning/extra_tune.json" if op_type == "extra_tune" else repo_path
        with open(save_file, 'r') as f:
            repo = json.loads(f.read())
            if len(tiling_param) != 0 and (get_repo(repo, [compute, shape, dtype]) is None or
                                           int(tuner.best_time) < int(
                        repo[compute][shape][dtype]["metadata"]["best_cycles"])):
                tuner.export_dim_configs_for_keys(config, save_file, False, [compute, shape, dtype, "metadata"])
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


def read_shapes_from_file(debug_mode, save_res, all_space, conf_of_set_dim, op_type):
    """read tuning shapes from file"""
    file = importlib.import_module('autotuning.shapes.' + op_type)
    shapes = file.shapes
    for _, shp in enumerate(shapes):
        do_profiling(shp, debug_mode, save_res, all_space, op_type, conf_of_set_dim)


def do_profiling(shp, debug_mode, save_res, all_space, op_type, conf_of_set_dim=None):
    """do profiling"""
    # remove undeleted JOB files for previous shapes
    if op_type == 'matmul':
        key = shp[2][0:-1]
        logger.debug("start profiling: [%s]", str(key))
        desc = MatmulCubeDesc(*key)
        jobs(op_type, desc, debug_mode, save_res, all_space, key.__str__(), conf_of_set_dim)
        logger.debug("end profiling: [%s]", str(key))
    elif op_type.startswith('conv_backprop'):
        key = shp[2]
        logger.debug("start profiling: [%s]", str(key))
        desc = ConvBackpropDesc(*key)
        jobs(op_type, desc, debug_mode, save_res, all_space, key.__str__(), conf_of_set_dim)
        logger.debug("end profiling: [%s]", str(key))
    elif op_type.startswith('conv'):
        key = shp[2]
        logger.debug("start profiling: [%s]", str(key))
        desc = ConvDesc(*key)
        jobs(op_type, desc, debug_mode, save_res, all_space, key.__str__(), conf_of_set_dim)
        logger.debug("end profiling: [%s]", str(key))
    else:
        key = shp
        logger.debug("start profiling: [%s]", str(key))
        desc = key
        jobs(op_type, desc, debug_mode, save_res, all_space, conf_of_set_dim=conf_of_set_dim)
        logger.debug("end profiling: [%s]", str(key))


def launch(op_type, debug_mode, save_res=False, desc=None, all_space=False):
    # get the existed tiling
    conf_of_set_dim = load_json_configs(op_type)

    if desc is None:
        read_shapes_from_file(debug_mode, save_res, all_space, conf_of_set_dim, op_type)
    else:
        shp = desc
        do_profiling(shp, debug_mode, save_res, all_space, op_type)
