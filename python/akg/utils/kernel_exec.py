#!/usr/bin/env python3
# coding: utf-8
# Copyright 2019-2021 Huawei Technologies Co., Ltd
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

"""util"""
import sys
import gc
import inspect
import datetime
import os
import uuid
import logging
import time
import random
import subprocess
import re
import tvm
from timeit import default_timer as timer
from threading import Thread
from functools import reduce
import numpy as np
from enum import IntEnum
import ctypes

import akg
import akg.tvm
from akg.tvm import autotvm
from akg.tvm import rpc
from akg.tvm import _api_internal
from akg.build_module import help_tiling_level
from akg.utils import result_analysis as ra_util
from akg.utils import format_transform as ft_util
from akg.utils import custom_tiling as ct_util
from akg.utils import validation_check as vc_util
from akg.utils.dsl_create import TensorUtils
from akg.backend.parsing_profiling_data import HWTSLogParser
from akg.backend.parsing_profiling_data import validate_and_normalize_path
from akg.backend import aic_model

sh = logging.StreamHandler(sys.stdout)
logging.getLogger().addHandler(sh)
logging.getLogger().setLevel(logging.INFO)

rpc_machine = {}
rpc_lb = {}

PERFORMANCE_TEST_FILE = "PERFORMANCE_TEST_FILE"
BINDS = "binds"
CUDA = "cuda"
CCE = "cce"
RANDOM_SEED_NUM = 20
PROF_ERROR_CODE = 9999999999

WGT_WIDTH = 16
INP_WIDTH = 16
OUT_WIDTH = 16
BLOCK_IN = 16
BLOCK_OUT = 16
BLOCK_REDUCE = 16
INP_ELEM_BYTES = (BLOCK_IN * BLOCK_REDUCE * INP_WIDTH // 8)
WGT_ELEM_BYTES = (BLOCK_OUT * BLOCK_REDUCE * WGT_WIDTH // 8)
OUT_ELEM_BYTES = (BLOCK_IN * BLOCK_OUT * OUT_WIDTH // 8)
GLB_ELEM_BYTES = (16 * OUT_WIDTH // 8)


class ReturnType(IntEnum):
    """Return Type IntEnum"""
    DEFAULT = 0
    FEAT = 1
    MOD = 2
    MOD_AND_FEAT = 3


def debug_mode(debug_flag):
    """
    Pass to enable tpu debug mode.

    Args:
        debug_flag (int): The dbeug flag to be passed.

    Returns:
        list of function, the pass to set to build_config(add_lower_pass=tpu.debug_mode(mode)).
    """
    # the number in pass_list such as 0,1,2,3 represents the order of the pass called
    pass_list = []
    if debug_flag == 1:
        from akg.tvm import ir_pass
        pass_list.append((0, ir_pass.inject_dma_intrin))
    return pass_list


def func_time_required(func_name):
    """Checking the Time Required for Function Running."""

    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = func_name(*args, **kwargs)
        t1 = time.time()
        logging.info("func_time_required func:%s, running:%lf seconds", func_name.__name__, (t1 - t0))
        return result

    return wrapper


def create_code(kernel_name, code_path=None, code=None, code_type=CCE):
    """
    Create cce or cuda file.

    Args:
        kernel_name: file name.
        code_path: file path.
        code: code.
        code_type: code type.
    """
    if code_type == CCE:
        postfix = ".cce"
    elif code_type == CUDA:
        postfix = ".cu"
    else:
        logging.info("the target code type %s is not supported.", code_type)

    if not code_path:
        code_path = "./"

    if code_type == CCE and len(code_path) > 4 and code_path[-4:].lower() == postfix:
        real_path = code_path
    elif code_type == CUDA and len(code_path) > 3 and code_path[-3:].lower() == postfix:
        real_path = code_path
    else:
        if code_path[-1] == r"/":
            real_path = code_path + kernel_name + postfix
        else:
            real_path = code_path + r"/" + kernel_name + postfix
    dir_path = r"/".join(real_path.split(r"/")[:-1])
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    with open(real_path, 'wt') as ss:
        ss.write(code)


def gen_name_kernel(kernel, dtype, shapes):
    """generate kernel name."""

    def _flat_array(srclist, dstlist):
        for i in srclist:
            if isinstance(i, (list, tuple)):
                _flat_array(i, dstlist)
            else:
                dstlist.append(i)

    res = ''
    flat = []
    _flat_array(shapes, flat)
    for s in flat:
        res = "%s%s'_'" % (res, s)
    res = "%s_%s%s" % (kernel, res, dtype)
    return res


def load_rpc_server_info(mode):
    """
    load rpc server host and port info.

    Args:
        mode (str): string of runtime choose, can set ca aic and rpc.
    """
    env_dic = os.environ
    if env_dic.get('RPC_HOST') and env_dic.get('RPC_PORT'):
        return None

    if mode == 'rpc_cloud':
        logging.error("runtime_mode=rpc_cloud must set 1980 host ip and port!")
        raise Exception("ERROR:runtime_mode=rpc_cloud must set 1980 host ip and port!")

    rpc_server_info_config = env_dic.get('RPC_SERVER_INFO_FILE')
    if not rpc_server_info_config:
        logging.error("runtime_mode=rpc must set RPC_SERVER_INFO_FILE for rpc server info config")
        raise Exception("ERROR:runtime_mode=rpc must set RPC_SERVER_INFO_FILE for rpc server info config")

    # load rpc server host and port info from local file.
    import json
    with open(rpc_server_info_config, 'r') as f:
        info = json.load(f)

    for i in info:
        rpc_machine[i] = info[i]
        rpc_lb[i] = 0.0
    return None


def dispatch(rank=0):
    """Function for lock waiting dispatch handle version 1."""

    def _sort_by_value(d):
        items = list(d.items())
        random.shuffle(items)
        items.sort(key=lambda x: x[1])
        return [item[0] for item in items]

    for k, v in rpc_lb.items():
        logging.info("######rpc_lb[%s]=%f", rpc_machine[k][0], v)
    lb_list = _sort_by_value(rpc_lb)
    if len(lb_list) > rank:
        return lb_list[rank]
    return lb_list[len(lb_list) - 1]


def commit(remote, weight):
    rpc_lb[remote] = weight


@func_time_required
def mod_launch_rpc_worker(mod, args, outputs, host, port, tuning=False):
    """internal RPC worker, should be called by mod_launch_rpc_thread."""
    logging.info("%s:====start connect to rpc ip: %s, rpc port: %d ",
                 datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), host, port)
    remote = rpc.connect(host, port, session_timeout=300)
    logging.info("%s:====connect to rpc ip: %s, rpc port: %d finished ",
                 datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), host, port)
    uuid_str = uuid.uuid4().hex
    temp_file_name = "stackvm_%s.o" % uuid_str
    mod.save(temp_file_name)
    remote.upload(temp_file_name)
    remote_mod = remote.load_module(temp_file_name)
    ctx = remote.cce()
    arg_list = []
    for a in args:
        arg_list.append(akg.tvm.nd.array(a, ctx))
    start_time = timer()
    remote_mod(*arg_list)
    ctx.sync()
    if os.path.exists(temp_file_name):
        os.remove(temp_file_name)
    out_list = []
    for i in outputs:
        out = arg_list[len(arg_list) + i if i < 0 else i].asnumpy()
        out_list.append(out)
    # this time measure is no accurate now, to be improved soon
    t = timer() - start_time
    if not tuning:
        return out_list[0] if len(out_list) == 1 else tuple(out_list)
    stat_info = {"run_time": t}
    return out_list[0] if len(out_list) == 1 else tuple(out_list), stat_info


def mod_launch_rpc_thread(mode, mod, args, outputs, results, need_retry, retry, tuning=False):
    """internal RPC thread, should be called by mod_launch_rpc_multithread."""
    remoteevb = '0'
    host = None
    port = None
    env_dic = os.environ
    if env_dic.get('RPC_HOST') and env_dic.get('RPC_PORT'):
        host = env_dic.get('RPC_HOST')
        port = int(env_dic.get('RPC_PORT'))
    else:
        if mode == 'rpc_cloud':
            logging.error("runtime_mode=rpc_cloud must set 1980 host ip and port!")
            raise Exception("ERROR:runtime_mode=rpc_cloud must set 1980 host ip and port!")
        remoteevb = dispatch(retry)
        host = rpc_machine[remoteevb][0]
        port = rpc_machine[remoteevb][1]

    start_time = timer()
    end_time = 0.0
    logging.debug("rpc ip: %s, rpc port: %d", host, port)
    try:
        out_list = mod_launch_rpc_worker(mod, args, outputs, host, port, tuning=tuning)
        end_time = timer()
        t = end_time - start_time
        if not env_dic.get('RPC_HOST'):
            commit(remoteevb, 20 if t > 20 else t)
        logging.info("===this round host is %s time is %f", host, (end_time - start_time))
        results[retry] = out_list
    except RuntimeError:
        need_retry[retry] = True
        end_time = timer()
        logging.error("===Failed! this round host is %s time is %f", host, (end_time - start_time))
        if not env_dic.get('RPC_HOST'):
            commit(remoteevb, end_time - start_time + 20 * (retry + 1))
        logging.error("rpc retry error: %d %s", retry, sys.exc_info())


def mod_launch_rpc(mode, mod, args, outputs, tuning=False):
    """
    launch rpc or rpc_cloud module with retry.

    Note:
        To minimize waiting time of struggler RPC servers, we wait for a short timeout and spawn
        a new thread after the timeout.
        In normal case, RPC would complete before the short timeout, so, only one thread will be created.
        When the RPC server is slow, we create multiple threads that run concurrently.
        We wait for the first thread that successfully completes its work and return the result.
        If a thread fails (an exception is raised), we spawn a new thread to retry.
        Newly spawned threads will use different RPC servers.
        We bound the maximum number of threads, i.e. maximum number of retries.
    """
    max_num_threads = 5

    import operator
    arg_filter = filter(lambda x: isinstance(x, np.ndarray), args)
    arg_tensor = list(arg_filter)
    tensor_size = reduce(operator.add, [reduce(operator.mul, arg.shape) for arg in arg_tensor])
    expected_upload_speed = 5e6
    expected_upload_time = int(tensor_size / expected_upload_speed)

    timeout_before_spawning_new_thread = 200 + expected_upload_time
    poll_interval = 1
    thread_timeout = 400 + expected_upload_time * 3

    load_rpc_server_info(mode)

    threads = [None] * max_num_threads
    results = [None] * max_num_threads
    need_retry = [None] * max_num_threads
    retried = [False] * max_num_threads
    for thread_index in range(max_num_threads):
        if thread_index > 0:
            logging.error("Thread %d run for %d seconds, spawn a new thread to retry",
                          (thread_index - 1), timeout_before_spawning_new_thread)
        threads[thread_index] = Thread(target=mod_launch_rpc_thread,
                                       args=(mode, mod, args, outputs, results, need_retry, thread_index, tuning))
        # daemonize the thread to prevent long running threads from hanging the whole process
        threads[thread_index].daemon = True
        threads[thread_index].start()
        poll_count = timeout_before_spawning_new_thread // poll_interval
        while poll_count > 0:
            poll_count -= 1
            # wait for the newly created thread, because it is most likely to complete first
            threads[thread_index].join(poll_interval)
            for poll_index in range(thread_index + 1):
                if not threads[poll_index].is_alive() and not need_retry[poll_index]:
                    return results[poll_index]
                if need_retry[poll_index] and not retried[poll_index]:
                    logging.error("Thread %d exit with error, spawn a new thread immediately", poll_index)
                    poll_count = 0
                    retried[poll_index] = True

    logging.error("All %d threads are created, poll the threads until the first one exits normally, \
                  or all threads exit abnormally or timeout", max_num_threads)
    poll_count = thread_timeout // poll_interval
    for _ in range(poll_count):
        threads[max_num_threads - 1].join(poll_interval)
        exit_thread_count = 0
        for poll_index in range(max_num_threads):
            if not threads[poll_index].is_alive() and not need_retry[poll_index]:
                return results[poll_index]
            if not threads[poll_index].is_alive():
                exit_thread_count += 1
            if exit_thread_count == max_num_threads:
                logging.error("All %d threads exit abnormally", max_num_threads)
                return None

    logging.error("All %d threads timeout", max_num_threads)
    return None


def profiling_mode_run(kernel_name, args, outputs, tuning, device_id):
    """
    Function for collecting cycle data from device.

    Args:
        kernel_name: name of kernel.
        args: list or tuple of numpy array.
        outputs: list or tuple of output argment index.
        tuning: tuning model.
        device_id: device_id on device.
    """
    tvm.get_global_func("ascend_start_profiling")(device_id)
    time_before_launch = time.time()
    output_data = ascend_run(kernel_name, args, outputs, device_id)
    tvm.get_global_func("ascend_stop_profiling")()

    cycle = profiling_analyse(device_id, time_before_launch)
    logging.info('=====parsing cycles==============================')
    if cycle != PROF_ERROR_CODE:
        logging.info(cycle)
    else:
        logging.error("OOPS, can't correctly parsing cycles!")
    TestUtils.record_cycle(cycle)
    logging.info('=====parsing cycles==============================')
    if tuning:
        return output_data, {'run_time': cycle}
    return output_data


def profiling_analyse(device_id, time_before_launch):
    """analyse profiling."""

    def exec_cmds_with_pipe(cmd_list):
        cmd_num = len(cmd_list)
        if cmd_num <= 1:
            raise RuntimeError("length of cmd_list should be greater than 1.")
        ps = []
        for i, cmd in enumerate(cmd_list):
            if i == 0:
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            else:
                p = subprocess.Popen(cmd, stdin=ps[-1].stdout, stdout=subprocess.PIPE)
            ps.append(p)
        for p in ps:
            p.wait()
        return ps[-1].communicate()

    if not isinstance(device_id, int):
        raise TypeError("device_id must be an integer.")

    try:
        public_path = os.getenv('PROFILING_DIR')
        if public_path is None:
            raise RuntimeError("Environment PROFILING_DIR not set!")

        public_path = validate_and_normalize_path(public_path)
        cmd_list = [
            ["find", public_path, "-iname", "*.log.%d" % device_id, "-printf", "'%T+\t%p\n'"],
            ["grep", "JOB"],
            ["sort", "-r"],
            ["head", "-n10"],
            ["awk", "{print $2}"],
            ["head", "-n1"],
        ]
        for _ in range(200):
            p = exec_cmds_with_pipe(cmd_list)
            if p[0].decode('utf8').strip() == '':
                time.sleep(1)
            else:
                break
        try:
            job_file = p[0].decode('utf8').strip().split('/')[-2]
        except BaseException:
            logging.warning("failed to decode profiling result")
            return None
        logging.debug("job file is: %s", job_file)

        file_abs_path = public_path + "/" + job_file
        file_create_time = os.path.getctime(file_abs_path)

        if file_create_time < time_before_launch:
            raise RuntimeError("The JOB file is too old")

        count = 0
        while count < 5:
            try:
                hwtslog_parser = HWTSLogParser(file_abs_path)
                return hwtslog_parser.execute()
            except BaseException:
                time.sleep(1)
                count += 1
    except SyntaxError as e:
        logging.error(e)
        return PROF_ERROR_CODE


def array_as_continue(arr):
    assert isinstance(arr, np.ndarray)
    arr = np.ascontiguousarray(arr, dtype=arr.dtype)
    assert arr.flags['C_CONTIGUOUS']
    return arr


def get_launch_args(arg_list, outputs):
    launch_args = []
    outputs = set(outputs)
    for i in range(len(arg_list)):
        arg = arg_list[i]
        if isinstance(arg, np.ndarray):
            data = arg.ctypes.data_as(ctypes.c_void_p)
            nbytes = arg.size * arg.dtype.itemsize
            is_output = 1 if i in outputs else 0
            launch_args.append(data)
            launch_args.append(nbytes)
            launch_args.append(is_output)
        else:
            launch_args.append(arg)
    return launch_args


def mod_launch_air(mod, args, outputs, device_id):
    """launch mod on kc_air."""

    ctx = akg.tvm.ndarray.cce(device_id)
    arg_list = []
    for a in args:
        if isinstance(a, np.ndarray):
            arg_list.append(akg.tvm.nd.array(a, ctx))
        elif isinstance(a, (list, tuple)):
            for aa in a:
                if isinstance(aa, np.ndarray):
                    arg_list.append(akg.tvm.nd.array(aa, ctx))
                else:
                    arg_list.append(aa)
        else:
            arg_list.append(a)
    for retry in range(3):
        need_retry = False
        try:
            mod(*arg_list)
            ctx.sync()
            out_list = []
            if not need_retry:
                for i in outputs:
                    out = arg_list[len(arg_list) + i if i < 0 else i].asnumpy()
                    out_list.append(out)
                return out_list[0] if len(out_list) == 1 else tuple(out_list)
        except RuntimeError:
            need_retry = True
            logging.error("kc_air retry error: %d %s", retry, sys.exc_info())
    logging.error("kc_air runtime error, please check!")
    return None


def ascend_run(kernel_name, args, outputs, device_id):
    """launch mod on ascend."""
    # Currently akg runs through this function in CCE RT mode
    arg_list = []
    for a in args:
        if isinstance(a, np.ndarray):
            arg_list.append(array_as_continue(a))
        elif isinstance(a, (list, tuple)):
            for aa in a:
                if isinstance(aa, np.ndarray):
                    arg_list.append(array_as_continue(a))
                else:
                    arg_list.append(aa)
        else:
            arg_list.append(a)
    outputs = [len(arg_list) + i if i < 0 else i for i in outputs]
    launch_args_list = get_launch_args(arg_list, outputs)
    tvm.get_global_func("ascend_run")(kernel_name, device_id, *launch_args_list)
    out_list = []
    for i in outputs:
        out = arg_list[i]
        out_list.append(out)
    return out_list[0] if len(out_list) == 1 else tuple(out_list)


def get_kernel_name_from_mod(mod):
    module = mod.imported_modules[0]
    code = module.get_source()
    kernel_name_end_pos = code.find("_kernel")
    kernel_name_start_pos = code[:kernel_name_end_pos].rfind(" ") + 1
    kernel_name = code[kernel_name_start_pos:kernel_name_end_pos]
    if not kernel_name:
        raise ValueError("fail to get kernel_name")
    return kernel_name

def mod_launch_ascend_profiling(mod, args, outputs=(-1,), tuning=False, device_id=-1):
    gc.collect()
    if device_id == -1:
        device_id = int(os.environ.get("DEVICE_ID", 0))
    kernel_name = get_kernel_name_from_mod(mod)
    return profiling_mode_run(kernel_name, args, outputs, tuning, device_id)

def mod_launch_gpu(mod, args, outputs=(-1,), tuning=False, device_id=-1, repeat_time=400):
    if device_id == -1:
        device_id = int(os.environ.get("DEVICE_ID", 0))
    ctx = akg.tvm.context(CUDA, device_id)
    mod_args = [akg.tvm.nd.array(a, ctx) for a in args]
    mod(*mod_args)
    out_list = [mod_args[len(args) + i if i < 0 else i].asnumpy() for i in outputs]
    if not tuning:
        return out_list[0] if len(out_list) == 1 else tuple(out_list)
    else:
        cycles = get_gpu_cycles(mod, *mod_args, device_id=device_id, repeat_time=repeat_time)
        return out_list[0] if len(out_list) == 1 else tuple(out_list), {'run_time': cycles}

@func_time_required
def mod_launch(mod, args, outputs=(-1,), tuning=False, device_id=-1, expect=None, repeat_time=400):
    """
    unified run CCE kernel api.

    Args:
        mod (Module): module for runtime
        args (Union[list, tuple]): list or tuple of numpy array.
        outputs (Union[list, tuple]): list or tuple of output argment index.
        tuning (bool): tuning model.
        device_id: device_id on device.
        expect: when mode in ["compile_cloud", "compile_mini"], return it.

    Returns:
        output numpy array, or tuple of numpy array if multi-output.
    """

    gc.collect()
    if device_id == -1:
        device_id = int(os.environ.get("DEVICE_ID", 0))
    module = mod.imported_modules[0]
    if module.type_key == CUDA:
        return mod_launch_gpu(mod, args, outputs, tuning, device_id, repeat_time)

    kernel_name = get_kernel_name_from_mod(mod)
    stat_info = {}
    profiling_mode = get_profiling_mode()
    if profiling_mode:
        return profiling_mode_run(kernel_name, args, outputs, tuning, device_id)
    mode = get_runtime_mode()
    if mode.startswith("aic"):
        output = aic_model.launch(mod, args, outputs)
        if not tuning:
            return output
        ra_util.get_ticks(stat_info)
        return output, stat_info
    if mode in ('rpc', 'rpc_cloud'):
        return mod_launch_rpc(mode, mod, args, outputs, tuning)

    # The air_cloud is the current default mode and needs to be modified in the future
    if mode == 'air_cloud':
        return ascend_run(kernel_name, args, outputs, device_id)

    if mode in ('ca', 'air', 'air_cloud'):
        return mod_launch_air(mod, args, outputs, device_id)
    if mode in ("compile_cloud", "compile_mini"):
        return expect
    if mode in ("csim", "ccesim", "cdiff"):
        from akg.backend.csim import csim_launch
        return csim_launch(args, outputs)
    if mode == "cpu":
        tvm_array = []
        ctx = akg.tvm.context("llvm", 0)
        for _, args_val in enumerate(args):
            tvm_temp = akg.tvm.nd.array(args_val, ctx)
            tvm_array.append(tvm_temp)
        mod(*tvm_array)
        return tvm_array[-1].asnumpy()

    raise ValueError("mode must be aic, rpc, aic_cloud, ca, compile_cloud, compile_mini, cpu, csim, ccesim or cdiff")


def gen_kernel_name(input_shapes, input_types, op_attrs=None, kernel_name="", attrs=None):
    """generate kernel name."""
    dir_max_length = 250
    shape_info = ''
    for _, (shape, dtype) in enumerate(zip(input_shapes, input_types)):
        if isinstance(shape, (list, tuple)) and shape and isinstance(shape[0], (list, tuple)):
            for _, tmp_shape in enumerate(shape):
                vc_util.check_shape(tmp_shape)
                tmp_shape = list(tmp_shape)
                str_tmp_shape = [str(tmp) for tmp in tmp_shape]
                shape_info = "%s_%s_%s" % (shape_info, dtype, '_'.join(str_tmp_shape))
        elif isinstance(shape, akg.tvm.tensor.Tensor):
            for tmp_shape in shape.shape:
                if isinstance(tmp_shape, akg.tvm.expr.Var):
                    str_shape = tmp_shape.name
                else:
                    str_shape = str(tmp_shape)
                shape_info = "%s_%s_%s" % (shape_info, dtype, '_'.join(str_shape))
        else:
            vc_util.check_shape(shape)
            if isinstance(shape, akg.tvm.expr.Var):
                shape = [shape]
            shape = list(shape)
            str_shape = [str(i) for i in shape]
            shape_info = "%s_%s_%s" % (shape_info, dtype, '_'.join(str_shape))

    if op_attrs is not None:
        for tmp in op_attrs:
            if isinstance(tmp, (list, tuple)):
                for ele in tmp:
                    if isinstance(ele, (list, tuple)):

                        str_tmp = [str(i) for i in ele]
                        shape_info = shape_info + '_' + '_'.join(str_tmp)
                    else:
                        shape_info = shape_info + '_' + str(ele)

            elif isinstance(tmp, (int, float)):
                shape_info = shape_info + '_' + str(tmp)

            elif isinstance(tmp, (str)):
                shape_info = shape_info + '_' + tmp

            elif isinstance(tmp, (np.ndarray)):
                shape = list(tmp.shape)
                str_shape = [str(i) for i in shape]
                shape_info = shape_info + '_' + '_'.join(str_shape)

    kernel_name = kernel_name + shape_info
    kernel_name = re.sub(r'[^0-9a-zA-Z]+', '_', kernel_name)
    if len(kernel_name) > dir_max_length:
        logging.info("Dir name %s exceed maximal length, use first %d char as dir name.", kernel_name, dir_max_length)
        kernel_name = kernel_name[:dir_max_length]

    # When the test cases is executed by multiple processes, different test cases of dynamic shape may generate
    # the same kernel_name.json and kernel_name.o in the kernel_meta directory, and different processes
    # will overlap and delete each other, resulting in failure.
    # This problem can be avoided by adding process id to kernel_name.
    if isinstance(attrs, dict) and (attrs.get("dynamic") or attrs.get("partial_dynamic")):
        pid_suffix = "_" + str(os.getpid())
        kernel_name = kernel_name + pid_suffix
        if len(kernel_name) > dir_max_length:
            real_kernel_name_len = dir_max_length - len(pid_suffix)
            kernel_name = kernel_name[:real_kernel_name_len] + pid_suffix
    return kernel_name


@func_time_required
def op_build_test(op_func, input_shapes, input_types, op_attrs=None, kernel_name="",
                  attrs=None, log_cce=False, dump_ir=True, dump_code=True,
                  polyhedral=True, tuning=False):
    """
    Return module from op_build with given inputs, distinguish tuning mode.

    Args:
        op_func (function returning an op or (op, [op_vars])): The op build function
        input_shapes(iterable of iterable of int): the dim sizes for input for op
        input_types (iterable of iterable of str): the dtypes for each input
        op_attrs (list or tuple): extra attributes for the op.
        kernel_name (str): name of op.
        attrs (dict): tiling parameter.
        log_cce (bool): False by default.
        dump_ir (bool): True by default.
        dump_code (bool): False by default.
        polyhedral (bool): True by default.
        tuning (bool): False by default.

    Return:
        module.
    """
    if isinstance(attrs, dict) and 'tuning' in attrs.keys():
        kernel_name = kernel_name
    else:
        kernel_name = gen_kernel_name(input_shapes, input_types, op_attrs, kernel_name, attrs)
    logging.debug('kernel_name---------- %s', str(kernel_name))
    mod = op_build(op_func, input_shapes, input_types, op_attrs, kernel_name,
                   attrs, log_cce, dump_ir, dump_code,
                   polyhedral, tuning)
    return mod


def recursive_copy(obj):
    """
    Copy a container object recursively

    Args:
        obj (list, tuple, dict or object): input container object.

    Return:
        copied object.
    """
    if isinstance(obj, list):
        return [recursive_copy(it) for it in obj]
    if isinstance(obj, tuple):
        return tuple([recursive_copy(it) for it in obj])
    if isinstance(obj, dict):
        copy_obj = dict()
        for key in obj:
            copy_obj[key] = recursive_copy(obj[key])
        return copy_obj
    return obj


def gen_inputs_and_shape_params(input_shapes, input_types, inputs, shape_params):
    """
    Generate akg.tvm.placeholder as inputs for op with given input_shapes and input_types

    Args:
        input_shapes(iterable of iterable of int): the dim sizes for input for op.
        input_types (iterable of iterable of str): the dtypes for each input.
        inputs (list): None by default.
        shape_params (list): None by default.

    """
    for i, (shape, dtype) in enumerate(zip(input_shapes, input_types)):
        if isinstance(shape, (list, tuple)) and shape and isinstance(shape[0], (list, tuple)):
            tmp_input = []
            for j, tmp_shape in enumerate(shape):
                tmp_input.append(akg.tvm.placeholder(tmp_shape, dtype, "input_%d_%d" % (i + 1, j + 1)))
                for tmp in tmp_shape:
                    if isinstance(tmp, akg.tvm.expr.Var):
                        shape_params.append(tmp)
            inputs.append(tmp_input)
        elif isinstance(shape, (list, tuple)) and shape and isinstance(shape[0], akg.tvm.expr.Var):
            inputs.append(akg.tvm.placeholder(shape, dtype, "input_%d" % (i + 1)))
            for tmp_shape in shape:
                if isinstance(tmp_shape, akg.tvm.expr.Var):
                    shape_params.append(tmp_shape)
        elif isinstance(shape, akg.tvm.tensor.Tensor):
            inputs.append(shape)
            for tmp_shape in shape.shape:
                shape_params.append(tmp_shape)
        else:
            inputs.append(akg.tvm.placeholder(shape, dtype, "input_%d" % (i + 1)))


def gen_attrs_params(op_attrs, attrs_params):
    """
    Parsing attrs given by op_attrs.

    Args:
        op_attrs (list or tuple): extra attributes for the op.
        attrs_params (list): None by default.

    """
    for tmp_attr in op_attrs:
        if isinstance(tmp_attr, (list, tuple)) and tmp_attr and isinstance(tmp_attr[0], akg.tvm.expr.Var):
            for attr_param in tmp_attr:
                if isinstance(attr_param, akg.tvm.expr.Var):
                    attrs_params.append(attr_param)
        elif isinstance(tmp_attr, akg.tvm.expr.Var):
            attrs_params.append(tmp_attr)


def get_dim_from_func_map(attrs, op_func, args, input_shapes, input_types, op_attrs):
    """
    Get tiling parameter from map defined in op_func.

    Args:
        attrs (dict): tiling parameter.
        op_func (function returning an op or (op, [op_vars])): The op build function.
        args (list): input tensors and attributes(if exists) of op_func.
        input_shapes (iterable of iterable of int): the dim sizes for input for op.
        input_types (iterable of iterable of str): the dtypes for each input.
        op_attrs (list or tuple): extra attributes for the op.
    """
    if attrs is None or 'dim' not in attrs or not attrs['dim']:
        dim_info = ""
        if attrs is None:
            attrs = dict()

        if op_func.__name__ in ct_util.set_dim_func_map.keys():
            value = ct_util.set_dim_func_map[op_func.__name__]
            if inspect.isfunction(value):
                dim_info = value(*args)
            elif isinstance(value, dict):
                key = []
                key.append(ft_util.convert_to_list(input_shapes))
                key.append(ft_util.convert_to_list(input_types))
                if op_attrs is not None:
                    key.append(op_attrs)
                key = str(tuple(key))

                if key in value.keys():
                    dim_info = ct_util.set_dims(value[key])
            else:
                raise RuntimeError("Registered set_dim_map is invalid. Must be a function or a dict!")
        if isinstance(dim_info, (list, tuple)):
            dim_info = dim_info[0]

        attrs['dim'] = dim_info
    return attrs


def parsing_output(output, attrs, compute_func, sch_tmpl, gpu_binds):
    """
    Parsing the outputs of op.

    Args:
        output (iterable of iterable of akg.tvm.tensor): the outputs of op.
        attrs (dict): tiling parameter.
        compute_func (function): None by default, func for doing compute_inline or other.
        sch_tmpl (dict): None by default.
        gpu_binds (dict): None by default.
    """
    if isinstance(output, (list, tuple)):
        from inspect import isfunction
        new_outputs = []
        for elem in output:
            if isfunction(elem):
                compute_func = elem
            elif isinstance(elem, dict):
                for key, value in elem.items():
                    if key not in attrs or not attrs[key]:
                        attrs[key] = value
            elif isinstance(elem, (list, tuple)):
                new_outputs += elem
            else:
                new_outputs.append(elem)

        output = new_outputs
    elif isinstance(output, dict):
        sch_tmpl = output
        output = sch_tmpl['output']
        gpu_binds = sch_tmpl['binds']
    return output, compute_func, sch_tmpl, gpu_binds


def gen_op_var(inputs, output, op_var):
    """
    Combine inputs and outputs about the op.

    Args:
        inputs(list): the inputs of op.
        output(list): the outputs of op.
        op_var (list): inputs and outputs for the op.
    """
    for xx in inputs:
        if isinstance(xx, list):
            for x in xx:
                op_var.append(x)
        else:
            op_var.append(xx)
    if isinstance(output, (list, tuple)):
        op_var = op_var + [i for i in output if TensorUtils.is_output_value(i)]
    else:
        if TensorUtils.is_output_value(output):
            op_var = op_var + [output]
    return op_var


def gen_shape_var(attrs_params, shape_params, shape_var):
    """
    Combine shape of inputs and extra attributes about the op.

    Args:
        attrs_params(list): shape of inputs for the op
        shape_params(list): extra attributes for the op
        shape_var (list): shape of inputs and extra attributes for the op.
    """
    if attrs_params:
        for i in attrs_params:
            if i not in shape_var:
                shape_var.append(i)
    for i in shape_params:
        if i not in shape_var:
            shape_var.append(i)


def gen_spaces_dim_key(op_func, args, s, op_var, kernel_name, attrs, polyhedral, tuning, target):
    """
    Generate tiling parameter.

    Args:
        op_func (function returning an op or (op, [op_vars])): The op build function.
        args (Union[list, tuple]): list or tuple of numpy array.
        s (dict): schedule of op.
        op_var (list): the akg.tvm.tensor of inputs and outputs for op.
        kernel_name (str): name of op.
        attrs (dict): tiling parameter.
        polyhedral (bool): True by default.
        tuning (bool): False by default.

    Return:
        tiling parameter.
    """
    set_dim_key = ""
    if op_func.__name__ in ct_util.set_dim_func_map.keys():
        func_ = ct_util.set_dim_func_map[op_func.__name__]
        if inspect.isfunction(func_):
            set_dim_key = func_(*args)[1]
    elif op_func.__name__ in ct_util.gen_key_func_map.keys():
        func_ = ct_util.gen_key_func_map[op_func.__name__]
        if inspect.isfunction(func_):
            set_dim_key = func_(*args)
    with akg.build_config(dump_pass_ir=True):
        spaces = akg.lower(s, op_var, name=kernel_name, attrs=attrs, polyhedral=polyhedral, tuning=tuning,
                           target=target)
        if set_dim_key == "":
            set_dim_key = str(args)
        return spaces, set_dim_key


def create_gpu_mod(sch_tmpl, s, op_func, op_var, shape_var, kernel_name, attrs, polyhedral, binds, dump_ir, dump_code,
                   tuning):
    """
    Return module for op of gpu.

    Args:
        sch_tmpl (dict): schedule of op and the others.
        s (dict): schedule of op.
        op_func (function returning an op or (op, [op_vars])): The op build function.
        op_var (list): the akg.tvm.tensor of inputs and outputs for op.
        shape_var (list): shape of inputs and extra attributes for the op.
        kernel_name (str): name of op.
        attrs (dict): tiling parameter.
        polyhedral (bool): True by default.
        binds (dict): BINDS
        dump_ir (bool): True by default.
        dump_code (bool): False by default.
        tuning (bool): False by default.

    Return:
        module.
    """

    if sch_tmpl is not None or (attrs and attrs.get("target", "cce") == "cuda"):
        if kernel_name == "":
            kernel_name = op_func.__name__ if sch_tmpl is None else sch_tmpl['op_name']

    target = CUDA

    if sch_tmpl is not None:
        if sch_tmpl['target'] != CUDA:
            raise ValueError("Only support cuda as target when using schedule template.")
        with akg.tvm.target.cuda() as target:
            if not tuning:
                s = sch_tmpl['schedule'](sch_tmpl['output'])
                with akg.tvm.build_config(dump_pass_ir=dump_ir):
                    mod = akg.build(s, op_var, "cuda", shape_var, name=kernel_name, attrs=attrs,
                                    polyhedral=False, binds=binds)
            else:
                @autotvm.template
                def _autotune_template():
                    s = sch_tmpl['schedule'](sch_tmpl['output'])
                    return (s, op_var)

                # create autotune task
                task = autotvm.task.create(_autotune_template,
                                           args=list(),
                                           target='cuda')

                print("task config: ", task.config_space)

                # set measure_option
                measure_option = autotvm.measure_option(
                    builder=autotvm.LocalBuilder(),
                    runner=autotvm.LocalRunner(repeat=5, min_repeat_ms=150, timeout=4)
                )

                # Begin tuning, log records to file `kernel_name.log`
                tuner = autotvm.tuner.RandomTuner(task)
                if not os.path.exists(kernel_name + '.log'):
                    tuner.tune(n_trial=len(task.config_space),
                               measure_option=measure_option,
                               callbacks=[autotvm.callback.log_to_file(kernel_name + '.log')])

                # query best config
                dispatch_context = autotvm.apply_history_best(kernel_name + '.log')
                best_config = dispatch_context.query(task.target, task.workload)
                print("\nBest config is:")
                print(best_config)

                # apply best config
                with autotvm.apply_history_best(kernel_name + '.log'):
                    s, op_var = _autotune_template()
                    mod = akg.build(s, op_var, "cuda", shape_var, name=kernel_name, attrs=attrs,
                                    polyhedral=False, binds=binds)
    else:
        with akg.build_config(dump_pass_ir=dump_ir):
            mod = akg.build(s, op_var, target, shape_var, name=kernel_name, attrs=attrs, polyhedral=polyhedral,
                            binds=binds)
    if dump_code:
        source_code = mod.imported_modules[0].get_source()
        create_code(kernel_name, "./", source_code, CUDA)
    return mod


def op_build(op_func, input_shapes, input_types, op_attrs=None, kernel_name="",
             attrs=None, log_cce=False, dump_ir=True, dump_code=True,
             polyhedral=True, tuning=False, ret_mode=ReturnType.MOD):
    """
    Return module built from op_func with given inputs.

    Args:
        op_func (function returning an op or (op, [op_vars])): The op build function.
        input_shapes(iterable of iterable of int): the dim sizes for input for op.
        input_types (iterable of iterable of str): the dtypes for each input.
        op_attrs (list or tuple): extra attributes for the op.
        kernel_name (str): name of op.
        attrs (dict): tiling parameter.
        log_cce (bool): False by default.
        dump_ir (bool): True by default.
        dump_code (bool): False by default.
        polyhedral (bool): True by default.
        tuning (bool): False by default.

    Return:
        module.
    """
    inputs = []
    shape_params = []  # save all the shape params for dynamic_shape cases
    gen_inputs_and_shape_params(input_shapes, input_types, inputs, shape_params)

    attrs_params = []
    if op_attrs is not None:
        args = inputs + op_attrs
        gen_attrs_params(op_attrs, attrs_params)
    else:
        args = inputs

    # backup inputs because the tensor names may be updated inside op_func
    inputs_backup = recursive_copy(inputs)

    output = op_func(*args)

    # restore inputs to make sure that tensor names are not changed by op_func
    inputs = inputs_backup
    # set dim
    attrs = get_dim_from_func_map(attrs, op_func, args, input_shapes, input_types, op_attrs)

    compute_func = None  # func which is defined in dsl for doing compute_inline or other
    sch_tmpl = None
    gpu_binds = None
    output, compute_func, sch_tmpl, gpu_binds = parsing_output(output, attrs, compute_func, sch_tmpl, gpu_binds)

    op_var = []
    op_var = gen_op_var(inputs, output, op_var)

    shape_var = []
    gen_shape_var(attrs_params, shape_params, shape_var)

    if sch_tmpl is not None:
        return create_gpu_mod(sch_tmpl, None, op_func, op_var, shape_var, kernel_name, attrs, polyhedral, gpu_binds,
                              dump_ir, dump_code, tuning)

    if isinstance(output, (list, tuple)):
        tmp = []
        for x in list(output):
            if isinstance(x, tuple):
                tmp.append(x[0].op)
            else:
                tmp.append(x.op)
        s = akg.tvm.create_schedule(tmp)
    else:
        s = akg.tvm.create_schedule(output.op)

    if compute_func is not None:
        compute_func(s)
        polyhedral = False

    target = CCE
    if attrs and attrs.get("target", "cce") == CUDA:
        target = CUDA

    level = attrs.get("help_tiling") if attrs and "help_tiling" in attrs else None
    if tuning or (level is not None and level > help_tiling_level['None']):
        return gen_spaces_dim_key(op_func, args, s, op_var, kernel_name, attrs, polyhedral, tuning, target)
    mode = get_runtime_mode()
    if mode == "cpu":
        mod = akg.tvm.build(s, op_var, "llvm")
        if not os.path.isdir("./cpu/ir/"):
            os.makedirs("./cpu/ir/", exist_ok=True)
        with os.fdopen(os.open("./cpu/ir/" + kernel_name + ".cc", os.O_WRONLY | os.O_CREAT, 0o400), 'w') as irf:
            irf.write(akg.tvm.lower(s, op_var, shape_var, simple_mode=True))
        return mod

    binds = None if not attrs else attrs.pop(BINDS, None)
    if ret_mode in [ReturnType.FEAT, ReturnType.MOD_AND_FEAT]:
        if binds is None:
            from akg.tvm import build_module
            binds, _ = build_module.get_binds(op_var)
        cfg = _api_internal._GetCurrentBuildConfig()
        stmt, args = _api_internal._Lower(s, op_var, shape_params, kernel_name,
                                          binds, attrs, False, True, False, target,
                                          cfg, True)
        from akg.utils.auto_tuning import get_features_from_stmts
        feature = get_features_from_stmts(stmts=[stmt], binds=[binds], n_skip_cache=0)[0]
        if ret_mode == ReturnType.FEAT:
            return feature
        mod = _api_internal._BuildStmtToModule(stmt, kernel_name, cfg, args, target)
        return mod, feature

    if target == CUDA:
        return create_gpu_mod(None, s, op_func, op_var, shape_var, kernel_name, attrs, polyhedral, binds, dump_ir,
                              dump_code, tuning)

    target = CCE
    with akg.build_config(dump_pass_ir=dump_ir):
        mod = akg.build(s, op_var, target, shape_var, name=kernel_name, attrs=attrs, polyhedral=polyhedral, binds=binds)

    source_code = mod.imported_modules[0].get_source()
    if log_cce:
        logging.debug("#################cce code####################")
        logging.debug(source_code)
    if dump_code:
        create_code(kernel_name, "./", source_code, target)
    return mod


def get_runtime_mode():
    """get runtime mode."""
    env_dic = os.environ
    if not env_dic.get('RUNTIME_MODE'):
        mode = 'rpc_cloud'
    else:
        mode = env_dic.get('RUNTIME_MODE')
    return mode


def get_profiling_mode():
    """get profiling mode."""
    env_dic = os.environ
    if env_dic.get('PROFILING_MODE') and env_dic.get('PROFILING_MODE').lower() == "true":
        return True
    return False


def product_is_mini():
    """check whether in mini environment."""
    mode = get_runtime_mode()
    if mode in ('rpc', 'air', 'aic', 'compile_mini'):
        return True
    return False


def get_available_devices_num():
    """get available devives num."""
    env_dic = os.environ
    try:
        return int(env_dic.get('DEVICE_TOTAL_NUM').lower()) if env_dic.get('DEVICE_TOTAL_NUM') else 1
    except NameError as e:
        logging.error(e)
        return 1


def get_device_id():
    """get device id."""
    env_dic = os.environ
    try:
        return int(env_dic.get('DEVICE_ID').lower()) if env_dic.get('DEVICE_ID') else 0
    except NameError as e:
        logging.error(e)
        return 0


def get_gpu_cycles(mod, *mod_args, device_id=0, repeat_time=400):
    """get gpu profiling cycles."""
    from akg.utils.result_analysis import gpu_profiling
    tcost = gpu_profiling(mod, *mod_args, repeat_time=repeat_time, device_id=device_id)
    return tcost


class TestUtils:
    """Class for getting cycle and core num."""

    @staticmethod
    def record_cycle(cycle):
        if os.environ.get(PERFORMANCE_TEST_FILE):
            result_file = os.environ.get(PERFORMANCE_TEST_FILE)
            with open(result_file, "a+") as f:
                f.write("{0}\n".format(cycle))

    @staticmethod
    def record_core(stmt):
        """Function for getting performance data from cores."""

        def get_core_num():
            core_num = 1
            if hasattr(stmt, 'attr_key') and stmt.attr_key == 'thread_extent':
                core_num = stmt.value
            return core_num

        if os.environ.get(PERFORMANCE_TEST_FILE):
            result_file = os.environ.get(PERFORMANCE_TEST_FILE)
            with open(result_file, "a+") as f:
                f.write("{0}; ".format(get_core_num()))
