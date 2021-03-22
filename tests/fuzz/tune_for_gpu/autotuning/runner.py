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

"""Runner for compile and execute a configs of an operator on device"""
import time
import multiprocessing
import logging
import json
import os
import subprocess
import time
from typing import NamedTuple
import numpy as np
from akg import composite
from akg.utils import custom_tiling as ct_util
from akg.utils import kernel_exec as utils
from .kernel_compiler import compile_kernel
from .test_data_generators import gen_data
from .tuning_utils import *

logger = logging.getLogger('fuzz.tune.autotuning.runner')

error_time_list = [
    9999999999.0,
    9999999998.0,
    9999999997.0,
    9999999996.0,
]

error_time_string = {
    error_time_list[0]: 'run_failed',
    error_time_list[1]: 'precision_error',
    error_time_list[2]: 'compile_failed',
    error_time_list[3]: 'timeout'
}

run_failed_time = error_time_list[0]
precision_error_time = error_time_list[1]
compile_fail_time = error_time_list[2]
timeout_time = error_time_list[3]


def get_attr_from_config(config, index_table):
    tiling = []
    attrs = {}
    tuning_dict = config._asdict()
    for key, value in tuning_dict.items():
        if key.startswith('tiling'):
            item = [value, 1]
            tiling.append(item)
        else:
            attrs[key] = value
    if len(tiling):
        tiling_param = []
        for i, element in enumerate(tiling):
            tiling_param.append(index_table[i] + element)
        dim_info = ct_util.set_dims(tuple(tiling_param))
        attrs['dim'] = dim_info
    else:
        print("No tiling info. Use auto tiling.")
    return attrs


class KernelRunner:
    """kernel runner
    This runner will compile and execute configs of an operator, and return their running times.

    Parameters
    ----------
    op_type: str
        The name of operator
    op_desc: NamedTuple
        The definition parameters of operator
    timeout: int
        Timeout for running one config
    repeat_times:
        Run one config repeat_times
    """

    def __init__(self, op_type: str, op_desc: NamedTuple,
                 index_table: list, self_attrs: list, timeout: int = 600,
                 repeat_times: int = 2, input_data=None,
                 expect=None, mod_output_param=None, is_all_space=True,
                 skip_config_set=None, need_tune_json=None):
        self.op_type = op_type
        self.op_desc = op_desc
        self._index_table = index_table
        self.self_attrs = self_attrs
        self.run_kernel_time = 0.0
        self.tune_self_attrs = True
        self.timeout = timeout
        self.repeat_times = repeat_times
        self.mod_output_param = mod_output_param
        self.is_all_space = is_all_space
        self.skip_config_set = skip_config_set
        self.need_tune_json = need_tune_json
        if input_data is None:
            self.input, self.expect = gen_data(op_type, op_desc)
            if isinstance(self.input, dict):
                self.input, self.mod_output_param = self.input['args'], self.input['outputs']
        else:
            self.input, self.expect = input_data, expect
        self.input_shape = [x.shape for x in self.input]

    def info(self):
        print('run kernel time:', self.run_kernel_time)

    def run_one_kernel(self, run_times, idx, config, best_time=np.inf, is_auto=False):
        """Compile and execute a config of the operator on device"""

        if json.dumps(config.input._asdict()) in self.skip_config_set:
            print("CONFIG SKIP:", json.dumps(config.input._asdict()))
            run_times[idx] = -1
            return

        time_one_kernel_start = time.time()
        logger.debug('compile %dth kernel', idx)
        gpu_devices_list = get_available_gpu_num()
        device_id = gpu_devices_list[idx % len(gpu_devices_list)]
        logger.debug('run %dth kernel', idx)
        logger.debug('++++++++++++++++++++++=device_id')
        logger.debug(device_id)
        logger.debug('++++++++++++++++++++++=device_id')
        try:
            time_start_build = time.time()
            logger.debug(config)
            if self.op_type in ["json", "extra_tune"]:
                if is_auto:
                    mod = composite.build(self.op_desc)
                    if self.op_type == "extra_tune":
                        del os.environ['MS_GRAPH_KERNEL_TILING']
                else:
                    attrs = get_attr_from_config(
                        config.input, self._index_table)
                    if os.environ['RUNTIME_MODE'] == "gpu":
                        attrs['target'] = "cuda"
                    mod = composite.build(self.op_desc, attrs, use_repo=False)
            else:
                mod = compile_kernel(self.op_type, self.op_desc, self.input_shape, self._index_table,
                                     None if is_auto else config.input, idx, need_tune_json=self.need_tune_json)
            time_end_build = time.time()
            logger.debug("build module time: %f",
                         time_end_build - time_start_build)
            logger.debug('finished compile %dth kernel', idx)
        except BaseException as e:
            logger.debug("Compile Failed: [%s] : %s", "origin" if is_auto else str(
                config.input), str(e))
            run_times[idx] = compile_fail_time
            return

        run_times[idx] = run_failed_time

        try:
            # NOTE: in gpu tuning, it is no need to use this repeat_times,
            # repeat_time has been setted in mod_launch in tuning mode.
            for _ in range(self.repeat_times):
                stat_info = {}
                try:
                    time_start_launch = time.time()
                    if self.mod_output_param is not None:
                        pass
                    else:
                        output, stat_info = utils.mod_launch(
                            mod, self.input, tuning=True, device_id=device_id, repeat_time=40)
                        if not np.allclose(output, self.expect, rtol=5e-03, atol=5e-03, equal_nan=True):
                            stat_info['run_time'] = precision_error_time
                            logger.debug("Precision Error: [%s]",
                                            "origin" if config is None else str(config.input))

                    time_end_launch = time.time()
                    logger.debug("mod launch time: %f",
                                 time_end_launch - time_start_launch)
                except BaseException as e:
                    logger.debug("Run Failed: [%s] : %s", str(
                        config.input), str(e))
                    stat_info['run_time'] = run_failed_time
                run_times[idx] = np.minimum(
                    run_times[idx], stat_info['run_time'])
        finally:
            logger.debug('end of %dth kernel', idx)
            time_one_kernel_end = time.time()
            logger.debug('run one kernel time: %f',
                         time_one_kernel_end - time_one_kernel_start)
        return

    def run(self, configs, best_time=np.inf, is_auto_set_dim=False, all_space=False):
        """Compile and execute a batch config of the operator on device"""
        start = time.time()
        logger.setLevel(logging.DEBUG)
        logger.debug("gen cce kernels batch: %d kernels", len(configs))
        subprocess.run("rm -rf ./jobs/JOB*", shell=True)

        process_jobs = []
        run_times = multiprocessing.Manager().list(
            np.full((len(configs),), compile_fail_time))
        for idx, config in enumerate(configs):
            p = multiprocessing.Process(target=self.run_one_kernel,
                                        args=(run_times, idx, config, best_time, is_auto_set_dim))
            process_jobs.append(p)
            p.start()
        timeout_error = False
        for idx, p in enumerate(process_jobs):
            if not timeout_error:
                p.join(timeout=self.timeout)
            if p.is_alive():
                timeout_error = True
                logger.debug("Timeout Error: [%s]", str(configs[idx].input))
                run_times[idx] = timeout_time
                p.terminate()

        process_end = time.time()
        logger.debug("process time: %f", process_end - start)
        # clean the profiling directory
        tune_device = int(os.environ['DEVICE_ID'])
        tune_num = int(os.environ['DEVICE_TOTAL_NUM'])
        if os.environ['RUNTIME_MODE'] == "gpu":
            subprocess.run("rm -rf cuda_meta_*", shell=True)
        else:
            pass
        end = time.time()
        logger.debug("run kernels time: %f", end - start)
        self.run_kernel_time += end - start

        for idx, config in enumerate(configs):
            if run_times[idx] not in error_time_list:
                logger.debug("KernelRunTime : [%s] : %s", str(
                    configs[idx].input), str(run_times[idx]))
            else:
                logger.debug("KernelRunTime : [%s] : %s",
                             str(configs[idx].input), str(error_time_string[run_times[idx]]))

        return run_times
