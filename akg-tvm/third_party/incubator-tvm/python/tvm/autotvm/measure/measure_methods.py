# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name,too-many-function-args,too-many-nested-blocks
"""
Functions that run on executor for measurement.

These functions are responsible for building the tvm module, uploading it to
remote devices, recording the running time costs, and checking the correctness of the output.
"""

import logging
import shutil
import os
import threading
import time
from random import getrandbits
from collections import namedtuple
import tempfile

import numpy as np

from ... import ir_pass, build, build_config, nd, TVMError, register_func, target as _target
from ...contrib import nvcc, ndk, tar

from ..util import get_const_tuple
from ..env import AutotvmGlobalScope
from ..task.space import InstantiationError

from .measure import MeasureResult, MeasureErrorNo, Builder, Runner
from .local_executor import LocalExecutor

logger = logging.getLogger('autotvm')

class BuildResult(namedtuple("BuildResult", ('filename', 'arg_info', 'error', 'time_cost'))):
    """
    Stores all the necessary inputs for a measurement.

    Parameters
    ----------
    filename : str
        The filename of generated library
    arg_info : Tuple
        The shape and dtype information of tvm tensor arguments
    error : Exception
        The error happens during compilation.
    time_cost : float
        The time cost of building
    """

class LocalBuilder(Builder):
    """Run compilation on local machine

    Parameters
    ----------
    timeout: float
        The timeout of a compilation
    n_parallel: int
        The number of tasks run in parallel. "None" will use all cpu cores
    build_func: callable or str
        If is 'default', use default build function
        If is 'ndk', use function for android ndk
        If is callable, use it as custom build function, expect lib_format field.
    """
    def __init__(self, timeout=10, n_parallel=None, build_func='default'):
        super(LocalBuilder, self).__init__(timeout, n_parallel)

        if isinstance(build_func, str):
            if build_func == 'default':
                build_func = tar.tar
            elif build_func == 'ndk':
                build_func = ndk.create_shared
            else:
                raise ValueError("Invalid build_func" + build_func)
        self.build_func = _wrap_build_func(build_func)
        self.executor = LocalExecutor(timeout=timeout)
        self.tmp_dir = tempfile.mkdtemp()

    def build(self, measure_inputs):
        results = []

        shutil.rmtree(self.tmp_dir)
        self.tmp_dir = tempfile.mkdtemp()

        for i in range(0, len(measure_inputs), self.n_parallel):
            futures = []
            for inp in measure_inputs[i:i + self.n_parallel]:
                ret = self.executor.submit(self.build_func,
                                           inp,
                                           self.tmp_dir,
                                           **self.build_kwargs)
                futures.append(ret)

            for future in futures:
                res = future.get()

                if isinstance(res, Exception):
                    # timeout or fleet error, return MeasureResult directly
                    results.append(MeasureResult((res,), MeasureErrorNo.BUILD_TIMEOUT,
                                                 self.timeout, time.time()))
                elif res.error is not None:
                    # instantiation error
                    if isinstance(res.error, InstantiationError):
                        results.append(MeasureResult((res.error,),
                                                     MeasureErrorNo.INSTANTIATION_ERROR,
                                                     res.time_cost, time.time()))
                    else:
                        if "InstantiationError" in str(res.error):
                            msg = str(res.error)
                            try:
                                msg = msg.split('\n')[-2].split(": ")[1]
                            except Exception:  # pylint: disable=broad-except
                                pass
                            results.append(MeasureResult((InstantiationError(msg),),
                                                         MeasureErrorNo.INSTANTIATION_ERROR,
                                                         res.time_cost, time.time()))
                        else:  # tvm error
                            results.append(MeasureResult((res.error,),
                                                         MeasureErrorNo.COMPILE_HOST,
                                                         res.time_cost, time.time()))
                else:
                    # return BuildResult
                    results.append(res)

        return results


def _build_func_common(measure_input, check_gpu=None, cuda_arch=None, build_option=None):
    """Common part for building a configuration"""
    target, task, config = measure_input
    with target:
        s, args = task.instantiate(config)

        # check invalidity of template and code hash consistency
        if not config.valid():
            raise InstantiationError(config.errors)

        opts = build_option or {}
        if check_gpu:  # Add verify pass to filter out invalid configs in advance.
            opts["add_lower_pass"] = [(2, gpu_verify_pass(**check_gpu))]
        if cuda_arch:
            set_cuda_target_arch(cuda_arch)

        # if target is vta, we need to use vta build
        if hasattr(measure_input.target, 'device_name') and \
            measure_input.target.device_name == 'vta':
            import vta
            func = vta.build(s, args, target_host=task.target_host)
        else:
            with build_config(**opts):
                func = build(s, args, target_host=task.target_host)
    return func, tuple((get_const_tuple(x.shape), x.dtype) for x in args)


def _wrap_build_func(build_func):
    """
    Wrap build_func to a function that can be used in measure.

    Parameters
    ----------
    build_func : The compilation function
        We expect fcompile to contain an attr "output_format"

    Returns
    -------
    wrapped_build_func : function
        The wrapped build function
    """
    if not hasattr(build_func, "output_format"):
        raise AttributeError("Expect build_func to have the attribute output_format.")
    output_format = build_func.output_format

    def _wrapped(measure_input, tmp_dir, **kwargs):
        """
        Wrapped build func.

        Parameters
        ----------
        measure_input: MeasureInput
            The input of measurement

        tmp_dir: str
            The path of temporary directory to export generated library
        """
        tic = time.time()
        try:
            filename = os.path.join(tmp_dir, "tmp_func_%0x.%s" % (
                getrandbits(64), output_format))
            # TODO(tvm-team) consider linline _build_func_common
            func, arg_info = _build_func_common(measure_input, **kwargs)
            func.export_library(filename, build_func)
        except Exception as e:  # pylint: disable=broad-except
            return BuildResult(None, None, e, time.time() - tic)
        return BuildResult(filename, arg_info, None, time.time() - tic)
    return _wrapped


@register_func
def tvm_callback_cuda_compile(code):
    """use nvcc to generate ptx code for better optimization"""
    curr_cuda_target_arch = AutotvmGlobalScope.current.cuda_target_arch
    # e.g., target arch could be [
    #   "-gencode", "arch=compute_52,code=sm_52",
    #   "-gencode", "arch=compute_70,code=sm_70"
    # ]
    target = "fatbin" if isinstance(curr_cuda_target_arch, list) else "ptx"
    ptx = nvcc.compile_cuda(code, target=target, arch=AutotvmGlobalScope.current.cuda_target_arch)
    return ptx


def set_cuda_target_arch(arch):
    """set target architecture of nvcc compiler

    Parameters
    ----------
    arch: str or list
        The argument of nvcc -arch. (e.g. "sm_51", "sm_62")
        it can also be a count of gencode arguments pass to nvcc command line,
        e.g., ["-gencode", "arch=compute_52,code=sm_52", "-gencode", "arch=compute_70,code=sm_70"]
    """
    AutotvmGlobalScope.current.cuda_target_arch = arch


def gpu_verify_pass(**kwargs):
    """Verify the validity of a gpu kernel.
    This pass will check memory usage and number of threads per block.
    """
    def verify_pass(stmt):
        valid = ir_pass.VerifyGPUCode(stmt, kwargs)
        if not valid:
            raise InstantiationError("Skipped because of invalid gpu kernel")
        return stmt
    return verify_pass
