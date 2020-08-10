#!/usr/bin/env python3
# coding: utf-8
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

"""op_build"""
import os
import fcntl
import types
import typing
import logging
import traceback
import akg.tvm
import akg
import akg.backend as cce
from akg.ms import save_gpu_param as gpu_utils
from akg.tvm import _api_internal
from akg.utils import validation_check as vc_util


BINDS = "binds"
MS_AKG_DUMP_IR = "MS_AKG_DUMP_IR"
MS_AKG_DUMP_CODE = "MS_AKG_DUMP_CODE"
MS_DAVINCI_KERNEL_PATH = "./kernel_meta/"


@vc_util.check_input_type(list, (list, tuple), (list, tuple), (types.FunctionType, type(None)), str, str, dict)
def op_build_to_func(opnames, computes, args, custom_schedule, device, kernel_name, attrs):
    """op_build_to_func"""
    if device not in ("aicore", "aicpu"):
        logging.error("Device %s is not in [aicore, aicpu].", device)
        return None

    polyhedral = True
    dump_ir = os.getenv(MS_AKG_DUMP_IR) == "on"

    try:
        tmp_outputs = [x.op for x in computes]
        s = akg.tvm.create_schedule(tmp_outputs)
        if custom_schedule:
            polyhedral = False
            custom_schedule(s)

        with akg.build_config(add_lower_pass=cce.debug_mode(0), dump_pass_ir=dump_ir):
            if attrs:
                binds = attrs.pop(BINDS, None)
                rst = akg.build_to_func(s, args, name=kernel_name, attrs=attrs, polyhedral=polyhedral,
                                        binds=binds, target=device)
            else:
                rst = akg.build_to_func(s, args, name=kernel_name, polyhedral=polyhedral, target=device)

    except Exception:
        logging.error(traceback.format_exc())
        return None
    return rst

@vc_util.check_input_type(list, (list, tuple), (list, tuple), (types.FunctionType, type(None)), str, str, dict)
def op_build(opnames, computes, args, custom_schedule, device, kernel_name, attrs):
    """op_build"""
    if device in ("aicore", "aicpu"):
        tmp_rst = op_build_to_func(opnames, computes, args, custom_schedule, device, kernel_name, attrs)
        return tmp_rst

    if device == "cuda":
        kernel_meta_path = "./cuda_meta_" + str(os.getpid()) + "/"
        cuda_path = os.path.realpath(kernel_meta_path)
        if not os.path.isdir(cuda_path):
            os.makedirs(cuda_path)
        if not opnames:
            logging.error("no opname given.")
            return None

        schedule_name = 'gpu_schedule_' + opnames[0]
        schedule_func = getattr(akg.ms.gpu, schedule_name)
        if not isinstance(schedule_func, (types.FunctionType, typing.Callable)):
            logging.error("no schedule func found %s", str(schedule_name))
            return None

        ptx_file = os.path.realpath(kernel_meta_path + kernel_name + ".ptx")
        if os.path.exists(ptx_file):
            os.remove(ptx_file)
        try:
            with open(ptx_file, 'at') as file:
                fcntl.flock(file.fileno(), fcntl.LOCK_EX)
                file.seek(0, 2)
                if file.tell() == 0:
                    s = schedule_func(computes)
                    foo = akg.tvm.build(s, args, device, name=kernel_name)
                    ptx_code = foo.imported_modules[0].get_source("ptx")
                    file.write(ptx_code)
                    json_file = os.path.realpath(kernel_meta_path + kernel_name + ".json")
                    kernel_info = (ptx_code, json_file, kernel_name)
                    gpu_utils.save_gpu_params(s, args, kernel_info)
            os.chmod(ptx_file, 0o400)
        except Exception:
            logging.error(traceback.format_exc())
            return None
        return True

    logging.error("Not support device %s.", device)
    return None
