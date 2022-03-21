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

"""aic model simulation util."""
import os
import subprocess
import json
import numpy as np
import akg.tvm
from akg.global_configs import get_kernel_meta_path


def launch(kernel, args, output=(-1,)):
    """
    simulated run CCE kernel by aic model.

    Args:
        kernel (str): str of kernel name, or CCE Module.
        args (Union[list, tuple]): list or tuple of numpy array.
        output (Union[list, tuple]): list or tuple of output argment index.
    Returns:
        output numpy array, or tuple of numpy array if multi-output.
    """

    def _check_exists(value, error_msg):
        if not value:
            raise RuntimeError(error_msg)

    def _mkdir(path):
        if not os.path.exists(path):
            os.mkdir(path)

    def _rmdir(path):
        if os.path.exists(path):
            os.remove(path)

    if isinstance(kernel, akg.tvm.module.Module):
        code = kernel.imported_modules[0].get_source()
        kernel_name = code.split("_kernel")[0].split(" ")[-1]
    else:
        kernel_name = kernel
    hbm_addr = 0x4000000
    hbm_unit = 0x1000000
    aic_model_path = os.getenv('AIC_MODEL_PATH')
    _check_exists(aic_model_path,
                  "AIC_MODEL_PATH environment variable is not set. Please set it to the dir of model_exe")
    aic_model_path = os.path.realpath(aic_model_path)
    # spec : target chip specification.
    spec_name = os.getenv('AIC_MODEL_SPEC_NAME')
    _check_exists(spec_name,
                  "AIC_MODEL_SPEC_NAME environment variable is not set. Please set it to the name of spec("
                  "It should be xxx.spec and the xxx.spec file is under the AIC_MODEL_PATH directory)")
    aic_out_path = os.path.realpath("aic_out")
    _mkdir(aic_out_path)
    calog_path = aic_out_path + "/calog"
    _mkdir(calog_path)

    model_path = aic_out_path + "/model"
    if not os.path.exists(model_path):
        subprocess.call(["ln", "-s", aic_model_path + "/model", model_path])

    kernel_meta_path = get_kernel_meta_path()
    kernel_meta_realpath = os.path.realpath(kernel_meta_path)
    _check_exists(kernel_meta_realpath,
                  "The parameter kernel_meta_realpath  can not be found, please check")

    o_name = kernel_meta_realpath + "/" + kernel_name + ".o"
    bin_name = aic_out_path + "/kernel.bin"
    subprocess.call(["aicore-elf-objcopy", "-O", "binary",
                    "-j", ".text", o_name, bin_name])

    load_dict = {}
    with open("%s/%s.json" % (kernel_meta_realpath, kernel_name), "r") as f:
        load_dict = json.load(f)

    arg_info = []
    desc = {"args": arg_info,
            "para_addr": hbm_addr,
            "bin_addr": hbm_addr + 0x100000,
            "bin": "kernel.bin",
            "block": load_dict["blockDim"],
            "spec": aic_model_path + '/' + spec_name,
            "path": aic_out_path}
    hbm_addr += hbm_unit

    for i, arg in enumerate(args):
        bin_name = "a_%d.bin" % (i)
        arg.tofile(os.path.join(aic_out_path, bin_name))
        info = {"bin": bin_name,
                "size": arg.size * arg.dtype.itemsize,
                "addr": hbm_addr,
                "out": False}
        arg_info.append(info)
        need_size = arg.size
        if need_size % hbm_unit:
            need_size += hbm_unit - (need_size % hbm_unit)
        hbm_addr += need_size
    for i in output:
        arg_info[len(arg_info) + i if i < 0 else i]['out'] = True

    config_path = aic_out_path + "/config.toml"
    _rmdir(config_path)
    with os.fdopen(os.open(config_path, os.O_WRONLY | os.O_CREAT, 0o400), 'w') as f:
        f.write('title="Sim Config"\n')
        f.write('log_open_value=0xffffffff\n')
        f.write('chip_version=1\n')
        f.write('block_dim=%d\n' % (desc['block']))
        f.write('specPathName="%s"\n' % (desc["spec"]))
        f.write('path="%s/"\n' % (desc["path"]))
        f.write('hbm_para_addr=0x%x\n' % (desc["para_addr"]))
        f.write('[BIN]\n')
        f.write('name="%s"\n' % (desc['bin']))
        f.write('addr=0x%x\n' % (desc['bin_addr']))
        for arg in arg_info:
            f.write('[[output_para_array]]\n' if arg['out']
                    else '[[input_para_array]]\n')
            f.write('name="%s"\n' % (arg['bin']))
            f.write('addr=0x%x\n' % (arg['addr']))
            f.write('valid=1\n')
            if arg['out']:
                f.write('size=0x%x\n' % (arg['size']))

    run_path = aic_out_path + "/run.sh"
    _rmdir(run_path)
    with os.fdopen(os.open(run_path, os.O_WRONLY | os.O_CREAT, 0o500), 'w') as f:
        f.write("cd " + aic_out_path + "\n")
        f.write("export DVCSPEC_DIR=" + aic_model_path + "\n")
        f.write(aic_model_path +
                "/v100_ca_tag_master --gtest_filter=test_st_case.test_st_ca\n")
    subprocess.call(["sh", aic_out_path + "/run.sh"])
    out_list = []
    for i, arg_ in enumerate(args):
        if arg_info[i]['out']:
            out_data = np.fromfile(os.path.join(
                aic_out_path, arg_info[i]['bin']), arg_.dtype)
            # strip unneeded data copied back by aic model
            if out_data.size > args[i].size:
                out_data = out_data[0:arg_.size]
            out_arg = out_data.reshape(arg_.shape)
            out_list.append(out_arg)
    return out_list[0] if len(out_list) == 1 else tuple(out_list)
