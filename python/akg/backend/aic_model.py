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

"""aic model simulation util."""
import os
import subprocess
import json
import numpy as np
import akg.tvm


class Spec():
    MINI = "davinci_mini.spec"
    LITE = "davinci_lite.spec"
    LITE2 = "davinci_lite2.spec"
    CLOUD = "davinci_cloud.spec"
    ORLANDO_CS = "orlando_cs.spec"
    PHEONIX_CS = "pheonix_cs.spec"


def launch(kernel, args, output=(-1,), kernel_meta_path='./kernel_meta', spec=Spec.MINI):
    """
    simulated run CCE kernel by aic model.

    Args:
        kernel (str): str of kernel name, or CCE Module.
        args (Union[list, tuple]): list or tuple of numpy array.
        output (Union[list, tuple]): list or tuple of output argment index.
        kernel_meta_path : kernel meta directory path of the kernel.
        spec : target chip specification.

    Returns:
        output numpy array, or tuple of numpy array if multi-output.
    """
    if isinstance(kernel, akg.tvm.module.Module):
        code = kernel.imported_modules[0].get_source()
        kernel_name = code.split("_kernel")[0].split(" ")[-1]
    else:
        kernel_name = kernel
    hbm_addr = 0x4000000
    hbm_unit = 0x1000000
    aic_model_path = os.getenv('AIC_MODEL_PATH')
    if not aic_model_path:
        msg = "AIC_MODEL_PATH environment variable is not set. Please set it to the dir of model_exe"
        raise RuntimeError(msg)
    aic_model_path = os.path.realpath(aic_model_path)
    if not os.path.exists(aic_model_path):
        msg = "The parameter aic_model_path can not be found, please check"
        raise RuntimeError(msg)

    aic_out_path = os.path.realpath("aic_out")
    if not os.path.exists(aic_out_path):
        os.mkdir(aic_out_path)
    calog_path = aic_out_path + "/calog"
    if not os.path.exists(calog_path):
        os.mkdir(calog_path)

    model_path = aic_out_path + "/model"
    if not os.path.exists(model_path):
        subprocess.call(["ln", "-s", aic_model_path + "/model", model_path])

    kernel_meta_realpath = os.path.realpath(kernel_meta_path)
    if not os.path.exists(kernel_meta_realpath):
        msg = "The parameter kernel_meta_realpath  can not be found, please check"
        raise RuntimeError(msg)

    o_name = kernel_meta_realpath + "/" + kernel_name + ".o"
    bin_name = aic_out_path + "/kernel.bin"
    subprocess.call(["aicore-elf-objcopy", "-O", "binary", "-j", ".text", o_name, bin_name])

    load_dict = {}
    with open("%s/%s.json" % (kernel_meta_realpath, kernel_name), "r") as f:
        load_dict = json.load(f)

    arg_info = []  # [{"bin": "xx.bin", "out" : False, "size":100, "addr": 200},]
    desc = {"args": arg_info,
            "para_addr": hbm_addr,
            "bin_addr": hbm_addr + 0x100000,
            "bin": "kernel.bin",
            "block": load_dict["blockDim"],
            "spec": aic_model_path + '/' + spec,
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
    if os.path.exists(config_path):
        os.remove(config_path)
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
            f.write('[[output_para_array]]\n' if arg['out'] else '[[input_para_array]]\n')
            f.write('name="%s"\n' % (arg['bin']))
            f.write('addr=0x%x\n' % (arg['addr']))
            f.write('valid=1\n')
            if arg['out']:
                f.write('size=0x%x\n' % (arg['size']))

    run_path = aic_out_path + "/run.sh"
    if os.path.exists(run_path):
        os.remove(run_path)
    with os.fdopen(os.open(run_path, os.O_WRONLY | os.O_CREAT, 0o500), 'w') as f:
        f.write("cd " + aic_out_path + "\n")
        f.write("export DVCSPEC_DIR=" + aic_model_path + "\n")
        f.write(aic_model_path + "/v100_ca_tag_master --gtest_filter=test_st_case.test_st_ca\n")
    subprocess.call(["sh", aic_out_path + "/run.sh"])
    out_list = []
    for i, arg_ in enumerate(args):
        if arg_info[i]['out']:
            out_data = np.fromfile(os.path.join(aic_out_path, arg_info[i]['bin']), arg_.dtype)
            if out_data.size > args[i].size:  # strip unneeded data copied back by aic model
                out_data = out_data[0:arg_.size]
            out_arg = out_data.reshape(arg_.shape)
            out_list.append(out_arg)
    return out_list[0] if len(out_list) == 1 else tuple(out_list)
