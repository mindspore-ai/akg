# Copyright 2021 Huawei Technologies Co., Ltd
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
import os
import sys
import time
import pytest
import logging
import argparse

CPU = "cpu"
GPU = "gpu"
ASCEND = "ascend"
FLAG_MAP = {
    CPU: "-m=platform_x86_cpu",
    GPU: "-m=platform_x86_gpu_training",
    ASCEND: "-m=platform_x86_ascend_training"
}
NETWORKS_TEST = "networks/test_network.py"

def get_files(dir):
    res = {}
    pwd = os.getcwd()
    for root, _, files in os.walk(dir):
        for f in files:
            if f.startswith("test_") and f.endswith(".py"):
                op_name = f.split("test_")[-1].split(".py")[0]
                res[op_name] = os.path.join(os.path.relpath(root, pwd), f)
    return res

def run_case(dir):
    parser = argparse.ArgumentParser(description='Run cases in st.')
    parser.add_argument("-e", "--environment", choices=['cpu', 'gpu', 'ascend'], type=str,
                        required=True, default="gpu", help="Hardware environment: cpu, gpu or ascend.")
    parser.add_argument("-o", "--op", type=str, default="all", nargs='+',
                        help="List of operators to test, default is all.")
    parser.add_argument("-l", "--level", type=str, default="all",
                        help="The level of cases: all, level0, level1, level2 ..., default is all.")
    parser.add_argument("-p", "--log_path", type=str, default="report",
                        help="The path directory of result.")
    parser.add_argument("-d", "--device_id", type=str, default="0", nargs='+',
                        help="The device ID of ascend or gpu performed by cases.")
    parser.add_argument("-n", "--network", action='store_true',
                        help="Run cases of network.")
    parser.add_argument("-v", "--verbose", action='store_true',
                        help="Whether to print IR and generated code.")
    parser.add_argument("-s", "--stop", action='store_true',
                        help="Whether to stop run others cases when an error occurs.")
    args = parser.parse_args()

    if args.verbose:
        os.environ["MS_DEV_DUMP_CODE"] = "on"
        os.environ["MS_DEV_DUMP_IR"] = "on"
    else:
        os.unsetenv("MS_DEV_DUMP_CODE")
        os.unsetenv("MS_DEV_DUMP_IR")

    devices = ",".join(args.device_id)
    if args.environment == GPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = devices
    elif args.environment == ASCEND:
        os.environ["DEVICE_ID"] = devices

    configures = ["-svq", "--disable-warnings"]
    if args.log_path:
        try:
            import imp
            imp.find_module("pytest_html")
            log_file = time.strftime("%Y%m%d_%H%M%S", time.localtime()) + ".html"
            log_path = "--html=" + args.log_path + "/" + log_file
            configures.append(log_path)
        except ImportError:
            pass
    
    if args.stop:
        configures.append("-x")
    banckend_flag = FLAG_MAP[args.environment]
    if args.level != "all":
        banckend_flag = banckend_flag + " and " + args.level
    configures.append(banckend_flag)

    if args.network:
        configures.append(NETWORKS_TEST)
    elif isinstance(args.op, str):
        configures.append("./")
    elif len(args.op) == 1 and args.op[0] == "all":
        configures.append("./")
    else:
        op_map = get_files(dir)
        for op in args.op:
            op_file = op_map.get(op, None)
            if op_file is None:
                logging.warning("Can't find %s op files!" % op)
                continue
            configures.append(op_file)

    pytest.main(configures)

if __name__ == '__main__':
    pwd = os.path.dirname(os.path.abspath(__file__))
    run_case(pwd)