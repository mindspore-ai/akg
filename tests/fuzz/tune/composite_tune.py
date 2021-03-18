# Copyright 2020 Huawei Technologies Co., Ltd
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

"""test composite json tuning"""
import os
import sys
import time
import getopt
import logging
from tests.fuzz.tune.autotuning.job import launch_json

logging.getLogger().setLevel(logging.WARNING)


def print_usage():
    logging.warning("Usage: python composite_tune.py <kernel_meta_path or info_file> [-f] -o <repo.json>")
    logging.warning("\nOptions:")
    logging.warning("<kernel_meta_path or info_file>")
    logging.warning("  kernel_meta directory or single file name, '*.info' or '*.json'.")
    logging.warning("-f")
    logging.warning("  tune for all jsons, including those already in repository")
    logging.warning("-o <repo.json>")
    logging.warning("  relative path of output repository")


def run_json_tuning(input_str, skip_exist, repo_path):
    time_start = time.time()
    debug_mode_ = False
    save_res_ = True
    all_space_ = False
    skip_file_ = False
    extra_tune_ = False
    self_attrs = None
    tuning_attrs = ['enable_pre_poly_loop_partition',
                    'enable_post_poly_loop_partition',
                    'enable_rewrite_scalar_compute',
                    'multicore_scalar_rearrange',
                    ]
    launch_json(debug_mode=debug_mode_,
                save_res=save_res_,
                input_str=input_str,
                repo_path=repo_path,
                all_space=all_space_,
                skip_exist=skip_exist,
                skip_file=skip_file_,
                extra_tune=extra_tune_,
                self_attrs=self_attrs,
                tuning_attrs=tuning_attrs)
    time_end = time.time()
    logging.debug("launch time: %f", time_end - time_start)


if __name__ == "__main__":
    try:
        if sys.argv[1] in ("-h", "--help"):
            sys.exit()
        input_str = sys.argv[1]
        if not os.path.isdir(input_str) and not os.path.isfile(input_str):
            logging.warning("ERROR: %s is not a directory or file.", input_str)
            sys.exit()
        options, args = getopt.getopt(sys.argv[2:], "-f-o:")
        skip_exist = True
        repo_path = None
        for option, value in options:
            if option == "-f":
                skip_exist = False
            elif option == "-o":
                repo_path = value
        if not repo_path:
            logging.warning("ERROR: Empty repository path.")
            sys.exit()
    except:
        print_usage()
        sys.exit()

    run_json_tuning(input_str, skip_exist, repo_path)
