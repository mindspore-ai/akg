#!/usr/bin/env python3
# coding: utf-8
# Copyright 2025 Huawei Technologies Co., Ltd
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

import sys
import numpy as np
import argparse

def myprint(a):
    len = a.shape[0]
    for i in range(0, int(len/32)):
        print(a[i*32:min((i+1)*32, len)])


def verify_result(args=None, output=None, golden=None):
    if args is None or (not args.int_list):
        relative_tol = 1e-2
        absolute_tol = 1e-2
        error_tol = 4e-3
    else:
        relative_tol = args.int_list[0]
        absolute_tol = args.int_list[1]
        error_tol = args.int_list[2]
    if output is None:
        output = np.fromfile(args.actual_file, dtype=args.dataType).reshape(-1)
    if golden is None:
        golden = np.fromfile(args.golden_file, dtype=args.dataType).reshape(-1)
    different_element_results = np.isclose(output,
                                           golden,
                                           rtol=relative_tol,
                                           atol=absolute_tol,
                                           equal_nan=True)
    different_element_indexes = np.where(different_element_results == False)[0]
    print("total different numbers: ", len(different_element_indexes))
    for index in range(len(different_element_indexes)):
        real_index = different_element_indexes[index]
        golden_data = golden[real_index]
        output_data = output[real_index]
        print(
            "data index: %06d, expected: %-.9f, actual: %-.9f, rdiff: %-.6f" %
            (real_index, golden_data, output_data,
             abs(output_data - golden_data) / golden_data))
        if index == 100:
            break
    error_ratio = float(different_element_indexes.size) / golden.size
    print("error ratio: %.4f, tolrence: %.4f" % (error_ratio, error_tol))
    return error_ratio <= error_tol

def parse_arguments():
    parser = argparse.ArgumentParser()
    # 定义前3个必选的字符串参数
    parser.add_argument('actual_file', type=str)
    parser.add_argument('golden_file', type=str)
    parser.add_argument('dataType', nargs='?', type=str, default='float16')
    parser.add_argument('int_list', nargs='*', type=float)
    return parser.parse_args()

if __name__ == '__main__':
    try:
        args = parse_arguments()
        res = verify_result(args)
        if not res:
            raise ValueError("[ERROR] result error")
        else:
            print("test pass")
    except Exception as e:
        print(e)
        sys.exit(1)
