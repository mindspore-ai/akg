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

"""parsing_profiling_data"""
import os
import struct
import re
import logging
import itertools
from akg import tvm

OUTPUT_FORMAT_DATA = "./output_format_data_hwts.txt"
max_time_consume = 9999999999


def get_log_slice_id(file_name):
    pattern = re.compile(r'(?<=slice_)\d+')
    slice_ = pattern.findall(file_name)
    index = re.findall(r'\d+', slice_[0])
    return int(index[0])


def get_file_join_name(input_path=None, file_name=None):
    """Function for getting join name from input path."""
    name_list = []
    file_join_name = ''
    if os.path.exists(input_path):
        files = os.listdir(input_path)
        for f in files:
            if file_name in f and not f.endswith('.done') and not f.endswith('.join'):
                name_list.append(f)
        # resort name_list
        name_list.sort(key=get_log_slice_id)

    if len(name_list) == 1:
        file_join_name = input_path + os.sep + name_list[0]
    elif len(name_list) > 1:
        file_join_name = input_path + os.sep + '%s.join' % file_name
        if os.path.exists(file_join_name):
            os.remove(file_join_name)
        with open(file_join_name, 'ab') as bin_data:
            for i in name_list:
                file = input_path + os.sep + i
                with open(file, 'rb') as txt:
                    bin_data.write(txt.read())
    return file_join_name


def fwrite_format(output_data_path=OUTPUT_FORMAT_DATA, data_source=None, is_start=False):
    if is_start and os.path.exists(OUTPUT_FORMAT_DATA):
        os.remove(OUTPUT_FORMAT_DATA)
    with open(output_data_path, 'a+') as f:
        if isinstance(data_source, (list, tuple)):
            for raw_data in data_source:
                if isinstance(raw_data, (list, tuple)):
                    raw_data = map(str, raw_data)
                    raw_data = " ".join(raw_data)
                f.write(raw_data)
                f.write("\n")
        else:
            f.write(data_source)
            f.write("\n")


def validate_and_normalize_path(
        path,
        check_absolute_path=False,
        allow_parent_dir=True,
):
    """
    Validates path and returns its normalized form.

    If path has a valid scheme, treat path as url, otherwise consider path a
    unix local path.

    Note:
        File scheme (rfc8089) is currently not supported.

    Args:
        path (str): Path to be normalized.
        check_absolute_path (bool): Whether check path scheme is supported.
        allow_parent_dir (bool): Whether allow parent dir in path.

    Returns:
        str, normalized path.
    """
    if not path:
        raise RuntimeError("The path is invalid!")

    path_str = str(path)
    if not allow_parent_dir:
        path_components = path_str.split("/")
        if ".." in path_components:
            raise RuntimeError("The parent path is not allowed!")

    # path does not have valid schema, treat it as unix local path.
    if check_absolute_path:
        if not path_str.startswith("/"):
            raise RuntimeError("The path is invalid!")
    try:
        # most unix systems allow
        normalized_path = os.path.realpath(path)
    except ValueError:
        raise RuntimeError("The path is invalid!")

    return normalized_path


class HWTSLogParser:
    """
    The Parser for hwts log files.

    Args:
         input_path (str): The profiling job path. Such as: '/var/log/npu/profiling/JOBAIFGJEJFEDCBAEADIFJAAAAAAAAAA".
         output_filename (str): The output data path and name. Such as: './output_format_data_hwts_0.txt'.
    """

    _source_file_target_old = 'hwts.log.data.45.dev.profiler_default_tag'
    _source_file_target = 'hwts.data'
    _dst_file_title = 'title:45 HWTS data'
    _dst_file_column_title = 'Type           cnt  Core_ID  Block_ID  Task_ID  Cycle_counter   Stream_ID'

    def __init__(self, input_path, output_filename=None, is_print=False):
        self._input_path = input_path
        self._output_filename = output_filename
        self._source_flie_name = self._get_source_file()
        self._is_print = is_print

    def _get_source_file(self):
        """Get hwts log file name, which was created by ada service."""
        input_paths = [self._input_path, os.path.join(self._input_path, "data")]
        file_targets = [self._source_file_target, self._source_file_target_old]
        for path, target in itertools.product(input_paths, file_targets):
            file_name = get_file_join_name(path, target)
            if file_name:
                return file_name
        msg = "Fail to find hwts log file, under profiling directory"
        raise RuntimeError(msg)

    @staticmethod
    def _parse_struct(ms_type, line, is_warn_res0_ov):
        content_format = ['QIIIIIIIIIIII', 'QIIQIIIIIIII', 'IIIIQIIIIIIII']
        stream_id = None
        syscnt = None
        if ms_type in ['000', '001', '010']:  # log type 0,1,2
            result = struct.unpack(content_format[0], line[8:])
            syscnt = result[0]
            stream_id = result[1]
        elif ms_type == '011':  # log type 3
            result = struct.unpack(content_format[1], line[8:])
            syscnt = result[0]
            stream_id = result[1]
        elif ms_type == '100':  # log type 4
            result = struct.unpack(content_format[2], line[8:])
            stream_id = result[2]
            if is_warn_res0_ov == '0':
                syscnt = result[4]
        return stream_id, syscnt

    def execute(self):
        """
        Execute the parser, get result data, and write it to the output file.

        Returns:
            bool, whether succeed to analyse hwts log.
        """

        log_type = ['Start of task', 'End of task', 'Start of block', 'End of block', 'Block PMU']

        result_data = ""

        self._source_flie_name = validate_and_normalize_path(self._source_flie_name)
        last_syscnt = 0
        cycles = 0

        kernel_label = tvm.get_global_func("ascend_get_kernel_label")()
        with open(self._source_flie_name, 'rb') as hwts_data:
            while True:
                # read 64 bit data
                line = hwts_data.read(64)
                if line:
                    if not line.strip():
                        continue
                else:
                    break
                byte_first_four = struct.unpack('BBHHH', line[0:8])
                # byte_first[0:4] refers to count. byte_first[4] refers to is_warn_res0_0v.
                # byte_first[5:8] refers to the type of ms.
                byte_first = bin(byte_first_four[0]).replace('0b', '').zfill(8)
                ms_type = byte_first[-3:]
                is_warn_res0_ov = byte_first[4]
                cnt = int(byte_first[0:4], 2)
                core_id = byte_first_four[1]
                blk_id, task_id = byte_first_four[3], byte_first_four[4]
                stream_id, syscnt = self._parse_struct(ms_type, line, is_warn_res0_ov)
                if stream_id is None:
                    logging.info("Profiling: invalid hwts log record type %s", ms_type)
                    continue
                if int(task_id) < 25000:
                    task_id = str(task_id)
                if kernel_label == (str(stream_id) + '_' + str(task_id)):
                    if log_type[int(ms_type, 2)] == "Start of task":
                        last_syscnt = syscnt
                    elif log_type[int(ms_type, 2)] == "End of task":
                        cycles += syscnt - last_syscnt

                if self._is_print:
                    result_data += ("%-14s %-4s %-8s %-9s %-8s %-15s %s\n" % (log_type[int(ms_type, 2)], cnt, core_id,
                                                                              blk_id, task_id, syscnt, stream_id))

        if self._is_print:
            fwrite_format(self._output_filename, data_source=self._dst_file_title, is_start=True)
            fwrite_format(self._output_filename, data_source=self._dst_file_column_title)
            fwrite_format(self._output_filename, data_source=result_data)

        return cycles if cycles != 0 else max_time_consume
