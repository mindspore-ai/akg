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
import subprocess
import struct
import re
from tabulate import tabulate

OUTPUT_FORMAT_DATA = "./output_format_data_hwts.txt"
BLOCK_LEN = 32
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


def get_first_runtime_task_trace(input_file=None):
    """Function for getting first task trace from runtime."""
    result_data = []
    format_ = "BBHIQHHHHII"
    format_last = "B"
    with open(input_file, 'rb') as bin_data:
        while True:
            line_ = bin_data.read(96)
            if line_:
                if not line_.strip():
                    continue
            else:
                break
            if len(line_) == 96:
                unpack_tuple = struct.unpack(format_, line_[0:32])
                char_string = line_[32:95].decode().strip(b'\x00'.decode())
                result_last = [hex(i) for i in struct.unpack(format_last, line_[95:96])]
                byte01 = bin(int(result_last[0].replace('0x', ''), 16)).replace('0b', '').zfill(8)
                persistant_1bit = byte01[-1]
                reserved_7bit = byte01[0:7]
                kernelname = char_string
                result_data.append((unpack_tuple[0], unpack_tuple[1], unpack_tuple[2], unpack_tuple[3],
                                    unpack_tuple[4], unpack_tuple[5], unpack_tuple[6], unpack_tuple[7],
                                    unpack_tuple[8], unpack_tuple[9], unpack_tuple[10],
                                    kernelname, persistant_1bit, reserved_7bit))
    return result_data


def get_44_tsch_fw_timeline(input_file=None):
    """Function for getting tsch_fw_timeline from input file."""
    result_data = []
    format_ = "BBHIHHHHQII"
    with open(input_file, 'rb') as bin_data:
        while True:
            line_ = bin_data.read(32)
            if line_:
                if not line_.strip():
                    continue
            else:
                break
            if len(line_) == 32:
                result_ = struct.unpack(format_, line_)
                result_data.append((result_[0], result_[1], result_[2], result_[3], result_[4], result_[5], result_[6],
                                    result_[7], result_[8], result_[9], result_[10]))
    return result_data


def get_43_ai_core_data(input_file=None):
    """Function for getting datas from aicore: ov/cnt/total_cyc/ov_cyc/pmu_cnt/stream_id."""
    result_data = []
    with open(input_file, 'rb') as ai_core_file:
        while True:
            line_ = ai_core_file.read(128)
            if line_:
                if not line_.strip():
                    continue
            else:
                break
            format_ = "BBHHHIIqqqqqqqqqqIIIIIIII"
            result_ = [hex(i) for i in struct.unpack(format_, line_)]
            byte01 = bin(int(result_[0].replace('0x', ''), 16)).replace('0b', '').zfill(8)
            ov = byte01[-4]
            cnt = byte01[0:4]
            total_cyc = int(result_[7].replace('0x', ''), 16)
            ov_cyc = int(result_[8].replace('0x', ''), 16)
            pmu_cnt = tuple(int(i.replace('0x', ''), 16) for i in result_[9:17])
            stream_id = int(result_[17].replace('0x', ''), 16)
            result_data.append((ov, cnt, total_cyc, ov_cyc, stream_id, pmu_cnt))
    return result_data


def get_last_tsch_training_trace(input_file=None):
    """Function for getting last tsch training trace from input file."""
    result_data = []
    format_ = "LLHHLL"
    with open(input_file, 'rb') as bin_data:
        while True:
            line_ = bin_data.read(20)
            if line_:
                if not line_.strip():
                    continue
            else:
                break
            if len(line_) == 20:
                result_ = struct.unpack(format_, line_)
                result_data.append((result_[0], result_[1], result_[3], result_[2], result_[4], result_[5]))
    return result_data


def get_45_hwts_log(input_file=None):
    """Function for getting hwts log from input file."""
    format_ = ['QIIIIIIIIIIII', 'QIIQIIIIIIII', 'IIIIQIIIIIIII']
    log_type = ['Start of task', 'End of task', 'Start of block', 'End of block', 'Block PMU']
    type1, type2, type3 = [], [], []
    with open(input_file, 'rb') as hwts_data:
        while True:
            line_ = hwts_data.read(64)
            if line_:
                if not line_.strip():
                    continue
            else:
                break
            byte_first_four = struct.unpack('BBHHH', line_[0:8])
            byte_first = bin(byte_first_four[0]).replace('0b', '').zfill(8)
            type_ = byte_first[-3:]
            is_warn_res0_ov = byte_first[4]
            cnt = int(byte_first[0:4], 2)
            core_id = byte_first_four[1]
            blk_id, task_id = byte_first_four[3], byte_first_four[4]
            if type_ in ['000', '001', '010']:  # log type 0,1,2
                result_ = struct.unpack(format_[0], line_[8:])
                syscnt = result_[0]
                stream_id = result_[1]
                type1.append((log_type[int(type_, 2)], cnt, core_id, blk_id, task_id, syscnt, stream_id))

            elif type_ == '011':  # log type 3
                result_ = struct.unpack(format_[1], line_[8:])
                syscnt = result_[0]
                stream_id = result_[1]
                if is_warn_res0_ov == '1':
                    warn_status = result_[3]
                else:
                    warn_status = None
                type2.append(
                    (log_type[int(type_, 2)], cnt, is_warn_res0_ov, core_id, blk_id, task_id, syscnt, stream_id,
                     warn_status))
                type1.append((log_type[int(type_, 2)], cnt, core_id, blk_id, task_id, syscnt, stream_id))
            elif type_ == '100':  # log type 4
                result_ = struct.unpack(format_[2], line_[8:])
                stream_id = result_[2]
                if is_warn_res0_ov == '0':
                    total_cyc = result_[4]
                    ov_cyc = None
                else:
                    total_cyc = None
                    ov_cyc = result_[4]
                pmu_events = result_[-8:]
                type3.append((log_type[int(type_, 2)], cnt, is_warn_res0_ov, core_id, blk_id, task_id, stream_id,
                              total_cyc, ov_cyc, pmu_events))
                type1.append((log_type[int(type_, 2)], cnt, core_id, blk_id, task_id, total_cyc, stream_id))

    return type1, type2, type3


def fwrite_format(output_data_path=OUTPUT_FORMAT_DATA, data_source=None, is_start=False):
    if is_start and os.path.exists(OUTPUT_FORMAT_DATA):
        os.remove(OUTPUT_FORMAT_DATA)
    with open(output_data_path, 'a+') as f:
        f.write(data_source)
        f.write("\n")


def parsing(source_path):
    """Function for parsing aicore data/tsch fw timeline data/HWTS data/last tsch training trace data."""
    # subprocess.run("cp -r %s ./jobs/" % source_path, shell=True)
    job_name = source_path.split('/')[-1]
    job_path = "/var/log/npu/profiling/" + job_name
    fwrite_format(data_source='====================starting  parse task ==================', is_start=True)
    result = get_file_join_name(input_path=job_path, file_name='runtime.host.runtime')
    if result:
        runtime_task_trace_data = get_first_runtime_task_trace(input_file=result)
        fwrite_format(data_source='====================first runtime task trace data==================')
        fwrite_format(data_source=tabulate(runtime_task_trace_data,
                                           ['mode', 'rpttype', 'bufsize', 'reserved', 'timestamp', 'eventname',
                                            'tasktype', 'streamid',
                                            'task_id', 'thread', 'device_id', 'kernelname', 'persistant_1bit',
                                            'reserved_7bit'],
                                           tablefmt='simple'))
    result = get_file_join_name(input_path=job_path, file_name='aicore.data.43.dev.profiler_default_tag')
    if result:
        ai_core_data = get_43_ai_core_data(input_file=result)
        fwrite_format(data_source='============================43 AI core data =========================')
        fwrite_format(data_source=tabulate(ai_core_data,
                                           ['Overflow', 'cnt', 'Total cycles', 'overflowed cycles', 'Stream ID',
                                            'PMU events'],
                                           tablefmt='simple'))
    result = get_file_join_name(input_path=job_path, file_name='ts_track.data.44.dev.profiler_default_tag')
    if result:
        tsch_fw_timeline_data = get_44_tsch_fw_timeline(input_file=result)
        fwrite_format(data_source='============================44 tsch fw timeline  data =========================')
        fwrite_format(data_source=tabulate(tsch_fw_timeline_data,
                                           ['mode', 'rptType', 'bufSize', 'reserved', 'task_type', 'task_state',
                                            'stream_id',
                                            'task_id', 'timestamp', 'thread', 'device_id'], tablefmt='simple'))
    result = get_file_join_name(input_path=job_path, file_name='hwts.log.data.45.dev.profiler_default_tag')
    start_time = 0
    end_time = 0
    if result:
        data_1, data_2, data_3 = get_45_hwts_log(input_file=result)
        fwrite_format(data_source='============================45 HWTS data ============================')
        for i in data_1:
            if i[0] == 'Start of task' and i[4] == 60000 and start_time == 0:
                start_time = i[5]
            if i[0] == 'End of task' and i[4] == 60000 and end_time == 0:
                end_time = i[5]

        fwrite_format(data_source=tabulate(data_1,
                                           ['Type', 'cnt', 'Core ID', 'Block ID', 'Task ID', 'Cycle counter',
                                            'Stream ID'],
                                           tablefmt='simple'))
        fwrite_format(data_source=tabulate(data_2,
                                           ['Type', 'cnt', 'WARN', 'Core ID', 'Block ID', 'Task ID', 'Cycle counter',
                                            'Stream ID', 'WARN Status'],
                                           tablefmt='simple'))
        fwrite_format(data_source=tabulate(data_3,
                                           ['Type', 'cnt', 'Overflow', 'Core ID', 'Block ID', 'Task ID', 'Stream ID',
                                            'Total cycles',
                                            'Overflowed cycles',
                                            'PMU events'], tablefmt='simple'))

    result = get_file_join_name(input_path=job_path, file_name='training_trace.dev.profiler_default_tag')
    if result:
        tsch_training_trace_data = get_last_tsch_training_trace(input_file=result)
        fwrite_format(data_source='============================last tsch training_trace data=========================')
        fwrite_format(data_source=tabulate(tsch_training_trace_data,
                                           ['id_lo', 'id_hi', 'stream_id', 'task_id', 'syscnt_lo', 'syscnt_hi'],
                                           tablefmt='simple'))

    try:
        time_consume = abs(int(start_time) - int(end_time))
        return time_consume if time_consume != 0 else max_time_consume
    except SyntaxError:
        return max_time_consume
