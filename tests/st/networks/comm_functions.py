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
import time
import logging
import pandas as pd
logging.basicConfig(level=logging.INFO)

save_file = "profiling_base.csv"

def uniform_units(data):
    columns = ['Avg', 'Min', 'Max']
    data_new = data.drop(index=0, inplace=False)
    for col in columns:
        cur_unit = data[col][0]
        if cur_unit == 's':
            data_new[col] = data_new[col].apply(lambda x: float(x) * 1000000)
        elif cur_unit == 'ms':
            data_new[col] = data_new[col].apply(lambda x: float(x) * 1000)
    return data_new

def data_cleaning(data, network, level):
    data = uniform_units(data)
    data = data[data['Type'].str.contains('GPU activities')]
    data = data[data['Name'].str.contains('Fused_')]
    data['Network'] = [network] * len(data['Name'])
    data['Level'] = [level] * len(data['Name'])
    col_names = list(data.columns)
    for i in range(len(col_names)):
        if col_names[i] in ['Avg', 'Max', 'Min']:
            col_names[i] = col_names[i] + '(us)'
    data.columns = col_names
    valid_cols = ['Network', 'Level', 'Name', 'Calls', 'Avg(us)', 'Max(us)', 'Min(us)']
    return data[valid_cols]

def get_output_files(home_dir):
    file_lists = []
    def get_all_files(out_dir):
        files = os.listdir(out_dir)
        for item in files:
            if item.startswith('.'):
                continue
            file_dir = os.path.join(out_dir, item)
            if os.path.isdir(file_dir):
                get_all_files(file_dir)
            elif os.path.isfile(file_dir):
                file_lists.append(file_dir)
    output_dir = os.path.join(home_dir, "output_gpu")
    get_all_files(output_dir)
    return file_lists

def find_op_profiling(csv_file, network, level, op_name):
    reader = pd.read_csv(csv_file, header=3, error_bad_lines=False)
    data = pd.DataFrame(reader)
    data = data_cleaning(data, network, level)

    if len(data['Name']) <= 0:
        raise ValueError("Please check the profiling file %s.csv of %s network." % (op_name, os.path.dirname(file).split('/')[-2]))
    return data

def process_results(directory):
    """
    Input:
        directory: the directory of 'output_gpu', which contains csv files for all networks fused operators.
    Return:
        the result data of all operators in directory of 'output_gpu'.
    """
    csv_files = get_output_files(directory)
    results = pd.DataFrame()
    logging.info("total numbers of csv files: %s", len(csv_files))
    for item in csv_files:
        op_name = os.path.basename(item).split('.')[0]
        network = os.path.dirname(item).split('/')[-2]
        test_level = os.path.dirname(item).split('/')[-1]
        with open(item, 'rb') as f:
            lines = f.readlines()
            if len(lines) < 5:
                logging.error("Have error in file: %s", item)
                info = 'Failed'
                s = pd.Series({'Network': network, 'Level': test_level, 'Name': op_name, 'Calls': info,
                    'Avg(us)': info, 'Max(us)': info, 'Min(us)': info})
                results = results.append(s, ignore_index=True)
                continue
        results = results.append(find_op_profiling(item, network, test_level, op_name), ignore_index=True)
    time_suffix = time.strftime("%Y%m%d", time.localtime())
    for col in list(results.columns):
        if col in ['Calls', 'Avg(us)', 'Max(us)', 'Min(us)']:
            results.rename(columns={col: col + '_' + time_suffix}, inplace=True)
    return results

def delete_old_data(data):
    columns = list(data.columns)
    save_count = 10
    if ((len(columns) - 3) // 4) < save_count: # Save only the last 'save_count' times.
        return data
    new_columns = columns[0:3] + columns[7:]
    return data[new_columns]

def update_data(data, csv_file):
    if not os.path.isfile(csv_file):
        return
    old_data = delete_old_data(pd.read_csv(csv_file))
    new_data = pd.merge(old_data, data, on=['Network', 'Level', 'Name'], suffixes=['_0', '_1'], how='outer', sort=False)
    new_data.to_csv(csv_file, index=False)

def save_data(directory):
    data = process_results(directory)
    results_csv = os.path.join(directory, save_file)
    if os.path.isfile(results_csv):
        update_data(data, results_csv)
    else:
        data.to_csv(results_csv, index=False)

def compare_base_line(pwd, new_csv, network, level, op_name):
    """
    Retuns: 
        False, if the new performance slower than the baseline and exceeds the threshold. Otherwise, return True
        For example: 
            new data is 18us, the baseline is 9us(less than 10us, so thresholds is 1),
            the ratio of degradation is ((18 - 9) / 9) >= 1,
            we can't tolerate this performance degradation, return 'False'.
    """
    keys = ["2", "5", "10", "100", "default"]
    values = [5.0, 2.0, 1.0, 0.5, 0.3]
    def get_threshold(data):
        threshold = 0
        for k,v in zip(keys, values):
            if k == "default":
                break
            if data < float(k):
                threshold = v
                break
        return threshold if threshold != 0 else values[-1]
    new = find_op_profiling(new_csv, network, level, op_name)
    kernel_name = new['Name'].iloc[0]
    base_line = pd.read_csv(os.path.join(pwd, save_file))
    query = base_line[base_line['Network'] == network]
    query = query[query['Level'] == level]
    query = query[query['Name'] == kernel_name]
    if len(query['Name']) == 1 and len(new['Name']) == 1:
        columns = list(query.columns)
        base_perf = float(query[columns[-3]].iloc[0])
        new_pref = float(new['Avg(us)'].iloc[0])
        if new_pref <= base_perf:
            return True
        ratio = (new_pref - base_perf) / base_perf
        if ratio < get_threshold(base_perf):
            return True
        else:
            logging.warning("-----------------------------------")
            logging.warning("new performance:")
            logging.warning(new[['Avg(us)', 'Max(us)', 'Min(us)']])
            logging.warning("-----------------------------------")
            logging.warning("base performance:")
            logging.warning(query[columns[-3:]])
            logging.warning("-----------------------------------")
            return False
    else:
        logging.info("baseline performance was not found!")
        return True