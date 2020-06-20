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
import sys
import logging
import json
from akg.ms import compilewithjson
logging.getLogger().setLevel(logging.INFO)

def completeJson(dict_json_origin, inputs, attr):
    dict_json = dict_json_origin
    if len(inputs) == 1:
        dict_json["input_desc"] = inputs
    else:
        list_list = []
        for i in range(len(inputs)):
            list_list.append([inputs[i]])
        dict_json["input_desc"] = list_list
    dict_json["attr"] = attr
    dict_json["op"] = dict_json["name"] + "_test"
    input_index = 0
    for input_desc_elem in dict_json["input_desc"]:
        dict_json["input_desc"][input_index][0]["tensor_name"] = "input_" + dict_json["input_desc"][input_index][0]["name"]
        input_index += 1
    return dict_json

def cce_run(run_func, *args, **kwargs):
    module = args[-1]['mod']
    if module == None:
        logging.info("build module failed")
        sys.exit(1)
    else:
        input, output, expect, runres = run_func(*args, **kwargs)
        if not runres:
            logging.info("test result wrong")
            sys.exit(1)
        else:
            logging.info("result right")
