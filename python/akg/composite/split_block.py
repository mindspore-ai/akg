#!/usr/bin/env python3
# coding: utf-8
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

import json

class Graph():
    def __init__(self, output):
        self.tensors = set(output)
        self.ops = []
        self.output_name = output
        self.input_name = []
        self.input = []
        self.core_num = 0
        self.output = []
        self.op_name = 'Fused'

def parallel_json_split(desc_d):
    """
    spilt merge_json to single graph json.
    Args:
        desc_d : dict of compute desciption
    Returns:
        List of subgraph json.
        List of input names.
        Dict of output names.
    """
    op_jsons = []

    # get some basic info to init subgraph
    composite_graph_id = desc_d['composite_graph']
    composite_id = desc_d['id']
    final_output_name = desc_d['parallel_fusion']['sub_graph']
    sub_graphs = []
    for i in range(len(final_output_name)):
        sub_graphs.append(Graph(final_output_name[i]))

    # traversal desc_d by reverse topological order to construct subgraph
    for i in range(len(desc_d['op_desc']) - 1, -1, -1):
        op_info = desc_d['op_desc'][i]
        for g in sub_graphs:
            for j in range(len(op_info['output_desc'])):
                if op_info['output_desc'][j]['tensor_name'] in g.tensors:
                    g.ops.append(op_info)
                    for input_info in op_info['input_desc']:
                        for sub_input_info in input_info:
                            g.tensors.add(sub_input_info['tensor_name'])

    # get subgraph original input
    if desc_d['input_desc']:
        for op_input in desc_d['input_desc']:
            for g in sub_graphs:
                if op_input[0]['tensor_name'] in g.tensors:
                    g.input.append(op_input)

    # get subgraph original output
    for op_output in desc_d['output_desc']:
        for g in sub_graphs:
            if op_output['tensor_name'] in g.tensors:
                g.output.append(op_output)

    # get subgraph core num info
    core_num_info = desc_d['parallel_fusion']['core_num']
    for idx in range(len(sub_graphs)):
        g = sub_graphs[idx]
        g.core_num = core_num_info[idx]

    # reverse ops order to generate a topology order subgraph
    for g in sub_graphs:
        g.ops = list(reversed(g.ops))
        g.op_name = desc_d['op']

    # get the original input of all subgraphs in order
    # suppose all original json input_args info satisfies this order
    input_tensor_names = [tensor[0]['tensor_name'] for tensor in desc_d['input_desc']] if desc_d['input_desc'] else []
    output_tensor_names = [tensor['tensor_name'] for tensor in desc_d['output_desc']] if desc_d['output_desc'] else []

    # construct subgraph json info
    op_result = []
    for g in sub_graphs:
        op_json_str = {}
        op_json_str['composite'] = True
        op_json_str['composite_graph'] = composite_graph_id
        op_json_str['id'] = composite_id
        op_json_str['op'] = g.op_name
        op_json_str['input_desc'] = g.input
        op_json_str['op_desc'] = g.ops
        op_json_str['output_desc'] = g.output
        op_json_str['core_num'] = g.core_num
        op_json_str['platform'] = "AKG"
        op_json_str['process'] = desc_d['process']
        op_result.append(op_json_str)

    # all sub json info saved in op_jsons list
    for idx in range(len(op_result)):
        single_op = op_result[idx]
        json_str = json.dumps(single_op, indent=4)
        op_jsons.append(json_str)
    return op_jsons, input_tensor_names, output_tensor_names
