#!/usr/bin/env python3
# coding: utf-8
# Copyright 2022 Huawei Technologies Co., Ltd
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
    def __init__(self):
        self.ops = []
        self.input_name = []
        self.input = []
        self.core_num = 0
        self.output = []
        self.op_name = ''
        self.transpose_inputs = []

def pass_through_json_split(desc_d, skip_ops_list):
    """
    split merge_json to single graph json
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
    
    input_tensor_names = [tensor[0]['tensor_name'] for tensor in desc_d['input_desc']] if desc_d['input_desc'] else []
    output_tensor_names = [tensor['tensor_name'] for tensor in desc_d['output_desc']] if desc_d['output_desc'] else []

    input_tensor_shapes = [tensor[0]['shape'] for tensor in desc_d['input_desc']] if desc_d['input_desc'] else []
    output_tensor_shapes = [tensor['shape'] for tensor in desc_d['output_desc']] if desc_d['output_desc'] else []
    
    sub_graphs = []
    for i in range(len(desc_d['op_desc'])):
        op_info = desc_d['op_desc'][i]
        op_name = op_info['name']
        if(op_name in skip_ops_list):
            continue
        g = Graph()
        g.ops.append(op_info)
        g.input = [t for t in op_info['input_desc'] if 'value' not in t[0].keys()]
        # g.input = op_info['input_desc']
        g.output = op_info['output_desc']
        g.op_name = op_name
        
        sub_graphs.append(g)
    
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
        # op_json_str['core_num'] = g.core_num
        op_json_str['platform'] = "AKG"
        op_json_str['process'] = desc_d['process']

        op_result.append(op_json_str)
    
    # all sub json info saved in op_jsons list
    for _, single_op in enumerate(op_result):
        json_str = json.dumps(single_op, indent=4)
        op_jsons.append(json_str)

    return op_jsons, input_tensor_names, output_tensor_names, input_tensor_shapes, output_tensor_shapes

def concat_json_split(desc_d):
    shapes_list = []
    axis = -1
    input_num = -1
    for op in desc_d['op_desc']:
        if op['name'] == 'Concat':
            for input in op['input_desc'][0]:
                shapes_list.append(input['shape'])
            for attr in op['attr']:
                if attr['name'] == 'axis':
                    axis = attr['value']
                elif attr['name'] == 'inputNums':
                    input_num = attr['value']
            break
    
    if(input_num != len(shapes_list)):
        print('wrong inputNums!')
        return None
    
    dim = len(shapes_list[0])
    for shape in shapes_list:
        if(len(shape) != dim):
            print('Concat wrong shapes!')
            return None
    
    if axis < 0 or axis >= dim:
        print('Concat wrong axis!')
        return None
    
    squeeze_shapes_list = []
    for shape in shapes_list:
        first_dim = 1
        for i in range(0, axis):
            first_dim *= shape[i]
        second_dim = shape[axis]
        third_dim = 1
        for i in range(axis+1, len(shape)):
            third_dim *= shape[i]
        squeeze_shapes_list.append([first_dim, second_dim, third_dim])
    
    first_dim, _, third_dim = squeeze_shapes_list[0]
    for shape in squeeze_shapes_list:
        if shape[0] != first_dim or shape[2] != third_dim:
            print('Concat wrong shapes!')
            return None
    
    return squeeze_shapes_list