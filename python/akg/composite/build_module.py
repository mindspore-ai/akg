#!/usr/bin/env python3
# coding: utf-8
# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

"""build module"""
import os
import json
from functools import reduce
import logging
import akg
from akg import tvm
from akg.tvm import _api_internal
from akg.topi.cuda.injective_single_kernel import schedule_injective
import topi
from akg.global_configs import get_dump_ir_flag


def should_enable_atomic_add(kernel_info):
    for op in kernel_info["op_desc"]:
        if not op["attr"]:
            continue
        for attr in op["attr"]:
            if attr["name"] == "enable_atomic_add" and attr["value"]:
                return True
    return False


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

class Liveness():
    def __init__(self):
        self.start = -1
        self.end = -1
        self.is_reduce = False
    def __str__(self):
        return "live_" + str(self.start) + "_" + str(self.end) + "_" + str(self.is_reduce)
    def __repr__(self):
        return "live_" + str(self.start) + "_" + str(self.end) + "_" + str(self.is_reduce)


def liveness_analysis(desc_d, req_map):
    req_liveness = dict((k, Liveness()) for k in req_map.keys())
    idx = len(desc_d['op_desc'])
    for i in range(len(desc_d['op_desc']) - 1, -1, -1):
        idx -= 1
        op_info = desc_d['op_desc'][i]
        for out_desc in op_info['output_desc']:
            out_name = out_desc['tensor_name']
            if out_name in req_liveness:
                if is_reduce(op_info['name']):
                    req_liveness[out_name].is_reduce = True   
                if req_liveness[out_name].end == -1:
                    req_liveness[out_name].end = idx
                    req_liveness[out_name].start = idx
                else:
                    req_liveness[out_name].start = idx
            for input_desc in op_info['input_desc']:
                for sub_input_desc in input_desc:
                    inp_name = sub_input_desc['tensor_name']
                    if inp_name in req_liveness and req_liveness[inp_name].end == -1:
                        req_liveness[inp_name].end = idx
                    if inp_name in req_liveness and req_liveness[inp_name].end > -1:
                        req_liveness[inp_name].start = idx
    # sort req_liveness by Liveness.end.
    sort_req_liveness = dict(sorted(req_liveness.items(), key=lambda x: x[1].end, reverse=True))
    return sort_req_liveness

def is_reduce(tensor_name):
    return tensor_name.startswith('Reduce')

def shared_memory_optimization(desc_d, req_map, outputs):
    sort_req_liveness = liveness_analysis(desc_d, req_map)
    sort_req_buf = list(sort_req_liveness.keys())
    alloc_map = dict()
    reuse_map = dict()
    reverse_reuse_map = dict()
    for i in range(len(sort_req_liveness)):
        reuse = False
        find_conflit = False
        ### TODO: the check is used due to the initialization clause position of reduce computation.
        if sort_req_liveness[sort_req_buf[i]].is_reduce:
            alloc_map[sort_req_buf[i]] = ['ALLOC', req_map[sort_req_buf[i]]]
            continue
        for j in range(len(sort_req_liveness) - 1, i, -1):
            # whether reuseable.
            # rule1: one buffer start larger equal to the reused buffer end.
            if sort_req_liveness[sort_req_buf[i]].start >= sort_req_liveness[sort_req_buf[j]].end:
                # rule2: sizes are compatiable.
                if req_map[sort_req_buf[i]] <= req_map[sort_req_buf[j]] and sort_req_buf[j] not in outputs:
                    # rule3: make sure the candidate reused buffer is not using by other conflict variable.
                    for item in reverse_reuse_map.get(sort_req_buf[j], []):
                        if (sort_req_liveness[item].end >= sort_req_liveness[sort_req_buf[i]].end) or (sort_req_liveness[item].end >= sort_req_liveness[sort_req_buf[i]].start):
                            find_conflit = True
                            break
                    if not find_conflit: 
                        if sort_req_buf[j] not in reverse_reuse_map:
                            reverse_reuse_map[sort_req_buf[j]] = [sort_req_buf[i]]
                        else:
                            reverse_reuse_map[sort_req_buf[j]].append(sort_req_buf[i])
                        # rule4: prefer to reuse buffer with same size.
                        if req_map[sort_req_buf[i]] == req_map[sort_req_buf[j]]:
                            reuse_map[sort_req_buf[i]] = [sort_req_buf[j], req_map[sort_req_buf[i]]]
                            reuse = True
                            break
                        else:
                            reuse_map[sort_req_buf[i]] = [sort_req_buf[j], req_map[sort_req_buf[i]]]
                            reuse = True
        if not reuse:
            alloc_map[sort_req_buf[i]] = ['ALLOC', req_map[sort_req_buf[i]]]
    return alloc_map, reuse_map

def is_tensor(op_info):
    return 'value' not in op_info


def parse_merged_json(desc_d, stitch_tensor_name, input_tensor_name, output_tensor_name):
    '''
    Parse merged json to get subgraph splitted by stitch nodes and input-output relationship of merged graph.

    Args:
        desc_d (dict): The dict of compute description.
        stitch_tensor_name (list[string]): The list of stitch node tensors.
            stitch nodes are regarded as edges of sub_graphs. The smallest number of sub_graph is the length of 
            stitch_tensor_name + 1.

        input_tensor_name (list[string]): The list of input tensors.
        output_tensor_name (list[string]): The list of output tensors.
            output tensors would be regarded as inter_output_tensor and final_output_tensor. The main difference
            of the two kinds of tensors is whether out-degree is zero, in which final_output_tensor is the tensor 
            with zero out-degree in merged graph and otherwise, it is inter_output_tensor.

    Returns:

        extra_subgraph_output (dict): The dict of extra output tensors for each sub_graph.
        final_output_list (list[string]): The list of final output tensors.
            output tensors in this list are are final_output_tensor and the subgraph they belong to doesn't
            include stitch nodes.
        final_output_within_graph (list[string]): The list of final output tensors.
            output tensors in this list are final_output_tensor and the subgraph they belong to also includes 
            stitch node.
            
    '''
    # Initialize sub_graph number as the smallest possible number of sub graph. 
    # sub graphs number might increase based on graph structure. 
    sub_graph_length = len(stitch_tensor_name)
    sub_graph_node = [set() for _ in range(sub_graph_length)]
    # use dict to save extra outputs for each sub_graph.
    extra_subgraph_output = dict(zip(stitch_tensor_name, [[] for _ in range(sub_graph_length)]))
    in_out_dict = {}
    inter_output_list = set()
    final_output_list = set()
    final_output_within_graph = []
    idx = 0
    final_output_graph = False
    for i in range(len(desc_d['op_desc']) - 1, -1, -1):
        op_info = desc_d['op_desc'][i]
        for out_desc in op_info['output_desc']:
            # switch to next subgraph if find stitch node.
            if out_desc['tensor_name'] in stitch_tensor_name:
                idx += 1
                cur_stitch_node = out_desc['tensor_name']
                # when current subgraph concludes final output and encounters with stitch node, increase number of subgraph.
                if final_output_graph:
                    final_output_list.add(cur_final_node)
                    final_output_within_graph.remove(cur_final_node)
                    sub_graph_length += 1
                    sub_graph_node += [set()]
                    final_output_graph = False
            
            # out_desc not in in_out_dict means out-degree is zero.
            if out_desc['tensor_name'] not in in_out_dict:
                final_output_graph = True
                cur_final_node = out_desc['tensor_name']
                final_output_within_graph.append(cur_final_node)

            sub_graph_node[idx].add(out_desc['tensor_name'])
            for input_desc in op_info['input_desc']:
                for sub_input_desc in input_desc:
                    sub_graph_node[idx].add(sub_input_desc['tensor_name'])
                    tmp_name = sub_input_desc['tensor_name']
                    if tmp_name in output_tensor_name:
                        inter_output_list.add(sub_input_desc['tensor_name'])
                    for subgraph in sub_graph_node[0: idx]:
                        extra_output = is_tensor(sub_input_desc) and tmp_name not in stitch_tensor_name and tmp_name not in input_tensor_name
                        used_by_other_sg = tmp_name in subgraph
                        used_as_output = tmp_name in output_tensor_name
                        extra_output = extra_output and (used_by_other_sg or used_as_output)
                        if extra_output and cur_stitch_node and not final_output_graph:
                            extra_subgraph_output[cur_stitch_node].insert(0, tmp_name)
                            break
                    if sub_input_desc['tensor_name'] not in in_out_dict:
                        in_out_dict[sub_input_desc['tensor_name']] = [out_desc['tensor_name']]
                    else:
                        in_out_dict[sub_input_desc['tensor_name']].append(out_desc['tensor_name'])
    
    return extra_subgraph_output, list(final_output_list), final_output_within_graph

def collect_subgraph_info(desc_d, sub_stitch_graphs, req_map, input_tensor_name, output_tensor_name, stitch_node_list):
    inplace_assign_map = {}
    fake_output_list = []
    # traversal desc_d by reverse topologically order.
    for i in range(len(desc_d['op_desc']) - 1, -1, -1):
        op_info = desc_d['op_desc'][i]
        if (op_info['name'] == "InplaceAssign"):
            inplace_assign_map[op_info['output_desc'][0]['tensor_name']] = op_info['input_desc'][0][0]['tensor_name']
            if (op_info['attr'][0]['name'] == 'fake_output' and op_info['attr'][0]['value'] == 1):
                fake_output_list.append(op_info['output_desc'][0]['tensor_name'])
        for sg in sub_stitch_graphs:
            added_output = []
            for out_desc in op_info['output_desc']:
                out_tensor_name = out_desc['tensor_name']
                if out_tensor_name in sg.tensors:
                    sg.ops.append(op_info)
                    if out_tensor_name in req_map:
                        if out_desc['shape']:
                            req_map[out_tensor_name] = reduce(lambda x, y: x * y, out_desc['shape'])
                        else:
                            req_map[out_tensor_name] = 1

                    if out_tensor_name in sg.output_name and out_tensor_name not in added_output:
                        sg.output.append(out_desc)
                        added_output.append(out_tensor_name)

                    for input_desc in op_info['input_desc']:
                        for sub_input_desc in input_desc:
                            if is_tensor(sub_input_desc):
                                input_name = sub_input_desc['tensor_name']
                                if input_name in output_tensor_name and input_name not in added_output:
                                    sg.output.insert(0, sub_input_desc)
                                    added_output.append(input_name)
                                if input_name in input_tensor_name and input_name not in sg.input_name:
                                    sg.input_name.append(sub_input_desc['tensor_name'])
                                    sg.input.append([sub_input_desc])
                                # stop expand subgraph when encounter with stitch node.
                                if input_name not in stitch_node_list:
                                    sg.tensors.add(sub_input_desc['tensor_name'])
                                # add extra input into subgraph.
                                elif input_name not in sg.output_name and input_name not in sg.input_name:
                                    sg.input_name.append(input_name)
                                    sg.input.append([sub_input_desc])
    return sub_stitch_graphs, inplace_assign_map, fake_output_list


def sub_graph_info(sub_graph, desc_d):
    # gather info for sub graph.
    op_json_str = {}
    op_json_str['composite'] = True
    op_json_str['composite_graph'] = desc_d['composite_graph']
    op_json_str['id'] = desc_d['id']
    op_json_str['op'] = sub_graph.op_name
    op_json_str['input_desc'] = sub_graph.input
    op_json_str['op_desc'] = sub_graph.ops
    op_json_str['output_desc'] = sub_graph.output
    op_json_str['platform'] = "AKG"
    op_json_str['process'] = desc_d['process']
    if 'sub_block_size' in desc_d['buffer_stitch']:
        op_json_str['blocksize'] = desc_d['buffer_stitch']['sub_block_size']

    json_str = json.dumps(op_json_str)
    return json_str

def stitch_json_split(desc_d):
    """
    split sub graph from merged json file.
    Using 'buffer_stitch' to store stitch info from graph kernel.
    Args:
        desc_d: dict of compute description
    Returns:
        List of spilted json info.
        List of original input.
        Dict of dominance info.
    """
    stitch_jsons = []

    input_tensor_name = [tensor[0]['tensor_name'] for tensor in desc_d['input_desc']]
    output_tensor_name = [tensor['tensor_name'] for tensor in desc_d['output_desc']]
    stitch_node = desc_d['buffer_stitch']['stitch_op']
    stitch_node_name = [node for stitchnode in stitch_node for node in stitchnode]
    extra_subgraph_output, final_output_list, final_output_within_graph = parse_merged_json(desc_d, stitch_node_name, input_tensor_name, output_tensor_name)

    # traverse extra_subgraph_output to save extra output into subgraph.
    stitch_node = []
    extra_list = []
    for item in extra_subgraph_output:
        cur_list = [item]
        for node in extra_subgraph_output[item]:
            if node not in extra_list:
                extra_list.append(node)
                cur_list.append(node)
        stitch_node.append(cur_list)
    stitch_node_name = [node for stitchnode in stitch_node for node in stitchnode]

    # initialize req_map
    req_op_size = [0] * len(stitch_node_name)
    req_map = dict(zip(stitch_node_name, req_op_size))
    # add final output within subgraph into the last initialized stitch sub_graph.
    stitch_node = stitch_node[:-1] + [stitch_node[-1] + final_output_within_graph]
    # add final output into stitch_op.
    stitch_node += [[op] for op in final_output_list if op not in stitch_node_name]
    stitch_node_list = [node for stitchnode in stitch_node for node in stitchnode]
    # each output tensor can only be parsed as output once in all subgraphs. 
    # All tensors in stitch_node_list will be put into output_name. 
    # Save other output tensors which are not in stitch_node_name for the output collection of subgraphs.
    complement_output = [tensor for tensor in output_tensor_name if tensor not in stitch_node_list]

    # initialize sub_stitch_graphs.
    sub_stitch_graphs = []
    for i, stitch_op in enumerate(stitch_node):
        sub_stitch_graphs.append(Graph(stitch_op))

    sub_stitch_graphs, inplace_assign_map, fake_output_list = collect_subgraph_info(desc_d, sub_stitch_graphs, req_map, input_tensor_name, complement_output, stitch_node_list)
    # reverse op order to generate topological subgraph
    for i, sg in enumerate(sub_stitch_graphs):
        sg.ops = list(reversed(sg.ops))
        sg.op_name = desc_d['op']
        stitch_json_str = sub_graph_info(sg, desc_d)
        if (os.getenv(get_dump_ir_flag()) == "on"):
            if not os.path.exists("stitch_info"):
                try:
                    os.mkdir("stitch_info")
                except OSError as err:
                    # 17, OSError: [Errno 17] File exists
                    if err.errno == 17:
                        pass
                    else:
                        raise err
            with open('stitch_info/' + sg.op_name + '_stitch_' + str(i + 1) + '.json', 'w+') as f:
                f.write(stitch_json_str)
            with open('stitch_info/' + sg.op_name + '_stitch.json', 'w+') as f:
                f.write(json.dumps(desc_d))
        stitch_jsons.append(stitch_json_str)
    
    clean_op_list = [fake_op for fake_op in fake_output_list if fake_op in stitch_node_name]
    # add fake outputs into output_tensor_name
    output_tensor_name += clean_op_list
    # start node for dominance tree is final_output_list + final_output_within_graph.
    start_node = final_output_list + final_output_within_graph
    alloc_map, reuse_map = shared_memory_optimization(desc_d, req_map, output_tensor_name)
    # remove fake output from alloc_map and store them into clean_op_map
    clean_op_map = dict()
    for fake_op in clean_op_list:
        clean_info = alloc_map[fake_op] if fake_op in alloc_map else reuse_map[fake_op]
        clean_op_map[inplace_assign_map[fake_op]] = clean_info
        alloc_map.pop(fake_op) if fake_op in alloc_map else reuse_map.pop(fake_op)

    if not alloc_map:
        alloc_map['EMPTY'] = []
    if not clean_op_map:
        clean_op_map['EMPTY'] = []
    if not reuse_map:
        reuse_map['EMPTY'] = []
    return stitch_jsons, input_tensor_name, output_tensor_name, alloc_map, reuse_map, clean_op_map


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


def generate_trait(desc):
    """ generate trait of kernel description """
    def generate_compute_trait():
        tensor_idx = {}
        counter = 0
        traits = []
        if desc['input_desc'] is not None:
            for in_desc in desc['input_desc']:
                tensor_idx[in_desc[0]['tensor_name']] = counter
                counter += 1
            traits = [str(len(desc['input_desc']))]
        for op in desc['op_desc'] if desc['op_desc'] is not None else []:
            input_idx = []
            for input_desc in op['input_desc']:
                if input_desc[0].get('value', None) is None:
                    input_idx.append(counter - tensor_idx[input_desc[0]['tensor_name']])
            input_idx.sort()
            input_idx_str = ''.join([str(i) for i in input_idx])
            op_trait = op['name'] + input_idx_str
            if op['name'] == "MatMul":
                for attr in op['attr']:
                    if attr['name'] == "transpose_a":
                        transpose_a = str(int(attr['value']))
                    if attr['name'] == "transpose_b":
                        transpose_b = str(int(attr['value']))
                op_trait += '_' + transpose_a + '_' + transpose_b
            traits.append(op_trait)
            tensor_idx[op['output_desc'][0]['tensor_name']] = counter
            counter += 1
        output_idx = []
        for out_desc in desc['output_desc'] if desc['output_desc'] is not None else []:
            output_idx.append(tensor_idx[out_desc['tensor_name']])
        output_idx.sort()
        traits.append(''.join([str(i) for i in output_idx]))
        return '.'.join(traits)

    def append_trait(traits, data):
        if traits and traits[-1].rstrip('-') == data:
            traits[-1] += '-'
        else:
            traits.append(data)

    def generate_shape_trait():
        traits = []
        for in_desc in desc['input_desc'] if desc['input_desc'] is not None else []:
            shape_s = '_'.join([str(i) for i in in_desc[0]['shape']])
            append_trait(traits, shape_s)
        for out_desc in desc['output_desc'] if desc['output_desc'] is not None else []:
            shape_s = '_'.join([str(i) for i in out_desc['shape']])
            append_trait(traits, shape_s)
        return '.'.join(traits)

    def generate_dtype_trait():
        traits = []
        for in_desc in desc['input_desc'] if desc['input_desc'] is not None else []:
            dtype = in_desc[0]['data_type']
            append_trait(traits, dtype)
        for out_desc in desc['output_desc'] if desc['output_desc'] is not None else []:
            dtype = out_desc['data_type']
            append_trait(traits, dtype)
        return '.'.join(traits)

    compute = generate_compute_trait()
    shape = generate_shape_trait()
    dtype = generate_dtype_trait()
    return compute, shape, dtype

def read_repo_file(repo_file):
    with open(repo_file, 'r') as f:
        repo = json.loads(f.read())
    return repo

def _get_repository_file_path(file):
    pwd = os.path.dirname(os.path.abspath(__file__))
    path = pwd + "/" + file
    if not os.path.exists(path):
        path = pwd + "/../config/" + file
        if not os.path.exists(path):
            raise FileNotFoundError("Can not find {} in directory {} and {}".format(file, pwd, pwd + "/../config"))
    return path

def _set_compute_attrs(desc_d_in, attr):
    desc_d = desc_d_in
    for i, op in enumerate(desc_d.get('op_desc')):
        if op.get('name') == "MatMul" and attr.get('bypass') not in (None, ''):
            desc_d['op_desc'][i]['attr'].append({'data_type': 'int32', 'name': 'bypass', 'value': attr['bypass']})
    desc_s = json.dumps(desc_d)
    return desc_d, desc_s

def _pragma_rmselfdep(kernel_info):
    for op in kernel_info["op_desc"]:
        if op['name'] == "MatMul":
            return False
    return True

def _enable_auto_inline(kernel_info):
    for op in kernel_info["op_desc"]:
        # For the MatMul/BatchMatMul with bias, the inline is necessary
        if op['name'] in ["MatMul", "BatchMatMul"]:
            return True
    # For the Ascend, turn 'enable_auto_inline' off for composite op by default.
    return False


def _build_to_module(desc_s_in, desc_d_in, attr=None, use_repo=True):
    """
    build kernel with compute description in json format
    Args:
       desc_s_in : str of compute description
       desc_d_in : dict of compute description
       attr   : dict of build attributes

    Returns:
       Module.
    """
    if os.getenv('MS_GRAPH_KERNEL_TILING'):
        repository = read_repo_file(str(os.getenv('MS_GRAPH_KERNEL_TILING')))
    else:
        file_path = _get_repository_file_path("repository.json")
        repository = read_repo_file(file_path)

    def get_matmul_desc(desc):
        for op in desc['op_desc'] if desc['op_desc'] is not None else []:
            if op['name'] in ["MatMul", "BatchMatMul"]:
                return op
        return None
    
    def get_attr(repo):
        if not isinstance(repo, dict) or len(repo) == 0:
            return {}
        for key, value in repo.items():
            if key == "attrs":
                return value
            else:
                return get_attr(value)
        
    def get_vague_repo(vague_key, vague_shape, repo, default=None, fuzzy_level=0):
        for compute, s_repo in repo.items():
            if compute.find(vague_key) != -1:
                for shape, t_repo in s_repo.items():
                    if fuzzy_level == 0 and shape.startswith(vague_shape):
                        return get_attr(t_repo)
                    if fuzzy_level == 1 and shape.find(vague_shape) != -1:
                        return get_attr(t_repo)
        return default
    
    def get_input_shape_str(desc):

        def append_trait(traits, data):
            if traits and traits[-1].rstrip('-') == data:
                traits[-1] += '-'
            else:
                traits.append(data)
        
        inputs = desc['input_desc']
        res = []
        for input_desc in inputs:
            shape = input_desc[0]['shape']
            shape_str = '_'.join([str(i) for i in shape])
            append_trait(res, shape_str)
        return '.'.join(res)

    def get_repo(keys, repo, default=None):
        for key in keys:
            repo = repo.get(key)
            if not repo:
                return default
        return repo
    if attr is None:
        attr = {'dim': ''}
    desc_d = desc_d_in
    desc_s = desc_s_in
    attr["pragma_rmselfdep"] = _pragma_rmselfdep(desc_d)
    attr["enable_auto_inline"] = _enable_auto_inline(desc_d)
    if use_repo:
        compute, shape, dtype = generate_trait(desc_d)
        repo_attr = get_repo([compute, shape, dtype, 'metadata', 'attrs'], repository, {})
        if not repo_attr:
            matmul_desc = get_matmul_desc(desc_d)
            if matmul_desc != None:
                input_shape_str = get_input_shape_str(matmul_desc)
                repo_attr = get_vague_repo("MatMul", input_shape_str, repository, {})
                print("use fuzzy attr: " ,repo_attr)
            else:
                repo_attr = get_repo([compute, 'metadata', 'attrs'], repository, {})
        for a in repo_attr:
            if not attr.get(a):
                attr[a] = repo_attr[a]
        if attr.get('dim') in (None, ''):
            tiling = get_repo([compute, shape, dtype, 'dim'], repository)
            if tiling:
                attr['dim'] = tiling
            elif 'online_tuning' in attr:
                from akg.auto_tune.composite_tuner import tune_composite
                best_config = tune_composite(desc_s_in,
                                             tune_level=attr["online_tuning"],
                                             repo_path=_get_repository_file_path("repository.json"),
                                             skip_exist=True)
                attr.update(best_config)
        desc_d, desc_s = _set_compute_attrs(desc_d, attr)

    if 'parallel_fusion' in desc_d or 'buffer_stitch' in desc_d:
        return _build_json_list_to_module(desc_d, attr, True, 'cce')
    func = tvm.get_global_func("composite_with_json")
    return func(desc_s, attr, True)

def _reducemax_pattern(kernel_info):
    for op in kernel_info['op_desc']:
        if op['name'] == 'ReduceMax':
            input_shape = op['input_desc'][0][0]['shape']
            batch_size = input_shape[0]
            reduce_size = batch_size * input_shape[1] * input_shape[2]
            return (True, reduce_size)
    return (False, 0)

def _is_batchmatmul(kernel_info):
    for op in kernel_info['op_desc']:
        if op['name'] == 'BatchMatMul':
            return True
    return False

def _set_tiling_attrs(out_shape, attrs):
    axis_len = len(out_shape)
    if axis_len < 3:
        return attrs
    if all(map(lambda x:x == 1, [out_shape[x] for x in range(axis_len - 2)])):
        return attrs
    if attrs.get('bind_block') in (None, ''):
        i = 0
        while out_shape[i] == 1:
            i += 1
        block_y = out_shape[i]
        block_x = out_shape[i + 1] if i < axis_len - 3 else 1
        attrs['bind_block'] = str(block_x) + ' ' + str(block_y)
    if attrs.get('dim') in (None, ''):
        batch_axis = 0
        for i in range(axis_len - 2):
            if out_shape[i] != 1:
                batch_axis += 1
        dim_list = [0, 0, 64, 64, 0, 0, 64, 64, 0, 0, 64, 4]
        dim_list = [0, 0, 1, 1] * batch_axis + dim_list
        i = 0
        while i < (len(dim_list) // 4):
            dim_list[i * 4 + 1] = i
            i += 1
        attrs['dim'] = ' '.join(str(x) for x in dim_list)
    return attrs

def _set_reducemax_attrs(desc_d, attrs):
    if _reducemax_pattern(desc_d)[0]:
        attrs['enable_tile_c0'] = True
        elem_per_thread = 4
        blockdim_x = 64
        blockdim_y = 16
        griddim_x = 1
        griddim_y = _reducemax_pattern(desc_d)[1] / (blockdim_y * elem_per_thread)
        attrs['dim'] = ' 0 0 128 64 0 1 128 128'
        attrs['bind_block'] = str(griddim_x) + ' ' + str(griddim_y)
        attrs['bind_thread'] = str(blockdim_x) + ' ' + str(blockdim_y)
    return attrs

def _json_need_split(desc_d, attrs):
    block_jsons = []
    input_tensor_name = []
    output_tensor_name = []
    attrs_list = []
    alloc_map_list = []
    reuse_map_list = []
    clean_op_map_list = []

    if 'parallel_fusion' in desc_d:
        block_jsons, input_tensor_name, output_tensor_name = parallel_json_split(desc_d)
        if desc_d["parallel_fusion"]["fusion_type"] == "block_pipeline_fusion":
            attrs["pipeline_groups"] = desc_d["parallel_fusion"]['type_info']
        for i, _ in enumerate(block_jsons):
            if 'buffer_stitch' in block_jsons[i]:
                stitch_jsons, _, _, alloc_map, reuse_map, clean_op_map = stitch_json_split(block_jsons[i])
                block_jsons[i] = stitch_jsons
                cur_attrs = _set_reducemax_attrs(json.loads(stitch_jsons), attrs.copy())
            else:
                alloc_map, reuse_map, clean_op_map = dict(), dict(), dict()
                cur_attrs = attrs.copy()

            cur_attrs["enable_atomic_add"] = should_enable_atomic_add(json.loads(block_jsons[i]))
            attrs_list.append(cur_attrs)
            alloc_map_list.append(alloc_map)
            reuse_map_list.append(reuse_map)
            clean_op_map_list.append(clean_op_map)
    elif 'buffer_stitch' in desc_d:
        stitch_jsons, input_tensor_name, output_tensor_name, alloc_map, reuse_map, clean_op_map = stitch_json_split(desc_d)
        block_jsons.append(stitch_jsons)
        attrs = _set_reducemax_attrs(desc_d, attrs)
        attrs_list.append(attrs)
        alloc_map_list.append(alloc_map)
        reuse_map_list.append(reuse_map)
        clean_op_map_list.append(clean_op_map)
    return block_jsons, input_tensor_name, output_tensor_name, attrs_list, alloc_map_list, reuse_map_list, clean_op_map_list

def _build_json_list_to_module(desc_d, attrs, poly, target):
    func = tvm.get_global_func("composite_with_json_list")
    block_jsons, input_tensor_name, output_tensor_name, attrs_list, alloc_map_list, reuse_map_list, \
    clean_op_map_list = _json_need_split(desc_d, attrs)
    return func(block_jsons, input_tensor_name, output_tensor_name, alloc_map_list, reuse_map_list, \
                clean_op_map_list, attrs_list, poly, target)

def _build_to_module_gpu(desc_s, desc_d, attrs=None, poly=False):
    """
    build kernel with compute description in json format
    Args:
       desc_s : str of compute description
       desc_d : dict of compute description
       attrs   : dict of build attributes

    Returns:
       Module.
    """
    if os.getenv('MS_GRAPH_KERNEL_TILING'):
        repository_gpu = read_repo_file(str(os.getenv('MS_GRAPH_KERNEL_TILING')))
    elif 'buffer_stitch' in desc_d:
        repository_gpu = {}
    else:
        file_path = _get_repository_file_path("repository_gpu.json")
        repository_gpu = read_repo_file(file_path)
    def get_repo(keys, default=None):
        repo = repository_gpu
        for key in keys:
            repo = repo.get(key)
            if not repo:
                return default
        return repo
    if attrs is None:
        attrs = {'dim': ''}
    compute, shape, dtype = generate_trait(desc_d)
    batchmatmul = _is_batchmatmul(desc_d)
    if batchmatmul:
        shape = "any_shape"
    repo_attr = get_repo([compute, shape, dtype, 'metadata', 'attrs'], {})
    if repo_attr and batchmatmul:
        repo_attr = _set_tiling_attrs(desc_d['output_desc'][0]['shape'], repo_attr)
    if not repo_attr:
        repo_attr = get_repo([compute, 'metadata', 'attrs'], {})
    for a in repo_attr:
        if not attrs.get(a):
            attrs[a] = repo_attr[a]
    attr_list = ['dim', 'bind_block', 'bind_thread']
    for item in attr_list:
        if attrs.get(item) in (None, ''):
            value = get_repo([compute, shape, dtype, item])
            if value:
                attrs[item] = value

    if 'parallel_fusion' in desc_d or 'buffer_stitch' in desc_d:
        return _build_json_list_to_module(desc_d, attrs, poly, 'cuda')
    func = tvm.get_global_func("composite_with_json")
    return func(desc_s, attrs, poly)

def _build(desc_s, desc_d, attrs=None, poly=True, use_repo=True):
    if attrs is None:
        attrs = dict()
    backend = desc_d['process']
    if "enable_atomic_add" not in attrs.keys():
        attrs["enable_atomic_add"] = should_enable_atomic_add(desc_d)
        if not poly:
            attrs["enable_atomic_add"] = False
    if backend == 'cuda':
        if poly:
            attrs["enable_akg_reduce_lib"] = True
        return _build_to_module_gpu(desc_s, desc_d, attrs, poly)
    else:
        return _build_to_module(desc_s, desc_d, attrs, use_repo)

def build(kernel_desc, attrs=None, poly=True, use_repo=True):
    """
    build kernel with compute description in json format
    Args:
       kernel_desc : str or dict of compute description
       attrs   : dict of build attributes

    Returns:
       Module.
    """
    if isinstance(kernel_desc, str):
        desc_s = kernel_desc
        desc_d = json.loads(kernel_desc)
    else:
        assert isinstance(kernel_desc, dict)
        desc_s = json.dumps(kernel_desc)
        desc_d = kernel_desc
    return _build(desc_s, desc_d, attrs, poly, use_repo)

def get_tiling_space(kernel_desc, level=1, attr=None):
    """
    get tiling space of composite kernel
    Args:
       kernel_desc : str of compute description
       level       : info level
       attr        : dict of build attributes

    Returns:
       Module.
    """
    if attr is None:
        attr = {}
    attr['help_tiling'] = level
    attr['tuning'] = 'on'
    if 'enable_auto_inline' not in attr:
        attr['enable_auto_inline'] = False
    attr['pragma_reschedule'] = 1
    func = tvm.get_global_func('composite_lower')
    ret = func(kernel_desc, attr)
    spaces = {}
    spaces['index'] = ret.index_table.asnumpy().tolist()
    spaces['c1_range'] = ret.c1_tile_range_table.asnumpy().tolist()
    spaces['c0_range'] = ret.c0_tile_range_table.asnumpy().tolist()
    spaces['c1_mod'] = ret.c1_tile_mod_table.asnumpy().tolist()
    spaces['c0_mod'] = ret.c0_tile_mod_table.asnumpy().tolist()
    if level >= 2:
        spaces['tuning_space'] = ret.tiling_candidate.asnumpy().tolist()
    return spaces

@tvm.register_func("akg_build_gpu_module")
def build_cuda(outputs, args, sch_name, kernel_name, attrs = False, poly = False, binds = None):
    s = select_cuda_scheduler(outputs, sch_name, poly)
    if attrs:
        attrs_t = dict(attrs.items())
    else:
        attrs_t = None
    dump_ir = os.getenv(get_dump_ir_flag()) == "on"
    with tvm.build_config(dump_pass_ir = dump_ir):
        mod = akg.build(s, list(args), "cuda", name = kernel_name, binds = binds, attrs = attrs_t, polyhedral=bool(poly))
        return mod

@tvm.register_func("select_cuda_scheduler")
def select_cuda_scheduler(outputs, sch_name, poly = False, grid_dims=0, block_dims=0, buffer_stitch=False):
    scheduler = {
        "injective" : topi.cuda.injective_single_kernel.schedule_injective,
        "reduce"    : topi.cuda.reduce_opt.schedule_reduce,
    }
    with tvm.target.cuda():
        if bool(poly):
            s = akg.tvm.create_schedule([x.op for x in list(outputs)])
        else:
            if grid_dims and block_dims and sch_name == "injective":
                s = scheduler[sch_name](outputs, grid_dims, block_dims, buffer_stitch=buffer_stitch)
            else:
                s = scheduler[sch_name](outputs, grid_dims, block_dims)
        return s
