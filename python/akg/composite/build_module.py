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
import akg
from akg import tvm
import topi
from akg.utils.kernel_exec import ReturnType
from .split_block import parallel_json_split
from .split_stitch import stitch_json_split, split_stitch_attr, combine_stitch_attr


def _should_enable_atomic_add(kernel_info):
    for op in kernel_info["op_desc"]:
        if not op["attr"]:
            continue
        for attr in op["attr"]:
            if attr["name"] == "enable_atomic_add" and attr["value"]:
                return True
    return False


def _reducemax_pattern(kernel_info):
    for op in kernel_info['op_desc']:
        if op['name'] == 'ReduceMax':
            input_shape = op['input_desc'][0][0]['shape']
            batch_size = input_shape[0]
            reduce_size = batch_size * input_shape[1] * input_shape[2]
            return True, reduce_size
    return False, 0


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
            if op['input_desc']:
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


def _set_compute_attrs(desc_d_in, attr):
    desc_d = desc_d_in
    for i, op in enumerate(desc_d.get('op_desc')):
        if op.get('name') == "MatMul" and attr.get('bypass') not in (None, ''):
            desc_d['op_desc'][i]['attr'].append({'data_type': 'int32', 'name': 'bypass', 'value': attr['bypass']})
    desc_s = json.dumps(desc_d)
    return desc_d, desc_s


def _get_feature(desc_s, attr):
    composite_lower = tvm.get_global_func("composite_lower")
    stmt, args = composite_lower(desc_s, attr)
    from akg.tvm import build_module
    binds, _ = build_module.get_binds(args)
    desc_d = json.loads(desc_s)
    target = desc_d.get("process")
    from akg.utils.auto_tuning import get_features_from_stmts
    feature = get_features_from_stmts(target=target, stmts=[stmt], binds=[binds], n_skip_cache=0)[0]
    return feature


def _build_for_tuning(desc_s, attrs, func):
    if attrs.get("ret_mode") == ReturnType.FEAT:
        return _get_feature(desc_s, attrs)
    elif attrs.get("ret_mode") in [ReturnType.DEFAULT, ReturnType.MOD]:
        return func(desc_s, attrs, True)
    elif attrs.get("ret_mode") == ReturnType.MOD_AND_FEAT:
        # get both module and feature
        attrs["ret_mode"] = ReturnType.FEAT
        feature = _get_feature(desc_s, attrs)
        attrs["ret_mode"] = ReturnType.MOD
        mod = func(desc_s, attrs, True)
        return mod, feature
    else:
        raise ValueError("ret_mode gets a wrong value: {}, should be in DEFAULT, FEAT, MOD, MOD_AND_FEAT".
                         format(attrs.get("ret_mode")))


def _set_tiling_attrs(out_shape, attrs):
    axis_len = len(out_shape)
    if axis_len < 3:
        return attrs
    if all(map(lambda x: x == 1, [out_shape[x] for x in range(axis_len - 2)])):
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
    backend = desc_d['process']
    if backend == 'cuda' and _reducemax_pattern(desc_d)[0]:
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


def _set_atomic_add_attrs(desc_d, attrs, poly):
    if "enable_atomic_add" not in attrs.keys():
        attrs["enable_atomic_add"] = _should_enable_atomic_add(desc_d)
        if not poly:
            attrs["enable_atomic_add"] = False
    return attrs


def _get_online_tune_attr(desc_s, attrs, repo_path, use_new_space=True):
    desc_d = json.loads(desc_s)
    if "buffer_stitch" in desc_d:
        from akg import auto_tune
        best_config = auto_tune.tune_stitch_segment(desc_s,
                                                    repo_path=repo_path)
    elif use_new_space:
        from akg import auto_tune
        task_options = auto_tune.TaskOptions(tune_level=attrs["online_tuning"],
                                             use_new_space=use_new_space,
                                             attrs=attrs,
                                             generate_trait=generate_trait,
                                             mode="online")
        best_config = auto_tune.tune_composite_v2(desc_s,
                                                  task_options=task_options)
    else:
        from tests.prev_version_auto_tune.composite_tuner import tune_composite
        best_config = tune_composite(desc_s,
                                     tune_level=attrs["online_tuning"],
                                     repo_path=repo_path,
                                     skip_exist=True)
    attrs.update(best_config)
    pop_keys = ["online_tuning", "help_tiling", "tuning", "use_new_space"]
    clean_attrs = {k: v for k, v in attrs.items() if k not in pop_keys}
    return clean_attrs


def get_attr_from_dict(keys, repo, default=None):
    # repo={key1:{key2:{key3:attr}}} , keys=[key1,key3,key3] return attr
    for key in keys:
        repo = repo.get(key)
        if not repo:
            return default
    return repo


def merge_attrs(attrs_a, attrs_b):
    # merge attrs_b into attrs_a if an attr in attrs_b but not in attrs_a
    attrs = attrs_a.copy()
    for i in attrs_b:
        if not attrs.get(i):
            attrs[i] = attrs_b[i]
    return attrs


def read_repo_file(repo_file):
    with open(repo_file, 'r') as f:
        repo = json.loads(f.read())
    return repo


def get_repository_file_path(file):
    # get the abosulte path for a file in currect dir, input is a file's name like "a.json"
    pwd = os.path.dirname(os.path.abspath(__file__))
    path = pwd + "/" + file
    if not os.path.exists(path):
        path = pwd + "/../config/" + file
        if not os.path.exists(path):
            raise FileNotFoundError("Can not find {} in directory {} and {}".format(file, pwd, pwd + "/../config"))
    return path


def _get_repository(file_name, desc_d, target=None):
    if os.getenv('MS_GRAPH_KERNEL_TILING'):
        repository = read_repo_file(str(os.getenv('MS_GRAPH_KERNEL_TILING')))
    elif 'buffer_stitch' in desc_d and target == 'cuda':
        repository = {}
    else:
        repository = read_repo_file(get_repository_file_path(file_name))
    return repository


def _get_repo_attr(desc_d, compute, shape, dtype, repo, batchmatmul):
    repo_attr = get_attr_from_dict([compute, shape, dtype, 'metadata', 'attrs'], repo, {})
    if repo_attr and batchmatmul:
        repo_attr = _set_tiling_attrs(desc_d['output_desc'][0]['shape'], repo_attr)
    if not repo_attr:
        repo_attr = get_attr_from_dict([compute, 'metadata', 'attrs'], repo, {})
    return repo_attr


def _json_need_split(desc_d, attrs, poly, target):
    block_jsons = []
    stitch_origin_jsons = []
    input_tensor_name = []
    output_tensor_name = []
    attrs_list = []
    alloc_map_list = []
    reuse_map_list = []
    clean_op_map_list = []

    if 'parallel_fusion' in desc_d:
        block_jsons, input_tensor_name, output_tensor_name = parallel_json_split(desc_d)
        stitch_origin_jsons = block_jsons
        if desc_d["parallel_fusion"]["fusion_type"] == "block_pipeline_fusion":
            attrs["pipeline_groups"] = desc_d["parallel_fusion"]['type_info']
        for i, _ in enumerate(block_jsons):
            if 'buffer_stitch' in block_jsons[i]:
                stitch_jsons, _, _, alloc_map, reuse_map, clean_op_map = stitch_json_split(block_jsons[i])
                block_jsons[i] = stitch_jsons
                cur_attrs = _set_reducemax_attrs(json.loads(stitch_jsons), attrs.copy())
                cur_attrs["enable_stitch_fusion"] = True
            else:
                alloc_map, reuse_map, clean_op_map = dict(), dict(), dict()
                cur_attrs = attrs.copy()

            cur_attrs["enable_atomic_add"] = _should_enable_atomic_add(json.loads(block_jsons[i]))
            if target == "cuda":
                all_ops = set([op['name'] for op in json.loads(block_jsons[i])['op_desc']])
                cur_attrs = _update_attrs_gpu(all_ops, cur_attrs, poly)
            attrs_list.append(cur_attrs)
            alloc_map_list.append(alloc_map)
            reuse_map_list.append(reuse_map)
            clean_op_map_list.append(clean_op_map)
    elif 'buffer_stitch' in desc_d:
        stitch_origin_jsons.append(json.dumps(desc_d))
        stitch_jsons, input_tensor_name, output_tensor_name, alloc_map, reuse_map, clean_op_map \
            = stitch_json_split(desc_d)
        block_jsons.append(stitch_jsons)
        attrs = _set_reducemax_attrs(desc_d, attrs)
        attrs["enable_stitch_fusion"] = True
        attrs_list.append(attrs)
        alloc_map_list.append(alloc_map)
        reuse_map_list.append(reuse_map)
        clean_op_map_list.append(clean_op_map)
    return block_jsons, stitch_origin_jsons, input_tensor_name, output_tensor_name, attrs_list, \
        alloc_map_list, reuse_map_list, clean_op_map_list


def _update_attrs_gpu(all_ops, attrs, poly):
    if poly:
        if any([i in all_ops for i in ['Argmax', 'Argmin']]):
            # disable auto_fuse and akg_reduce_lib for argmax and argmin
            attrs["enable_akg_reduce_lib"] = False
            attrs["enable_auto_fuse"] = False
        elif "enable_akg_reduce_lib" not in attrs.keys():
            attrs["enable_akg_reduce_lib"] = True

        if "pragma_enable_matmul" not in attrs.keys() and any([i in all_ops for i in ["BatchMatMul", "MatMul", "Conv2D"]]):
            attrs['pragma_enable_matmul'] = True
            attrs['enable_auto_inline'] = False
        if "pragma_enable_conv_tensor_core" not in attrs.keys() and "Conv2D" in all_ops:
            attrs["pragma_enable_conv_tensor_core"] = True
            attrs["enable_auto_fuse"] = False
    return attrs


def _update_attrs_ascend(all_ops, attr):
    attr["pragma_reschedule"] = 1
    attr["pragma_rmselfdep"] = all([i not in all_ops for i in ["BatchMatMul", "MatMul"]])
    # For the MatMul/BatchMatMul with bias, the inline is necessary
    # For the Ascend, turn 'enable_auto_inline' off for composite op by default.
    attr["enable_auto_inline"] = any([i in all_ops for i in ["BatchMatMul", "MatMul"]])
    return attr


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
    repository_gpu = _get_repository("repository_gpu.json", desc_d)
    all_ops = set([op['name'] for op in desc_d['op_desc']])

    def update_attr(desc_d, attrs):
        if attrs is None:
            attrs = {'dim': ''}
        compute, shape, dtype = generate_trait(desc_d)
        batchmatmul = "BatchMatMul" in all_ops
        if batchmatmul:
            shape = "any_shape"
        repo_attr = _get_repo_attr(desc_d, compute, shape, dtype, repository_gpu, batchmatmul)
        attrs = merge_attrs(attrs, repo_attr)
        attr_list = ['dim', 'bind_block', 'bind_thread']
        for item in attr_list:
            if attrs.get(item) in (None, ''):
                value = get_attr_from_dict([compute, shape, dtype, item], repository_gpu)
                if value:
                    attrs[item] = value
        if attrs.get('dim') in (None, '') and 'online_tuning' in attrs:
            attrs = _get_online_tune_attr(desc_s, attrs, get_repository_file_path("repository_gpu.json"))
        return desc_d, attrs

    if 'parallel_fusion' in desc_d or 'buffer_stitch' in desc_d:
        block_jsons, stitch_origin_jsons, input_tensor_name, output_tensor_name, attrs_list, \
            alloc_map_list, reuse_map_list, clean_op_map_list = _json_need_split(desc_d, attrs, poly, 'cuda')
        if 'parallel_fusion' in desc_d:
            for i, [cur_json, cur_attr] in enumerate(zip(block_jsons, attrs_list)):
                cur_desc_d, attrs_list[i] = update_attr(json.loads(cur_json), cur_attr)
                block_jsons[i] = json.dumps(cur_desc_d)
        else:
            desc_d, attrs = update_attr(desc_d, attrs)
        func = tvm.get_global_func("composite_with_json_list")
        return func(block_jsons, stitch_origin_jsons, input_tensor_name, output_tensor_name,
                    alloc_map_list, reuse_map_list, clean_op_map_list, attrs_list, poly, "cuda")

    desc_d, attrs = update_attr(desc_d, attrs)
    attrs = _update_attrs_gpu(all_ops, attrs, poly)
    func = tvm.get_global_func("composite_with_json")
    if "ret_mode" in attrs:
        return _build_for_tuning(desc_s, attrs, func)
    return func(desc_s, attrs, poly)


def _build_to_module_ascend(desc_s_in, desc_d_in, attr=None, use_repo=True):
    """
    build kernel with compute description in json format
    Args:
       desc_s_in : str of compute description
       desc_d_in : dict of compute description
       attr   : dict of build attributes

    Returns:
       Module.
    """

    repository = _get_repository("repository.json", desc_d_in)

    def _auto_set_single_block(desc_d, attr):
        if not attr.get("enable_multicore", None) and desc_d.get("extra", None):
            if desc_d["extra"].get("BlockMode", "") == "single_block":
                attr["enable_multicore"] = 0
        return attr

    def update_attr(desc_s, desc_d, attr, given_attrs=None, support_online_tuning=True):
        if attr is None:
            attr = {'dim': ''}
        all_ops = set([op['name'] for op in desc_d['op_desc']])
        attr = _update_attrs_ascend(all_ops, attr)
        attr = _auto_set_single_block(desc_d, attr)
        if given_attrs is not None:
            for key, value in given_attrs.items():
                if not attr.get(key):
                    attr[key] = value
            _, desc_s = _set_compute_attrs(desc_d, attr)
        elif use_repo:
            compute, shape, dtype = generate_trait(desc_d)
            repo_attr = _get_repo_attr(desc_d, compute, shape, dtype, repository, False)
            attr = merge_attrs(attr, repo_attr)
            if attr.get('dim') in (None, ''):
                tiling = get_attr_from_dict([compute, shape, dtype, 'dim'], repository)
                if tiling:
                    attr['dim'] = tiling
                elif support_online_tuning and 'online_tuning' in attr:
                    attr = _get_online_tune_attr(desc_s_in, attr, get_repository_file_path("repository.json"))
            _, desc_s = _set_compute_attrs(desc_d, attr)
        return desc_s, attr

    def get_parallel_repo(desc_d):
        compute, shape, dtype = generate_trait(desc_d)
        repo_attr = get_attr_from_dict([compute, shape, dtype, 'BlockPlan'], repository, {})
        return repo_attr

    def get_stitch_repo(desc_d):
        compute, shape, dtype = generate_trait(desc_d)
        repo_attr = get_attr_from_dict([compute, shape, dtype], repository, {})
        return repo_attr

    if 'parallel_fusion' in desc_d_in or 'buffer_stitch' in desc_d_in:
        block_jsons, stitch_origin_jsons, input_tensor_name, output_tensor_name, attrs_list, \
            alloc_map_list, reuse_map_list, clean_op_map_list = _json_need_split(desc_d_in, attr, True, "cce")
        if 'parallel_fusion' in desc_d_in:
            parallel_repo = get_parallel_repo(desc_d_in)
            if parallel_repo:
                # "BlockPlan" should be: [{"block_plan": x1, attr1: x2, attr2: x3}, ...]
                for i, [cur_json, cur_attr, cur_plan] in enumerate(zip(block_jsons, attrs_list, parallel_repo)):
                    # When BlockPlan is active, the body should be run as single block
                    cur_attr["enable_multicore"] = 0
                    block_jsons[i], attrs_list[i] = update_attr(cur_json, json.loads(cur_json), cur_attr,
                                                                cur_plan["attrs"], False)
            else:
                for i, [cur_json, cur_attr] in enumerate(zip(block_jsons, attrs_list)):
                    block_jsons[i], attrs_list[i] = update_attr(cur_json, json.loads(cur_json), cur_attr, None, False)
        else:
            if attrs_list[0].get("peeling") is None:
                # Read buffer stitch attr from repo
                stitch_repo = get_stitch_repo(desc_d_in)
                if stitch_repo.get("peeling") is not None:
                    attrs_list[0].update(stitch_repo)
                elif "online_tuning" in attr:
                    # If buffer stitch attr not in repo, use online tuning
                    tuning_attr = _get_online_tune_attr(desc_s_in, attrs_list[0],
                                                        get_repository_file_path("repository.json"))
                    attrs_list[0].update(tuning_attr)
            # Update sub json attr
            common_attr, sub_attr = split_stitch_attr(attrs_list[0], len(block_jsons[0]))
            for i, cur_json in enumerate(block_jsons[0]):
                block_jsons[0][i], sub_attr[i] = update_attr(cur_json, json.loads(cur_json), sub_attr[i], {})
            attrs_list[0] = combine_stitch_attr(common_attr, sub_attr)
        func = tvm.get_global_func("composite_with_json_list")
        return func(block_jsons, stitch_origin_jsons, input_tensor_name, output_tensor_name,
                    alloc_map_list, reuse_map_list, clean_op_map_list, attrs_list, True, "cce")

    desc_s, attr = update_attr(desc_s_in, desc_d_in, attr)
    func = tvm.get_global_func("composite_with_json")
    if "ret_mode" in attr:
        return _build_for_tuning(desc_s, attr, func)
    return func(desc_s, attr, True)


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
    if attrs is None:
        attrs = dict()
    backend = desc_d['process']
    attrs = _set_atomic_add_attrs(desc_d, attrs, poly)
    if backend == 'cuda':
        return _build_to_module_gpu(desc_s, desc_d, attrs, poly)
    else:
        return _build_to_module_ascend(desc_s, desc_d, attrs, use_repo)


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
    desc_d = json.loads(kernel_desc)
    backend = desc_d['process']
    all_ops = set([op['name'] for op in desc_d['op_desc']])
    if backend == "cuda":
        attr = _update_attrs_gpu(all_ops, attr, True)
    else:
        attr = _update_attrs_ascend(all_ops, attr)
    func = tvm.get_global_func('composite_lower')
    ret = func(kernel_desc, attr)
    spaces = {}
    if attr.get("use_new_space", False):
        spaces['tune_space'] = ret
    else:
        spaces['index'] = ret.index_table.asnumpy().tolist()
        spaces['c1_range'] = ret.c1_tile_range_table.asnumpy().tolist()
        spaces['c0_range'] = ret.c0_tile_range_table.asnumpy().tolist()
        spaces['c1_mod'] = ret.c1_tile_mod_table.asnumpy().tolist()
        spaces['c0_mod'] = ret.c0_tile_mod_table.asnumpy().tolist()
        if level >= 2:
            spaces['tuning_space'] = ret.tiling_candidate.asnumpy().tolist()
    return spaces
