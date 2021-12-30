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
from akg.utils.kernel_exec import ReturnType
from .split_stitch import split_stitch_attr
from .construct_args import get_construct_args, get_tune_construct_args, \
    should_enable_attr, get_stmt_for_tune, add_attrs_in_segment_infos
from .construct_args import ConstructType, ConstructKey


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
            for op_out_desc in op['output_desc'] if op['output_desc'] is not None else []:
                tensor_idx[op_out_desc['tensor_name']] = counter
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


def _get_feature(target, segment_tree, segment_infos):
    tune_composite = tvm.get_global_func("tune_composite")
    stmt, args = tune_composite(target, True, segment_tree, segment_infos)
    from akg.tvm import build_module
    binds, _ = build_module.get_binds(args)
    from akg.utils.auto_tuning import get_features_from_stmts
    feature = get_features_from_stmts(target=target, stmts=[stmt], binds=[binds], n_skip_cache=0)[0]
    return feature


def _build_for_tuning(attrs, func, target, segment_tree, segment_infos):
    def _setup_for_feature(segment_infos):
        feature_segment_infos = segment_infos.copy()
        if attrs.get("ret_mode") != ReturnType.FEAT:
            feature_segment_infos = add_attrs_in_segment_infos(feature_segment_infos, "ret_mode", ReturnType.FEAT)
        feature_segment_infos = get_stmt_for_tune(feature_segment_infos)
        return feature_segment_infos

    if attrs.get("ret_mode") == ReturnType.FEAT:
        segment_infos = _setup_for_feature(segment_infos)
        return _get_feature(target, segment_tree, segment_infos)
    elif attrs.get("ret_mode") in [ReturnType.DEFAULT, ReturnType.MOD]:
        return func(target, True, segment_tree, segment_infos)
    elif attrs.get("ret_mode") == ReturnType.MOD_AND_FEAT:
        # get both module and feature
        feature_segment_infos = _setup_for_feature(segment_infos)
        feature = _get_feature(target, segment_tree, feature_segment_infos)
        segment_infos = add_attrs_in_segment_infos(segment_infos, "ret_mode", ReturnType.MOD)
        mod = func(target, True, segment_tree, segment_infos)
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


def _set_attrs(desc_d, attrs, poly):
    if "enable_atomic_add" not in attrs.keys():
        attrs["enable_atomic_add"] = should_enable_attr(desc_d, "enable_atomic_add")
        if not poly:
            attrs["enable_atomic_add"] = False
    if "is_csr" not in attrs.keys():
        attrs["is_csr"] = should_enable_attr(desc_d, "is_csr")
    if "csr_avg_row" not in attrs.keys():
        attrs["csr_avg_row"] = should_enable_attr(desc_d, "csr_avg_row")
    return attrs


def _get_online_tune_attr(desc_s, attrs, repo_path, use_new_space=True):
    try:
        import auto_tune
    except ImportError:
        raise ImportError("Import auto_tune fail, please install auto_tune using pip")

    desc_d = json.loads(desc_s)
    if "buffer_stitch" in desc_d:
        best_config = auto_tune.tune_stitch_segment(desc_s,
                                                    repo_path=repo_path)
    elif use_new_space:
        task_options = auto_tune.TaskOptions(tune_level=attrs["online_tuning"],
                                             use_new_space=use_new_space,
                                             attrs=attrs,
                                             generate_trait=generate_trait,
                                             mode="online",
                                             enable_transfer=True)
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
    if not os.path.exists(repo_file):
        return {}
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
        # Close general tot by default
        enable_general_tot = False
        if "has_tot_ops" not in attrs.keys() and any([i in all_ops for i in ["Gather", "TensorScatterAdd"]]):
            attrs["has_tot_ops"] = enable_general_tot
    return attrs

def _update_attrs_cpu(all_ops, attrs, poly):
    if not poly:
        return attrs
    if "pragma_enable_matmul" not in attrs.keys() and any([i in all_ops for i in ["BatchMatMul", "MatMul"]]):
        attrs['pragma_enable_matmul'] = True
    if "feature" not in attrs.keys() and any([i in all_ops for i in ["BatchMatMul", "MatMul"]]):
        attrs["feature"] = "avx"
    return attrs

def _update_attrs_ascend(all_ops, attr):
    attr["pragma_rmselfdep"] = all([i not in all_ops for i in ["BatchMatMul", "MatMul"]])
    # For the MatMul/BatchMatMul with bias, the inline is necessary
    # For the Ascend, turn 'enable_auto_inline' off for composite op by default.
    attr["enable_auto_inline"] = any([i in all_ops for i in ["BatchMatMul", "MatMul"]])
    return attr


def _build_to_module(desc_s, desc_d, attrs=None, poly=True):
    """
    build kernel with compute description in json format
    Args:
       desc_s : str of compute description
       desc_d : dict of compute description
       attrs   : dict of build attributes

    Returns:
       Module.
    """
    def _update_attr_by_repo(desc_s, attrs):
        desc_d = json.loads(desc_s)
        process = desc_d["process"]
        file_name = "repository_" + process + ".json"
        repository = _get_repository(file_name, desc_d)
        all_ops = set([op["name"] for op in desc_d["op_desc"]])

        if attrs is None:
            attrs = {"dim": ""}
        compute, shape, dtype = generate_trait(desc_d)
        batchmatmul = "BatchMatMul" in all_ops
        if batchmatmul:
            shape = "any_shape"
        repo_attr = _get_repo_attr(desc_d, compute, shape, dtype, repository, batchmatmul)
        attrs = merge_attrs(attrs, repo_attr)
        attr_list = ["dim", "bind_block", "bind_thread"] if process == "cuda" else ["dim"]
        for item in attr_list:
            if attrs.get(item) in (None, ""):
                value = get_attr_from_dict([compute, shape, dtype, item], repository)
                if value:
                    attrs[item] = value
        if attrs.get("dim") in (None, "") and "online_tuning" in attrs:
            attrs = _get_online_tune_attr(desc_s, attrs, get_repository_file_path(file_name))
        return desc_d, attrs

    def _post_update_attr(desc_s, attrs, poly):
        desc_d, attrs = _update_attr_by_repo(desc_s, attrs)
        all_ops = set([op["name"] for op in desc_d["op_desc"]])
        if desc_d["process"] == "cuda":
            attrs = _update_attrs_gpu(all_ops, attrs, poly)
        elif desc_d["process"] == "cpu":
            attrs = _update_attrs_cpu(all_ops, attrs, poly)
        return attrs

    def _common_postprocess(_, json_str_list, attrs_list, poly):
        for i, (cur_json_str, cur_attr) in enumerate(zip(json_str_list, attrs_list)):
            attrs_list[i] = _post_update_attr(cur_json_str, cur_attr, poly)
        return json_str_list, attrs_list

    def _stitch_postprocess(desc_d, json_str_list, attrs_list, poly):
        for i, cur_attr in enumerate(attrs_list):
            attrs_list[i] = _post_update_attr(json.dumps(desc_d), cur_attr, poly)
        return json_str_list, attrs_list

    post_funcs = {
        ConstructType.PARALLEL: _common_postprocess,
        ConstructType.STITCH: _stitch_postprocess,
        ConstructType.NORMAL: _common_postprocess,
        ConstructType.TOT: _common_postprocess
    }
    segment_tree, segment_infos = get_construct_args(desc_s, attrs, post_funcs)
    process = desc_d["process"]

    func = tvm.get_global_func("lower_composite_to_module")
    if "ret_mode" in attrs and poly:
        return _build_for_tuning(attrs, func, process, segment_tree, segment_infos)
    return func(process, poly, segment_tree, segment_infos)


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

    def _update_attr_by_repo(desc_s, desc_d, attr, given_attrs=None, support_online_tuning=True):
        def _auto_set_single_block(desc_d, attr):
            if not attr.get("enable_multicore", None) and desc_d.get("extra", None):
                if desc_d["extra"].get("BlockMode", "") == "single_block":
                    attr["enable_multicore"] = 0
            return attr
        if attr is None:
            attr = {'dim': ''}
        all_ops = set([op['name'] for op in desc_d['op_desc']])
        attr = _update_attrs_ascend(all_ops, attr)
        attr = _auto_set_single_block(desc_d, attr)
        if given_attrs is not None:
            for key, value in given_attrs.items():
                if not attr.get(key):
                    attr[key] = value
        elif use_repo:
            compute, shape, dtype = generate_trait(desc_d)
            repo_attr = _get_repo_attr(desc_d, compute, shape, dtype, repository, False)
            attr = merge_attrs(attr, repo_attr)
            if attr.get('dim') in (None, ''):
                tiling = get_attr_from_dict([compute, shape, dtype, 'dim'], repository)
                if tiling:
                    attr['dim'] = tiling
                elif support_online_tuning and 'online_tuning' in attr:
                    attr = _get_online_tune_attr(desc_s, attr, get_repository_file_path("repository.json"))
            _, desc_s = _set_compute_attrs(desc_d, attr)
        return desc_s, attr

    def _get_parallel_repo(desc_d):
        compute, shape, dtype = generate_trait(desc_d)
        repo_attr = get_attr_from_dict([compute, shape, dtype, 'BlockPlan'], repository, {})
        return repo_attr

    def _get_stitch_repo(desc_d):
        compute, shape, dtype = generate_trait(desc_d)
        repo_attr = get_attr_from_dict([compute, shape, dtype], repository, {})
        return repo_attr

    def _parallel_postprocess(desc_d, json_str_list, attrs_list, _):
        parallel_repo = _get_parallel_repo(desc_d)
        if parallel_repo:
            # "BlockPlan" should be: [{"block_plan": x1, attr1: x2, attr2: x3}, ...]
            for i, [cur_json, cur_attr, cur_plan] in enumerate(zip(json_str_list, attrs_list, parallel_repo)):
                # When BlockPlan is active, the body should be run as single block
                cur_attr["enable_multicore"] = 0
                json_str_list[i], attrs_list[i] = _update_attr_by_repo(cur_json, json.loads(cur_json), cur_attr,
                                                                       cur_plan[ConstructKey.ATTRS], False)
        else:
            for i, [cur_json, cur_attr] in enumerate(zip(json_str_list, attrs_list)):
                json_str_list[i], attrs_list[i] = _update_attr_by_repo(
                    cur_json, json.loads(cur_json), cur_attr, None, False)

        return json_str_list, attrs_list

    def _stitch_postprocess(desc_d, stitch_jsons, attrs_list, _):
        def _stitch_combine_attrs(common_attr, sub_attrs):
            combine_attrs = []
            for i, a in enumerate(sub_attrs):
                new_sub_attrs = {}
                for k, v in common_attr.items():
                    new_sub_attrs[k] = v
                if a:
                    key = "sub_attr_" + str(i + 1)
                    new_sub_attrs[key] = {}
                    for k, v in a.items():
                        new_sub_attrs[key][k] = v
                combine_attrs.append(new_sub_attrs)
            return combine_attrs

        origin_stitch_attrs = attrs_list[0]
        if origin_stitch_attrs.get("peeling") is None:
            # Read buffer stitch attr from repo
            stitch_repo = _get_stitch_repo(desc_d)
            if stitch_repo.get("peeling") is not None:
                origin_stitch_attrs.update(stitch_repo)
            elif "online_tuning" in attr:
                # If buffer stitch attr not in repo, use online tuning
                tuning_attr = _get_online_tune_attr(json.dumps(desc_d), origin_stitch_attrs,
                                                    get_repository_file_path("repository.json"))
                origin_stitch_attrs.update(tuning_attr)
        # Update sub json attr
        common_attr, stitch_sub_attrs = split_stitch_attr(origin_stitch_attrs, len(stitch_jsons))
        for i, cur_json_str in enumerate(stitch_jsons):
            stitch_jsons[i], stitch_sub_attrs[i] = _update_attr_by_repo(
                cur_json_str, json.loads(cur_json_str), stitch_sub_attrs[i], {})
        stitch_attrs = _stitch_combine_attrs(common_attr, stitch_sub_attrs)

        return stitch_jsons, stitch_attrs

    def _normal_postprocess(desc_d, json_str_list, attrs_list, poly):
        _ = (desc_d, poly)  # For unused warning...
        for i, (cur_json_str, cur_attr) in enumerate(zip(json_str_list, attrs_list)):
            json_str_list[i], attrs_list[i] = _update_attr_by_repo(
                cur_json_str, json.loads(cur_json_str), cur_attr)
        return json_str_list, attrs_list

    post_funcs = {
        ConstructType.PARALLEL: _parallel_postprocess,
        ConstructType.STITCH: _stitch_postprocess,
        ConstructType.NORMAL: _normal_postprocess,
    }
    segment_tree, segment_infos = get_construct_args(desc_s_in, attr, post_funcs)
    process = desc_d_in["process"]

    func = tvm.get_global_func("lower_composite_to_module")
    if "ret_mode" in attr:
        return _build_for_tuning(attr, func, process, segment_tree, segment_infos)
    return func(process, True, segment_tree, segment_infos)


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
    attrs = _set_attrs(desc_d, attrs, poly)
    if "enable_elementwise_flatten" not in attrs.keys():
        attrs["enable_elementwise_flatten"] = False
    if backend == 'aicore':
        return _build_to_module_ascend(desc_s, desc_d, attrs, use_repo)
    else:
        return _build_to_module(desc_s, desc_d, attrs, poly)


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
    elif backend == "cpu":
        attr = _update_attrs_cpu(all_ops, attr, True)
    else:
        attr = _update_attrs_ascend(all_ops, attr)

    segment_tree, segment_infos = get_tune_construct_args(kernel_desc, attr)
    tune_composite = tvm.get_global_func("tune_composite")
    ret = tune_composite(backend, True, segment_tree, segment_infos)
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
