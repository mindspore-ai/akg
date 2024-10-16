#!/usr/bin/env python3
# coding: utf-8
# Copyright 2020-2023 Huawei Technologies Co., Ltd
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
from collections.abc import Iterable
import akg
import math
from akg import tvm
from tvm.autotvm.env import AutotvmGlobalScope
from akg.utils.util import parse_workspace_map
from akg.utils.tbe_codegen_utils import build_tbe_codegen
from akg.utils.kernel_exec import ReturnType, is_symbolic_tiling
from .split_stitch import split_stitch_attr
from .construct_args import ConstructType, ConstructKey
from .construct_args import get_construct_args, get_tune_construct_args, \
    should_enable_attr, get_stmt_for_tune, add_attrs_in_segment_infos
from utils.util import get_ascend_type

matmul_keys = [
    "m0", # [max(16,m)%m0==0, m0%16==0?]
    "k0", # [max(16,m)%m0==0,]
    "n0", # [max(16,m)%m0==0,]
    "swizzlCount",#[1-blockDim],
    "swizzlDirect"#[0,1] nz or zn format
]

matmul_calc_keys = [
    "mLoop",#=max(16,m)/m0
    "kLoop",#=max(16,m)/m0
    "nLoop",#=max(16,m)/m0
    "coreLoop",#=mLoop/nLoop/kLoop
]

pa_keys = [
    "headSplit",
]

kv_keys = [
    "num_token_tile",
    "n_burst"
]

def generate_trait(desc):
    """
    generate trait of kernel description
    """

    def get_op_trait(op, counter, tensor_idx):
        input_idx = []
        if op['input_desc']:
            for input_desc in op['input_desc']:
                if input_desc[0].get('value', None) is None:
                    input_idx.append(counter - tensor_idx[input_desc[0]['tensor_name']])
        input_idx.sort()
        input_idx_str = ''.join(str(i) for i in input_idx)
        op_trait = op['name'] + input_idx_str
        if  op['name'].find("MatMul") != -1:
            for attr in op['attr']:
                if attr['name'] == "transpose_a":
                    transpose_a = str(int(attr['value']))
                if attr['name'] == "transpose_b":
                    transpose_b = str(int(attr['value']))
            op_trait += '_' + transpose_a + '_' + transpose_b
        return op_trait

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
            op_trait = get_op_trait(op, counter, tensor_idx)
            traits.append(op_trait)
            for op_out_desc in op['output_desc'] if op['output_desc'] is not None else []:
                tensor_idx[op_out_desc['tensor_name']] = counter
                counter += 1
        output_idx = []
        for out_desc in desc['output_desc'] if desc['output_desc'] is not None else []:
            output_idx.append(tensor_idx.get(out_desc.get('tensor_name', "")))
        output_idx.sort()
        traits.append(''.join(str(i) for i in output_idx))
        return '.'.join(traits)

    def append_trait(traits, data):
        if traits and traits[-1].rstrip('-') == data:
            traits[-1] += '-'
        else:
            traits.append(data)

    def generate_shape_trait():
        traits = []
        for in_desc in desc['input_desc'] if desc['input_desc'] is not None else []:
            shape_s = '_'.join(str(i) for i in in_desc[0]['shape'])
            append_trait(traits, shape_s)
        for out_desc in desc['output_desc'] if desc['output_desc'] is not None else []:
            shape_s = '_'.join(str(i) for i in out_desc['shape'])
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
    if all(map(lambda x: x == 1, list(out_shape[x] for x in range(axis_len - 2)))):
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


def _update_target_info(desc_d, attr):
    target_info = desc_d.get("target_info")
    if not target_info:
        return attr

    process = desc_d.get("process")
    if process == "cuda":
        # auto detect proper gpu device type according to compute capability description
        if target_info.get("compute_capability") == "8.0":
            attr["device_type"] = "a100"
        elif target_info.get("compute_capability") == "7.0":
            attr["device_type"] = "v100"
    elif process == "cpu":
        if target_info.get("feature"):
            attr["feature"] = target_info.get("feature")

    return attr


def _update_compile_attr(desc_d, attr):
    # For user defined akg compile attr

    attr = _update_target_info(desc_d, attr)

    if desc_d.get('op_desc') is None:
        return attr
    for op in desc_d.get('op_desc'):
        op_attrs = op.get("attr", {})
        if not isinstance(op_attrs, Iterable):
            continue
        compile_attrs = list(item.get("value", "") for item in op_attrs if isinstance(
            item, dict) and item.get("name", "") == "func_compile_attrs")
        if compile_attrs:
            attrs_dict = json.loads(compile_attrs[0])
            for key in attrs_dict:
                attr.update({key: attrs_dict[key]})

    return attr

def update_tuned_attrs(desc_d, attrs):
    """update attrs from tuning to build process, like 'tuned_dim' -> 'dim'
    """
    tuned_attrs_list = ["tuned_dim", "tuned_bind_block", "tuned_bind_thread"]
    if desc_d.get("op_desc", None):
        return attrs
    for op in desc_d.get("op_desc"):
        if op.get("attr", None):
            continue
        for a in op.get("attr"):
            if a["name"] in tuned_attrs_list:
                name = a["name"][6:] # remove 'tuned_'
                attrs[name] = attrs.get(name, a["value"])
    return attrs


def update_dynmaic_batch_attrs(desc_d, attrs):
    """update attrs related to dynamic batch
    """
    if "dynamic_input_index" in desc_d:
        attrs["dynamic_input_index"] = desc_d["dynamic_input_index"]
    else:
        attrs["dynamic_input_index"] = ""
    return attrs


def _set_attrs(desc_d, attrs, poly):
    if "enable_atomic_add" not in attrs.keys():
        attrs["enable_atomic_add"] = should_enable_attr(desc_d, "enable_atomic_add")
        if not poly:
            attrs["enable_atomic_add"] = False
    if "is_csr" not in attrs.keys():
        attrs["is_csr"] = should_enable_attr(desc_d, "is_csr")
    if "enable_approximate_read" not in attrs.keys():
        attrs["enable_approximate_read"] = should_enable_attr(desc_d, "enable_approximate_read")
    if "enable_elementwise_flatten" not in attrs.keys():
        attrs["enable_elementwise_flatten"] = False
    attrs["enable_symbolic_tiling"] = is_symbolic_tiling(desc_d['op'])
    attrs["process"] = desc_d["process"]
    attrs = update_tuned_attrs(desc_d, attrs)
    attrs = update_dynmaic_batch_attrs(desc_d, attrs)
    if desc_d["process"] == "cpu":
        attrs["pack_matrix_b"] = False if should_enable_attr(desc_d, "pack_b") else True
    return _update_compile_attr(desc_d, attrs)


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
    """
    :param keys: [key1,key3,key3]
    :param repo: {key1:{key2:{key3:attr}}}
    :return: attr
    """
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


def read_repo_file(repo_file, is_json_load=True):
    if not os.path.exists(repo_file):
        return {}
    with open(repo_file, 'r') as f:
        repo = f.read()
    return json.loads(repo) if is_json_load else repo


def _get_default_repository_file(process):
    filename = "repository.json" if process == "aicore" else "repository_%s.json" % process
    # get the abosulte path for a file in currect dir, input is a file's name like "a.json"
    pwd = os.path.dirname(os.path.abspath(__file__))
    path_str = pwd + "/" + filename
    if not os.path.exists(path_str):
        path_str = pwd + "/../config/" + filename
        if not os.path.exists(path_str):
            raise FileNotFoundError("Can not find {} in directory {} and {}".format(filename, pwd, pwd + "/../config"))
    return path_str


def _get_repository(desc_d, attrs):
    if os.getenv('MS_GRAPH_KERNEL_TILING'):
        return read_repo_file(str(os.getenv('MS_GRAPH_KERNEL_TILING')))
    if 'buffer_stitch' in desc_d and attrs.get("process") == 'cuda':
        return {}
    if "repository_path" in attrs:
        filepath = os.path.join(os.path.realpath(attrs["repository_path"]), "repo_op_tiling.json")
        if os.path.exists(filepath):
            return read_repo_file(filepath)
    process = attrs.get("process", "aicore")
    return read_repo_file(_get_default_repository_file(process))


def _get_repo_attr(desc_d, compute, shape, dtype, repo, batchmatmul):
    repo_attr = get_attr_from_dict([compute, shape, dtype, 'metadata', 'attrs'], repo, {})
    if repo_attr and batchmatmul:
        repo_attr = _set_tiling_attrs(desc_d['output_desc'][0]['shape'], repo_attr)
    if not repo_attr:
        repo_attr = get_attr_from_dict([compute, 'metadata', 'attrs'], repo, {})
    return repo_attr


def _update_attrs_gpu(all_ops, attrs, poly):
    if poly:
        if any(i in all_ops for i in ['Argmax', 'Argmin']):
            # disable auto_fuse and akg_reduce_lib for argmax and argmin
            attrs["enable_akg_reduce_lib"] = False
            attrs["enable_auto_fuse"] = False
        elif "enable_akg_reduce_lib" not in attrs.keys():
            attrs["enable_akg_reduce_lib"] = True

        if "pragma_enable_matmul" not in attrs.keys() and any(
                i in all_ops for i in ["BatchMatMul", "MatMul", "Conv2D"]):
            attrs['pragma_enable_matmul'] = True
            attrs['enable_auto_inline'] = False
        if "pragma_enable_conv_tensor_core" not in attrs.keys() and "Conv2D" in all_ops:
            attrs["pragma_enable_conv_tensor_core"] = True
            attrs["enable_auto_fuse"] = False
        # Close general tot by default
        enable_general_tot = False
        if "has_tot_ops" not in attrs.keys() and any(i in all_ops for i in ["Gather", "TensorScatterAdd"]):
            attrs["has_tot_ops"] = enable_general_tot
    return attrs


def _update_attrs_cpu(all_ops, attrs, poly):
    if not poly:
        return attrs
    if "pragma_enable_matmul" not in attrs.keys() and any(i in all_ops for i in ["BatchMatMul", "MatMul"]):
        attrs['pragma_enable_matmul'] = True
        attrs['enable_auto_inline'] = False
        attrs['pragma_enable_schedule_maximize_coincidence'] = True
    if any([i in all_ops for i in ["Conv2D"]]):
        attrs["enable_auto_fuse"] = False
        attrs["pragma_enable_conv2d_direct"] = True
    if any([i in all_ops for i in ["Pool2D"]]):
        attrs["enable_auto_fuse"] = False
    if "feature" not in attrs.keys() and any([i in all_ops for i in ["BatchMatMul", "MatMul"]]):
        attrs["feature"] = "avx"
    return attrs


def _update_attrs_ascend(all_ops, attr):
    attr["pragma_rmselfdep"] = all(i not in all_ops for i in ["BatchMatMul", "MatMul"])
    # For the MatMul/BatchMatMul with bias, the inline is necessary
    # For the Ascend, turn 'enable_auto_inline' off for composite op by default.
    attr["enable_auto_inline"] = any(i in all_ops for i in ["BatchMatMul", "MatMul"])
    attr["multicore_loop_switch_hoist"] = "UnsortedSegmentSum" not in all_ops
    return attr

def _cpp_build(attrs, process, poly, segment_tree, segment_infos):
    if attrs.get("is_tbe_codegen"):
        func = tvm.get_global_func("lower_composite")
    else:
        func = tvm.get_global_func("lower_composite_to_module")

    if "ret_mode" in attrs:
        return _build_for_tuning(attrs, func, process, segment_tree, segment_infos)

    res = func(process, poly, segment_tree, segment_infos)
    return res


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

    process = desc_d["process"]
    file_name = "repository_" + process + ".json"

    def _update_attr_by_repo(desc_s, attrs):
        desc_d = json.loads(desc_s)
        process = desc_d["process"]
        attrs.update({"process": process})
        repository = _get_repository(desc_d, attrs)
        all_ops = set(op["name"] for op in desc_d["op_desc"])

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
            attrs = _get_online_tune_attr(desc_s, attrs, _get_default_repository_file(process))
        return desc_d, attrs

    def _post_update_attr(desc_s, attrs, poly):
        desc_d, attrs = _update_attr_by_repo(desc_s, attrs)
        all_ops = set(op["name"] for op in desc_d["op_desc"])
        if process == "cuda":
            attrs = _update_attrs_gpu(all_ops, attrs, poly)
        elif process == "cpu":
            attrs = _update_attrs_cpu(all_ops, attrs, poly)
        return attrs

    def _common_postprocess(_, json_str_list, attrs_list, poly):
        for i, (cur_json_str, cur_attr) in enumerate(zip(json_str_list, attrs_list)):
            attrs_list[i] = _post_update_attr(cur_json_str, cur_attr, poly)
        return json_str_list, attrs_list

    def _get_stitch_repo(desc_d):
        compute, shape, dtype = generate_trait(desc_d)
        repo_attr = get_attr_from_dict([compute, shape, dtype], _get_repository(file_name, desc_d), {})
        return repo_attr

    def _stitch_postprocess(desc_d, json_str_list, attrs_list, _):
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
                        new_sub_attrs.get(key)[k] = v
                combine_attrs.append(new_sub_attrs)
            return combine_attrs

        origin_stitch_attrs = attrs_list[0]
        if origin_stitch_attrs.get("peeling") is None:
            # Read buffer stitch attr from repo
            stitch_repo = _get_stitch_repo(desc_d)
            if stitch_repo.get("peeling") is not None:
                origin_stitch_attrs.update(stitch_repo)
            elif "online_tuning" in attrs:
                # If buffer stitch attr not in repo, use online tuning
                tuning_attr = _get_online_tune_attr(json.dumps(desc_d), origin_stitch_attrs,
                                                    _get_default_repository_file(process))
                origin_stitch_attrs.update(tuning_attr)
        # Update sub json attr
        common_attr, stitch_sub_attrs = split_stitch_attr(origin_stitch_attrs, len(json_str_list))
        # common_attr.update({'peeling': '0 1', 'fold_dim': False})
        for i, cur_attr in enumerate(stitch_sub_attrs):
            stitch_sub_attrs[i] = _post_update_attr(json.dumps(desc_d), cur_attr, poly)
        stitch_attrs = _stitch_combine_attrs(common_attr, stitch_sub_attrs)

        return json_str_list, stitch_attrs

    post_funcs = {
        ConstructType.PARALLEL: _common_postprocess,
        ConstructType.STITCH: _stitch_postprocess,
        ConstructType.NORMAL: _common_postprocess,
        ConstructType.TOT: _common_postprocess,
        ConstructType.CONCAT: _common_postprocess
    }
    segment_tree, segment_infos = get_construct_args(desc_s, attrs, post_funcs)
    process = desc_d["process"]

    return _cpp_build(attrs, process, poly, segment_tree, segment_infos)

def _build_to_module_ascend(desc_s_in, desc_d_in, attr, use_repo=True):
    """
    build kernel with compute description in json format
    Args:
       desc_s_in : str of compute description
       desc_d_in : dict of compute description
       attr   : dict of build attributes

    Returns:
       Module.
    """
    repository = _get_repository(desc_d_in, attr)

    def _update_attr_by_repo(desc_s, desc_d, attr, given_attrs=None, support_online_tuning=True):
        def _auto_set_single_block(desc_d, attr):
            if not attr.get("enable_multicore", None) and desc_d.get("extra", None):
                if desc_d["extra"].get("BlockMode", "") == "single_block":
                    attr["enable_multicore"] = 0
            return attr

        if attr is None:
            attr = {'dim': ''}
        all_ops = set(op['name'] for op in desc_d['op_desc'])
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
                    attr = _get_online_tune_attr(desc_s, attr, _get_default_repository_file("aicore"))
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
                        new_sub_attrs.get(key)[k] = v
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
                                                    _get_default_repository_file("aicore"))
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
    process = desc_d_in["process"]
    kernel_name = desc_d_in['op']
    ascend_type = get_ascend_type(desc_d_in)
    ascend_type_to_section = {"Ascend910A": "1.6", "Ascend310P3": "1.7",
                              "Ascend910B1": "2.1", "Ascend910B2": "2.2",
                              "Ascend910B3": "2.3", "Ascend910B4": "2.4",
                              "Ascend910_9391": "2.5", "Ascend910_9381": "2.5",
                              "Ascend910_9392": "2.5", "Ascend910_9382": "2.5",
                              "Ascend910_9372": "2.5", "Ascend910_9361": "2.5"}
    if ascend_type is not None:
        section = ascend_type_to_section.get(ascend_type, "1.6")
        config_func = akg.tvm.get_global_func("cce.set_product_section")
        config_func(section)
        if section >= "2.1":
            attr["is_tbe_codegen"] = True
            attr["pragma_modshift"] = True
    segment_tree, segment_infos = get_construct_args(desc_s_in, attr, post_funcs)

    if desc_d_in.get("backend"):
        backend_func = akg.tvm.get_global_func("cce.set_backend")
        backend_func(desc_d_in["backend"])

    if desc_d_in.get("enable_cce_lib"):
        attr["enable_cce_lib"] = True
        repository = None
        if os.getenv('MS_GRAPH_KERNEL_TILING'):
            repository = read_repo_file(str(os.getenv('MS_GRAPH_KERNEL_TILING')))
        return _build_to_module_ascend_lib(desc_s_in, kernel_name, repository)

    poly = True
    res = _cpp_build(attr, process, poly, segment_tree, segment_infos)
    if attr.get("is_tbe_codegen"):
        stmt_json = akg.tvm.save_json(res[0], "0.8.0")
        args_json = []
        for buf in res[1]:
            args_json.append(akg.tvm.save_json(buf, "0.8.0"))
        
        workspace_dict = parse_workspace_map(res[2])
        if workspace_dict is not None:
            attr["workspace"] = workspace_dict

        is_success = build_tbe_codegen(kernel_name, stmt_json, args_json, attr, ascend_type)
        if not is_success:
            raise TypeError("npu_inference codegen failed.")
        akg.tvm.get_global_func("build_host_cce")(res[1], res[2], kernel_name)
        return kernel_name
    return res

def _build_to_module_ascend_lib(desc_s_in, kernel_name, repository=None):
    
    def _convert_dim_to_attr(dim, tiling_info, compute):
        f = akg.tvm.get_global_func("cce.current_product_conf_core")
        coreNum = f("Core_num")
        if coreNum == 0:
            raise RuntimeError("Get the cce product value is None")
        if "MatMul" in compute:
            batch = tiling_info.get("batch_size")
            m = tiling_info.get("M")
            n = tiling_info.get("N")
            k = tiling_info.get("K")
            matmul_dict = dict()
            dim_list = dim.split(" ")
            if len(dim_list) < 4:
                return {}
            cnt =  0
            for i in range(2, len(dim_list), 4):
                key = matmul_keys[cnt]
                value = dim_list[i]
                matmul_dict[key] = int(value)
                cnt += 1
            mLoop = (max(16, m) - 1) // matmul_dict.get("m0") + 1
            nLoop = (max(16, n) - 1) // matmul_dict.get("n0") + 1
            kLoop = (max(16, k) - 1) // matmul_dict.get("k0") + 1
            coreLoop = batch * mLoop * nLoop
            blockDim = min(coreLoop, coreNum)
            matmul_dict["mLoop"] = mLoop
            matmul_dict["kLoop"] = kLoop
            matmul_dict["nLoop"] = nLoop
            matmul_dict["coreLoop"] = coreLoop
            matmul_dict["blockDim"] = blockDim
            return matmul_dict
        elif "PagedAttention" in compute or "PagedAttentionMask" in compute:
            num_heads = tiling_info.get("num_heads")
            num_tokens = tiling_info.get("num_tokens")
            pa_dict = dict()
            dim_list = dim.split(" ")
            if len(dim_list) < 4:
                return {}
            cnt =  0
            for i in range(2, len(dim_list), 4):
                key = pa_keys[cnt]
                value = dim_list[i]
                pa_dict[key] = int(value)
                cnt += 1
            head_split = pa_dict.get("headSplit")  # Tuning<<---- [1, num_heads]
            loopLen = (num_heads + head_split - 1) // head_split
            block = min(loopLen * num_tokens, coreNum)
            pa_dict["blockDim"] = block
            return pa_dict
        elif "ReshapeAndCache" in compute:
            num_tokens = tiling_info.get("num_tokens")

            kv_dict = dict()
            dim_list = dim.split(" ")
            cnt =  0
            for i in range(2, len(dim_list), 4):
                key = kv_keys[cnt]
                value = dim_list[i]
                kv_dict[key] = int(value)
                cnt += 1
            if "num_token_tile" in kv_dict:
                block_dim = str((int(num_tokens) - 1) // kv_dict.get("num_token_tile") + 1)
            else:
                block_dim = num_tokens
            
            kv_dict["block_dim"] = block_dim
            kv_dict["kernel_version"] = 2
            return kv_dict
        else:
            return {}
    
    def _get_all_shape(shapes):
        shape_split = shapes.split(".")
        shape_list = []
        for shape in shape_split:
            if "-" in shape:
                tmp_shape = shape.split("-")[0]
                for _ in range(shape.count("-") + 1):
                    shape_list.append(tmp_shape)
            else:
                shape_list.append(shape)
        return shape_list
    
    def _get_tiling_info(desc_s):
        compute, shape, dtype = generate_trait(desc_s)
        tiling_info = {}
        if "MatMul" in compute:
            trans_a = compute.split("_")[1]
            trans_b = compute.split("_")[-1].split(".")[0]

            shape_list = _get_all_shape(shape)
            bias_flag = int(len(shape_list) > 3)
            tensor_A = shape_list[0]
            tensor_B = shape_list[1]

            tensor_A_split = tensor_A.split("_")

            if trans_a == "1":
                M = int(tensor_A_split[-1])
                K = int(tensor_A_split[-2])
            else:
                M = int(tensor_A_split[-2])
                K = int(tensor_A_split[-1])

            tensor_B_split = tensor_B.split("_")
            if trans_b == "1":
                N = int(tensor_B_split[-2])
            else:
                N = int(tensor_B_split[-1])
            batch_size = 1
            if len(tensor_A_split) > 2 and len(tensor_B_split) == 2:
                # some bmm in onnx will have multi batch such as [b1,b0,m,k]*[k,n]
                batch_sizes = [int(i) for i in tensor_A.split("_")[:-2]]
                # change bmm with shape [b1,b0,m,k]*[k,n] to matmul with shape [b1*b0*m,k]*[k,n]
                for b in batch_sizes:
                    M = M * b
                batch_size = 1
            elif len(tensor_A_split) == len(tensor_B_split) and len(tensor_B_split) > 2:
                # common bmm [b1,b0,m,k]*[b1,b0,k,n]
                batch_sizes = [int(i) for i in tensor_A.split("_")[:-2]]
                for b in batch_sizes:
                    batch_size = batch_size * b
            op_type = "MatMul"
            if bias_flag:
                op_type = "MatMulMix"
            tensor_a_type = str(dtype.split("-")[0])
            tiling_info = {"batch_size":batch_size, "M": M, "N": N, "K": K, "trans_a": int(trans_a), "trans_b": int(trans_b),
                           "tensor_A_type": tensor_a_type, "bias_flag": bias_flag, "op_type": op_type}
        elif "PagedAttention" in compute or "PagedAttentionMask" in compute:
            shape_list = _get_all_shape(shape)
            query = shape_list[0]
            key_cache = shape_list[1]
            table_shape = shape_list[3]

            num_tokens = int(query.split("_")[0])
            num_heads = int(query.split("_")[1])
            embedding_size = int(query.split("_")[2])
            num_blocks = int(key_cache.split("_")[0])
            block_size = int(key_cache.split("_")[1])
            kv_heads = int(key_cache.split("_")[2])

            max_num_blocks_per_query = int(table_shape.split("_")[1])
            tor = float(1.0 / math.sqrt(1.0 * embedding_size))

            tiling_info = {"num_tokens": num_tokens, "num_heads": num_heads, "embedding_size": embedding_size, 
                           "num_blocks": num_blocks, "block_size": block_size, "max_num_blocks_per_query": max_num_blocks_per_query,
                           "tor": tor, "kv_heads": kv_heads, "op_type": "PagedAttention"}
            if "PagedAttentionMask" in compute:
                mask_shape = shape_list[5]
                tiling_info["mask"] = list(map(int, mask_shape.split("_")))
                tiling_info["op_type"] = "PagedAttentionMask"
        elif "ReshapeAndCache" in compute:
            shape_list = _get_all_shape(shape)
            kv = shape_list[0]

            num_tokens = int(kv.split("_")[0])
            num_heads = int(kv.split("_")[1])
            head_size = int(kv.split("_")[2])

            tiling_info = {"num_tokens": num_tokens, "num_heads": num_heads, "head_size": head_size, 
                           "op_type": "ReshapeAndCache"}
        elif "Transpose" in compute:
            shape_list = _get_all_shape(shape)
            input_shape = shape_list[0]

            axis_1 = int(input_shape.split("_")[0])
            axis_2 = int(input_shape.split("_")[1])
            axis_3 = int(input_shape.split("_")[2])

            tiling_info = {"op_type": "AddReshapeTranspose", "axis_1":axis_1, "axis_2": axis_2, "axis_3":axis_3}
        
        tiling_info["use_repo"] = 0
        if  repository is not None:
            repo_attr = get_attr_from_dict([compute, shape, dtype, 'metadata', 'attrs', 'dim'], repository, {})
            if len(repo_attr) > 0:
                repo_attr = _convert_dim_to_attr(repo_attr, tiling_info, compute)
                tiling_info = merge_attrs(tiling_info, repo_attr)
                tiling_info["use_repo"] = 1
        tiling_info["arch"] = json.loads(desc_s_in)["target_info"]["arch"]
        return tiling_info

    func = tvm.get_global_func("build_cce_lib")
    tiling_info = _get_tiling_info(json.loads(desc_s_in))
    func(kernel_name, tiling_info, None)
    return kernel_name

def _set_backend(desc_d):
    desc_d_process = desc_d
    for i, op in enumerate(desc_d.get("op_desc")):
        op_attrs = op.get("attr", [])
        op_name = op.get("name", "")
        if op_name != "UnsortedSegmentSum":
            continue
        op_attrs.append({'data_type': 'string', 'name': 'process', 'value': desc_d['process']})
        op["attr"] = op_attrs
        desc_d_process["op_desc"][i] = op
    desc_s = json.dumps(desc_d_process)
    return desc_s


def _set_cuda_compute_capability(desc_d):
    """set the compute_capability from info file"""
    sm_str = "".join(desc_d.get("target_info", {}).get("compute_capability", "0.0").split("."))
    if sm_str != "00":
        AutotvmGlobalScope.current.cuda_target_arch = "sm_" + sm_str


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
        desc_d = json.loads(kernel_desc)
    else:
        if not isinstance(kernel_desc, dict):
            raise TypeError("kernel_desc should be a dict, but get a {}".format(type(kernel_desc)))
        desc_d = kernel_desc

    from akg.ms.info_version_adapt import InfoVersionAdapt
    info_adapter = InfoVersionAdapt(desc_d)
    ret = info_adapter.run()
    if not ret:
        raise RuntimeError(info_adapter.msg)
    desc_s = _set_backend(desc_d)

    if attrs is None:
        attrs = dict()
    backend = desc_d['process']
    attrs = _set_attrs(desc_d, attrs, poly)
    # set compute_capability for cuda backend
    if backend == "cuda":
        _set_cuda_compute_capability(desc_d)
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
    from akg.ms.info_version_adapt import InfoVersionAdapt
    info_adapter = InfoVersionAdapt(desc_d)
    if not info_adapter.run():
        raise RuntimeError(info_adapter.msg)
    kernel_desc = _set_backend(desc_d)
    backend = desc_d['process']
    all_ops = set(op['name'] for op in desc_d['op_desc'])
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
