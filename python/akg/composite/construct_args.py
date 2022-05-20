# coding: utf-8
# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
import json
from os import stat
from .split_block import parallel_json_split
from .split_stitch import stitch_json_split
from .split_pass_through import pass_through_json_split, concat_json_split


class ConstructType:
    P = "P"
    NORMAL = "Normal"
    TOT = "Tot"
    PARALLEL = "Parallel"
    STITCH = "Stitch"
    TUNE = "Tune"
    ELEMANY = "ElemAny"
    UNKNOWN = "Unknow"
    CONCAT = "PassThroughConcat"


class ConstructKey:
    ALLOC_MAP = "alloc_map"
    REUSE_MAP = "reuse_map"
    CLEAN_OP_MAP = "clean_op_map"
    STITCH_ORIGIN_JSON = "stitch_origin_json"
    SEGMENT_TREE = "segment_tree"
    JSON_STR = "json_str"
    ATTRS = "attrs"
    KERNEL_INPUTS = "kernel_inputs"
    KERNEL_OUTPUTS = "kernel_outputs"
    GET_STMT = "get_stmt"
    OUTPUT_TENSOR_SHAPES = "output_tensor_shapes"
    INPUT_TENSOR_SHAPES = "input_tensor_shapes"
    KERNEL_NAME = "kernel_name"
    CONCAT_SHAPES = "concat_shapes"


def _reducemax_pattern(kernel_info):
    """
    Check ReduceMax and return reduce_size when true.
    """
    for op in kernel_info['op_desc']:
        if op['name'] == 'ReduceMax':
            input_shape = op['input_desc'][0][0]['shape']
            batch_size = input_shape[0]
            reduce_size = batch_size * input_shape[1] * input_shape[2]
            return True, reduce_size
    return False, 0


def should_enable_attr(kernel_info, key):
    """
    Check whether enable the attribute denoted by key for this kernel or not.
    """
    for op in kernel_info["op_desc"]:
        if not op["attr"]:
            continue
        for attr in op["attr"]:
            if attr["name"] == key and attr["value"]:
                return True
    return False


def update_attrs(kernel_info, key, attrs):
    for op in kernel_info["op_desc"]:
        if not op["attr"]:
            continue
        for attr in op["attr"]:
            if attr["name"] == key:
                attrs[key] = attr["value"]
                return
    return


def _set_reducemax_attrs(desc_d, attrs):
    """
    Add addition attributes for ReduceMax.
    """
    backend = desc_d['process']
    if backend == 'cuda' and _reducemax_pattern(desc_d)[0]:
        attrs['enable_tile_c0'] = True
        elem_per_thread = 4
        blockdim_x = 64
        blockdim_y = 16
        griddim_x = 1
        griddim_y = _reducemax_pattern(desc_d)[1] / (blockdim_y * elem_per_thread)
        attrs['dim'] = ' 0 0 128 64 b1 t1 0 1 128 128 b0 t0'
        attrs['bind_block'] = ''.join([str(griddim_x), ' ', str(griddim_y)])
        attrs['bind_thread'] = ''.join([str(blockdim_x), ' ', str(blockdim_y)])
    return attrs


class ConstructNode:
    """
    Lower node for kernel construction.
    """
    def __init__(self, parent, ntype, nid):
        r"""
        Initialization.
        Args:
            parent (ConstructNode): Parent node object.
            ntype (str): Current node's type.
            nid (int): Current node index for its type.
        """
        self.subs = []
        self.ntype = ntype
        self.nid = nid
        if parent:
            parent.subs.append(self)

    def name(self):
        """
        Get this node's name.
        """
        return "{}{}".format(self.ntype, self.nid)


class ConstructTree:
    """
    Lower tree for kernel construction.
    """

    def __init__(self):
        self.root = None
        self.type_collect = {}
        self.knot_stack = []
        self.nodes = []

    def add(self, ntype: str):
        """
        Add new lower node.
        """
        nid = self.type_collect.get(ntype, -1) + 1
        if not self.knot_stack:
            new_node = ConstructNode(None, ntype, nid)
            self.root = new_node
        else:
            new_node = ConstructNode(self.knot_stack[-1], ntype, nid)
        self.nodes.append(new_node)
        self.type_collect[ntype] = nid

    def add_knot(self, ntype: str):
        """
        Add new knot for this tree, and get ready to add more child nodes.
        """
        self.add(ntype)
        self.set_last_as_knot()

    def set_last_as_knot(self):
        """
        Set last lower node as current tree knot.
        """
        self.knot_stack.append(self.nodes[-1])

    def pop_knot(self):
        """
        Pop out current tree knot.
        """
        if self.knot_stack:
            self.knot_stack.pop()

    def get_construct_type(self):
        """
        Get whole construction type string by tracing this construct tree.
        """
        def _recursive_get(node):
            res = node.name()
            if node.subs:
                sub_res = []
                for sub in node.subs:
                    sub_res.append(_recursive_get(sub))
                res += "[{}]".format(",".join(sub_res))
            return res

        if not self.root:
            return ""

        return _recursive_get(self.root)


class ChildNodeTrace:
    """
    Custom with environment for latter child in construct tree.
    """

    def __init__(self, ct: ConstructTree):
        self.trace = ct

    def __enter__(self):
        return self.trace

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.trace.pop_knot()


class BaseNodeAnalyze:
    @staticmethod
    def get_name():
        """
        Return type name.
        """
        raise RuntimeError("Not implemented.")

    @staticmethod
    def check_type(desc_d):
        r"""
        Check whether the json object is this type.

        Args:
            desc_d (dict): Json detail dictionary.

        Returns:
            String, whether the json is this type. Return type string if type check pass, or return UNKNOWN.
        """
        _ = (desc_d,)  # For unused warning...
        raise RuntimeError("Not implemented.")

    @staticmethod
    def extract_infos(desc_d, attrs):
        r"""
        Extract children's jsons, attributes and segment infos.

        Args:
            desc_d (dict): Json detail dictionary.
            attrs (dict): Attributes dictionary.

        Returns:
            sub_jsons (list): List of strings for children json.
            sub_attrs (list): List of dictionaries for children attributes.
            segment_infos (dict): Extracted informations for segment lower.
        """
        _ = (desc_d, attrs)  # For unused warning...
        raise RuntimeError("Not implemented.")

    @staticmethod
    def analyze_children(analyze_json_func, analye_res, ct, type_infos, poly, func):
        r"""
        Process children of type.

        Args:
            analyze_json_func (function): Analyze function for recursive call.
            analye_res (dict): Output dictionary for whole analyzation.
            ct (ConstructTree): ConstructTree object.
            type_infos (dict): Current extracted type infos(including jsons, attrs and segment_infos).
            poly (bool): Is enable polyhedral.
            func (function): Extract functions interface.
        """
        _ = (analyze_json_func, analye_res, ct, type_infos, poly, func)  # For unused warning...
        raise RuntimeError("Not implemented.")

    @staticmethod
    def multi_children_analyze(json_type, analyze_json_func, analye_res, ct, type_infos, poly, func):
        """
        Analyze multiple children.
        """
        with ChildNodeTrace(ct) as akt:
            akt.add_knot(json_type)
            jsons_list, attrs_list, _ = type_infos
            for json_s, attr in zip(jsons_list, attrs_list):
                analyze_json_func(analye_res, akt, json_s, attr, poly, func)

    @staticmethod
    def leaf_analyze(json_type, ct):
        """
        Analyze leaf information.
        """
        with ChildNodeTrace(ct) as akt:
            akt.add_knot(json_type)
            akt.add(ConstructType.P)


class NormalNodeAnalyze(BaseNodeAnalyze):
    @staticmethod
    def get_name():
        """
        Return type name.
        """
        return ConstructType.NORMAL

    @staticmethod
    def check_type(_):
        """
        Check whether the json object is NORMAL type.
        """
        # Final check.
        return NormalNodeAnalyze.get_name()

    @staticmethod
    def extract_infos(desc_d, attrs):
        """
        Extract NORMAL's children information, including jsons, attributes and segment infos.
        """
        total_jsons = [json.dumps(desc_d)]
        total_attrs = [attrs]
        return total_jsons, total_attrs, {}

    @staticmethod
    def analyze_children(analyze_json_func, analye_res, ct, type_infos, poly, func):
        """
        Process children of NORMAL.
        """
        _ = (analyze_json_func, analye_res, type_infos, poly, func)  # For unused warning...
        BaseNodeAnalyze.leaf_analyze(NormalNodeAnalyze.get_name(), ct)


class ParallelNodeAnalyze(BaseNodeAnalyze):
    @staticmethod
    def get_name():
        """
        Return type name.
        """
        return ConstructType.PARALLEL

    @staticmethod
    def check_type(desc_d):
        """
        Check whether the json object is PARALLEL type.
        """
        if "parallel_fusion" in desc_d:
            return ParallelNodeAnalyze.get_name()
        return ConstructType.UNKNOWN

    @staticmethod
    def extract_infos(desc_d, attrs):
        """
        Extract PARALLEL's children information, including jsons, attributes and segment infos.
        """
        def _update_bool(key, kernel_info, target_dict):
            new_bool = should_enable_attr(kernel_info, key)
            target_bool = target_dict.get(key, None)
            target_dict[key] = target_bool | new_bool

        block_jsons, input_tensor_name, output_tensor_name = parallel_json_split(desc_d)
        if desc_d["parallel_fusion"]["fusion_type"] == "block_pipeline_fusion":
            attrs["pipeline_groups"] = desc_d["parallel_fusion"]['type_info']
        attrs_list = []
        for i, _ in enumerate(block_jsons):
            cur_attrs = dict()
            for key in attrs:
                cur_attrs.update({key : attrs.get(key)})
            _update_bool("enable_atomic_add", json.loads(block_jsons[i]), cur_attrs)
            _update_bool("is_csr", json.loads(block_jsons[i]), cur_attrs)
            attrs_list.append(cur_attrs)
        total_jsons = block_jsons
        total_attrs = attrs_list
        type_name = ParallelNodeAnalyze.get_name()
        segment_infos = {type_name: {ConstructKey.KERNEL_INPUTS: input_tensor_name,
                                     ConstructKey.KERNEL_OUTPUTS: output_tensor_name}}
        return total_jsons, total_attrs, segment_infos

    @staticmethod
    def analyze_children(analyze_json_func, analye_res, ct, type_infos, poly, func):
        """
        Process children of PARALLEL.
        """
        type_name = ParallelNodeAnalyze.get_name()
        BaseNodeAnalyze.multi_children_analyze(type_name, analyze_json_func, analye_res, ct, type_infos, poly, func)


class StitchNodeAnalyze(BaseNodeAnalyze):
    @staticmethod
    def get_name():
        """
        Return type name.
        """
        return ConstructType.STITCH

    @staticmethod
    def check_type(desc_d):
        """
        Check whether the json object is STITCH type.
        """
        if "buffer_stitch" in desc_d:
            stitch_nodes = list(node for node_list in desc_d["buffer_stitch"]["stitch_op"] for node in node_list)
            for op in desc_d["op_desc"]:
                op_outputs = list(out["tensor_name"] for out in op["output_desc"])
                if set(stitch_nodes).intersection(set(op_outputs)):
                    return StitchNodeAnalyze.get_name()
        return ConstructType.UNKNOWN

    @staticmethod
    def extract_infos(desc_d, attrs):
        """
        Extract STITCH's children information, including jsons, attributes and segment infos.
        """
        stitch_jsons, input_tensor_name, output_tensor_name, alloc_map, reuse_map, clean_op_map \
            = stitch_json_split(desc_d)
        attrs = _set_reducemax_attrs(desc_d, attrs)
        attrs["enable_stitch_fusion"] = True
        new_attrs = dict()
        for key in attrs:
            new_attrs.update({key : attrs.get(key)})
        attrs_list = list(new_attrs for i, _ in enumerate(stitch_jsons))
        stitch_origin_json = json.dumps(desc_d)
        total_jsons = stitch_jsons
        total_attrs = attrs_list
        type_name = StitchNodeAnalyze.get_name()
        segment_infos = {
            type_name: {
                ConstructKey.KERNEL_INPUTS: input_tensor_name, ConstructKey.KERNEL_OUTPUTS: output_tensor_name,
                ConstructKey.ALLOC_MAP: alloc_map, ConstructKey.REUSE_MAP: reuse_map,
                ConstructKey.CLEAN_OP_MAP: clean_op_map,
                ConstructKey.STITCH_ORIGIN_JSON: stitch_origin_json
            }
        }
        return total_jsons, total_attrs, segment_infos

    @staticmethod
    def analyze_children(analyze_json_func, analye_res, ct, type_infos, poly, func):
        """
        Process children of STITCH.
        """
        type_name = StitchNodeAnalyze.get_name()
        BaseNodeAnalyze.multi_children_analyze(type_name, analyze_json_func, analye_res, ct, type_infos, poly, func)


class TotNodeAnalyze(BaseNodeAnalyze):
    @staticmethod
    def get_name():
        """
        Return type name.
        """
        return ConstructType.TOT

    @staticmethod
    def check_type(desc_d):
        """
        Check whether the json object is TOT type.
        """
        all_ops = set(list(op['name'] for op in desc_d['op_desc']))
        if any(list(i in all_ops for i in ["Gather", "TensorScatterAdd"])):
            return TotNodeAnalyze.get_name()
        return ConstructType.UNKNOWN

    @staticmethod
    def extract_infos(desc_d, attrs):
        """
        Extract TOT's children information, including jsons, attributes and segment infos."""
        total_jsons = [json.dumps(desc_d)]
        total_attrs = [attrs]
        return total_jsons, total_attrs, {}

    @staticmethod
    def analyze_children(analyze_json_func, analye_res, ct, type_infos, poly, func):
        """
        Process children of TOT.
        """
        _ = (analyze_json_func, analye_res, type_infos, poly, func)  # For unused warning...
        BaseNodeAnalyze.leaf_analyze(TotNodeAnalyze.get_name(), ct)

class PassThroughNodeAnalyze(BaseNodeAnalyze):
    def __init__(self, op_name, skip_ops_list):
        self.op_name = op_name
        self.skip_ops_list = skip_ops_list
        
    def get_name(self):
        return "PassThrough" + self.op_name
    
    def check_type(self, desc_d):
        """Check whether the json object is PassThrough type."""
        if  self.op_name in desc_d['op']:
            return self.get_name()
        return ConstructType.UNKNOWN
    
    def extract_infos(self, desc_d, attrs):
        """Extract PassThrough's children information, including jsons, attributes and segment infos."""
        self.total_jsons, self.input_tensor_names, self.output_tensor_names, self.input_tensor_shapes, self.output_tensor_shapes = pass_through_json_split(desc_d, self.skip_ops_list)
        
        self.total_attrs = [attrs.copy() for _ in range(len(self.total_jsons))]
        type_name = self.get_name()

        self.segment_infos = {
            type_name: {
                ConstructKey.KERNEL_NAME: desc_d['op'],
                ConstructKey.KERNEL_INPUTS: self.input_tensor_names,
                ConstructKey.KERNEL_OUTPUTS: self.output_tensor_names,
                ConstructKey.INPUT_TENSOR_SHAPES: self.input_tensor_shapes,
                ConstructKey.OUTPUT_TENSOR_SHAPES: self.output_tensor_shapes,
            }
        }

        return self.total_jsons, self.total_attrs, self.segment_infos
    
    def analyze_children(self, analyze_json_func, analye_res, ct, type_infos, poly, func):
        """Process children of PassThrough."""
        type_name = self.get_name()
        BaseNodeAnalyze.multi_children_analyze(type_name, analyze_json_func, analye_res, ct, type_infos, poly, func)

class ConcatNodeAnalyze(PassThroughNodeAnalyze):
    def __init__(self):
        super().__init__("Concat", ["Concat"])
    
    def extract_infos(self, desc_d, attrs):
        super().extract_infos(desc_d, attrs)
        concat_shapes = concat_json_split(desc_d)
        self.segment_infos[self.get_name()][ConstructKey.CONCAT_SHAPES] = concat_shapes
        return self.total_jsons, self.total_attrs, self.segment_infos

class ElemAnyNodeAnalyze(BaseNodeAnalyze):
    @staticmethod
    def get_name():
        """
        Return type name.
        """
        return ConstructType.ELEMANY

    @staticmethod
    def check_type(desc_d):
        """
        Check whether the json object is ELEMANY type.
        """
        if "ElemAny" in set(op['name'] for op in desc_d['op_desc']):
            return ElemAnyNodeAnalyze.get_name()
        return ConstructType.UNKNOWN

    @staticmethod
    def extract_infos(desc_d, attrs):
        """
        Extract ELEMANY's children information, including jsons, attributes and segment infos."""
        total_jsons = [json.dumps(desc_d)]
        total_attrs = [attrs]
        return total_jsons, total_attrs, {}

    @staticmethod
    def analyze_children(analyze_json_func, analye_res, ct, type_infos, poly, func):
        """
        Process children of ELEMANY.
        """
        _ = (analyze_json_func, analye_res, type_infos, poly, func)  # For unused warning...
        BaseNodeAnalyze.leaf_analyze(ElemAnyNodeAnalyze.get_name(), ct)

class AnalyzeUtils:
    check_cls_list = [ParallelNodeAnalyze, StitchNodeAnalyze, ConcatNodeAnalyze(), TotNodeAnalyze, ElemAnyNodeAnalyze, NormalNodeAnalyze]

    @staticmethod
    def check_json_type(desc_d):
        for check_cls in AnalyzeUtils.check_cls_list:
            type_result = check_cls.check_type(desc_d)
            if type_result != ConstructType.UNKNOWN:
                return type_result

    @staticmethod
    def extract_infos_by_type(jtype, desc_d, attrs):
        for cls in AnalyzeUtils.check_cls_list:
            if cls.get_name() == jtype:
                return cls.extract_infos(desc_d, attrs)
        raise RuntimeError("Not support type: {}".format(jtype))

    @staticmethod
    def analyze_children_by_type(jtype, analyze_json_func, analye_res, ct, type_infos, poly, func):
        for cls in AnalyzeUtils.check_cls_list:
            if cls.get_name() == jtype:
                cls.analyze_children(analyze_json_func, analye_res, ct, type_infos, poly, func)
                return
        raise RuntimeError("Not support type: {}".format(jtype))


def analyze_json(analye_res: dict, ct: ConstructTree, json_str, attrs, poly, extract_func):
    def _update_res(res, need_update):
        def _update(res, name, new_append):
            value = res.get(name, [])
            value.append(new_append)
            res[name] = value
        jsons_list, attrs_list, segment_infos = need_update
        # Only update json and attrs when in leaf node.
        if len(jsons_list) == 1 and len(attrs_list) == 1:
            _update(res, ConstructType.P, {ConstructKey.JSON_STR: jsons_list[0], ConstructKey.ATTRS: attrs_list[0]})
        if not isinstance(segment_infos, dict):
            raise ValueError("Need update infos should be a dictionary, but got {}".format(type(segment_infos)))
        for k, vs in segment_infos.items():
            _update(res, k, vs)

    json_type, type_infos = extract_func(json.loads(json_str), attrs, poly)
    _update_res(analye_res, type_infos)

    AnalyzeUtils.analyze_children_by_type(json_type, analyze_json, analye_res, ct, type_infos, poly, extract_func)


def _extract_type_infos(desc_d, attrs, poly, post_funcs: dict):
    jtype = AnalyzeUtils.check_json_type(desc_d)
    total_jsons, total_attrs, segment_infos = AnalyzeUtils.extract_infos_by_type(jtype, desc_d, attrs)

    if post_funcs.get(jtype, None):
        total_jsons, total_attrs = post_funcs[jtype](desc_d, total_jsons, total_attrs, poly)

    return jtype, (total_jsons, total_attrs, segment_infos)


def get_construct_args(json_str, attrs, post_funcs):
    segment_infos = {}
    ct = ConstructTree()
    analyze_json(segment_infos, ct, json_str, attrs, True,
                lambda d, a, p: _extract_type_infos(d, a, p, post_funcs))
    segment_tree = ct.get_construct_type()
    return segment_tree, segment_infos


def get_tune_construct_args(kernel_desc, attr):
    desc_d = json.loads(kernel_desc)
    json_type = AnalyzeUtils.check_json_type(desc_d)
    if json_type not in (ConstructType.NORMAL, ConstructType.TOT):
        raise RuntimeError("Not support tune for {} now".format(json_type))

    segment_infos = {ConstructType.P: [{ConstructKey.JSON_STR: kernel_desc, ConstructKey.ATTRS: attr}]}
    segment_tree = "{}0[P0]".format(json_type)
    return segment_tree, segment_infos


def get_stmt_for_tune(segment_infos):
    segment_infos[ConstructType.TUNE] = [{ConstructKey.GET_STMT: True}]
    return segment_infos


def add_attrs_in_segment_infos(segment_infos, key, value):
    for p_info in segment_infos[ConstructType.P]:
        new_attrs = p_info.get(ConstructKey.ATTRS, {})
        new_attrs[key] = value
        p_info[ConstructKey.ATTRS] = new_attrs
    return segment_infos
