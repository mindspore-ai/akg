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

"""Automatic differentiation of tensor expressions."""
import akg
from akg.tvm._ffi.function import _init_api
from akg.tvm._ffi.node import NodeBase, register_node
from akg.utils.format_transform import get_shape

_init_api("akg.autodiff")


@akg.tvm.register_func("akg.autodiff.export_to_DOT")
def export_to_dot(tensors, filename="test.dot"):
    """
    Export computation tree of tensors to a DOT file.

    Args:
        tensors: A single/list/array of input tensors.
        filename: the name of the DOT file to be generated.
    """

    def export_tensor_shape(a_shape):
        result = "("
        for _, a_shp in enumerate(a_shape):
            result = "{}{}, ".format(result, str(a_shp.value))
        result = result + ")"
        return result

    def recursive_collect_nodes(tensor, exported_op_nodes, repeat_name):
        if tensor in exported_op_nodes:
            return exported_op_nodes, repeat_name

        if not exported_op_nodes:
            exported_op_nodes = {tensor: tensor.op.name}
        else:
            if tensor.op.name in exported_op_nodes.values():
                exported_op_nodes[tensor] = tensor.op.name + '_r' + str(repeat_name)
                repeat_name = repeat_name + 1
            else:
                exported_op_nodes[tensor] = tensor.op.name
        # exported_op_nodes[tensor] contains the name in DOT for "tensor"
        # If name is duplicated, a postfix '-r' + number is add to the end
        for child in tensor.op.input_tensors:
            if child not in exported_op_nodes:
                exported_op_nodes, repeat_name = recursive_collect_nodes(
                    child, exported_op_nodes, repeat_name)
        return exported_op_nodes, repeat_name

    def export_node_name(tensor):
        if isinstance(tensor.op, akg.tvm.tensor.ComputeOp):
            if isinstance(tensor.op.body[0], akg.tvm.expr.Reduce):
                tensor_opcode_name = 'Reduce'
            elif isinstance(tensor.op.body[0], akg.tvm.expr.Mul):
                tensor_opcode_name = '*'
            elif isinstance(tensor.op.body[0], akg.tvm.expr.Add):
                tensor_opcode_name = '+'
            elif isinstance(tensor.op.body[0], akg.tvm.expr.Sub):
                tensor_opcode_name = '-'
            elif isinstance(tensor.op.body[0], akg.tvm.expr.Div):
                tensor_opcode_name = '/'
            elif isinstance(tensor.op.body[0], akg.tvm.expr.Call):
                tensor_opcode_name = 'Call {}'.format(tensor.op.body[0].name)
            elif isinstance(tensor.op.body[0], akg.tvm.expr.Cast):
                tensor_opcode_name = 'Cast: {} => {}'.format(tensor.op.input_tensors[0].dtype, tensor.dtype)
            else:
                tensor_opcode_name = 'Unsupported yet OP'
            tensor_node_name = '    "{}" [label = "{}\\n{}; {}\\n{}"; shape = ellipse; style = filled; color = lightgrey];\
                '.format(exported_op_nodes[tensor], exported_op_nodes[tensor],
                         export_tensor_shape(tensor.shape), tensor.dtype, tensor_opcode_name)
        else:  # isinstance(tensor.op,akg.tvm.tensor.PlaceholderOp):
            tensor_node_name = '    "{}" [label = "{}\\n{}"; shape = box; style = filled; color = \
                lightseagreen];'.format(exported_op_nodes[tensor], exported_op_nodes[tensor],
                                        export_tensor_shape(tensor.shape))
        return tensor_node_name

    def recursive_export_nodes_name(tensor, f, exported_op_nodes):
        for child in tensor.op.input_tensors:
            recursive_export_nodes_name(child, f, exported_op_nodes)

        if isinstance(tensor.op, akg.tvm.tensor.ComputeOp) and \
                isinstance(tensor.op.body[0], (akg.tvm.expr.Mul, akg.tvm.expr.Add, akg.tvm.expr.Sub,
                                               akg.tvm.expr.Div)) and len(tensor.op.input_tensors) < 2:
            if isinstance(tensor.op.body[0].a, akg.tvm.expr.FloatImm):
                tensor_node_name = '    "Const_a_{}" [label = "{}\\n{}"; shape = box; style = filled; color \
                    = lightseagreen];'.format(exported_op_nodes[tensor], str(tensor.op.body[0].a.value),
                                              tensor.op.body[0].a.dtype)
                f.write(tensor_node_name)
                f.write("\n")
            if isinstance(tensor.op.body[0].b, akg.tvm.expr.FloatImm):
                tensor_node_name = '    "Const_b_{}" [label = "{}\\n{}"; shape = box; style = filled; color = lightseagreen];'\
                    .format(exported_op_nodes[tensor], str(tensor.op.body[0].b.value), tensor.op.body[0].b.dtype)
                f.write(tensor_node_name)
                f.write("\n")
        f.write(export_node_name(tensor))
        f.write("\n")

    def recursive_export_edges(tensor, f, exported_op_nodes, exported_edges):
        to_name = '"' + exported_op_nodes[tensor] + '"'
        for child in tensor.op.input_tensors:
            recursive_export_edges(child, f, exported_op_nodes, exported_edges)
            from_name = '"{}"'.format(exported_op_nodes.get(child))
            if (from_name, to_name) not in exported_edges:
                exported_edges.add((from_name, to_name))
                f.write('    {} -> {}   [label = "{}"];\n'.format(from_name,
                                                                  to_name, export_tensor_shape(child.shape)))
        if isinstance(tensor.op, akg.tvm.tensor.ComputeOp) and \
                isinstance(tensor.op.body[0], (akg.tvm.expr.Mul, akg.tvm.expr.Add, akg.tvm.expr.Sub, akg.tvm.expr.Div)) and \
                len(tensor.op.input_tensors) < 2:
            if isinstance(tensor.op.body[0].a, akg.tvm.expr.FloatImm):
                from_name = '"Const_a_{}"'.format(exported_op_nodes[tensor])
                if (from_name, to_name) not in exported_edges:
                    exported_edges.add((from_name, to_name))
                    f.write('    {} -> {}   [label = "(const)"];\n'.format(from_name, to_name))
            if isinstance(tensor.op.body[0].b, akg.tvm.expr.FloatImm):
                from_name = '"Const_b_{}"'.format(exported_op_nodes[tensor])
                if (from_name, to_name) not in exported_edges:
                    exported_edges.add((from_name, to_name))
                    f.write('    {} -> {}   [label = "(const)"];\n'.format(from_name, to_name))
        return exported_edges

    with open(filename, "w+") as f_out:
        f_out.write('digraph G {\n  ration = compress;\n  nodesep = 0.1;  rankdir = BT\n')

        exported_op_nodes = dict()  # dict of {tensor, tensor_name}
        exported_edges = set()
        repeat_name = 0

        if isinstance(tensors, akg.tvm.container.Array):
            list_tensors = list(x for x in tensors)
        else:
            if isinstance(tensors, akg.tvm.tensor.Tensor):
                list_tensors = [tensors]
            else:
                list_tensors = []

        for a_tensor in list_tensors:
            exported_op_nodes, repeat_name = recursive_collect_nodes(a_tensor, exported_op_nodes, repeat_name)
            recursive_export_nodes_name(a_tensor, f_out, exported_op_nodes)
            exported_edges = recursive_export_edges(a_tensor, f_out, exported_op_nodes, exported_edges)

        f_out.write("\n}\n")


variable_map = {}


def register_variables(name, input_var, output_var):
    """
    register variables as a dictionary.
    """
    if not isinstance(name, str):
        raise ValueError("key {} is not str.".format(name))
    variable_map[name] = [output_var, input_var]


def get_variables(name):
    """
    get variables from dictionary.
    """
    if isinstance(name, str):
        if not name in variable_map:
            raise ValueError("value to key {} is empty.".format(name))
        return variable_map.get(name)
    raise ValueError("key {} is not str.".format(name))


@register_node
class DifferentiationResult(NodeBase):
    """
    Result of differentiation.

    Args:
        result (list[tvm.tensor.Tensor]):
            The requested adjoints, i.e. the Jacobians or gradients of the given output
            wrt to the given inputs.
        adjoints (dict[tvm.tensor.Tensor, tvm.tensor.Tensor]):
            A map from tensors to the corresponding adjoints (including internal nodes).
        adjoint_summands (dict[tvm.tensor.Tensor, dict[tvm.tensor.Tensor, tvm.tensor.Tensor]]):
            Single summands of the adjoints.
    """

    # Here we convert tvm Maps to dicts because Map compares keys by reference which is
    # wrong for tvm.tensor.Tensors. Hopefully, in the future Map gets fixed somehow, and these properties
    # may be removed then.

    @property
    def adjoints(self):
        res = NodeBase.__getattr__(self, 'adjoints')
        return dict(res.items())

    @property
    def adjoint_summands(self):
        res = NodeBase.__getattr__(self, 'adjoint_summands')
        return {k: dict(v.items()) for k, v in res.items()}

    def _check_not_empty(self):
        if not self.result:
            raise ValueError("The result of differentiation does not contain any explicitly "
                             "requested results, so using it as an iterable is probably a mistake. "
                             "Please explicitly use res.adjoints to get adjoints or res.result to "
                             "get the empty list.")

    def __getitem__(self, i):
        self._check_not_empty()
        return self.result[i]

    def __len__(self):
        self._check_not_empty()
        return len(self.result)


def differentiate(output, inputs=None, head=None, ad_attrs=None, new_pld_array=None, override=None, fdiff=None):
    """
    Perform operator-level automatic differentiation.

    Args:
        output (tvm.tensor.Tensor): The tensor to differentiate.
        inputs (list[tvm.tensor.Tensor]): The list of input tensors.
            When the list is empty or None, will perform differentiation with respect to all tensors the output depends
            on (i.e. will compute all adjoints and populate the corresponding dict, but the list of results will be
            empty). Default: None.
        head (tvm.tensor.Tensor): The adjoint of the output.
            in other words, some tensors, by which the Jacobians will be multiplied. Its shape must be of the form
            `prefix + output.shape`. For example, if the shape of `output` is (2, 3), the shape of `head` could
            be (2, 3), (?, 2, 3) and etc.
            If `None` is passed, the identity tensor of shape `output.shape + output.shape` will be used.
            Default: None.
        ad_attrs (dict): The additional attributes for the auto-differentiate computation. Default: None.
        new_pld_array (list): List of additional variables which could be used in differentiation. Default: None.
        override (dict): A dictionary to override differentiation for certain tensors.
            Override is a dictionary with types: {tvm.tensor.Tensor: (list[tvm.tensor.Tensor],
            callable[tvm.tensor.Tensor, list[tvm.tensor.Tensor], tvm.tensor.Tensor, list[tvm.tensor.Tensor]])}.
            This dict maps tensors `t` to pairs `(dependencies, custom_diff)` where `dependencies` is a list of
            tensors which are considered to be inputs of `t` (which may differ from the immediate inputs),
            and `custom_diff` is a custom differentiation function which will be called as
            `custom_diff(t, dependencies, adjoint, new_pld_array)` and should return a list of adjoints
            corresponding to dependencies.
            Note that this function differs from the one required for `fdiff`
            in that it takes a list of inputs instead of a single input
            and returns a list of adjoints instead of a single adjoint. Default: None.
        fdiff (callable[tvm.tensor.Tensor, tvm.tensor.Tensor, tvm.tensor.Tensor, tvm.tensor.Tensor]): The default
            function performing differentiation and multiplication, by default `akg.autodiff.DiffBuildingBlock` is used.
            The function must accept parameters:

                - `output` - an output tensor

                - `input` - an input tensor

                - `head` - the adjoint of the output tensor

                - `ad_attrs` - the additional attributes for the auto-differentiate computation

                - `new_pld_array` - the additional tensors with information for the auto-differentiate computation

            The result should be `head` multiplied by the Jacobians of `output` wrt `input`. Default: None.

    Returns:
        DifferentiationResult.
        class DifferentiationResult is used to represent a differentiation result, including:
            - result (list[tvm.tensor.Tensor]):
              The requested adjoints, i.e. the Jacobians or gradients of the given output
              with respect to the given inputs.

            - adjoints (dict{tvm.tensor.Tensor: tvm.tensor.Tensor}):
              A dict from tensors to the corresponding adjoints (including internal nodes).

            - adjoint_summands (dict{tvm.tensor.Tensor: dict{tvm.tensor.Tensor: tvm.tensor.Tensor}}):
              Single summands of the adjoints.

    Raises:
        ValueError: If the shape of `head` is invalid.

    Examples:
        >>> x = akg.tvm.placeholder((32, 3, 28, 28), name='x')
        >>> w1 = akg.tvm.placeholder((10, 3, 3, 3), name='w1')
        >>> z1 = akg.topi.nn.conv2d(x, w1, 1, 0, 1)
        >>> z2 = akg.topi.nn.flatten(z1)
        >>> y = akg.topi.sum(z2)
        >>>
        >>> # produce gradients
        >>> [dw1, dw2] = akg.differentiate(y, [x, w1])
        >>>
        >>> # produce Jacobians
        >>> [jw1, jw2] = akg.differentiate(z2, [x, w1])
        >>>
        >>> # produce Jacobians, the head adjoint for z2 is provided manually
        >>> [dw1, dw2] = akg.differentiate(z2, [x, w1], akg.topi.full_like(z2, 1.0))
        >>>
        >>> # produce gradients wrt all inputs
        >>> res = akg.differentiate(y)
        >>> dw1 = res.adjoints[x]
        >>> dw2 = res.adjoints[w1]
        >>>
        >>> # a custom differentiation function
        >>> head = akg.tvm.placeholder((1,), name = 'head')
        >>> def my_fdiff(out, inp, head, ad_attrs, new_pld_array):
        >>>     return [akg.tvm.compute(inp[0].shape, lambda ax0, ax1, ax2, ax3: head[ax0, ax3 + ax2*26 + ax1*676])]
        >>>
        >>> # using a custom differentiation function only for z2
        >>> res = akg.differentiate(y, [x, w1], head, None, None, override={z2: ([z1], my_fdiff)})
    """

    # check whether head shape is compatible with output shape.
    if head is not None:
        output_shape = get_shape(output)
        head_shape = get_shape(head)
        output_dim = len(output_shape)
        head_last_shape = head_shape[-output_dim:]
        if head_last_shape != output_shape:
            raise ValueError("operands could not be broadcast together with head shape %s and output shape %s" %
                             (str(head_shape), str(output_shape)))

    if inputs is None:
        inputs = []

    if override is not None:
        override_deps = []

    if fdiff is None:
        fdiff = akg.autodiff.DiffBuildingBlock

    if override is not None:
        def modified_fdiff(out, inp, head, ad_attrs, new_pld_array, override=override, old_fdiff=fdiff, cache=None):
            if cache is None:
                cache = {}
            if out in override:
                if (out, head) not in cache:
                    cache[(out, head)] = override[out][1](
                        out, override[out][0], head, ad_attrs, new_pld_array)
                idx = override[out][0].index(inp)
                return cache.get((out, head), {})[idx]
            return old_fdiff(out, inp, head, ad_attrs, new_pld_array)

        fdiff = modified_fdiff

        override_deps = {t: deps for t, (deps, _) in override.items()}
        return akg.autodiff.Differentiate(output, inputs, head, ad_attrs, None, fdiff, override_deps)

    if new_pld_array is None:
        return akg.autodiff.Differentiate(output, inputs, head, ad_attrs, [], fdiff)
    return akg.autodiff.Differentiate(output, inputs, head, ad_attrs, new_pld_array, fdiff)
