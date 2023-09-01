# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Hybrid Script Parser"""

# 2023.08.16 - Support parser for python 3.9 and up.
# 2022.02.15 - Support grid as new loop mode.
# 2022.01.19 - Support negative extent for range.
# 2021.12.15 - Support block_realize intrin in with scope.
# 2021.10.21 - Support reverse order loop range.
# 2019.12.30 - Modify parser.py, add TensorIntrinSubscriptParser, generate_one_assign, visit_Assign,
#              and some visit_ function, modify _floordiv.

import ast
import operator
import logging
import sys
import types
import numbers

from enum import Enum

from .util import _internal_assert, _apply_indices
from . import calls
from . import util
from .intrin import _TensorIntrin
from .preprocessor import determine_variable_usage
from ..api import all as _all
from ..api import any as _any

from ..container import Array
from ..tensor import Tensor, Operation
from .. import _api_internal as _tvm_internal
from .. import expr as _expr
from .. import stmt as _stmt
from .. import make as _make
from .. import api  as _api
from .. import ir_pass as _ir_pass
from collections.abc import Iterable
from ..intrin import call_pure_intrin


def concat_list_to_block(lst):
    """Concatenate a list of Python IR nodes to HalideIR Block"""
    if not lst:
        return util.make_nop()
    n = len(lst)
    if n == 1:
        return lst[0]
    body = lst[n - 1]
    for i in range(1, n):
        stmt = lst[n - 1 - i]
        if isinstance(stmt, _stmt.AssertStmt):
            body = _make.AssertStmt(stmt.condition, stmt.message, body)
        elif isinstance(stmt, _stmt.LetStmt):
            body = _make.LetStmt(stmt.var, stmt.value, body)
        else:
            body = _make.Block(stmt, body)
    return body


def visit_list_to_block(visit, lst):
    """Visit and concatenate a list of Python IR nodes to HalideIR Block"""
    lst = [visit(stmt) for stmt in lst if not util.is_docstring(stmt)]
    lst = [
        _make.Evaluate(stmt) if isinstance(stmt, _expr.Expr) else stmt
        for stmt in lst
    ]
    lst = [stmt for stmt in lst if not _ir_pass.Equal(stmt, util.make_nop())]
    if not lst:
        return util.make_nop()
    return concat_list_to_block(lst)


class Symbol(Enum):
    """Enumerates types in the symbol table"""
    Callable = 0
    Input = 1
    OutputBuffer = 2
    GlobalBuffer = 3
    LocalBuffer = 4
    SharedBuffer = 5
    ConstVar = 6
    BufferVar = 7
    LoopVar = 8
    ConstLoopVar = 9
    ThreadBind = 10
    LetBindVar = 11
    UbBuffer = 12
    RegBuffer = 13
    LoopVarTuple = 14


class TensorIntrinSubscriptParser(ast.NodeVisitor):

    def __init__(self, parent_parser):
        self._parent_parser = parent_parser

    def visit_Subscript(self, node):
        """ Visit the ast.Subscript which is used as an argument to a tensor
        intrinsics call. Should return the tensor being visited as well as the
        region where the tensor intrin is operating on. The definition of
        "Region" is in include/tvm/expr.h .
        """
        _internal_assert(isinstance(node.value, ast.Name),
                         'node.value has to be an ast.Name!')
        tensor = self._parent_parser.visit(node.value)
        # This call should return a list of either a tuple of lower bound (lb)
        # and upper bound (ub) or an index. For example,
        # [(lb, ub), idx, idx, (lb, ub), ...], both lb and ub should be an Expr.
        region = self.visit(node.slice)
        for i, idx_range in enumerate(region):
            if isinstance(idx_range, tuple):
                _internal_assert(
                    len(idx_range) == 2,
                    'idx_range can only contain lb and ub!')
                lb, ub = idx_range
                lb = _api.const(0, 'int32') if lb is None else lb
                ub = tensor.shape[i] if ub is None else ub
                region[i] = _api.Range(lb, ub)
            else:
                region[i] = _make.range_by_min_extent(idx_range, 1)

        return tensor, region

    def visit_Index(self, node):
        """ For the case of a[i, j].
        """
        return [idx for idx in self._parent_parser.visit(node)]

    def visit_ExtSlice(self, node):
        """ For the case of a[i:i+4, j:j+4].
        """
        region = []
        for _, dim in enumerate(node.dims):
            idx_range = (self.visit(dim) if isinstance(dim, ast.Slice) else
                         self._parent_parser.visit(dim))
            region.extend(idx_range)
        return region

    def visit_Slice(self, node):
        """ For the case of a[i:i+4] and also to support visit_ExtSlice.
        """
        _internal_assert(node.step is None, 'Slice.step is not supported!')
        lb = (self._parent_parser.visit(node.lower)
              if node.lower is not None else None)
        ub = (self._parent_parser.visit(node.upper)
              if node.upper is not None else None)
        return [(lb, ub)]


def _floordiv(x, y):
    if isinstance(x, _expr.ExprOp) or isinstance(y, _expr.ExprOp):
        return _api.floordiv(x, y)
    return operator.floordiv(x, y)


def _floormod(x, y):
    if isinstance(x, _expr.ExprOp) or isinstance(y, _expr.ExprOp):
        return _api.floormod(x, y)
    return operator.mod(x, y)


class HybridParser(ast.NodeVisitor):
    """Python AST visitor pass which finally lowers it to HalideIR"""


    _binop_maker = {
        ast.Add     : operator.add,
        ast.Sub     : operator.sub,
        ast.Mult    : operator.mul,
        ast.Div     : operator.div if sys.version_info[0] == 2 else operator.truediv,
        ast.FloorDiv: _floordiv,
        ast.Mod     : _floormod,
        ast.BitOr   : operator.or_,
        ast.BitAnd  : operator.and_,
        ast.BitXor  : operator.xor,
        ast.Gt      : operator.gt,
        ast.GtE     : operator.ge,
        ast.Lt      : operator.lt,
        ast.LtE     : operator.le,
        ast.Eq      : operator.eq,
        ast.NotEq   : operator.ne,
        ast.And     : _all,
        ast.Or      : _any,
        ast.Pow     : operator.pow,
    }


    _unaryop_maker = {
        ast.USub   : operator.neg,
        ast.Invert : operator.invert,
        ast.Not    : operator.not_
    }


    def __init__(self, args, usage, symbols, closure_vars, func_name=None):
        """
        Parameters
        ----------
        args: A list of tvm.placeholder or tvm.var
            Provided by the user, the argument list of the function to be lowered.

        usage: A dict of variables used in last in this function
            Provided by last lower pass, which collects this information

        symbols : list of str
            The symbol list of the global context of the function.

        closure_vars: dict
            A dict of external name reference captured by this function.

        Returns
        -------
        func_name: str
            The name of the function to be lowered; if not provided,
            the compiler will use the name in the AST
        """
        self.args = list(args)
        self.arg_buffers = dict()
        self.arg_regions = dict()
        self.usage = usage.copy()

        self.symbols = {} # Symbol table
        for k, v in symbols.items():
            if isinstance(v, types.FunctionType):
                self.add_symbol(k, Symbol.Callable, v)

        self.closure_vars = closure_vars

        self.binds = {} # Thread binds
        self.device = 0 # Is it generating device

        self.func_name = func_name # The name of the function to be lowered
        self.outputs = [] # Output tensors' name
        self.output_buffers = dict()
        self.output_regions = dict()
        self.side_effect = set() # Tensors with side effects
        self.parsed_body = None # The parsed HalideIR body
        self.returned = False # If this function has a valid return
        self.slice_mapping = dict()
        self.dim_idx = 0


    def add_symbol(self, key, ty, val): #pylint: disable=invalid-name
        """Add value to the symbol table context"""
        if key in self.symbols.keys():
            old = str(self.symbols[key])
            new = str((ty, val))
            _internal_assert(False,
                             "Name conflict in symbol table! [%s] %s -> %s" % (key, old, new))

        self.symbols[key] = ty, val

        if ty == Symbol.ThreadBind:
            if val.var.name not in self.binds.keys():
                self.binds[val.var.name] = val
                return
            val_ = self.binds[val.var.name]
            _internal_assert(_ir_pass.Equal(val_.dom.extent, val.dom.extent),
                             "Thread extents should be uniform!")
            self.symbols[key] = ty, val_


    def wrap_up_realize(self, node, body):
        """Wrap up all the variables which will no longer be used"""
        to_pop = []
        for key, val in self.usage.items():
            _, level, _ = val
            if level != node:
                continue
            _internal_assert(key in self.symbols.keys(), "Unknown symbol %s!" % key)

            ty, entry = self.symbols[key] #pylint: disable=invalid-name
            if ty in [Symbol.Input, Symbol.OutputBuffer]:
                continue
            elif 'Buffer' in ty.name:
                _buf = entry
                _scope = 'global' if ty is Symbol.BufferVar else ty.name[:-6].lower()
                if ty is Symbol.UbBuffer:
                    _scope = 'local.UB'
                elif ty is Symbol.RegBuffer:
                    _scope = 'local.REG'
                to_pop.append(key)
            else:
                continue

            if _scope == 'global':
                body = self.wrap_up_binds(body)

            _domain = [_make.range_by_min_extent(0, i) for i in _buf.shape]
            _dtype = _buf.dtype
            _true = _api.convert(True)
            body = _make.Realize(_buf.op, _buf.value_index, _dtype, _domain, _true, body)
            body = _make.AttrStmt(_buf.op, 'realize_scope',  _api.convert(_scope), body, None)

        for elem in to_pop:
            self.symbols.pop(elem)

        return body


    def wrap_up_binds(self, body):
        for _, iter_var in self.binds.items():
            ext = iter_var.dom.extent
            body = _make.AttrStmt(iter_var, 'thread_extent', ext, body, None)
        self.binds = {}
        return body


    #pylint: disable=invalid-name, missing-docstring
    def visit_Module(self, node):
        _internal_assert(len(node.body) == 1, \
                         "Only one-function source code will be fed to this parser!")
        return self.visit(node.body[0])


    def visit_FunctionDef(self, node):
        _internal_assert(len(node.args.args) == len(self.args), \
                         "The number of arguments passed to the \
                         function should be the same as it is defined!")
        if self.func_name is None:
            self.func_name = node.name
        for idx, arg in enumerate(node.args.args):
            _attr = 'id' if sys.version_info[0] < 3 else 'arg' # To make py2 and 3 compatible
            self.add_symbol(getattr(arg, _attr), Symbol.Input, self.args[idx])
        res = visit_list_to_block(self.visit, node.body)
        res = self.wrap_up_realize(node, res)
        return self.wrap_up_binds(res)


    def visit_Expr(self, node):
        return self.visit(node.value)


    def visit_Name(self, node):
        name = node.id
        if sys.version_info[0] == 2 and name in ['True', 'False']:
            return _api.convert(ast.literal_eval(name))

        if name in self.closure_vars:
            return _api.convert(self.closure_vars[name])

        ty, entry = self.symbols[name]
        _internal_assert(name in self.symbols, "Unknown symbol %s!" % name)
        if ty in [Symbol.LoopVar, Symbol.Input, Symbol.ConstLoopVar, Symbol.LoopVarTuple]:
            return entry
        if ty is Symbol.ThreadBind:
            return entry.var
        if ty is Symbol.ConstVar:
            return entry if isinstance(node.ctx, ast.Load) else None
        if ty is Symbol.LetBindVar:
            return entry
        if ty is Symbol.BufferVar:
            if isinstance(node.ctx, ast.Load):
                return _make.Call(entry.dtype, entry.name, [_api.const(0, 'int32')], \
                                  _expr.Call.Halide, entry.op, entry.value_index)
            return entry, [_api.const(0, 'int32')]
        # Do I need any assertion here?
        return entry


    def visit_Num(self, node):
        if isinstance(node.n, numbers.Integral):
            dtype = "int32"
        elif isinstance(node.n, float):
            dtype = "float32"
        else:
            _internal_assert(isinstance(node.n, bool),
                             "The data type should be one of (int, float, bool)")
            dtype = "bool"
        return _api.const(node.n, dtype)


    def visit_NameConstant(self, node):
        return _api.convert(node.value)


    def visit_AugAssign(self, node):
        buf = self.visit(node.target)
        rhs = self.visit(node.value)
        if isinstance(buf, tuple):
            _internal_assert(len(buf) == 2, "LHS is supposed to be (buf, args)!")
            buf, args = buf
        else:
            args = [_api.const(0, 'int32')]
        _internal_assert(isinstance(buf, Tensor), "LHS is supposed to be Tensor!")

        read = _make.Call(buf.dtype, buf.name, args, _expr.Call.Halide, buf.op, buf.value_index)
        value = HybridParser._binop_maker[type(node.op)](read, rhs)

        return _make.Provide(buf.op, 0, value, args)

    def slice_collection(self, lhs):
        order = 0
        if isinstance(lhs, ast.Subscript):
            args = self.visit(lhs.slice)

            if not isinstance(args, Iterable):
                args = [args]

            for arg in args:
                if isinstance(arg, _expr.Call) and arg.name == "Slice":
                    arg_name = arg.__str__()
                    iter_var = _api.var("slice_" + str(self.dim_idx))
                    self.dim_idx += 1
                    if arg_name in self.slice_mapping:
                        var_list = self.slice_mapping[arg_name]
                        self.slice_mapping[arg_name] = var_list + \
                            [[iter_var, order], ]
                    else:
                        self.slice_mapping[arg_name] = [[iter_var, order]]
                    order += 1

    def generate_one_assign(self, lhs, rhs):
        if isinstance(rhs, _expr.Expr):
            rhs = _ir_pass.Simplify(rhs)
        if isinstance(lhs, ast.Name):
            #TODO: support defined intermediate buffer later
            lhs_ = lhs
            lhs = lhs.id
            if lhs in self.symbols.keys():
                ty, _ = self.symbols[lhs]
                _internal_assert(ty != Symbol.LoopVar, \
                                 "Loop variable cannot be overwritten!")
            decl, _, rw = self.usage[lhs]
            if decl == lhs_:
                _internal_assert(lhs not in self.symbols.keys(),
                                 "This value should not be defined before this point!")
                if isinstance(rhs, tuple):
                    shape, dtype, scope = rhs
                    ph = _api.placeholder(shape, dtype=dtype, name=lhs)
                    self.add_symbol(lhs, getattr(Symbol, scope.title() + "Buffer"), ph)
                    if scope == 'output':
                        self.outputs.append(lhs)
                    return util.make_nop()
                if isinstance(rhs, util.halide_imm_types) and ast.Store not in rw:
                    self.add_symbol(lhs, Symbol.ConstVar, rhs)
                else:
                    _internal_assert(self.device == 0,
                                     "Single variable not supported in devices' side!\n" + \
                                     "If you are using GPU, please allocate a 'local' spad " + \
                                     "outside the bind body")
                    if ast.Store not in rw:
                        # this value is only be assigned once, we can use let binding
                        self.add_symbol(
                            lhs, Symbol.LetBindVar,
                            _api.var(name=lhs,
                                     dtype=rhs.dtype))
                    else:
                        ph = _api.placeholder((1, ), dtype=rhs.dtype, name=lhs)
                        self.add_symbol(lhs, Symbol.BufferVar, ph)
            lhs = self.visit(lhs_)
            if isinstance(lhs, (list, tuple)):
                buf, args = lhs
                return _make.Provide(buf.op, 0, rhs, args)
            elif lhs is not None:
                # assign to a let binding var
                _internal_assert(
                    isinstance(lhs, _expr.Var),
                    "Lhs can only be a var for let binding.")
                return _make.LetStmt(lhs, rhs, util.make_nop())
            else:
                return util.make_nop()

        lhs, args = self.visit(lhs)
        _internal_assert(isinstance(lhs, Tensor), \
                         "An array access's LHS is expected to be a expr.Call!")
        def _check_if_slice(args):
            for arg in args:
                if isinstance(arg,
                              _expr.Call) and arg.name in ("Ellipsis", "Slice"):
                    return True
            return False

        if not _check_if_slice(args):
            res = _make.Provide(lhs.op, lhs.value_index, rhs, args)
            return res

        # This is a numpy-like Slice in hybrid!
        # Reference: https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html

        # List of tuple (iter_var, lower, upper, step).
        expand_args = []
        ellipsis_dim_idx = 0

        def _const(num):
            return _api.const(num, 'int32')

        slice_count = {}
        for arg in args:
            if isinstance(arg,
                          _expr.Call) and arg.name == "Ellipsis":
                ellipsis_cross_dims = len(lhs.shape) - len(args) + 1
                for _ in range(ellipsis_cross_dims):
                    lower, upper, step = _const(
                        0), lhs.shape[ellipsis_dim_idx], _const(1)
                    expand_args.append(
                        (_api.var("ellipsi_" + str(ellipsis_dim_idx)), lower, upper, step))
                    ellipsis_dim_idx += 1
            elif isinstance(arg,
                            _expr.Call) and arg.name == "Slice":
                lower, upper, step = arg.args
                arg_name = arg.__str__()
                if arg_name not in slice_count:
                    slice_count[arg_name] = 0
                iter_var, _ = self.slice_mapping[arg_name][slice_count[arg_name]]
                slice_count[arg_name] += 1
                expand_args.append(
                    (iter_var, lower, upper, step))
            else:
                expand_args.append([arg])
        res = _make.Provide(lhs.op, lhs.value_index, rhs,
                            [arg[0] for arg in expand_args])
        for idx in range(len(expand_args), 0, -1):
            arg = expand_args[idx - 1]
            if len(arg) == 4:
                iter_var, lower, upper = arg[:3]
                res = _make.For(iter_var, lower, upper,
                                _stmt.For.Parallel, 0, res)
        return res

    def visit_Assign(self, node):
        _internal_assert(len(node.targets) == 1, "Internal Error")

        lhs = node.targets[0]
        self.slice_collection(lhs)
        rhs = self.visit(node.value)

        if isinstance(rhs, Operation):
            rmap = {}
            if isinstance(lhs, ast.Tuple):
                lhs = lhs.elts
            else:
                lhs = (lhs,)
            _internal_assert(len(lhs) == rhs.num_outputs,
                             "Unable to detuple the outs to targets")
            for i in range(rhs.num_outputs):
                _internal_assert(isinstance(lhs[i], ast.Name),
                                 "You should bind a pure name to the tensors")
                self.add_symbol(lhs[i].id, Symbol.GlobalBuffer, rhs.output(i))
                rmap[rhs.outputs[i].op] = rhs.output(i)
            return util.replace_io(rhs.body, rmap)

        if isinstance(lhs, ast.Tuple):
            # do not support nested tuple
            _internal_assert(
                isinstance(rhs,
                           (tuple, list, Array)) and len(lhs.elts) == len(rhs),
                "The numbers of elements mismatch between lhs and rhs in tuple assignment"
            )
            assign_list = [
                self.generate_one_assign(lhs.elts[i], rhs[i])
                for i in range(len(lhs.elts))
            ]
            return concat_list_to_block(assign_list)
        else:
            assign_stmt = self.generate_one_assign(lhs, rhs)
            self.slice_mapping.clear()

            return assign_stmt

    def visit_Index(self, node):
        return self.visit(node.value)


    def visit_Attribute(self, node):
        _internal_assert(isinstance(node.value, ast.Name), "For atrribute access, only both names are supported so far!")
        buf = self.visit(node.value)
        return getattr(buf, node.attr)

    def visit_Ellipsis(self, node):
        return call_pure_intrin("handle", "Ellipsis")

    def visit_Slice(self, node):
        lower = self.visit(node.lower)
        upper = self.visit(node.upper)
        if node.step:
            step = self.visit(node.step)
        else:
            step = _api.const(1, 'int32')
        return call_pure_intrin("handle", "Slice", lower, upper, step)

    def visit_ExtSlice(self, node):
        dims = []
        for dim in node.dims:
            dims.append(self.visit(dim))
        return dims

    def visit_Subscript(self, node):
        if isinstance(node.value, ast.Name) and node.value.id in self.symbols.keys() and \
                self.symbols[node.value.id][0] == Symbol.LoopVarTuple:
            # when the name is a loop var tuple
            # deal with subscripts to get the index of the tensor
            loop_var_tuple = self.symbols[node.value.id][1]

            def ast_num_to_index(node):
                if isinstance(node, ast.Num):
                    index = node.n
                    _internal_assert(
                        isinstance(index, numbers.Integral),
                        "All indices are supposed to be an integer")
                    return index
                else:
                    _internal_assert(
                        isinstance(node, ast.UnaryOp),
                        "All indices are supposed to be an integer")
                    _internal_assert(
                        isinstance(node.op, ast.USub),
                        "All indices are supposed to be an integer")
                    return -ast_num_to_index(node.operand)

            if isinstance(node.slice, ast.Index):
                # when the slice is an index, get the number of the index
                index = ast_num_to_index(node.slice.value)
                return loop_var_tuple[index]
            elif isinstance(node.slice, ast.Slice):
                # when the slice is a slice, get the lower and upper bound of the index
                lower = 0 if node.slice.lower is None else ast_num_to_index(node.slice.lower)
                upper = loop_var_tuple.__len__() if node.slice.upper is None else ast_num_to_index(node.slice.upper)

                return loop_var_tuple[lower:upper]

        args_list = self.visit(node.slice)
        if not isinstance(args_list, Iterable):
            args_list = [args_list]
        args = []
        for arg in args_list:
            if isinstance(arg, list):
                args = args + arg
            else:
                args.append(arg)

        if isinstance(node.value, ast.Name):
            if node.value.id in self.closure_vars:
                args = ast.literal_eval(str(args))
                return _api.convert(
                    _apply_indices(self.closure_vars[node.value.id], args))

            buf = self.visit(node.value)
            if isinstance(buf, (Array, list)):
                for i in args:
                    if isinstance(i, numbers.Integral):
                        buf = buf[i]
                    else:
                        _internal_assert(
                            isinstance(i, (_expr.IntImm, _expr.UIntImm)),
                            "All indices are supposed to be constants")
                        buf = buf[i.value]

                return buf
            expand_args = []
            slice_count = {}
            current_order = -1
            for arg in args:
                if isinstance(arg,
                              _expr.Call) and arg.name == "Slice":
                    arg_name = arg.__str__()
                    _internal_assert(arg_name in self.slice_mapping,
                                     "Can't deal with {}: slice in tuple assignment or not appeared in LHS".format(arg_name))
                    if arg_name not in slice_count:
                        slice_count[arg_name] = 0
                    _internal_assert(len(self.slice_mapping[arg_name]) > slice_count[arg_name],
                                     "Can't deal with {}: more slice on RHS then that on LHS".format(arg_name))
                    iter_var, order = self.slice_mapping[arg_name][slice_count[arg_name]]
                    _internal_assert(order > current_order,
                                     "The order of slices on LHS and that on RHS doesn't match: {}".format(arg_name))
                    slice_count[arg_name] += 1
                    current_order = order
                    expand_args.append(iter_var)
                else:
                    expand_args.append(arg)
            if isinstance(node.ctx, ast.Load):
                return _make.Call(buf.dtype, buf.name, expand_args, _expr.Call.Halide,
                                  buf.op, buf.value_index)
            return buf, args

        shape = self.visit(node.value)
        _internal_assert(
            len(args) == 1,
            "For 'shape' access the argument should be only one!")
        args = args[0]
        _internal_assert(isinstance(args, (_expr.IntImm, _expr.UIntImm)),
                         "So far only constant shape access supported!")
        return shape[args.value]

    def visit_With(self, node):
        if sys.version_info[0] < 3:
            context = node.context_expr
            # option = node.optional_vars
        else:
            _internal_assert(len(node.items) == 1, "Only one with element is supported so far!")
            context = node.items[0].context_expr
            # option = node.items[0].optional_vars
        _internal_assert(isinstance(context, ast.Call), "The object must be a Python func call!")
        block = visit_list_to_block(self.visit, node.body)
        if context.func.id == "attr":
            args = [self.visit(i) for i in context.args]
            return _make.AttrStmt(_api._IterVar(None, block.loop_var.name, 0), args[0],
                                  _api.convert(args[1]), block, None)
        elif context.func.id == "allocate":
            lhs = node.items[0].optional_vars
            rhs = self.visit(context)
            _ = self.generate_one_assign(lhs, rhs)
            return _make.AttrStmt(rhs, "type",
                                  _api.convert("inline"), block, None)
        elif context.func.id == "block_realize":
            args = [self.visit(i) for i in context.args]
            return _make.AttrStmt(args[0].op, "block_realize",
                                  _api.convert(True), block, None)
        else:
            raise ValueError("unsupported function in With scope")


    def visit_If(self, node):
        cond = self.visit(node.test)

        # Return no IfThenElse if proven
        if isinstance(cond, _expr.UIntImm):
            if cond.value:
                return visit_list_to_block(self.visit, node.body)
            if node.orelse:
                return visit_list_to_block(self.visit, node.orelse)
            return util.make_nop()

        if_body = visit_list_to_block(self.visit, node.body)

        if node.orelse:
            else_body = visit_list_to_block(self.visit, node.orelse)
        else:
            else_body = None
        return _make.IfThenElse(cond, if_body, else_body)


    def visit_IfExp(self, node):
        cond = self.visit(node.test)
        if_body = self.visit(node.body)
        else_body = self.visit(node.orelse)
        return _make.Select(cond, if_body, else_body)


    def visit_Compare(self, node):
        _internal_assert(len(node.ops) == len(node.comparators),
                         "#compare ops != #comparators")
        ops = [self.visit(node.left)]
        ops += [self.visit(i) for i in node.comparators]
        res = []
        for i in range(len(node.ops)):
            lhs = ops[i]
            rhs = ops[i + 1]
            res.append(HybridParser._binop_maker[type(node.ops[i])](lhs, rhs))
        return _all(*res)


    def visit_BoolOp(self, node):
        n = len(node.values)
        if n == 1:
            _internal_assert(isinstance(node.op, ast.Not), \
                             "Unary is supposed to be not!")
            return operator.not_(self.visit(node.values[0]))
        _internal_assert(isinstance(node.op, (ast.And, ast.Or)), \
                         "Binary is supposed to be and/or!")
        values = [self.visit(i) for i in node.values]
        return HybridParser._binop_maker[type(node.op)](*values)


    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        return HybridParser._unaryop_maker[type(node.op)](operand)


    def visit_BinOp(self, node):
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)

        if isinstance(node.op, ast.Add) and isinstance(lhs, (Iterable, Array)) and isinstance(rhs, (Iterable, Array)):
            # deal with add in the index to connect two parts of the index list
            arg_list = []
            for arg in lhs:
                arg_list.append(arg)
            for arg in rhs:
                arg_list.append(arg)

            return arg_list

        if rhs.dtype != lhs.dtype:
            if (isinstance(rhs, _expr.ConstExpr)):
                rhs = _api.const(rhs.value, dtype=lhs.dtype)
            elif (isinstance(lhs, _expr.ConstExpr)):
                lhs = _api.const(lhs.value, dtype=rhs.dtype)
            else:
                _internal_assert(False, "Mismatch type!")
        return HybridParser._binop_maker[type(node.op)](lhs, rhs)


    def visit_Call(self, node):
        # Yet, no function pointer supported
        _internal_assert(isinstance(node.func, ast.Name), \
                         "Only id-function function call is supported so far!")

        func_id = node.func.id
        # Tensor intrins.
        if (hasattr(calls, func_id) and
                isinstance(getattr(calls, func_id), _TensorIntrin)):
            subscript_parser = TensorIntrinSubscriptParser(self)
            all_tensors = []
            all_regions = []
            for arg in node.args:
                tensor, region = subscript_parser.visit(arg)
                all_tensors.append(tensor)
                all_regions.append(region)
            return getattr(calls,
                           func_id)(*(all_tensors+all_regions),
                                    input_buffer_map=self.arg_buffers,
                                    output_buffer_map=self.output_buffers,
                                    input_region_map=self.arg_regions,
                                    output_region_map=self.output_regions)
        args = [self.visit(i) for i in node.args]
        # Scalar Intrinsics'
        if hasattr(calls, func_id):
            return getattr(calls, func_id)(func_id, args)
        # Contexts'
        _internal_assert(func_id in self.symbols.keys(), \
                         "The function called (%s) is not in the context either!" % func_id)
        ty, entry = self.symbols[func_id]
        _internal_assert(ty is Symbol.Callable, \
                         "Are you sure what you call is a function?!")
        outs = entry(*args)
        op = outs.op if isinstance(outs, Tensor) else outs[0].op
        return op


    def visit_For(self, node):
        iter_var, low, ext, for_type, step = self.visit(node.iter)
        _internal_assert(isinstance(node.target, ast.Name), \
                         "The loop iterator should be a variable!")
        if isinstance(ext, _expr.Expr):
            ext = _ir_pass.Simplify(ext)

        _name = node.target.id

        if isinstance(for_type, tuple):
            low = _ir_pass.Simplify(low)
            _internal_assert(isinstance(low, _expr.ConstExpr) and
                             isinstance(ext, _expr.ConstExpr) and
                             isinstance(step, _expr.ConstExpr),
                             "Const range should start from a const " + \
                             "and iterate const times")

            low, ext, step = low.value, ext.value, step.value
            if ext > 114514:
                logging.log(logging.CRITICAL, \
                            '[Warning] Are you sure to unroll a large loop in Python?')

            bodies = []
            for i in range(low, low + ext, step):
                self.add_symbol(_name, Symbol.ConstLoopVar, i)
                body = visit_list_to_block(self.visit, node.body)
                body = self.wrap_up_realize(node, body)
                bodies.append(body)
                self.symbols.pop(_name)
            return concat_list_to_block(bodies)

        if isinstance(low, list):
            # if the lower bound is a list, we are in nested loops for grid
            # for each dim of the grid, generate a loop iter var
            loop_level = low.__len__()
            iter_var = []
            for i in range(loop_level):
                iter_var.append(_api.var(_name + "_" + str(i)))
            self.add_symbol(_name, Symbol.LoopVarTuple, iter_var)
            _body = visit_list_to_block(self.visit, node.body)
        elif iter_var is None:
            _internal_assert(_ir_pass.Equal(step, _api.const(1, dtype='int32')) or
                             _ir_pass.Equal(
                                 step, _api.const(-1, dtype='int32')),
                             "The loop step should be +/-1!")
            _internal_assert(for_type is not None,
                             "The loop iterating function parse error!")
            offset = iter_var = _api.var(_name)
            if _ir_pass.Equal(step, _api.const(-1, 'int32')):
                ext = ext * step
                offset = low - iter_var
            elif not _ir_pass.Equal(low, _api.const(0, 'int32')):
                offset = iter_var + low
            self.add_symbol(_name, Symbol.LoopVar, offset)
            _body = visit_list_to_block(self.visit, node.body)
        else:
            _internal_assert(for_type is None,
                             "The loop bind function parse error!")
            self.add_symbol(_name, Symbol.ThreadBind, iter_var)
            self.device += 1
            _body = visit_list_to_block(self.visit, node.body)
            self.device -= 1

        _body = self.wrap_up_realize(node, _body)

        if for_type is None:
            res = _body
        elif isinstance(low, list):
            # generate one loop for each dim of the grid
            for i in range(low.__len__()-1, -1, -1):
                _body = _make.For(iter_var[i], _api.const(0, 'int32'), ext[i], for_type, 0, _body)
            res = _body
        else:
            _internal_assert(not isinstance(for_type, tuple), \
                            "Micro expansion should be handled before!")
            res = _make.For(iter_var, _api.const(0, 'int32'), ext, for_type, 0, _body)

        self.symbols.pop(_name)
        return res


    def visit_Return(self, node):
        _internal_assert(all(ty != Symbol.LoopVar for ty, _ in self.symbols.values()), \
                         "Return should not be in a loop body!")
        ids = []
        if isinstance(node.value, ast.Name):
            ids = [node.value.id]
        else:
            _internal_assert(isinstance(node.value, ast.Tuple), \
                             "You should return either a single tensor or a tuple")
            _internal_assert(all(isinstance(i, ast.Name) for i in node.value.elts), \
                             "What do you return?")
            ids = [i.id for i in node.value.elts]
        _internal_assert(len(set(ids)) == len(ids), "Duplicated tensors in the return tuples")
        if len(ids) < len(self.outputs):
            logging.log(logging.CRITICAL, '[Warning] Not all the output buffers returned!')
        self.outputs = [self.symbols[i][1] for i in ids]
        self.returned = True
        return util.make_nop()


    def visit_Tuple(self, node):
        return tuple(self.visit(i) for i in node.elts)


    def visit_Str(self, node):
        return node.s


    def visit_Assert(self, node):
        test = self.visit(node.test)
        mesg = _api.convert(self.visit(node.msg))
        return _make.AssertStmt(test, mesg, util.make_nop())


def parse_python(src, args, symbols, closure_vars):
    """The helper function of calling the AST visitor

    Parameters
    ----------
    src : ast.node or str
        If an ast.node, then directly lower it.
        If a str, then parse it to ast and lower it.

    args : list of Tensors or Vars
        The argument lists to the function.
        It is NOT encouraged to write a function without arguments.
        It is NOT encouraged to write a function with side effect.

    symbols : list of str
        The symbol list of the global context of the function.

    closure_vars: dict
        A dict of external name reference captured by this function.

    Returns
    -------
    root : Stmt
        The result Halide IR and the parser class instance.
    """
    root = ast.parse(src) if isinstance(src, str) else src
    _internal_assert(root, ast.AST)
    var_usage = determine_variable_usage(root, args, symbols, closure_vars)
    parser = HybridParser(args, var_usage, symbols, closure_vars)
    parser.parsed_body = parser.visit(root)
    _internal_assert(parser.returned, 'No valid return found in the function body!')
    return parser


def source_to_op(src, args, symbols, closure_vars):
    """Another level of wrapper

    Parameters
    ----------
    src : ast.node or str
        If an ast.node, then directly lower it.
        If a str, then parse it to ast and lower it.

    args : list of Tensors or Vars
        The argument lists to the function.
        It is NOT encouraged to write a function without arguments.
        It is NOT encouraged to write a function with side effect.

    symbols : list of str
        The symbol list of the global context of the function.

    closure_vars: dict
        A dict of external name reference captured by this function.

    Returns
    -------
    res : list of output tensors
        The result of output tensors of the formed OpNode.
    """
    # add local hybrid script definition to symbols
    for k, v in closure_vars.items():
        if isinstance(v, types.FunctionType):
            symbols[k] = v

    parser = parse_python(src, args, symbols, closure_vars)

    input_tensors = []
    for i in args:
        if isinstance(i, Tensor):
            input_tensors.append(i)
    op = _tvm_internal._HybridOp(parser.func_name, "HybridOp", None, input_tensors,
                                 parser.outputs,
                                 parser.arg_buffers, parser.output_buffers,
                                 parser.arg_regions, parser.output_regions,
                                 parser.parsed_body)
    res = [op.output(i) for i in range(len(parser.outputs))]
    return res[0] if len(res) == 1 else res
