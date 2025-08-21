#!/usr/bin/env python3
# coding: utf-8
# Copyright 2025 Huawei Technologies Co., Ltd
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

import ast
import inspect
import operator
from ast import NodeTransformer, NodeVisitor, parse, fix_missing_locations
from contextlib import contextmanager
from swft.utils.util import is_scalar, is_tensor
from .c_expression import push_to_list, compile_ckernel
from .instruction import Instruction
from .scalar import Scalar


def custom_and(*args):
    res = Scalar("BOOL")
    Instruction("SAND", (args[0], args[1], ), (res, ), None)()
    for arg in args[2:]:
        new_res = Scalar("BOOL")
        Instruction("SAND", (res, arg, ), (new_res, ), None)()
        res = new_res
    return res


def custom_or(*args):
    res = Scalar("BOOL")
    Instruction("SOR", (args[0], args[1], ), (res, ), None)()
    for arg in args[2:]:
        new_res = Scalar("BOOL")
        Instruction("SOR", (res, arg, ), (new_res, ), None)()
        res = new_res
    return res


def custom_not(arg):
    res = Scalar("BOOL")
    Instruction("SNOT", (arg, ), (res, ), None)()
    return res


@contextmanager
def code_block_context(name, cond=None):
    if not cond:
        Instruction(name + "_START", (), (), None)()
        yield
        Instruction(name + "_END", (), (), None)()
    elif isinstance(cond, Scalar) and (not cond.has_value()):
        Instruction(name + "_START", (cond,), (), None)()
        yield
        Instruction(name + "_END", (), (), None)()
    else:
        raise TypeError("for if (condition), condition must be scalar.")


def update_name(obj, name):
    if is_scalar(obj) or is_tensor(obj):
        obj.update_name(name)

def continue_():
    Instruction("CONTINUE", (), (), None)()

class RemoveControlFlowAndInjectContext(NodeTransformer):
    def __init__(self):
        self.block_counter = 0
        self.iter_var_set = set()
        self.in_dynamic_loop = False

    def visit_For(self, node):
        var_names = []
        if node.iter.func.id != "dynamic_loop":
            if isinstance(node.target, ast.Name):
                var_names = [node.target.id]
            elif isinstance(node.target, ast.Tuple):
                var_names = [
                    el.id for el in node.target.elts if isinstance(el, ast.Name)]
        else:
            self.in_dynamic_loop = True
        for var in var_names:
            self.iter_var_set.add(var)

        node = self.generic_visit(node)
        for var in var_names:
            self.iter_var_set.remove(var)
        return node

    def visit_Continue(self, node):
        if self.in_dynamic_loop:
            new_node = ast.Expr(
                value=ast.Call(
                    func=ast.Name(id='continue_', ctx=ast.Load()),
                    args=[],
                    keywords=[]
                )
            )
            return new_node
        return node

    def is_constant_expression(self, node):
        if isinstance(node, ast.Name):
            return node.id in self.iter_var_set
        elif isinstance(node, (ast.Constant, ast.Num)):
            return True
        elif isinstance(node, ast.UnaryOp):
            return self.is_constant_expression(node.operand)
        elif isinstance(node, ast.BinOp):
            return self.is_constant_expression(node.left) and self.is_constant_expression(node.right)
        elif isinstance(node, ast.Compare):
            if not self.is_constant_expression(node.left):
                return False
            for op, comparator in zip(node.ops, node.comparators):
                if not self.is_constant_expression(comparator):
                    return False
            return True
        elif isinstance(node, ast.BoolOp):
            return all(self.is_constant_expression(value) for value in node.values)
        else:
            return False

    def visit_If(self, node):
        node = self.generic_visit(node)
        if self.is_constant_expression(node.test):
            return node

        if_body = self.wrap_with_context(node.body, "IF", node.test, node)
        else_body = self.wrap_with_context(node.orelse, "ELSE", None, node)
        return if_body + else_body

    def visit_BoolOp(self, node):
        if isinstance(node.op, ast.And):
            func_name = "custom_and"
        elif isinstance(node.op, ast.Or):
            func_name = "custom_or"
        else:
            return self.generic_visit(node)

        self.generic_visit(node)

        return ast.Call(
            func=ast.Name(id=func_name, ctx=ast.Load()),
            args=node.values,
            keywords=[]
        )

    def visit_UnaryOp(self, node):
        if isinstance(node.op, ast.Not):
            self.generic_visit(node)

            return ast.Call(
                func=ast.Name(id="custom_not", ctx=ast.Load()),
                args=[node.operand],
                keywords=[]
            )
        return self.generic_visit(node)

    def wrap_with_context(self, statements, name, cond, node):
        if not statements:
            return []

        if not cond:
            ctx_expr = ast.Call(
                func=ast.Name(id='code_block_context', ctx=ast.Load()),
                args=[ast.Constant(value=name)],
                keywords=[],
                lineno=node.lineno,
                col_offset=node.col_offset
            )
        else:
            ctx_expr = ast.Call(
                func=ast.Name(id='code_block_context', ctx=ast.Load()),
                args=[ast.Constant(value=name), cond],
                keywords=[],
                lineno=node.lineno,
                col_offset=node.col_offset
            )

        with_item = ast.withitem(context_expr=ctx_expr, optional_vars=None)
        with_node = ast.With(items=[with_item], body=statements)

        return [with_node]


class ConstantFolding(NodeTransformer):
    BIN_OP_MAP = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.LShift: operator.lshift,
        ast.RShift: operator.rshift,
        ast.BitOr: operator.or_,
        ast.BitXor: operator.xor,
        ast.BitAnd: operator.and_,
    }

    UNARY_OP_MAP = {
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
        ast.Invert: operator.invert,
        ast.Not: operator.not_,
    }

    COMP_OP_MAP = {
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.Is: operator.is_,
        ast.IsNot: operator.is_not,
        ast.In: lambda a, b: a in b,
        ast.NotIn: lambda a, b: a not in b,
    }

    BOOL_OP_MAP = {
        ast.And: operator.and_,
        ast.Or: operator.or_,
    }

    def visit_BinOp(self, node):
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)
        if isinstance(node.left, ast.Constant) and isinstance(node.right, ast.Constant):
            op_func = self.BIN_OP_MAP.get(type(node.op))
            if op_func is not None:
                left_val = node.left.value
                right_val = node.right.value
                result = op_func(left_val, right_val)
                return ast.Constant(value=result, kind=None)
        return node

    def visit_UnaryOp(self, node):
        node.operand = self.visit(node.operand)
        if isinstance(node.operand, ast.Constant):
            op_func = self.UNARY_OP_MAP.get(type(node.op))
            if op_func is not None:
                operand_val = node.operand.value
                result = op_func(operand_val)
                return ast.Constant(value=result, kind=None)
        return node

    def visit_Compare(self, node):
        node.left = self.visit(node.left)
        node.comparators = [self.visit(comp) for comp in node.comparators]
        all_constants = (isinstance(node.left, ast.Constant) and
                         all(isinstance(c, ast.Constant) for c in node.comparators))
        if all_constants and len(node.ops) == len(node.comparators):
            left_val = node.left.value
            result = True
            for op, comparator in zip(node.ops, node.comparators):
                op_func = self.COMP_OP_MAP.get(type(op))
                if op_func is None:
                    return node
                comp_result = op_func(left_val, comparator.value)
                result = result and comp_result
                left_val = comparator.value
            return ast.Constant(value=result, kind=None)
        return node

    def visit_BoolOp(self, node):
        node.values = [self.visit(v) for v in node.values]
        all_constants = all(isinstance(v, ast.Constant) for v in node.values)
        if all_constants and node.values:
            op_func = self.BOOL_OP_MAP.get(type(node.op))
            if op_func is not None:
                result = node.values[0].value
                for value in node.values[1:]:
                    result = op_func(result, value.value)
                    if isinstance(node.op, ast.And) and not result:
                        break
                    if isinstance(node.op, ast.Or) and result:
                        break
                return ast.Constant(value=result, kind=None)
        return node

    def visit_If(self, node):
        node.test = self.visit(node.test)
        node.body = [self.visit(stmt) for stmt in node.body]
        node.orelse = [self.visit(stmt) for stmt in node.orelse]
        if isinstance(node.test, ast.Constant):
            if node.test.value:
                return node.body
            if node.orelse:
                return node.orelse
            return []
        return node

    def visit_IfExp(self, node):
        node.test = self.visit(node.test)
        node.body = self.visit(node.body)
        node.orelse = self.visit(node.orelse)

        if isinstance(node.test, ast.Constant):
            if node.test.value:
                return node.body
            return node.orelse
        return node


class ScalarCopy(NodeTransformer):
    def __init__(self):
        super().__init__()
        self.scalar_vars = set()
        self.defined_scalar_vars = set()

    def isScalarAssign(self, func):
        if ((isinstance(func, ast.Name) and func.id == "move_to_scalar") or
            (isinstance(func, ast.Attribute) and func.attr == "copy" and
             isinstance(func.value, ast.Call) and isinstance(func.value.func, ast.Name) and
             func.value.func.id == "Scalar")):
            return True
        return False

    def visit_Assign(self, node):
        if (isinstance(node.targets[0], ast.Name) and
                node.targets[0].id in self.defined_scalar_vars):
            new_expr = ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id=node.targets[0].id, ctx=ast.Load()),
                        attr='load',
                        ctx=ast.Load()
                    ),
                    args=[node.value],
                    keywords=[]
                )
            )
            return new_expr

        if (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and
                node.value.func.id == "Scalar"):
            var_name = node.targets[0].id
            new_value = ast.Call(
                func=ast.Attribute(
                    value=node.value,
                    attr="copy",
                    ctx=ast.Load(),
                ),
                args=[],
                keywords=[],
            )
            self.scalar_vars.add(var_name)
            self.defined_scalar_vars.add(var_name)
            return ast.Assign(targets=node.targets, value=new_value)

        if (isinstance(node.value, ast.Call) and self.isScalarAssign(node.value.func)):
            var_name = node.targets[0].id
            self.scalar_vars.add(var_name)
            self.defined_scalar_vars.add(var_name)
            return node

        if (isinstance(node.targets[0], ast.Name) and
                node.targets[0].id in self.scalar_vars):
            if (isinstance(node.value, ast.Name) and
                    node.value.id in self.scalar_vars):
                new_value = ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id=node.value.id, ctx=ast.Load()),
                        attr="copy",
                        ctx=ast.Load(),
                    ),
                    args=[],
                    keywords=[],
                )
                return ast.Assign(targets=node.targets, value=new_value)
            else:
                self.scalar_vars.remove(node.targets[0].id)
        return node


class NameTensor(ast.NodeTransformer):
    def visit_Assign(self, node):
        node = self.generic_visit(node)
        target_names = []
        for target in node.targets:
            if isinstance(target, ast.Name):
                target_names.append(target.id)
            elif isinstance(target, ast.Tuple):
                for element in target.elts:
                    if isinstance(element, ast.Name):
                        target_names.append(element.id)
            elif isinstance(target, ast.Subscript):
                continue
            elif isinstance(target, ast.Attribute):
                continue

        if not target_names:
            return node

        update_calls = []
        for name in target_names:
            call = ast.Expr(value=ast.Call(
                func=ast.Name(id='update_name', ctx=ast.Load()),
                args=[
                    ast.Name(id=name, ctx=ast.Load()),
                    ast.Constant(value=name)
                ],
                keywords=[]
            ))
            update_calls.append(call)
        return [node] + update_calls

    def visit_Module(self, node):
        node = self.generic_visit(node)
        new_body = []
        for stmt in node.body:
            if isinstance(stmt, list):
                new_body.extend(stmt)
            else:
                new_body.append(stmt)
        node.body = new_body
        return node

    def visit_FunctionDef(self, node):
        node = self.generic_visit(node)
        new_body = []
        for stmt in node.body:
            if isinstance(stmt, list):
                new_body.extend(stmt)
            else:
                new_body.append(stmt)
        node.body = new_body
        return node


class JITTransformer(NodeTransformer):
    def visit_FunctionDef(self, node):
        node = self.generic_visit(node)
        new_decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Name) and decorator.func.id == 'jit':
                    continue
                if (isinstance(decorator.func, ast.Attribute) and
                    isinstance(decorator.func.value, ast.Name) and
                    decorator.func.value.id == 'swft' and decorator.func.attr == 'jit'):
                    continue
            new_decorators.append(decorator)
        node.decorator_list = new_decorators
        return node


def transform(transform_types, tree):
    for transform_type in transform_types:
        transformer = transform_type()
        transformed_tree = transformer.visit(tree)
        fix_missing_locations(transformed_tree)
        tree = transformed_tree
    return tree


def compile_func(func, globalv):
    original_source = inspect.getsource(func)
    tree = parse(original_source)
    transform_types = [JITTransformer, ConstantFolding,
                       RemoveControlFlowAndInjectContext, ScalarCopy, NameTensor]
    transformed_tree = transform(transform_types, tree)
    namespace = {}
    globalv["code_block_context"] = code_block_context
    globalv["custom_and"] = custom_and
    globalv["custom_or"] = custom_or
    globalv["custom_not"] = custom_not
    globalv["update_name"] = update_name
    globalv["continue_"] = continue_
    exec(compile(transformed_tree, "<ast>", "exec"), globalv, namespace)
    modified_func_name = func.__name__
    modified_func = namespace[modified_func_name]
    return modified_func
