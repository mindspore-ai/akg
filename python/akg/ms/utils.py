# Copyright 2019-2022 Huawei Technologies Co., Ltd
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

"""utils"""

import logging
import functools

# input format begin
DEFAULT = "DefaultFormat"
NCHW = "NCHW"
NHWC = "NHWC"
HWCN = "HWCN"
NC1HWC0 = "NC1HWC0"
FRAC_Z = "FracZ"
# input format end

# fusion type begin
ELEMWISE = "ELEMWISE"
CONVLUTION = "CONVLUTION"
COMMREDUCE = "COMMREDUCE"
SEGMENT = "SEGMENT"
OPAQUE = "OPAQUE"
# fusion type end

BINDS = "binds"

ALL = "all"


def _process_reg_targets(targets):
    if targets is None:
        targets = ALL
    if not isinstance(targets, (list, tuple)):
        targets = [targets]
    for _, target in enumerate(targets):
        if not isinstance(target, str):
            raise TypeError("targets should be of type None, str or list/tuple of str, but got {} with type {}"
                            .format(targets, type(targets)))
    targets = list(set(targets))
    return targets


def reg_op(op_name, targets=None):
    """
    Register operator.

    :param op_name: str. The name of operator to be registered.
    :param targets: None, str or list/tuple of str. The supported targets of operator, if None,
        means support all targets.
    :return:
    """

    targets = _process_reg_targets(targets)
    if not targets:
        raise ValueError("targets can not be empty!")

    def decorator(op):
        registered_ops = getattr(reg_op, "registered_ops", {})
        if op_name not in registered_ops:
            registered_ops[op_name] = {}
        for target in targets:
            if target in registered_ops[op_name]:
                logging.warning("Op [{}] for target {} already exists in the registry and will be overridden!"
                                .format(op_name, target))
            registered_ops[op_name][target] = op
        setattr(reg_op, "registered_ops", registered_ops)

        @functools.wraps(op)
        def wrapper(*args, **kwargs):
            return op(*args, **kwargs)

        return wrapper

    return decorator


def get_op(op_name, target):
    """
    Get operator from registry.

    :param op_name: str. The name of operator.
    :param target: None or str. The target of operator.
    :return: The operator.
    """
    if target is None:
        target = ALL
    if not isinstance(target, str):
        raise TypeError("target should be of type None or str, but got {} with type {}".format(target, type(target)))

    registered_ops = getattr(reg_op, "registered_ops", None)
    if not isinstance(registered_ops, dict) or registered_ops.get(op_name) is None:
        raise ValueError("Op [{}] for target {} is not found in the registry! Please register it first."
                         .format(op_name, target))
    op = registered_ops[op_name].get(target, registered_ops[op_name].get(ALL))
    if op is None:
        raise ValueError("Op [{}] for target {} is not found in the registry! Please register it first."
                         .format(op_name, target))
    return op
