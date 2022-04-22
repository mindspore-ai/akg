# Copyright 2019-2021 Huawei Technologies Co., Ltd
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


def reg_op(op_name, target=None):
    """
    Register operator.

    :param op_name: str. The name of operator to be registered.
    :param target: None or str. The supported target of operator, if None, means support all targets.
    :return:
    """
    def decorator(op):
        registered_ops = getattr(reg_op, "registered_ops", {})
        if op_name in registered_ops:
            logging.warning("Op [{}] already exists in the registry and will be overridden!".format(op_name))
        registered_ops[op_name] = {"op": op, "target": target}
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
    registered_ops = getattr(reg_op, "registered_ops", None)
    if not isinstance(registered_ops, dict) or registered_ops.get(op_name) is None:
        raise ValueError("Op [{}] not found! Please register it first.".format(op_name))
    registered_target = registered_ops[op_name].get("target")
    if registered_target is not None and registered_target != target:
        raise ValueError("Op [{}] for target {} not found! Please register it first.".format(op_name, target))
    return registered_ops[op_name].get("op")
