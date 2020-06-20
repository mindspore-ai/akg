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

"""parser the config params"""
from __future__ import absolute_import as _abs

import akg.tvm


def cce_product_(product):
    """
    dynamic load the product params.

    Args:
        product (str): product name.
    """
    product = product.lower()
    if isinstance(product, str):
        versions = product.split(".")
        if len(versions) != 4 and len(versions) != 5:
            raise RuntimeError("Donot support specify the product %s" % product)

        if product.startswith("3.5."):
            product = "3.5"
        elif product.startswith("3.3."):
            product = "3.3"
        elif product.startswith("1.1."):
            product = "1.1"
        elif product.startswith("1.2."):
            product = "1.2"
        elif product.startswith("1.6."):
            product = "1.6"
        else:
            raise RuntimeError("Donot support specify the product %s" % product)
    else:
        raise RuntimeError("The Supported product type error")

    cur_cce_product_params = CceProductParams()
    cur_cce_product_params.cce_product = product

    # set section to conf
    f = akg.tvm.get_global_func("cce.set_product_section")
    f(product)

    return cur_cce_product_params


def get_value(product, key):
    """
    call global func to get product value.

    Args:
        product (str): product name.
        key (str): key.
    """
    if "Buffer" in key:
        f = akg.tvm.get_global_func("cce.product_conf_buffer")

        value = f(product, key)
        if value == 0:
            raise RuntimeError("Get the cce product value is 0")

        return value
    if "Compiler" in key:
        f = akg.tvm.get_global_func("cce.product_conf_compiler")

        value = f(product, key)
        if value == "":
            raise RuntimeError("Get the cce product value is None")

        return value
    if "Intrinsic" in key:
        f = akg.tvm.get_global_func("cce.product_conf_intrinsic")

        value = f(product, key)
        if value == "":
            raise RuntimeError("Get the cce product value is None")

        return value
    if "Core" in key:
        f = akg.tvm.get_global_func("cce.product_conf_core")

        value = f(product, key)
        if value == 0:
            raise RuntimeError("Get the cce product value is None")

        return value
    return None


class CceProductParams():
    """define Cce Product Params class."""

    _instance = None
    cce_product = None
    # set false to switch off aicpuos feature
    enable_aicpuos = True

    def __init__(self):
        pass

    # singleton pattern
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = object.__new__(cls, *args, **kwargs)
        return cls._instance

    def get_params_(self, key):
        """get parameters."""
        if self.cce_product is None:
            raise RuntimeError("not set product info")

        value = get_value(self.cce_product, key)

        # if product supports os
        if key == "Compiler_aicpu_support_os":
            # string to bool
            value = bool(value == "true")

        return value


def set_status_check(bl):
    """
    call global func to set debug mode to add status special register check code to check if the compute overflow.

    Args:
        bl (bool): when true, the code will print the check code.
    """
    if not isinstance(bl, bool):
        raise TypeError("The input value type must be boolean")

    f = akg.tvm.get_global_func("cce.status_check")

    f(bl)
