#!/usr/bin/env python3
# coding: utf-8
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

"""
Additional IR Pass for CCE
"""
from __future__ import absolute_import as _abs
import sys
import os
import logging

def AKGAddPath():
    """akg add path."""
    pwd = os.path.dirname(os.path.realpath(__file__))
    tvm_path = os.path.realpath(pwd)
    if tvm_path not in sys.path:
        sys.path.insert(0, tvm_path)
    else:
        sys.path.remove(tvm_path)
        sys.path.insert(0, tvm_path)


# --- Python 3.12+ Compatibility Update ---
# Python 3.12 *removed* the legacy import hooks `find_module`, and need to
# migrate to `find_spec`. Refer to:
# https://docs.python.org/3/whatsnew/3.10.html#deprecated
# https://docs.python.org/3/whatsnew/3.12.html#importlib
# https://docs.python.org/3.12/reference/import.html#the-meta-path
# https://docs.python.org/3/library/sys.html#sys.meta_path
# https://docs.python.org/3/library/importlib.html#importlib.abc.MetaPathFinder
if sys.version_info >= (3, 12):
    import importlib.abc
    import importlib.machinery

    class AKGMetaPathFinder(importlib.abc.MetaPathFinder):
        """class AKGMetaPath finder."""

        def find_spec(self, fullname, path, target=None):
            """Find the ModuleSpec for the given fullname."""
            rname = None
            if fullname.startswith("akg.tvm"):
                rname = fullname[4:]
            elif fullname.startswith("akg.topi"):
                rname = fullname[4:]
            elif fullname == "akg.topi.cce.cce_extended_op_build":
                logging.warning(
                    "akg error: 'akg.topi.cce.cce_extended_op_build' has been deprecated, please using "
                    "'akg.topi.cce.te_op_build' instead "
                )
                return None

            if rname:
                return importlib.machinery.ModuleSpec(fullname, AKGMetaPathLoader(rname))

            return None

else:
    class AKGMetaPathFinder:
        """class AKGMetaPath finder."""

        def find_module(self, fullname, path=None):
            """method akg find module."""
            if fullname.startswith("akg.tvm"):
                rname = fullname[4:]
                return AKGMetaPathLoader(rname)
            if fullname.startswith("akg.topi"):
                rname = fullname[4:]
                return AKGMetaPathLoader(rname)
            if fullname == "akg.topi.cce.cce_extended_op_build":
                logging.warning("akg error: 'akg.topi.cce.cce_extended_op_build' has been deprecated, please using "
                                "'akg.topi.cce.te_op_build' instead ")
            return None


# `load_module` method will be removed in Python3.15, and need to migrate to
# `exec_module`. Refer to:
# https://docs.python.org/3/whatsnew/3.12.html#pending-removal-in-python-3-15
# https://docs.python.org/3.12/library/importlib.html#importlib.abc.Loader
# https://docs.python.org/3.12/reference/import.html#loaders
class AKGMetaPathLoader:
    """class AKGMetaPathLoader loader."""
    def __init__(self, rname):
        self.__rname = rname

    def load_module(self, fullname):
        if self.__rname in sys.modules:
            sys.modules.pop(self.__rname)
        AKGAddPath()
        __import__(self.__rname, globals(), locals())
        self.__target_module = sys.modules[self.__rname]
        sys.modules[fullname] = self.__target_module
        return self.__target_module


# pylint: disable=missing-function-docstring
def schedule(sch, target = 'cuda'):
    def decorator(func):
        def wrapper(*args, **kwargs):
            binds = None
            output = func(*args, **kwargs)
            if isinstance(output, tuple):
                attrs = [t for t in output if isinstance(t, dict)]
                for attr in attrs:
                    if "binds" in attr.keys():
                        binds = attr['binds']
                # pylint: disable=consider-using-generator
                output = tuple([t for t in output if not isinstance(t, dict)])
            return {'schedule' : sch, 'target' : target, 'output' : output, 'binds': binds, 'op_name' : func.__name__}
        return wrapper
    return decorator

sys.meta_path.insert(0, AKGMetaPathFinder())

# pylint: disable=wrong-import-position
from . import autodiff
from .build_module import build, build_to_func, lower, build_config
from .autodiff import differentiate
from .autodiff import get_variables
from .autodiff import register_variables
from . import lang
from .utils.dump_cuda_meta import dump_cuda_meta
from .utils.dump_cpu_meta import dump_cpu_meta
from .utils.dump_ascend_meta import tvm_callback_cce_postproc

__all__ = ["differentiate"]
