#!/usr/bin/env python3
# coding: utf-8
# Copyright 2019-2023 Huawei Technologies Co., Ltd
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
The build utils in python.
This module provides the functions to transform schedule to
LoweredFunc and compiled Module.
"""
from __future__ import absolute_import as _abs
import sys
import logging
from akg.utils import validation_check as vc_util
import akg.tvm
from akg.tvm import _api_internal
from akg.tvm import schedule


help_tiling_level = {
    "None": 0, "General": 1, "Candidates": 2, "Tuning": 3
}
EMPTY_CODE = 0
L0_DEFAULT_TILING = 1


def dump_tiling_info(level, tuning_spaces = None):
    """Dump tiling info."""
    if tuning_spaces is None:
        return
    logging.getLogger().setLevel(logging.INFO)
    if level >= help_tiling_level["General"]:
        logging.info("==========General tiling help info=============")
        indice = tuning_spaces["index"]
        if isinstance(indice, list):
            for i in range(len(indice)):
                info = "index %d, axis %d, l1_tile_ranges [%d, %d](jump by %d),l0_tile_ranges [%d, %d](jump by %d)"
                logging.info(info, tuning_spaces["index"][i][0], tuning_spaces["index"][i][1],
                             tuning_spaces["c1_range"][i][0], tuning_spaces["c1_range"][i][1],
                             tuning_spaces["c1_mod"][i][0], tuning_spaces["c0_range"][i][0],
                             tuning_spaces["c0_range"][i][1], tuning_spaces["c0_mod"][i][0],
                             )
            idx_to_str = {0: "x", 1: "y", 2: "z"}
            for i in range(len(tuning_spaces["thread_range"])):
                info = "[thread.%s] range [%d, %d](jump by %d), "
                logging.info(info, idx_to_str[i], tuning_spaces["thread_range"][i][0], tuning_spaces["thread_range"][i][1],
                             tuning_spaces['thread_mod'][i][0], )
            for i in range(len(tuning_spaces["block_range"])):
                info = "[block.%s]  range [%d, %d](jump by %d)"
                logging.info(info, idx_to_str[i], tuning_spaces["block_range"][i][0],
                             tuning_spaces["block_range"][i][1], tuning_spaces['block_mod'][i][0],)
            logging.info("===============================================")
        elif isinstance(indice, int) and indice == EMPTY_CODE:
            logging.info("Empty tiling space.")

    if level >= help_tiling_level["Candidates"]:
        logging.info("")
        logging.info("==========Detailed tiling help info(Only L1)=============")
        logging.info("index 0 has %d candidate(s) tiling factors", len(tuning_spaces["tuning_space"]))
        tuning_spaces_len = len(tuning_spaces["tuning_space"])
        for i in range(tuning_spaces_len):
            info = "candidate %d:("
            for l1_candidate in tuning_spaces["tuning_space"][i]:
                info += ("(" + str(l1_candidate) + ", " + str(L0_DEFAULT_TILING) + "),")
            info += ")"
            logging.info(info, i)
    logging.info("=============================================================")
    logging.info("")
    logging.info("Please read this tiling help info and set tiling factor.")
    logging.info("And then set attr \"help_tiling\" value to 0 and re-run.")
    logging.info("Exit.")
    sys.exit()


def build_config(**kwargs):
    """build config."""
    return akg.tvm.build_config(**kwargs)


@vc_util.check_input_type(schedule.Schedule, (list, tuple), (list, tuple), str,
                          (dict, type(None)), (dict, type(None)), bool, bool, bool, str)
def lower(sch, args, shape_params=None, name="default_function", binds=None, attrs=None,
          simple_mode=False, polyhedral=False, tuning=False, target="cce"):
    """Lowering function."""
    tmp_binds = None
    if binds is not None:
        tmp_binds = None if not bool(binds) else binds
    tmp_attrs = None
    if attrs is not None:
        tmp_attrs = None if not bool(attrs) else attrs
    if shape_params is None:
        shape_params = []

    cfg = _api_internal._GetCurrentBuildConfig()
    ret = _api_internal._Lower(sch, args, shape_params, name,
                               tmp_binds, tmp_attrs, simple_mode,
                               polyhedral, tuning, target, cfg)
    if tmp_attrs is None:
        tmp_attrs = {}
    level = tmp_attrs.get("help_tiling")
    if attrs.get("use_new_space", False):
        # new space: constraints format
        print("NEW SPACE: ", ret)
        space = dict()
        space['tune_schedule'] = ret.tune_schedule
        space['tune_constraints'] = ret.tune_constraints
        return space
    elif tuning or (level is not None and level > help_tiling_level['None']):
        level = help_tiling_level['Tuning'] if tuning else level
        tuning_spaces = {}
        tuning_spaces["index"] = ret.index_table.asnumpy().tolist()
        tuning_spaces["c1_range"] = ret.c1_tile_range_table.asnumpy().tolist()
        tuning_spaces["c0_range"] = ret.c0_tile_range_table.asnumpy().tolist()
        tuning_spaces["c1_mod"] = ret.c1_tile_mod_table.asnumpy().tolist()
        tuning_spaces["c0_mod"] = ret.c0_tile_mod_table.asnumpy().tolist()
        tuning_spaces["thread_range"] = ret.gpu_thread_range_table.asnumpy().tolist()
        tuning_spaces["block_range"] = ret.gpu_block_range_table.asnumpy().tolist()
        tuning_spaces["thread_mod"] = ret.gpu_thread_mod_table.asnumpy().tolist()
        tuning_spaces["block_mod"] = ret.gpu_block_mod_table.asnumpy().tolist()
        if level >= help_tiling_level["Candidates"]:
            tuning_spaces["tuning_space"] = ret.tiling_candidate.asnumpy().tolist()
        if not tuning:
            dump_tiling_info(level, tuning_spaces)
    return ret


@vc_util.check_input_type(schedule.Schedule, (list, tuple), (list, tuple, type(None)), str,
                          (dict, akg.tvm.container.Map, type(None)), (dict, type(None)), bool, str)
def build_to_func(inputs, args, shape_params=None, name="default_function",
                  binds=None, attrs=None, polyhedral=False, target="cce"):
    """Build module."""
    tmp_binds = None
    if binds is not None:
        tmp_binds = None if not bool(binds) else binds
    tmp_attrs = None
    if attrs is not None:
        tmp_attrs = None if not bool(attrs) else attrs
    for arg in args:
        vc_util.tensor_max_size_check(arg)
    if shape_params is None:
        shape_params = []
    cfg = _api_internal._GetCurrentBuildConfig()
    return _api_internal._BuildToFunc(inputs, args, shape_params, name, tmp_binds, tmp_attrs,
                                      polyhedral, target, cfg)

@vc_util.check_input_type(schedule.Schedule, (list, tuple), str, (list, tuple), str,
                          (dict, akg.tvm.container.Map, type(None)), (dict, type(None)), bool)
def build(inputs, args, target='cce', shape_params=None, name="default_function",
          binds=None, attrs=None, polyhedral=False):
    tmp_rst = build_to_func(inputs, args, shape_params=shape_params, name=name, binds=binds,
                            attrs=attrs, polyhedral=polyhedral, target=target)

    return _api_internal._BuildToModule(tmp_rst, target)
