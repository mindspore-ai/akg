#!/usr/bin/env python3
# coding: utf-8
# Copyright 2021 Huawei Technologies Co., Ltd
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
auto tuner function
"""
from akg.auto_tune.job import launch_json


def tune_composite(json_str, tune_level=0, repo_path="repo.json", skip_exist=False):
    """
    tune composite
    Args:
       json_str    : str of compute description
       tune_level  : interger value to specify the tuning level: higher level expects better performance while lower level tunes much faster
       repo_path   : the path of repo file to save tuning result
       skip_exist  : whether skip tuning when there is already previous tuning result found in repo file
    Returns:
       attrs       : the best config
    """
    iter_times = [80, 160, 320] if tune_level == 0 else [15, 15, 15]
    debug_mode = False
    save_res = True
    all_space = False
    skip_file = False
    extra_tune = False
    self_attrs = None
    tuning_attrs = ['enable_pre_poly_loop_partition',
                    'enable_post_poly_loop_partition']

    best_results = launch_json(debug_mode=debug_mode, save_res=save_res, input_str=json_str, repo_path=repo_path, all_space=all_space,
                              skip_exist=skip_exist, skip_file=skip_file, extra_tune=extra_tune, self_attrs=self_attrs, tuning_attrs=tuning_attrs, iter_times=iter_times)

    attrs = {}
    if best_results is not None and len(best_results) == 1:
        bst = best_results[0]
        if bst is not None:
            if isinstance(bst, dict):
                attrs = bst.get("metadata", {}).get("attrs", {})
            elif bst.best_config is not None:
                best_config = best_results[0].best_config.input
                index_table = best_results[0].index_table
                from akg.auto_tune.runner import get_attr_from_config
                attrs = get_attr_from_config(best_config, index_table)
    return attrs
