# Copyright 2022 Huawei Technologies Co., Ltd
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

def get_layout_list(data_layout="NCHW8c"):
    """Get the layout as str list.
    For example, get_layout_list("NCHW8c") = list("N", "C", "H", "W", "8c")
    """
    vec = []
    cur_str = ""
    for ch in data_layout:
        cur_str += ch
        if not ch.isdigit():
            vec.append(cur_str)
            cur_str = ""
    return vec


def get_alpha_only(s):
    """Get the aplha from string.
    For example get_alpha_only("8c") = "c"
    """
    for ch in s:
        if ch.isalpha():
            return ch


def get_tiled_pair(layout, exclude=None):
    """Get the idx for tiled pair.
    For example, layout can be list("N", "C", "H", "W", "8c")
    """
    if exclude == None:
      exclude = list()
    layout_len = len(layout)
    for i in range(layout_len):
        for j in range(i + 1, layout_len):
            if layout[i].lower() == get_alpha_only(layout[j]) and layout[i] not in exclude:
                return i, j
    return -1, -1


def get_idx_by_char(layout, ch):
    """Get the idx by specific char."""
    layout_len = len(layout)
    for i in range(layout_len):
        if layout[i] == ch:
            return i
    raise ValueError("can not find the char={} form layout={}".format(
        ch, layout
    ))


def get_tile_by_char(layout, ch):
    """Get int("x") form NCHW[x]c."""
    layout_len = len(layout)
    for i in range(layout_len):
        if ch in layout[i]:
            return int(layout[i][:-1])