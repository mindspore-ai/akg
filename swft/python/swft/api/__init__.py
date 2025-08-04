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

from .compute import mmad, vadd, vsub, vmul, vdiv, vmax, vmin, vand, vor, \
    vadds, vmuls, vmaxs, vmins, vexp, vsqrt, vrelu, vln, vrec, vabs, vnot, \
    vconv, vconv_s42f16, vconv_s42s8, vbrcb, vector_dup, vcmax, vcmin, vcadd, \
    vsubs, vdivs, vcmpv, vcmpvs, where
from .context import get_block_idx
from .move import move_to_gm, move_to_ub, move_to_l1, move_to_l0A, move_to_l0B, move_to_l0C, move_to_scalar, move_scalar_to_ub
from .transdata import nd_to_nz, nz_to_nd, transpose, reshape, nchw_to_nc1hwc0, change_view, transpose_to_gm
from .slicedata import slice, split_to_ub, slice_to_l1, split_to_l1, split_to_l0A, split_to_l0B, concat, concat_to_l1, concat_to_gm, slice_to_ub, pad_to_ub, insert_to_gm, slice_to_l0A, slice_to_l0B
from .sync import sync_cores
from .gatherandscatter import vgather
from .sort import vconcat, vsort16, vmrgsort4, vextract
from .composite import tanh, arange
from swft.runtime import exec_kernel