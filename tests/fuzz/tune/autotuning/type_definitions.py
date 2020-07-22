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

"""operator description and config param definitions"""
from collections import namedtuple

# op desc
ConvDesc = namedtuple("ConvDesc", ['fmap_shape', 'filter_shape', 'pad', 'stride', 'dilation', 'use_bias'])

ConvBackpropDesc = namedtuple("ConvBackpropDesc", ['fmap_shape', 'filter_shape', 'pad', 'stride', 'dilation'])

MatmulCubeDesc = namedtuple("MatmulCubeDesc", ["x_shape", "y_shape", "bias", "left_format", "right_format",
                                               "out_format", "adj_x", "adj_y", "dtype", "bias_dtype", "out_dtype"])

# config param definitions
ConvConfig = namedtuple('ConvConfig', ['tile_h', 'tile_co', 'tile_m', 'tile_k', 'tile_n', 'tile_w', 'bypass'])
ConvBackpropInputConfig = namedtuple('ConvBackpropInputConfig',
                                     ['tile_h', 'tile_co', 'tile_m', 'tile_k', 'tile_n', 'tile_w'])
ConvBackpropFilterConfig = namedtuple('ConvBackpropFilterConfig',
                                      ['tile_ci', 'tile_kh', 'tile_kw', 'tile_co', 'tile_batch',
                                       'tile_h', 'tile_w', 'tile_m', 'tile_k', 'tile_n'])
MatmulCubeConfig = namedtuple('MatmulCubeConfig', ['n_l1', 'n_l0', 'm_l1', 'm_l0', 'k_l1', 'k_l0', 'bypass'])

EmptyConfig = namedtuple('empty', [])
