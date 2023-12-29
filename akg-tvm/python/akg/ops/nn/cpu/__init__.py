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

from .conv_utils import get_channel_inners, pack_data, unpack_nchwc_to_nchw
from .layout_transform_utils import get_layout_list, get_alpha_only, \
    get_tiled_pair, get_idx_by_char, get_tile_by_char
from .conv2d import conv2d_nchwc
from .depthwise_conv2d import depthwise_conv2d_nchwc
from .layout_transform import layout_transform
from .pooling import pooling
from .global_pooling import global_pooling
