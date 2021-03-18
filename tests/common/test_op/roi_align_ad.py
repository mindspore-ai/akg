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

"""operator dsl function: roi_align_ad"""
import akg
import akg.topi



def roi_align_ad(head, data, rois, pooled_size, spatial_scale, sample_ratio):
    output = akg.topi.vision.rcnn.roi_align.roi_align_nchw(data, rois, pooled_size, spatial_scale, sample_ratio)
    _jacs = list(akg.differentiate(output, [data], head))
    return _jacs[0]
