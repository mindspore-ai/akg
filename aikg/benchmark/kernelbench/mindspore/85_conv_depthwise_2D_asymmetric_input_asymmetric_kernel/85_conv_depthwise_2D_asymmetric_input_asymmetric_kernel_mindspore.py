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
import mindspore as ms
from mindspore import nn

class Model(nn.Cell):
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size_h: int, kernel_size_w: int, stride_h: int = 1, stride_w: int = 1, padding_h: int = 0, padding_w: int = 0, dilation_h: int = 1, dilation_w: int = 1, groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, in_channels, (kernel_size_h, kernel_size_w), stride=(stride_h, stride_w), padding=(padding_h, padding_w), dilation=(dilation_h, dilation_w), groups=in_channels, bias=bias)
        
    def construct(self, x: ms.Tensor) -> ms.Tensor:
        
        return self.conv2d(x)

# Test code
batch_size = 16
in_channels = 3
out_channels = in_channels
kernel_size_h = 3
kernel_size_w = 5
width = 256
height = 128
stride_h = 1
stride_w = 1
padding_h = 0
padding_w = 0
dilation_h = 1
dilation_w = 1
groups = in_channels

def get_inputs():
    x = ms.ops.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size_h, kernel_size_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups]