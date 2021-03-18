# Copyright 2020 Huawei Technologies Co., Ltd
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

"""__init__"""
from .notequal import NotEqual
from .equal import Equal
from .greater_equal import GreaterEqual
from .less_equal import LessEqual
from .tile import Tile
from .cast import Cast
from .relu6 import ReLU6
from .logical_and import LogicalAnd
from .logical_not import LogicalNot
from .logical_or import LogicalOr
from .relu6_grad import ReLU6Grad
from .squeeze import Squeeze
from .squeeze_grad import SqueezeGrad, gpu_schedule_SqueezeGrad
from .sub import Sub
from .mul import Mul
from .hsigmoid import HSigmoid
from .hsigmoid_grad import HSigmoidGrad
from .hswish import HSwish
from .hswish_grad import HSwishGrad
from .assign import Assign
