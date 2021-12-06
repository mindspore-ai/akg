# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

"""operator dsl function: pooling"""
import akg
import akg.utils as utils
from akg.utils.format_transform import get_shape
from akg.ops.nn.ascend import MaxPool, Avgpool

def _pooling_compute(x, window, stride,
                     mode=0, pad_mode=5, pad=(0, 0, 0, 0)):
    """compute for pooling"""
    # convert mode&pad_mode to str
    if mode == 0:
        mode = "MAX"
    elif mode == 1:
        mode = "AVG"
    else:
        raise RuntimeError("Invalid mode parameters, mode must set 0 or 1.")

    if pad_mode == 5:
        pad_mode = "VALID"
    elif pad_mode == 6:
        pad_mode = "SAME"
    else:
        raise RuntimeError("Invalid pad_mode parameters, pad_mode must set 5 or 6.")

    # check pad
    if pad not in ((0, 0, 0, 0), [0, 0, 0, 0]):
        raise RuntimeError("Not support pad now!")

    in_size_h = x.shape[2].value
    in_size_w = x.shape[3].value

    window = list(window)

    if window[0] >= in_size_h and window[1] >= in_size_w:
        window[0] = in_size_h
        window[1] = in_size_w
        pad_mode = "VALID"
        stride = [1, 1]

    if mode == "MAX":
        res = MaxPool(x, window, stride, pad_mode)
    else:
        # AVG
        res = Avgpool(x, window, stride, pad_mode)

    return res


@utils.check_input_type(akg.tvm.tensor.Tensor,
                          (list, tuple), (list, tuple), (int, type(None)),
                          (int, type(None)), (list, tuple, type(None)),
                          (bool, type(None)), (int, type(None)))
def pooling(x, window, stride,
            mode=0, pad_mode=5, pad=(0, 0, 0, 0),
            global_pooling=False, ceil_mode=0):
    """
    Pooling operation, including MaxPool and AvgPool.

    Args:
        x (tvm.tensor.Tensor): Input tensor, only support float16
                               dtype, and NC1HWC0 format.
        window (Union[list, tuple]): Pooling window, only support pooling
                                     in H or W.
        stride (Union[list, tuple]): Pooling stride, only support pooling
                                     in H or W.
        mode (int): Mode of pooling, support MaxPool and AvgPool. 0 for MaxPool,
                    1 for AvgPool.
        pad_mode (int): Mode of padding, 5 for VALID, 6 for SAME.
        pad (Union[list, tuple]): Implicit padding size to up/down/left/right.
        global_pooling (bool): Global pooling flag, invalid now, should be False.
        ceil_mode (int): Round_mode params, invalid now, should be 0.

    Returns:
        A tvm.tensor.Tensor with same dtype as input.
    """
    utils.check_shape(get_shape(x))
    utils.ops_dtype_check(x.dtype, utils.DtypeForDavinci.FLOAT16)

    if len(window) != 2:
        raise RuntimeError("Invalid shape params, window shape must be 2 dims, "
                           "including window_h and window_w.")
    if len(stride) != 2:
        raise RuntimeError("Invalid shape params, stride shape must be 2 dims, "
                           "including stride_h and stride_w.")

    if global_pooling or ceil_mode != 0:
        raise RuntimeError("Not support global_pooling and ceil_mode for now.")

    return _pooling_compute(x, window, stride, mode, pad_mode, pad)
