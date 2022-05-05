# Copyright 2019-2021 Huawei Technologies Co., Ltd
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

"""focal_loss"""

import akg.tvm
import akg.topi
import akg.utils as utils
from akg.utils import custom_tiling as ct_util
from akg.utils.format_transform import get_shape
from akg.ops.math import reduce_max
from akg.utils.kernel_exec import product_is_mini

focal_loss_set_dim_map = {
    str(((8,4718,6),  "float16", "float16")) : ((1,1), (674,674), (16,16)),
    str(((8,4718,6),  "float32", "float32")) : ((1,1), (674,674), (16,16)),
    str(((8,4718,12), "float16", "float16")) : ((1,1), (674,674), (34,34)),
    str(((8,4718,12), "float32", "float32")) : ((1,1), (1,1), (34,34)),
    str(((8,4718,12), "float16", "int32")) : ((1,1), (674,674), (34,34)),
    str(((8,4718,12), "float32", "int32")) : ((1,1), (1,1), (34,34)),
    str(((8,8732,6),  "float16", "float16")) : ((1,1), (148,148), (16,16)),
    str(((8,8732,6),  "float32", "float32")) : ((1,1), (1,1), (16,16)),
    str(((8,59,6),    "float16", "float16")) : ((1,1), (59,59), (16,16)),
    str(((8,59,6),    "float32", "float32")) : ((1,1), (59,59), (16,16)),
    str(((8,8732,6),  "float16", "int32"))   : ((1,1), (236,236), (16,16)),
    str(((32, 8732, 6), "float16", "float16")): ((1, 1), (236, 236), (16, 16)),
}


def focal_loss_set_dim_func(prediction, target):
    """setdim function"""
    key = []
    key.append(tuple(prediction.shape))
    key.append(prediction.dtype)
    key.append(target.dtype)
    hash_key = str(tuple(key))

    if hash_key in focal_loss_set_dim_map.keys():
        return ct_util.set_dims(focal_loss_set_dim_map[hash_key]), hash_key
    else:
        return "", hash_key

@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, float)
def focal_loss(prediction, tar, gamma):
    """
    Calculate loss by focalloss.
    
    See Source: <a href="https://arxiv.org/abs/1708.02002">Focal Loss for Dense Object Detection;
                Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár</a>
    
    This op fuses activation function (`softmax`) and loss function (`focalloss`) together.
    
    .. math::
        p = softmax(x) \\
        FL(p) = -(1-p)^{\\gamma}log(p)
    
    Args:
        prediction (tvm.tensor.Tensor): The predicted logits for each class,
            type is float32 or float16 and shape is `(batch_size, num_anchors, num_clases)`,
        tar (tvm.tensor.Tensor): The one-hot encoded classification targets,
            type is float32, float16 or int32 and shape is `(batch_size, num_anchors, num_classes)`,
        gamma (float): positive float number.
    
    Returns:
        tvm.tensor.Tensor, has the same type as inputs with shape `(batch_size, num_anchors)`.
    """

    utils.check_shape(prediction, length=3, tensor_name="prediction")
    utils.check_shape(tar, length=3, tensor_name="target")
    utils.ops_dtype_check(prediction.dtype, utils.DtypeForDavinci.ALL_FLOAT)
    utils.ops_dtype_check(tar.dtype, [utils.DtypeForDavinci.ALL_FLOAT, utils.DtypeForDavinci.INT32])
    utils.check_greater("gamma", "zero", gamma, 0)

    dim_info, _ = focal_loss_set_dim_func(prediction, tar)
    attrs = {"dim": dim_info}

    dtype = prediction.dtype

    if product_is_mini() and dtype == 'float32':
        prediction = akg.topi.cast(prediction, "float16")
        tar = akg.topi.cast(tar, "float16")

    axis = -1
    shape = get_shape(prediction)

    maxv = reduce_max(prediction, axis=axis, keepdims=True, target=utils.CCE)

    k1 = akg.tvm.reduce_axis((0, shape[-1]), name="k1")
    expsum = akg.tvm.compute(shape[:-1], lambda *i: akg.tvm.sum(
        akg.tvm.exp(prediction(*i, k1) - maxv(*i, 0)), axis=k1), name="expsum")

    gamma = akg.tvm.const(gamma, prediction.dtype)
    one = akg.tvm.const(1, prediction.dtype)

    def cal_focalloss(*i):
        x = prediction(*i) - maxv(*i[:-1], 0)
        pred = akg.tvm.exp(x - akg.tvm.log(expsum(*i[:-1])))  # softmax(x)
        log_p = x - akg.tvm.log(expsum(*i[:-1]))  # logsoftmax(x)
        neg_pred_pow = akg.tvm.exp(akg.tvm.log(one - pred) * gamma)  # (1-pred)^gamma
        loss = akg.tvm.const(-1, prediction.dtype) * tar(*i) * neg_pred_pow * log_p
        return loss
    loss = akg.tvm.compute(shape, cal_focalloss, name="loss")

    loss = akg.topi.sum(loss, axis=axis)

    if product_is_mini() and dtype == 'float32':
        loss = akg.topi.cast(loss, "float32")

    return loss, attrs
