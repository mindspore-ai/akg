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

"""focalloss_grad"""
import akg
import akg.tvm
import akg.topi
from akg.utils import custom_tiling as ct_util
import akg.utils as utils
from akg.utils.format_transform import get_shape
from akg.ops.math.ascend import RecPositive

focalloss_grad_set_dim_map = {
    str(((32, 8732, 6), "float16")): ((1, 1), (236, 236), (21, 21)),
    str(((8, 4718, 12), "float32")): ((1, 1), (1, 1), (48, 48)),

}


def focalloss_grad_set_dim_func(prediction):
    """setdim function"""
    key = []
    key.append(tuple(prediction.shape))
    key.append(prediction.dtype)
    hash_key = str(tuple(key))

    if hash_key in focalloss_grad_set_dim_map.keys():
        return ct_util.set_dims(focalloss_grad_set_dim_map[hash_key]), hash_key
    else:
        return "", hash_key


def focal_loss_2_classification_bwd(labels, logits, grad, alpha=0.5, gamma=2):
    """focalloss for 2 classification"""
    batch_size = get_shape(labels)[0]
    pred = akg.topi.sigmoid(logits)
    log_p = akg.topi.log(pred)
    neg_log_p = akg.topi.log(1 - pred)
    pred_pow = akg.topi.power(pred, gamma)
    neg_pred_pow = akg.topi.power(1 - pred, gamma)
    d_labels = akg.tvm.compute((batch_size,),
        lambda i: (-alpha * neg_pred_pow[i] * log_p[i] + (1 - alpha) * pred_pow[i] * neg_log_p[i]) * grad[i])
    d_logits = akg.tvm.compute((batch_size,),
        lambda i: (-labels[i] * alpha *
            (-log_p[i] * gamma * neg_pred_pow[i] * pred[i] + neg_pred_pow[i] * (1 - pred[i])) +
            (labels[i] - 1) * (1 - alpha) *
            (gamma * pred_pow[i] * (1 - pred[i]) * neg_log_p[i] - pred_pow[i] * pred[i])) * grad[i])
    return d_labels, d_logits


@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, (float, int))
def focal_loss_bwd(prediction, labels, grad, gamma):
    """
    Gradient for focal_loss.
    
    Args:
        prediction (tvm.tensor.Tensor): The predicted logits for each class,
            type is float32 or float16, shape is `(batch_size, num_anchors, num_clases)`.
        labels (tvm.tensor.Tensor): The one-hot encoded classification targets,
            type is float32, float16 or int32, shape is `(batch_size, num_anchors, num_classes)`.
        grad (tvm.tensor.Tensor): The input gradient of backpropagation,
            type is float32 or float16, shape is `(batch_size, num_anchors)`.
        gamma (float): Positive float number.
    
    Returns:
    tvm.tensor.Tensor, has the same type and shape as `prediction`, representing `d_prediction`.
    """

    utils.check_shape(prediction, length=3, tensor_name="prediction")
    utils.check_shape(labels, length=3, tensor_name="labels")
    utils.ops_dtype_check([prediction.dtype, grad.dtype], utils.DtypeForDavinci.ALL_FLOAT)
    utils.ops_dtype_check(labels.dtype, [utils.DtypeForDavinci.ALL_FLOAT, utils.DtypeForDavinci.INT32])
    utils.check_greater("gamma", "zero", gamma, 0)

    dim_info, _ = focalloss_grad_set_dim_func(prediction)
    attrs = {"dim": dim_info}

    gamma = akg.tvm.const(gamma, dtype=prediction.dtype)

    # softmax: (x-max(x))/(sum(x-max(x)))
    axis = -1
    max_pred = akg.lang.ascend.reduce_max(prediction, axis=axis, keepdims=True)
    max_broadcast = akg.lang.ascend.broadcast(max_pred, prediction.shape)
    data_sub = akg.lang.ascend.vsub(prediction, max_broadcast)
    data_exp = akg.lang.ascend.vexp(data_sub)
    data_expsum = akg.lang.ascend.sum(data_exp, axis, keepdims=True)
    data_expsum_broadcast = akg.lang.ascend.broadcast(data_expsum, prediction.shape)
    para = RecPositive(data_expsum_broadcast)
    pred = akg.lang.ascend.vmul(data_exp, para)

    # logsoftmax: x - max - log(sum(exp(x - max)))
    log_sum_1 = akg.topi.log(data_expsum)
    log_sum = akg.lang.ascend.broadcast(log_sum_1, prediction.shape)
    log_p = akg.lang.ascend.vsub(data_sub, log_sum)

    one_sub_pred = akg.tvm.compute(pred.shape,
        lambda *indice: akg.tvm.const(1, pred.dtype) - pred(*indice),
        name="one_sub_pred")

    # (1-pred)^gamma
    vlog_t = akg.tvm.compute(pred.shape, lambda *indice: akg.tvm.log(one_sub_pred(*indice)), name="vlog_t")
    vmuls_t = akg.tvm.compute(pred.shape, lambda *indice: vlog_t(*indice) * gamma, name="vmuls_t")
    neg_pred_pow = akg.tvm.compute(pred.shape, lambda *indice: akg.tvm.exp(vmuls_t(*indice)), name="neg_pred_pow")

    def d_logits1_compute(_labels, _log_p, _neg_pred_pow, _pred, _one_sub_pred):
        _t1 = gamma * _log_p * _neg_pred_pow * _pred  # gamma * log(pred) * (1-pred)^gamma * pred
        _t2 = _neg_pred_pow * _one_sub_pred   # (1-pred)^(gamma+1)
        _t3 = _t1 - _t2
        _out = _labels * _t3
        return _out

    d_logits1 = akg.tvm.compute(pred.shape,
        lambda *i: d_logits1_compute(labels(*i), log_p(*i), neg_pred_pow(*i), pred(*i), one_sub_pred(*i)),
        name="d_logits1")

    def d_logits2_tmp_compute(_labels, _vlog_t, _pred, _log_p, _neg_pred_pow):
        # gamma * (1-pred)^(gamma-1) * pred * log(pred)
        _t1 = gamma * akg.tvm.exp(_vlog_t * (gamma - 1)) * _pred * _log_p
        _t2 = _neg_pred_pow - _t1
        _out = _labels * _t2
        return _out

    tmp1 = akg.tvm.compute(pred.shape,
        lambda *i: d_logits2_tmp_compute(labels(*i), vlog_t(*i), pred(*i), log_p(*i), neg_pred_pow(*i)),
        name="tmp1")
    tmp2 = akg.lang.ascend.sum(tmp1, axis=-1, keepdims=True)
    d_logits2 = akg.tvm.compute(pred.shape, lambda i, j, k: tmp2[i, j, 0] * pred[i, j, k], name="d_logits2")

    d_logits = akg.tvm.compute(pred.shape,
        lambda i, j, k: (akg.tvm.expr.Select(labels[i, j, k] > akg.tvm.const(0.5, labels.dtype),
            d_logits1[i, j, k], d_logits2[i, j, k])) * grad[i, j],
        name="d_logits")
    return d_logits, attrs
