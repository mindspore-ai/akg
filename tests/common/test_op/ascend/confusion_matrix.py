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

"""confusion_matrix"""
import akg.tvm
import akg.utils as utils

@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, int, (str, type(None)))
def confusion_matrix(actual, predict, num_class, target=utils.CCE):
    """
    Computes the confusion matrix from predictions and labels.
    
    The matrix columns represent the prediction labels and the rows represent the real labels.
    The confusion matrix is always a 2-D array of shape `(num_class, num_class)`.
    Both `predict` and `actual` must be 1-D arrays of the same shape in order for this function to work.
    
    Args:
        actual (tvm.tensor.Tensor): 1-D tensor of type int32.
        predict (tvm.tensor.Tensor): 1-D tensor of type int32.
        num_class (int): The number of valid labels for a given classification task.
    
    Returns:
        tvm.tensor.Tensor, with shape `(num_class, num_class)` representing the confusion matrix.
    """
    utils.check_shape(actual, length=1, tensor_name="actual")
    utils.check_shape(predict, length=1, tensor_name="predict")
    utils.check_equal("length of actual", "length of predict", actual.shape[0].value, predict.shape[0].value)
    utils.ops_dtype_check([actual.dtype, predict.dtype], utils.DtypeForDavinci.INT32)

    N = num_class
    K = actual.shape[0].value
    k = akg.tvm.reduce_axis((0, K), "k")

    # reduce over first axis
    tmp = akg.tvm.compute((K, N, N),
        lambda k, i, j: akg.tvm.expr.Select(akg.tvm.all(i == actual[k], j == predict[k]), 1, 0),
        name="tmp")
    output = akg.tvm.compute((N, N), lambda i, j: akg.tvm.sum(tmp[k][i][j], axis=k))

    return output
