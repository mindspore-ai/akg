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

"""operator dsl function:prelu"""

import akg.tvm
import akg.topi
from akg.utils import validation_check as vc_util
from akg.utils import custom_tiling as ct_util
from akg.utils import kernel_exec as utils

add_set_dim_map = {
    #
    str(((1, 64, 112, 112), (64,), "float16")): ((1, 1), (64, 1), (112, 1), (112, 1)),
    str(((1, 64, 56, 56), (64,), "float16")): ((1, 1), (64, 1), (56, 1), (56, 1)),
    str(((1, 128, 56, 56), (128,), "float16")): ((1, 1), (128, 1), (56, 1), (56, 1)),
    str(((1, 128, 28, 28), (128,), "float16")): ((1, 1), (128, 1), (28, 1), (28, 1)),
    str(((1, 256, 28, 28), (256,), "float16")): ((1, 1), (256, 1), (28, 1), (28, 1)),
    str(((1, 256, 14, 14), (256,), "float16")): ((1, 1), (256, 1), (14, 1), (14, 1)),
    str(((1, 512, 14, 14), (512,), "float16")): ((1, 1), (512, 1), (14, 1), (14, 1)),
    str(((1, 512, 7, 7), (512,), "float16")): ((1, 1), (512, 1), (7, 1), (7, 1)),
    #
    str(((1, 64, 112, 112), (1,), "float16")): ((1, 1), (64, 1), (112, 1), (112, 1)),
    str(((1, 64, 56, 56), (1,), "float16")): ((1, 1), (64, 1), (56, 1), (56, 1)),
    str(((1, 128, 56, 56), (1,), "float16")): ((1, 1), (128, 1), (56, 1), (56, 1)),
    str(((1, 128, 28, 28), (1,), "float16")): ((1, 1), (128, 1), (28, 1), (28, 1)),
    str(((1, 256, 28, 28), (1,), "float16")): ((1, 1), (256, 1), (28, 1), (28, 1)),
    str(((1, 256, 14, 14), (1,), "float16")): ((1, 1), (256, 1), (14, 1), (14, 1)),
    str(((1, 512, 14, 14), (1,), "float16")): ((1, 1), (512, 1), (14, 1), (14, 1)),
    str(((1, 512, 7, 7), (1,), "float16")): ((1, 1), (512, 1), (7, 1), (7, 1)),
    #
    str(((1, 64, 112, 112), (64,), "float32")): ((1, 1), (64, 1), (112, 1), (112, 1)),
    str(((1, 64, 56, 56), (64,), "float32")): ((1, 1), (64, 1), (56, 1), (56, 1)),
    str(((1, 128, 56, 56), (128,), "float32")): ((1, 1), (128, 1), (56, 1), (56, 1)),
    str(((1, 128, 28, 28), (128,), "float32")): ((1, 1), (128, 1), (28, 1), (28, 1)),
    str(((1, 256, 28, 28), (256,), "float32")): ((1, 1), (256, 1), (28, 1), (28, 1)),
    str(((1, 256, 14, 14), (256,), "float32")): ((1, 1), (256, 1), (14, 1), (14, 1)),
    str(((1, 512, 14, 14), (512,), "float32")): ((1, 1), (512, 1), (14, 1), (14, 1)),
    str(((1, 512, 7, 7), (512,), "float32")): ((1, 1), (512, 1), (7, 1), (7, 1)),
    #
    str(((1, 64, 112, 112), (1,), "float32")): ((1, 1), (64, 1), (112, 1), (112, 1)),
    str(((1, 64, 56, 56), (1,), "float32")): ((1, 1), (64, 1), (56, 1), (56, 1)),
    str(((1, 128, 56, 56), (1,), "float32")): ((1, 1), (128, 1), (56, 1), (56, 1)),
    str(((1, 128, 28, 28), (1,), "float32")): ((1, 1), (128, 1), (28, 1), (28, 1)),
    str(((1, 256, 28, 28), (1,), "float32")): ((1, 1), (256, 1), (28, 1), (28, 1)),
    str(((1, 256, 14, 14), (1,), "float32")): ((1, 1), (256, 1), (14, 1), (14, 1)),
    str(((1, 512, 14, 14), (1,), "float32")): ((1, 1), (512, 1), (14, 1), (14, 1)),
    str(((1, 512, 7, 7), (1,), "float32")): ((1, 1), (512, 1), (7, 1), (7, 1)),
    #
    str(((128, 64, 112, 112), (64,), "float16")): ((1, 1), (1, 1), (16, 1), (112, 1)),
    str(((128, 64, 56, 56), (64,), "float16")): ((1, 1), (1, 1), (56, 1), (56, 1)),
    str(((128, 128, 56, 56), (128,), "float16")): ((1, 1), (1, 1), (56, 1), (56, 1)),
    str(((128, 128, 28, 28), (128,), "float16")): ((1, 1), (1, 1), (28, 1), (28, 1)),
    str(((128, 256, 28, 28), (256,), "float16")): ((1, 1), (1, 1), (28, 1), (28, 1)),
    str(((128, 256, 14, 14), (256,), "float16")): ((1, 1), (1, 1), (14, 1), (14, 1)),
    str(((128, 512, 14, 14), (512,), "float16")): ((1, 1), (1, 1), (14, 1), (14, 1)),
    str(((128, 512, 7, 7), (512,), "float16")): ((1, 1), (1, 1), (7, 1), (7, 1)),
    #
    str(((128, 64, 112, 112), (1,), "float16")): ((1, 1), (1, 1), (112, 1), (112, 1)),
    str(((128, 64, 56, 56), (1,), "float16")): ((1, 1), (1, 1), (56, 1), (56, 1)),
    str(((128, 128, 56, 56), (1,), "float16")): ((1, 1), (1, 1), (56, 1), (56, 1)),
    str(((128, 128, 28, 28), (1,), "float16")): ((1, 1), (1, 1), (28, 1), (28, 1)),
    str(((128, 256, 28, 28), (1,), "float16")): ((1, 1), (1, 1), (28, 1), (28, 1)),
    str(((128, 256, 14, 14), (1,), "float16")): ((1, 1), (1, 1), (14, 1), (14, 1)),
    str(((128, 512, 14, 14), (1,), "float16")): ((1, 1), (1, 1), (14, 1), (14, 1)),
    str(((128, 512, 7, 7), (1,), "float16")): ((1, 1), (1, 1), (7, 1), (7, 1)),
    #
    str(((128, 64, 112, 112), (64,), "float32")): ((1, 1), (1, 1), (112, 1), (112, 1)),
    str(((128, 64, 56, 56), (64,), "float32")): ((1, 1), (1, 1), (56, 1), (56, 1)),
    str(((128, 128, 56, 56), (128,), "float32")): ((1, 1), (1, 1), (56, 1), (56, 1)),
    str(((128, 128, 28, 28), (128,), "float32")): ((1, 1), (1, 1), (28, 1), (28, 1)),
    str(((128, 256, 28, 28), (256,), "float32")): ((1, 1), (1, 1), (28, 1), (28, 1)),
    str(((128, 256, 14, 14), (256,), "float32")): ((1, 1), (1, 1), (14, 1), (14, 1)),
    str(((128, 512, 14, 14), (512,), "float32")): ((1, 1), (1, 1), (14, 1), (14, 1)),
    str(((128, 512, 7, 7), (512,), "float32")): ((1, 1), (1, 1), (7, 1), (7, 1)),
    #
    str(((128, 64, 112, 112), (1,), "float32")): ((1, 1), (1, 1), (112, 1), (112, 1)),
    str(((128, 64, 56, 56), (1,), "float32")): ((1, 1), (1, 1), (56, 1), (56, 1)),
    str(((128, 128, 56, 56), (1,), "float32")): ((1, 1), (1, 1), (56, 1), (56, 1)),
    str(((128, 128, 28, 28), (1,), "float32")): ((1, 1), (1, 1), (28, 1), (28, 1)),
    str(((128, 256, 28, 28), (1,), "float32")): ((1, 1), (1, 1), (28, 1), (28, 1)),
    str(((128, 256, 14, 14), (1,), "float32")): ((1, 1), (1, 1), (14, 1), (14, 1)),
    str(((128, 512, 14, 14), (1,), "float32")): ((1, 1), (1, 1), (14, 1), (14, 1)),
    str(((128, 512, 7, 7), (1,), "float32")): ((1, 1), (1, 1), (7, 1), (7, 1)),
}


def add_set_dim_func(dy, A, w):
    shape1 = [x.value for x in dy.shape]
    shape2 = [x.value for x in w.shape]
    hash_key = gen_set_dim_key(dy, shape1, shape2)
    return [ct_util.set_dims_by_key(hash_key, add_set_dim_map), hash_key]


def gen_set_dim_key(dy, shape1, shape2):
    key = str((tuple(shape1), tuple(shape2), dy.dtype))
    return key


@ct_util.reg_set_dim_func(add_set_dim_func)
def prelu_grad(dy, A, w):
    """
    brief Computes backgrad prelu value of a tensor.

    \f[
    dw = sum(dy * \\partial(prelu(A)) / \\partial w)
    dA = A > 0 ? dy : dy * w
    \f]

    param inputs akg.tvm.Tensor of type float16, float32

    return akg.tvm.Tensor of same type and shape as inputs
    """
    shape = [x.value for x in dy.shape]
    dtype = dy.dtype
    shape1 = [x.value for x in A.shape]
    dtype1 = A.dtype
    shape2 = [x.value for x in w.shape]
    dtype2 = w.dtype
    assert len(shape) == 4, "only support 4-dim pooling"  # NCHW
    assert len(shape1) == 4, "only support 4-dim pooling"  # NCHW
    assert len(shape2) == 1, "only support 1-dim a"
    assert (shape2[0] == shape1[1] or shape2[0] == 1), "there is only two values are legitimate: 1, or the number of channels at input. Default: 1"
    assert (shape[0] == shape1[0] and shape[1] == shape1[1] and shape[2] == shape1[2] and shape[3] == shape1[3]), "dim number must be equal"

    check_list = ["float16", "float32"]
    if not (dtype1.lower() in check_list and dtype2.lower() in check_list and dtype.lower() in check_list):
        raise RuntimeError("tile_cce only support %s while dtype is %s and %s and %s" % (",".join(check_list), dtype, dtype1, dtype2))
    vc_util.check_shape(shape)
    vc_util.check_shape(shape1)
    vc_util.check_shape(shape2)

    def grad_dsl():
        w_reshape = akg.topi.reshape(w, (1, shape2[0], 1, 1))
        w_broadcast = akg.topi.broadcast_to(w_reshape, shape1)
        dA = akg.tvm.compute(shape,
                         lambda *i: akg.tvm.if_then_else(
                             A(*i) >= akg.tvm.const(0, dtype),
                             dy(*i), dy(*i) * w_broadcast(*i)
                         ))

        # dy * \partial(prelu(A)) / \partial w
        dw_intermediate = akg.tvm.compute(shape,
                                      lambda *i: akg.tvm.if_then_else(
                                          A(*i) >= akg.tvm.const(0, dtype),
                                          akg.tvm.const(0, dtype), dy(*i) * A(*i)
                                      ))

        # hybrid accuracy: sum use float32, other use fp16
        # if dtype.lower() is not "float32":
        #     dw_intermediate = akg.topi.cast(dw_intermediate, "float32")

        if shape2[0] == 1:
            # all channel share one w
            #dw = akg.topi.sum(dw_intermediate)
            dw = akg.topi.sum(dw_intermediate, axis=3)
            dw = akg.topi.sum(dw, axis=2)
            dw = akg.topi.sum(dw, axis=1)
            dw = akg.topi.sum(dw, axis=0)
            # dw = akg.topi.sum(dw_intermediate, axis=1)
            # dw = akg.topi.sum(dw, axis=2)
            # dw = akg.topi.sum(dw, axis=1)
            # dw = akg.topi.sum(dw, axis=0)
            #dw = akg.tvm.compute(shape, lambda *indice: akg.tvm.sum(dw_intermediate(*indice), axis=[0,1,2,3]), name="dw")
            #dw = akg.lang.cce.sum(dw_intermediate, axis=3, keepdims=False)
            #dw = akg.lang.cce.sum(dw_intermediate, axis=2, keepdims=False)
            #dw = akg.lang.cce.sum(dw_intermediate, axis=1, keepdims=False)
            #dw = akg.lang.cce.sum(dw_intermediate, axis=0, keepdims=False)
        else:
            # all channel use separate w
            # dw = akg.topi.sum(dw_intermediate, axis=[0,2,3]) # Accuracy is not up to standard
            dw = akg.topi.sum(dw_intermediate, axis=3)
            dw = akg.topi.sum(dw, axis=2)
            dw = akg.topi.sum(dw, axis=0)
            # dw = akg.topi.sum(dw_intermediate, axis=1)
            # dw = akg.topi.sum(dw, axis=1)
            # dw = akg.topi.sum(dw, axis=0)

        # hybrid accuracy: sum use float32, other use fp16
        # if dtype.lower() is not "float32":
        #     dw = akg.topi.cast(dw, "float16")

        return dA, dw

    attrs = {"pragma_checkcoincident": 0, "pragma_modshift": 1}
    return grad_dsl(), attrs
