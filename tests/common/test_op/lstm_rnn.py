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

"""lstm_rnn"""

from akg.ops.math.tanh import tanh
from test_op.relu6 import relu6
from test_op.sigmoid import sigmoid
from test_op.dense import dense
from test_op.split import split


def lstmcell(inputs, hx, cx, w_ih, w_hh, b_ih, b_hh, use_bias=True):
    """
    Computes the hidden and state variables of a Long Short Term Memory (lstm) cell.

    Args:
        input:  akg.tvm.Tensor of type float16, float32 with shape [batch, input_size].
        hx:     akg.tvm.Tensor for hidden variable from previous cell with shape [batch, hidden_size].
        cx:     akg.tvm.Tensor for state variable from previous cell with shape [batch, hidden_size].
        w_ih:   akg.tvm.Tensor for input weights with shape [4*hidden_size, input_size].
        w_hh:   akg.tvm.Tensor for hidden weights with shape [4*hidden_size, hidden_size].
        b_ih:   akg.tvm.Tensor for input bias with shape [4*hidden_size].
        b_hh:   akg.tvm.Tensor for hidden bias with shape [4*hidden_size].

    Returns:
        hy:     akg.tvm.Tensor for hidden variable of current cell. 
        cy:     akg.tvm.Tensor for state variable of current cell.
    """
    w_i_ih, w_f_ih, w_c_ih, w_o_ih = split(w_ih, 4, 0)
    b_i_ih, b_f_ih, b_c_ih, b_o_ih = split(b_ih, 4)
    w_i_hh, w_f_hh, w_c_hh, w_o_hh = split(w_hh, 4, 0)
    b_i_hh, b_f_hh, b_c_hh, b_o_hh = split(b_hh, 4)

    # gates:[batch, 4*hidden_size] ih*wh+bias
    # ingate, forgetgate, cellgate, outgate = split(gates, 4, 1)
    i = dense(inputs, w_i_ih, b_i_ih, use_bias) + dense(hx, w_i_hh, b_i_hh, use_bias)
    f = dense(inputs, w_f_ih, b_f_ih, use_bias) + dense(hx, w_f_hh, b_f_hh, use_bias)
    c = dense(inputs, w_c_ih, b_c_ih, use_bias) + dense(hx, w_c_hh, b_c_hh, use_bias)
    o = dense(inputs, w_o_ih, b_o_ih, use_bias) + dense(hx, w_o_hh, b_o_hh, use_bias)

    cy = (sigmoid(f) * cx) + (sigmoid(i) * tanh(c))
    hy = sigmoid(o) * tanh(cy)

    return hy, cy

def rnn_tanh_cell(inputs, hidden, w_ih, w_hh, b_ih, b_hh, use_bias=True):
    """
    RNN cell with tanh non-linearity.

    Args:
        inputs:  akg.tvm.Tensor of type float16, float32.
        hidden:  akg.tvm.Tensor for hidden variable from previous cell.
        w_ih:   akg.tvm.Tensor for input weights.
        w_hh:   akg.tvm.Tensor for hidden weights.
        b_ih:   akg.tvm.Tensor for input bias.
        b_hh:   akg.tvm.Tensor for hidden bias.

    Returns:
        h:      akg.tvm.Tensor for hidden output variable of current cell.
    """ 
    igates = dense(inputs, w_ih, b_ih, use_bias)
    hgates = dense(hidden, w_hh, b_hh, use_bias)
    h = tanh(igates + hgates)
    return h

def rnn_relu_cell(inputs, hidden, w_ih, w_hh, b_ih, b_hh, use_bias=True):
    """
    RNN cell with relu non-linearity.

    Args:
        inputs:  akg.tvm.Tensor of type float16, float32.
        hidden:  akg.tvm.Tensor for hidden variable from previous cell.
        w_ih:   akg.tvm.Tensor for input weights.
        w_hh:   akg.tvm.Tensor for hidden weights.
        b_ih:   akg.tvm.Tensor for input bias.
        b_hh:   akg.tvm.Tensor for hidden bias.

    Returns:
        h:      akg.tvm.Tensor for hidden output variable of current cell.
    """
    igates = dense(inputs, w_ih, b_ih, use_bias)
    hgates = dense(hidden, w_hh, b_hh, use_bias)
    h = relu6(igates + hgates)
    return h
