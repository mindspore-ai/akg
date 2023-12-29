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

"""lstm_rnn_gard"""
import akg.topi
import akg.tvm
from akg.utils.format_transform import get_shape
from akg.ops.math.ascend import Tanh
from tests.common.test_op.ascend.dense import dense
from akg.ops.array.ascend import Concat
from akg.ops.array.ascend import Split
from tests.common.test_op.ascend.sigmoid import sigmoid


def lstmcell_grad_h(input, hx, cx, w_ih, w_hh, b_ih, b_hh, dh, dc, target="cce"):
    """
    Computes dh w.r.t. dw, db, dcx, dhx, dx.

    Args:
        input: akg.tvm.Tensor of type float16, float32.
        hx:    akg.tvm.Tensor for hidden variable from previous cell.
        cx:    akg.tvm.Tensor for state variable from previous cell.
        w_ih:  akg.tvm.Tensor for input weights.
        w_hh:  akg.tvm.Tensor for hidden weights.
        b_ih:  akg.tvm.Tensor for input bias. 
        b_hh:  akg.tvm.Tensor for hidden bias.

    Returns:
        dw_ih:    akg.tvm.Tensor for dh/dw_ih.
        dw_hh:    akg.tvm.Tensor for dh/dw_hh.
        db_ih:    akg.tvm.Tensor for dh/db_ih.
        db_hh:    akg.tvm.Tensor for dh/db_hh.
        dcx:      akg.tvm.Tensor for dh/dcx.
        dhx:      akg.tvm.Tensor for dh/dhx.
        dx:       akg.tvm.Tensor for dh/dx.
    """
    # things from fwd
    batch, input_size = get_shape(input)
    _, hidden_size = get_shape(hx)
    xh = akg.topi.concatenate((hx, input), 1)
    whl = [w_ih, w_hh]
    W = Concat(whl, 1)  # [4*hidden_size, input_size+hidden_size]

    gates = dense(input, w_ih, b_ih, True) + dense(hx, w_hh, b_hh, True)

    ingate_in, forgetgate_in, cellgate_in, outgate_in = Split(gates, 4, 1)

    ingate = sigmoid(ingate_in)
    forgetgate = sigmoid(forgetgate_in)
    cellgate = Tanh(cellgate_in)
    outgate = sigmoid(outgate_in)
    cy = (forgetgate * cx) + (ingate * cellgate)
    tanh_cy = Tanh(cy)
    #hy = outgate * tanh_cy

    # starts bwd
    # head * dh/do shape [n,]
    doutgate = dh * tanh_cy
    doutgate_in = outgate * (1 - outgate) * doutgate
    kk = akg.tvm.reduce_axis((0, batch))
    dWo = akg.tvm.compute((hidden_size, hidden_size + input_size), lambda i, j:
                      akg.tvm.sum(xh[kk, j] * doutgate_in(kk, i), axis=kk), name="dWo")

    dtanh_cy = dh * outgate
    dc = (1 - tanh_cy * tanh_cy) * dtanh_cy

    dingate = cellgate * dc
    dingate_in = ingate * (1 - ingate) * dingate
    kk3 = akg.tvm.reduce_axis((0, batch))
    dWi = akg.tvm.compute((hidden_size, hidden_size + input_size), lambda i, j:
                      akg.tvm.sum(xh[kk3, j] * dingate_in(kk3, i), axis=kk3), name="dWi")

    dforgetgate = dc * cx
    dforgetgate_in = forgetgate * (1 - forgetgate) * dforgetgate
    kk2 = akg.tvm.reduce_axis((0, batch))
    dWf = akg.tvm.compute((hidden_size, hidden_size + input_size), lambda i, j:
                      akg.tvm.sum(xh[kk2, j] * dforgetgate_in(kk2, i), axis=kk2), name="dWf")

    dcellgate = ingate * dc
    dcellgate_in = (1 - cellgate * cellgate) * dcellgate
    kk4 = akg.tvm.reduce_axis((0, batch))
    dWc = akg.tvm.compute((hidden_size, hidden_size + input_size), lambda i, j:
                      akg.tvm.sum(xh[kk4, j] * dcellgate_in(kk4, i), axis=kk4), name="dWc")

    dW = akg.topi.concatenate((dWi, dWf, dWc, dWo))

    db = akg.topi.concatenate((dingate_in, dforgetgate_in, dcellgate_in, doutgate_in), 1)

    kk5 = akg.tvm.reduce_axis((0, 4 * hidden_size))
    dxh = akg.tvm.compute((batch, hidden_size + input_size), lambda i, j:
                      akg.tvm.sum(W[kk5, j] * db[i, kk5], axis=kk5), name="dxh")
    dhx = akg.tvm.compute((batch, hidden_size), lambda i, j: dxh[i, j], name="dhx")
    dx = akg.tvm.compute((batch, input_size), lambda i, j: dxh[i, j + hidden_size], name="dx")

    dcx = forgetgate * dc

    dw_ih = akg.tvm.compute(w_ih.shape, lambda i, j: dW[i, j])
    #dw_hh = akg.tvm.compute(w_hh.shape, lambda i, j: dW[i, j + input_size])

    bhr = akg.tvm.reduce_axis((0, batch))

    db_ih = akg.tvm.compute((4 * hidden_size,), lambda i: akg.tvm.sum(db[i, bhr], axis=bhr), name="dbih")

    bir = akg.tvm.reduce_axis((0, batch))

    db_hh = akg.tvm.compute((4 * hidden_size,), lambda i: akg.tvm.sum(db[i, bir], axis=bir), name="dbhh")

    return dw_ih, w_hh, db_ih, db_hh, dcx, dhx, dx


def lstmcell_grad_c(input, hx, cx, w_ih, w_hh, b_ih, b_hh, dc, target="cce"):
    """
    Computes dc w.r.t. dw, db, dcx, dhx, dx.

    Args:
        input: akg.tvm.Tensor of type float16, float32.
        hx:    akg.tvm.Tensor for hidden variable from previous cell.
        cx:    akg.tvm.Tensor for state variable from previous cell.
        w_ih:  akg.tvm.Tensor for input weights.
        w_hh:  akg.tvm.Tensor for hidden weights.
        b_ih:  akg.tvm.Tensor for input bias. 
        b_hh:  akg.tvm.Tensor for hidden bias.

    Returns:
        dw_ih:    akg.tvm.Tensor for dc/dw_ih.
        dw_hh:    akg.tvm.Tensor for dc/dw_hh.
        db_ih:    akg.tvm.Tensor for dc/db_ih.
        db_hh:    akg.tvm.Tensor for dc/db_hh.
        dcx:      akg.tvm.Tensor for dc/dcx.
        dhx:      akg.tvm.Tensor for dc/dhx.
        dx:       akg.tvm.Tensor for dc/dx.
    """
    # things from fwd
    whl = [w_ih, w_hh]
    W = Concat(whl, 1)  # [4*hidden_size, input_size+hidden_size]
    b = b_ih + b_hh

    batch, input_size = get_shape(input)
    _, hidden_size = get_shape(hx)
    xh = akg.topi.concatenate((hx, input), 1)
    t = akg.topi.nn.dense(xh, W, b)
    temp_i = akg.tvm.compute((batch, hidden_size), lambda i, j: t(i, j), name="temp_i")
    i = sigmoid(temp_i)
    temp_f = akg.tvm.compute((batch, hidden_size), lambda i, j: t(i, j + hidden_size), name="temp_f")
    f = sigmoid(temp_f)
    temp_c_ = akg.tvm.compute((batch, hidden_size), lambda i, j: t(i, j + 2 * hidden_size), name="temp_c")
    c_ = Tanh(temp_c_)

    # starts bwd
    # head * dh/do shape [n,]
    dtemp_o = akg.tvm.compute((batch, hidden_size), lambda *i: 0)
    dWo = akg.tvm.compute((hidden_size, hidden_size + input_size), lambda i, j: 0, name="dWo")

    df = dc * cx
    dtemp_f = f * (1 - f) * df
    kk2 = akg.tvm.reduce_axis((0, batch))
    dWf = akg.tvm.compute((hidden_size, hidden_size + input_size), lambda i, j:
                      akg.tvm.sum(xh[kk2, j] * dtemp_f(kk2, i), axis=kk2), name="dWf")

    di = c_ * dc
    dtemp_i = i * (1 - i) * di
    kk3 = akg.tvm.reduce_axis((0, batch))
    dWi = akg.tvm.compute((hidden_size, hidden_size + input_size), lambda i, j:
                      akg.tvm.sum(xh[kk3, j] * dtemp_i(kk3, i), axis=kk3), name="dWi")

    dc_ = i * dc
    dtemp_c_ = (1 - c_ * c_) * dc_
    kk4 = akg.tvm.reduce_axis((0, batch))
    dWc = akg.tvm.compute((hidden_size, hidden_size + input_size), lambda i, j:
                      akg.tvm.sum(xh[kk4, j] * dtemp_c_(kk4, i), axis=kk4), name="dWc")

    dW = akg.topi.concatenate((dWi, dWf, dWc, dWo))

    db = akg.topi.concatenate((dtemp_i, dtemp_f, dtemp_c_, dtemp_o), 1)

    kk5 = akg.tvm.reduce_axis((0, 4 * hidden_size))
    dxh = akg.tvm.compute((batch, hidden_size + input_size), lambda i, j:
                      akg.tvm.sum(W[kk5, j] * db[i, kk5], axis=kk5), name="dxh")
    dhx = akg.tvm.compute((batch, hidden_size), lambda i, j: dxh[i, j], name="dhx")
    dx = akg.tvm.compute((batch, input_size), lambda i, j: dxh[i, j + hidden_size], name="dx")

    dcx = f * dc

    dw_ih = akg.tvm.compute(w_ih.shape, lambda i, j: dW[i, j])
    #dw_hh = akg.tvm.compute(w_hh.shape, lambda i, j: dW[i, j + input_size])

    bhr = akg.tvm.reduce_axis((0, batch))

    db_ih = akg.tvm.compute((4 * hidden_size,), lambda i: akg.tvm.sum(db[i, bhr], axis=bhr), name="dbih")

    bir = akg.tvm.reduce_axis((0, batch))

    db_hh = akg.tvm.compute((4 * hidden_size,), lambda i: akg.tvm.sum(db[i, bir], axis=bir), name="dbhh")

    return dw_ih, w_hh, db_ih, db_hh, dcx, dhx, dx


def rnn_tanh_cell_grad(input, hidden, w_ih, w_hh, b_ih, b_hh, grad, target="cce"):
    """
    Computes dgrad w.r.t. dinput (di), dhidden_input (dhid), dweights (dWih, dWhh), dbias (db).

    Args:
        input:  akg.tvm.Tensor of type float16, float32 with shape [batch, input_size].
        hidden: akg.tvm.Tensor for hidden variable from previous cell with shape [batch, hidden_size].
        w_ih:   akg.tvm.Tensor for input weights with shape [hidden_size, input_size].
        w_hh:   akg.tvm.Tensor for hidden weights with shape [hidden_size, hidden_size].
        b_ih:   akg.tvm.Tensor for input bias with shape [hidden_size].
        b_hh:   akg.tvm.Tensor for hidden bias with shape [hidden_size].
        grad:   akg.tvm.Tensor representing dy with shape [batch, hidden_size].

    Returns:
        di:     akg.tvm.Tensor for dy/di.
        dhid:   akg.tvm.Tensor for dy/dhid.
        dWih:   akg.tvm.Tensor for dy/dWih (input weights).
        dWhh:   akg.tvm.Tensor for dy/dWhh (hidden weights).
        db:     akg.tvm.Tensor for dy/db.
    """
    batch, input_size = get_shape(input)
    _, hidden_size = get_shape(hidden)
    igates = akg.topi.nn.dense(input, w_ih, b_ih)
    hgates = akg.topi.nn.dense(hidden, w_hh, b_hh)
    h = Tanh(igates + hgates)

    dh = (1 - h * h) * grad
    kk = akg.tvm.reduce_axis((0, batch))
    dWih = akg.tvm.compute((hidden_size, input_size), lambda i, j:
                       akg.tvm.sum(input[kk, j] * dh(kk, i), axis=kk), name="dWih")
    kk2 = akg.tvm.reduce_axis((0, batch))
    dWhh = akg.tvm.compute((hidden_size, hidden_size), lambda i, j:
                       akg.tvm.sum(hidden[kk2, j] * dh(kk2, i), axis=kk2), name="dWhh")
    kk3 = akg.tvm.reduce_axis((0, hidden_size))
    di = akg.tvm.compute((batch, input_size), lambda i, j: akg.tvm.sum(w_ih[kk3, j] * dh[i, kk3], axis=kk3), name="di")
    kk4 = akg.tvm.reduce_axis((0, hidden_size))
    dhid = akg.tvm.compute((batch, hidden_size), lambda i, j: akg.tvm.sum(w_hh[kk4, j] * dh[i, kk4], axis=kk4), name="dhid")
    db = akg.topi.sum(dh, 0)
    return di, dhid, dWih, dWhh, db
    # dbih/dbhh are the same and returning it twice causes CCEbuild to fail due to some SSA error
    # return di, dhid, dWih, dWhh, db, db


def rnn_relu_cell_grad(input, hidden, w_ih, w_hh, b_ih, b_hh, grad, target="cce"):
    """
    Computes dgrad w.r.t. dinput (di), dhidden_input (dhi), dweights (dWih, dWhh), dbias (db).

    Args:
        input:  akg.tvm.Tensor of type float16, float32 with shape [batch, input_size].
        hidden: akg.tvm.Tensor for hidden variable from previous cell with shape [batch, hidden_size].
        w_ih:   akg.tvm.Tensor for input weights with shape [hidden_size, input_size].
        w_hh:   akg.tvm.Tensor for hidden weights with shape [hidden_size, hidden_size].
        b_ih:   akg.tvm.Tensor for input bias with shape [hidden_size].
        b_hh:   akg.tvm.Tensor for hidden bias with shape [hidden_size].
        grad:   akg.tvm.Tensor representing dy with shape [batch, hidden_size].

    Returns:
        di:     akg.tvm.Tensor for dy/di.
        dhi:    akg.tvm.Tensor for dy/dhi.
        dWih:   akg.tvm.Tensor for dy/dWih (input weights).
        dWhh:   akg.tvm.Tensor for dy/dWhh (hidden weights).
        db:     akg.tvm.Tensor for dy/db.
    """
    batch, input_size = get_shape(input)
    _, hidden_size = get_shape(hidden)
    igates = akg.topi.nn.dense(input, w_ih, b_ih)
    hgates = akg.topi.nn.dense(hidden, w_hh, b_hh)
    h = akg.topi.nn.relu(igates + hgates)

    dh = akg.tvm.compute((batch, hidden_size), lambda *i: grad(*i) * akg.tvm.expr.Select(h(*i) >= 0, 1, 0), name="dh")

    kk = akg.tvm.reduce_axis((0, batch))
    dWih = akg.tvm.compute((hidden_size, input_size), lambda i, j:
                       akg.tvm.sum(input[kk, j] * dh(kk, i), axis=kk), name="dWih")
    kk2 = akg.tvm.reduce_axis((0, batch))
    dWhh = akg.tvm.compute((hidden_size, hidden_size), lambda i, j:
                       akg.tvm.sum(hidden[kk2, j] * dh(kk2, i), axis=kk2), name="dWhh")
    kk3 = akg.tvm.reduce_axis((0, hidden_size))
    di = akg.tvm.compute((batch, input_size), lambda i, j:
                     akg.tvm.sum(w_ih[kk3, j] * dh[i, kk3], axis=kk3), name="di")
    kk4 = akg.tvm.reduce_axis((0, hidden_size))
    dhi = akg.tvm.compute((batch, hidden_size), lambda i, j:
                      akg.tvm.sum(w_hh[kk4, j] * dh[i, kk4], axis=kk4), name="dhi")
    db = akg.topi.sum(dh, 0)
    return di, dhi, dWih, dWhh, db
    # dbih/dbhh are the same and returning it twice causes CCEbuild to fail due to some SSA error
    # return di, dhi, dWih, dWhh, db, db
