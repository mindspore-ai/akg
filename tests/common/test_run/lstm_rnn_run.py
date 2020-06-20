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

import numpy as np
from tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from test_op import lstm_rnn
from gen_random import random_gaussian

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def lstmcell_run(shape, dtype, kernel_name="lstmcell", attrs={}):
    # shape: batch_size, input_size, hidden_size
    batch_size = shape[0]
    input_size = shape[1]
    hidden_size = shape[2]
    W_ih_shape = (4 * hidden_size, input_size,)
    W_hh_shape = (4 * hidden_size, hidden_size,)
    b_shape = (4 * hidden_size,)
    c_prev_shape = (batch_size, hidden_size,)
    h_prev_shape = (batch_size, hidden_size,)
    x_shape = (batch_size, input_size,)

    print("lstmcell - ")
    op_attrs = []

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(lstm_rnn.lstmcell,
                                  [x_shape, h_prev_shape, c_prev_shape, W_ih_shape, W_hh_shape, b_shape, b_shape, ],
                                  [dtype, dtype, dtype, dtype, dtype, dtype, dtype],
                                  op_attrs=op_attrs, kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            W_hh, W_ih, b_hh, b_ih, c, c_prev, expect_c, expect_h, h, h_prev, x = gen_lstmcell_data(W_hh_shape,
                                                                                                    W_ih_shape,
                                                                                                    b_shape,
                                                                                                    c_prev_shape,
                                                                                                    dtype, h_prev_shape,
                                                                                                    hidden_size,
                                                                                                    x_shape)
            return mod, (expect_c, expect_h), {"args": (x, h_prev, c_prev, W_ih, W_hh, b_ih, b_hh, h, c), 'outputs': (-2, -1), 'tuning': False}
        else:
            return mod
    else:
        mod = utils.op_build_test(lstm_rnn.lstmcell,
                                  [x_shape, h_prev_shape, c_prev_shape, W_ih_shape, W_hh_shape, b_shape, b_shape, ],
                                  [dtype, dtype, dtype, dtype, dtype, dtype, dtype],
                                  op_attrs=op_attrs, kernel_name=kernel_name, attrs=attrs)
        W_hh, W_ih, b_hh, b_ih, c, c_prev, expect_c, expect_h, h, h_prev, x = gen_lstmcell_data(W_hh_shape, W_ih_shape,
                                                                                                b_shape, c_prev_shape,
                                                                                                dtype, h_prev_shape,
                                                                                                hidden_size, x_shape)
        h, c = utils.mod_launch(mod, (x, h_prev, c_prev, W_ih, W_hh, b_ih, b_hh, h, c), outputs=(-2, -1),
                                expect=(expect_h, expect_c))

        assert_res = True
        assert_res &= compare_tensor(h, expect_h, rtol=5e-02, atol=5e-02, equal_nan=True)
        print("act_h_output = ", h)
        print("expect_h_output = ", expect_h)
        print("LSTM_cell assert_res_h = ", assert_res)
        assert_res &= compare_tensor(c, expect_c, rtol=5e-02, atol=5e-02, equal_nan=True)
        print("act_c_output = ", c)
        print("expect_c_output = ", expect_c)
        print("LSTM_cell assert_res = ", assert_res)
        # input("Press ENTER...")

        return (x, h_prev, c_prev, W_ih, W_hh, b_ih, b_hh), (h, c), (expect_h, expect_c), assert_res


def gen_lstmcell_data(W_hh_shape, W_ih_shape, b_shape, c_prev_shape, dtype, h_prev_shape, hidden_size, x_shape):
    W_ih = random_gaussian(W_ih_shape, miu=0.1, sigma=0.1).astype(dtype)
    W_hh = random_gaussian(W_hh_shape, miu=0.1, sigma=0.1).astype(dtype)
    b_ih = random_gaussian(b_shape, miu=0.1, sigma=0.1).astype(dtype)
    b_hh = random_gaussian(b_shape, miu=0.1, sigma=0.1).astype(dtype)
    c_prev = random_gaussian(c_prev_shape, miu=0.1, sigma=0.1).astype(dtype)
    h_prev = random_gaussian(h_prev_shape, miu=0.1, sigma=0.1).astype(dtype)
    x = random_gaussian(x_shape, miu=0.1, sigma=0.1).astype(dtype)
    np_hx = np.concatenate((x, h_prev), axis=1)
    np_wx = np.concatenate((W_ih, W_hh), axis=1)
    np_t = np.dot(np_hx, np_wx.transpose(1, 0)) + b_ih + b_hh
    np_i = sigmoid(np_t[:, 0: hidden_size])
    np_f = sigmoid(np_t[:, hidden_size: 2 * hidden_size])
    np_c_ = np.tanh(np_t[:, 2 * hidden_size: 3 * hidden_size])
    np_o = sigmoid(np_t[:, 3 * hidden_size: 4 * hidden_size])
    expect_c = np.add(np.multiply(np_f, c_prev), np.multiply(np_i, np_c_))
    expect_h = np.multiply(np_o, np.tanh(expect_c))
    h = np.full(h_prev.shape, np.nan, dtype)
    c = np.full(c_prev.shape, np.nan, dtype)
    return W_hh, W_ih, b_hh, b_ih, c, c_prev, expect_c, expect_h, h, h_prev, x


def rnn_tanh_cell_run(shape, dtype, kernel_name="rnn_tanh_cell", attrs={}):
    # shape: batch_size, input_size, hidden_size
    batch_size = shape[0]
    input_size = shape[1]
    hidden_size = shape[2]
    W_ih_shape = (hidden_size, input_size,)
    W_hh_shape = (hidden_size, hidden_size,)
    b_shape = (hidden_size,)
    h_prev_shape = (batch_size, hidden_size,)
    x_shape = (batch_size, input_size,)

    print("rnn_tanh_cell - ")
    op_attrs = []

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(lstm_rnn.rnn_tanh_cell,
                                  [x_shape, h_prev_shape, W_ih_shape, W_hh_shape, b_shape, b_shape, ],
                                  [dtype, dtype, dtype, dtype, dtype, dtype],
                                  op_attrs=op_attrs, kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            W_hh, W_ih, b_hh, b_ih, expect, h, h_prev, x = gen_rnn_tanh_cell_data(W_hh_shape, W_ih_shape, b_shape,
                                                                                  dtype, h_prev_shape, x_shape)
            return mod, expect, (x, h_prev, W_ih, W_hh, b_ih, b_hh, h)
        else:
            return mod
    else:
        mod = utils.op_build_test(lstm_rnn.rnn_tanh_cell,
                                  [x_shape, h_prev_shape, W_ih_shape, W_hh_shape, b_shape, b_shape, ],
                                  [dtype, dtype, dtype, dtype, dtype, dtype],
                                  op_attrs=op_attrs, kernel_name=kernel_name, attrs=attrs)
        W_hh, W_ih, b_hh, b_ih, expect, h, h_prev, x = gen_rnn_tanh_cell_data(W_hh_shape, W_ih_shape, b_shape, dtype,
                                                                              h_prev_shape, x_shape)
        h = utils.mod_launch(mod, (x, h_prev, W_ih, W_hh, b_ih, b_hh, h))
        assert_res = compare_tensor(h, expect, rtol=5e-02, atol=5e-02, equal_nan=True)
        # print("act_h_output = ", h)
        # print("expect_h_output = ", expect)
        print("RNN_tanh_cell assert_res = ", assert_res)

        return (x, h_prev, W_ih, W_hh, b_ih, b_hh), h, expect, assert_res


def gen_rnn_tanh_cell_data(W_hh_shape, W_ih_shape, b_shape, dtype, h_prev_shape, x_shape):
    x = random_gaussian(x_shape, miu=0.1, sigma=0.1).astype(dtype)
    h_prev = random_gaussian(h_prev_shape, miu=0.1, sigma=0.1).astype(dtype)
    W_ih = random_gaussian(W_ih_shape, miu=0.1, sigma=0.1).astype(dtype)
    W_hh = random_gaussian(W_hh_shape, miu=0.1, sigma=0.1).astype(dtype)
    b_ih = random_gaussian(b_shape, miu=0.1, sigma=0.1).astype(dtype)
    b_hh = random_gaussian(b_shape, miu=0.1, sigma=0.1).astype(dtype)
    np_igates = np.dot(x, W_ih.transpose(1, 0)) + b_ih
    np_hgates = np.dot(h_prev, W_hh.transpose(1, 0)) + b_hh
    expect = np.tanh(np_igates + np_hgates)
    h = np.full(h_prev.shape, np.nan, dtype)
    return W_hh, W_ih, b_hh, b_ih, expect, h, h_prev, x


def rnn_relu_cell_run(shape, dtype, kernel_name="rnn_relu_cell", attrs={}):
    # shape: batch_size, input_size, hidden_size
    batch_size = shape[0]
    input_size = shape[1]
    hidden_size = shape[2]
    W_ih_shape = (hidden_size, input_size,)
    W_hh_shape = (hidden_size, hidden_size,)
    b_shape = (hidden_size,)
    h_prev_shape = (batch_size, hidden_size,)
    x_shape = (batch_size, input_size,)

    print("rnn_relu6_cell - ")
    op_attrs = []

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(lstm_rnn.rnn_relu_cell,
                                  [x_shape, h_prev_shape, W_ih_shape, W_hh_shape, b_shape, b_shape, ],
                                  [dtype, dtype, dtype, dtype, dtype, dtype],
                                  op_attrs=op_attrs, kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            W_hh, W_ih, b_hh, b_ih, expect, h, h_prev, x = gen_rnn_relu_cell_data(W_hh_shape, W_ih_shape, b_shape,
                                                                                  dtype, h_prev_shape, x_shape)
            return mod, expect, (x, h_prev, W_ih, W_hh, b_ih, b_hh, h)
        else:
            return mod
    else:
        mod = utils.op_build_test(lstm_rnn.rnn_relu_cell,
                                  [x_shape, h_prev_shape, W_ih_shape, W_hh_shape, b_shape, b_shape, ],
                                  [dtype, dtype, dtype, dtype, dtype, dtype],
                                  op_attrs=op_attrs, kernel_name=kernel_name, attrs=attrs)
        W_hh, W_ih, b_hh, b_ih, expect, h, h_prev, x = gen_rnn_relu_cell_data(W_hh_shape, W_ih_shape, b_shape, dtype,
                                                                              h_prev_shape, x_shape)
        h = utils.mod_launch(mod, (x, h_prev, W_ih, W_hh, b_ih, b_hh, h))
        assert_res = compare_tensor(h, expect, rtol=5e-02, atol=5e-02, equal_nan=True)
        # print("act_h_output = ", h)
        # print("expect_h_output = ", expect)
        print("RNN_relu6_cell assert_res = ", assert_res)

        return (x, h_prev, W_ih, W_hh, b_ih, b_hh), h, expect, assert_res


def gen_rnn_relu_cell_data(W_hh_shape, W_ih_shape, b_shape, dtype, h_prev_shape, x_shape):
    x = random_gaussian(x_shape, miu=0.1, sigma=0.1).astype(dtype)
    h_prev = random_gaussian(h_prev_shape, miu=0.1, sigma=0.1).astype(dtype)
    W_ih = random_gaussian(W_ih_shape, miu=0.1, sigma=0.1).astype(dtype)
    W_hh = random_gaussian(W_hh_shape, miu=0.1, sigma=0.1).astype(dtype)
    b_ih = random_gaussian(b_shape, miu=0.1, sigma=0.1).astype(dtype)
    b_hh = random_gaussian(b_shape, miu=0.1, sigma=0.1).astype(dtype)
    np_igates = np.dot(x, W_ih.transpose(1, 0)) + b_ih
    np_hgates = np.dot(h_prev, W_hh.transpose(1, 0)) + b_hh
    np_t = np_igates + np_hgates
    # relu6
    zero = np.full(np_t.shape, 0, dtype)
    six = np.full(np_t.shape, 6, dtype)
    max = np.maximum(np_t, zero)
    expect = np.minimum(max, six)
    h = np.full(h_prev.shape, np.nan, dtype)
    return W_hh, W_ih, b_hh, b_ih, expect, h, h_prev, x
