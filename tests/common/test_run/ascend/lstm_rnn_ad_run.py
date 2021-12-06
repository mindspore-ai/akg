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

from akg.utils import kernel_exec as utils
from tests.common.tensorio import compare_tensor
import numpy as np
from tests.common.test_op.ascend.lstm_rnn_grad import lstmcell_grad_h
from tests.common.test_op.ascend.lstm_rnn_grad import lstmcell_grad_c, rnn_tanh_cell_grad, rnn_relu_cell_grad
from tests.common.test_op.ascend.lstm_rnn_ad import rnncell_tanh_ad, rnncell_relu_ad, lstmcell_h_ad, lstmcell_c_ad
from tests.common.gen_random import random_gaussian

def np_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def init_lstmcell_shapes(shape):
    # shape: batch_size, input_size, hidden_size
    batch_size, input_size, hidden_size = shape
    W_ih_shape = (4 * hidden_size, input_size,)
    W_hh_shape = (4 * hidden_size, hidden_size,)
    b_shape = (4 * hidden_size,)
    c_prev_shape = (batch_size, hidden_size,)
    h_prev_shape = (batch_size, hidden_size,)
    x_shape = (batch_size, input_size,)

    gradh_shape = (batch_size, hidden_size,)
    gradc_shape = (batch_size, hidden_size,)

    return x_shape, c_prev_shape, h_prev_shape, W_ih_shape, W_hh_shape, b_shape, b_shape, gradh_shape, gradc_shape


def init_lstmcell_data(shapes, dtype):

    x_shape, c_prev_shape, h_prev_shape, W_ih_shape, W_hh_shape, b_ih_shape, b_hh_shape, gradh_shape, gradc_shape = shapes
    batch_size, input_size = x_shape
    _, hidden_size = c_prev_shape

    np_miu = 1.0 / hidden_size
    np_sigma = np_miu / 4.0
    W_ih = random_gaussian(W_ih_shape, miu=np_miu, sigma=np_sigma).astype(dtype)
    W_hh = random_gaussian(W_hh_shape, miu=np_miu, sigma=np_sigma).astype(dtype)
    b_ih = random_gaussian(b_ih_shape, miu=np_miu, sigma=np_sigma).astype(dtype)
    b_hh = random_gaussian(b_hh_shape, miu=np_miu, sigma=np_sigma).astype(dtype)
    c_prev = random_gaussian(c_prev_shape, miu=np_miu, sigma=np_sigma).astype(dtype)
    h_prev = random_gaussian(h_prev_shape, miu=np_miu, sigma=np_sigma).astype(dtype)
    x = random_gaussian(x_shape, miu=np_miu, sigma=np_sigma).astype(dtype)

    gradh = random_gaussian(gradh_shape, miu=5 * np_miu, sigma=5 * np_sigma).astype(dtype)
    gradc = random_gaussian(gradc_shape, miu=5 * np_miu, sigma=5 * np_sigma).astype(dtype)

    dW_ih = np.full(W_ih.shape, np.nan, dtype)
    dW_hh = np.full(W_hh.shape, np.nan, dtype)
    db_ih = np.full(b_ih.shape, np.nan, dtype)
    db_hh = np.full(b_hh.shape, np.nan, dtype)
    dc_prev = np.full(c_prev.shape, np.nan, dtype)
    dh_prev = np.full(h_prev.shape, np.nan, dtype)
    dx = np.full(x.shape, np.nan, dtype)
    return x, c_prev, h_prev, W_ih, W_hh, b_ih, b_hh, gradh, gradc, dW_ih, dW_hh, db_ih, db_hh, dc_prev, dh_prev, dx


def lstm_backward_data_h(np_input, hx, cx, w_ih, w_hh, b_ih, b_hh, gradh):
    np_igates = np.dot(np_input, w_ih.transpose(1, 0)) + b_ih
    np_hgates = np.dot(hx, w_hh.transpose(1, 0)) + b_hh
    np_gates = np_igates + np_hgates

    np_ingate, np_forgetgate, np_cellgate, np_outgate = np.split(np_gates, 4, axis=1)
    np_w_ih_i, np_w_ih_f, np_w_ih_c, np_w_ih_o = np.split(w_ih, 4, axis=0)
    np_b_ih_i, np_b_ih_f, np_b_ih_c, np_b_ih_o = np.split(b_ih, 4, axis=0)
    np_w_hh_i, np_w_hh_f, np_w_hh_c, np_w_hh_o = np.split(w_hh, 4, axis=0)
    np_b_hh_i, np_b_hh_f, np_b_hh_c, np_b_hh_o = np.split(b_hh, 4, axis=0)

    np_sigm_ingate = np_sigmoid(np_ingate)
    np_sigm_forgetgate = np_sigmoid(np_forgetgate)
    np_tanh_cellgate = np.tanh(np_cellgate)
    np_sigm_outgate = np_sigmoid(np_outgate)

    np_c_out = np_sigm_forgetgate * cx + np_sigm_ingate * np_tanh_cellgate
    np_h_out = np_sigm_outgate * np.tanh(np_c_out)

    expect_dsigm_outgate = gradh * np.tanh(np_c_out)
    expect_dc_out = np_sigm_outgate * gradh * (1.0 - np.tanh(np_c_out) * np.tanh(np_c_out))

    expect_dsigm_forgetgate = expect_dc_out * cx
    expect_dcx = np_sigm_forgetgate * expect_dc_out
    expect_dsigm_ingate = expect_dc_out * np_tanh_cellgate
    expect_dtanh_cellgate = np_sigm_ingate * expect_dc_out

    expect_dingate = expect_dsigm_ingate * np_sigm_ingate * (1 - np_sigm_ingate)
    expect_dforgetgate = expect_dsigm_forgetgate * np_sigm_forgetgate * (1 - np_sigm_forgetgate)
    expect_dcellgate = expect_dtanh_cellgate * (1 - np_tanh_cellgate * np_tanh_cellgate)
    expect_doutgate = expect_dsigm_outgate * np_sigm_outgate * (1.0 - np_sigm_outgate)

    expect_dw_ih_i = np.dot(expect_dingate.transpose(1, 0), np_input)
    expect_dw_ih_f = np.dot(expect_dforgetgate.transpose(1, 0), np_input)
    expect_dw_ih_c = np.dot(expect_dcellgate.transpose(1, 0), np_input)
    expect_dw_ih_o = np.dot(expect_doutgate.transpose(1, 0), np_input)
    expect_dw_ih = np.concatenate((expect_dw_ih_i, expect_dw_ih_f, expect_dw_ih_c, expect_dw_ih_o), axis=0)

    expect_db_ih_i = np.sum(expect_dingate, axis=0)
    expect_db_ih_f = np.sum(expect_dforgetgate, axis=0)
    expect_db_ih_c = np.sum(expect_dcellgate, axis=0)
    expect_db_ih_o = np.sum(expect_doutgate, axis=0)
    expect_db_ih = np.concatenate((expect_db_ih_i, expect_db_ih_f, expect_db_ih_c, expect_db_ih_o), axis=0)

    expect_dinput = np.dot(expect_dingate, np_w_ih_i) + np.dot(expect_dforgetgate, np_w_ih_f) +\
        np.dot(expect_dcellgate, np_w_ih_c) + np.dot(expect_doutgate, np_w_ih_o)

    expect_dw_hh_i = np.dot(expect_dingate.transpose(1, 0), hx)
    expect_dw_hh_f = np.dot(expect_dforgetgate.transpose(1, 0), hx)
    expect_dw_hh_c = np.dot(expect_dcellgate.transpose(1, 0), hx)
    expect_dw_hh_o = np.dot(expect_doutgate.transpose(1, 0), hx)
    expect_dw_hh = np.concatenate((expect_dw_hh_i, expect_dw_hh_f, expect_dw_hh_c, expect_dw_hh_o), axis=0)
    expect_db_hh = expect_db_ih

    expect_dhx = np.dot(expect_dingate, np_w_hh_i) + np.dot(expect_dforgetgate, np_w_hh_f) +\
        np.dot(expect_dcellgate, np_w_hh_c) + np.dot(expect_doutgate, np_w_hh_o)

    return [expect_dinput, expect_dhx, expect_dcx, expect_dw_ih, expect_dw_hh, expect_db_ih, expect_db_hh]


def lstm_backward_data_c(np_input, hx, cx, w_ih, w_hh, b_ih, b_hh, gradc):
    np_igates = np.dot(np_input, w_ih.transpose(1, 0)) + b_ih
    np_hgates = np.dot(hx, w_hh.transpose(1, 0)) + b_hh
    np_gates = np_igates + np_hgates

    np_ingate, np_forgetgate, np_cellgate, np_outgate = np.split(np_gates, 4, axis=1)
    np_w_ih_i, np_w_ih_f, np_w_ih_c, np_w_ih_o = np.split(w_ih, 4, axis=0)
    np_b_ih_i, np_b_ih_f, np_b_ih_c, np_b_ih_o = np.split(b_ih, 4, axis=0)
    np_w_hh_i, np_w_hh_f, np_w_hh_c, np_w_hh_o = np.split(w_hh, 4, axis=0)
    np_b_hh_i, np_b_hh_f, np_b_hh_c, np_b_hh_o = np.split(b_hh, 4, axis=0)

    np_sigm_ingate = np_sigmoid(np_ingate)
    np_sigm_forgetgate = np_sigmoid(np_forgetgate)
    np_tanh_cellgate = np.tanh(np_cellgate)
    np_sigm_outgate = np_sigmoid(np_outgate)

    np_c_out = np_sigm_forgetgate * cx + np_sigm_ingate * np_tanh_cellgate

    expect_dc_out = gradc
    expect_dsigm_outgate = np_sigm_outgate * 0.0

    expect_dsigm_forgetgate = expect_dc_out * cx
    expect_dcx = np_sigm_forgetgate * expect_dc_out
    expect_dsigm_ingate = expect_dc_out * np_tanh_cellgate
    expect_dtanh_cellgate = np_sigm_ingate * expect_dc_out

    expect_dingate = expect_dsigm_ingate * np_sigm_ingate * (1.0 - np_sigm_ingate)
    expect_dforgetgate = expect_dsigm_forgetgate * np_sigm_forgetgate * (1.0 - np_sigm_forgetgate)
    expect_dcellgate = expect_dtanh_cellgate * (1.0 - np_tanh_cellgate * np_tanh_cellgate)
    expect_doutgate = expect_dsigm_outgate * np_sigm_outgate * (1.0 - np_sigm_outgate)

    expect_dw_ih_i = np.dot(expect_dingate.transpose(1, 0), np_input)
    expect_dw_ih_f = np.dot(expect_dforgetgate.transpose(1, 0), np_input)
    expect_dw_ih_c = np.dot(expect_dcellgate.transpose(1, 0), np_input)
    expect_dw_ih_o = np.dot(expect_doutgate.transpose(1, 0), np_input)
    expect_dw_ih = np.concatenate((expect_dw_ih_i, expect_dw_ih_f, expect_dw_ih_c, expect_dw_ih_o), axis=0)

    expect_db_ih_i = np.sum(expect_dingate, axis=0)
    expect_db_ih_f = np.sum(expect_dforgetgate, axis=0)
    expect_db_ih_c = np.sum(expect_dcellgate, axis=0)
    expect_db_ih_o = np.sum(expect_doutgate, axis=0)
    expect_db_ih = np.concatenate((expect_db_ih_i, expect_db_ih_f, expect_db_ih_c, expect_db_ih_o), axis=0)

    expect_dinput = np.dot(expect_dingate, np_w_ih_i) + np.dot(expect_dforgetgate, np_w_ih_f) +\
        np.dot(expect_dcellgate, np_w_ih_c) + np.dot(expect_doutgate, np_w_ih_o)

    expect_dw_hh_i = np.dot(expect_dingate.transpose(1, 0), hx)
    expect_dw_hh_f = np.dot(expect_dforgetgate.transpose(1, 0), hx)
    expect_dw_hh_c = np.dot(expect_dcellgate.transpose(1, 0), hx)
    expect_dw_hh_o = np.dot(expect_doutgate.transpose(1, 0), hx)
    expect_dw_hh = np.concatenate((expect_dw_hh_i, expect_dw_hh_f, expect_dw_hh_c, expect_dw_hh_o), axis=0)
    expect_db_hh = expect_db_ih

    expect_dhx = np.dot(expect_dingate, np_w_hh_i) + np.dot(expect_dforgetgate, np_w_hh_f) +\
        np.dot(expect_dcellgate, np_w_hh_c) + np.dot(expect_doutgate, np_w_hh_o)

    return [expect_dinput, expect_dhx, expect_dcx, expect_dw_ih, expect_dw_hh, expect_db_ih, expect_db_hh]


def lstmcell_grad_h_run(shape, dtype, kernel_name="lstm_grad_h", attrs={}):

    shapes = init_lstmcell_shapes(shape)
    print("lstmcell_grad_h - shapes:", shapes)
    mod = utils.op_build_test(lstmcell_grad_h,
                              shapes, [dtype, dtype, dtype, dtype, dtype, dtype, dtype, dtype, dtype],
                              op_attrs=[], kernel_name='lstmcell_grad_h', attrs=attrs)

    np_input, cx, hx, w_ih, w_hh, b_ih, b_hh, gradh, gradc, dw_ih, dw_hh, db_ih, db_hh, dcx, dhx, dx =\
        init_lstmcell_data(shapes, dtype)

    dw_ih, dw_hh, db_ih, db_hh, dcx, dhx, dx = utils.mod_launch(mod, (np_input, hx, cx, w_ih, w_hh, b_ih, b_hh, gradh, gradc, dw_ih, dw_hh, db_ih, db_hh, dcx, dhx, dx),
                                                                outputs=(-7, -6, -5, -4, -3, -2, -1))

    # verification code
    return None, None, None, True


def lstmcell_grad_c_run(shape, dtype, kernel_name="lstm_grad_c", attrs={}):

    shapes = init_lstmcell_shapes(shape)
    print("lstmcell_grad_c - shapes:", shapes)
    mod = utils.op_build_test(lstmcell_grad_c,
                              init_lstmcell_shapes(shape), [dtype, dtype, dtype, dtype, dtype, dtype, dtype, dtype],
                              op_attrs=[], kernel_name='lstmcell_grad_c', attrs=attrs)
    # print(mod.imported_modules[0].get_source())

    np_input, cx, hx, w_ih, w_hh, b_ih, b_hh, gradh, gradc, dw_ih, dw_hh, db_ih, db_hh, dcx, dhx, dx =\
        init_lstmcell_data(shapes, dtype)

    # dw_ih, dw_hh, db_ih, db_hh, dcx, dhx, dx = utils.mod_launch(mod, (np_input, hx, cx, w_ih, w_hh, b_ih, b_hh, gradh, gradc, dw_ih, dw_hh, db_ih, db_hh, dcx, dhx, dx),
    #     outputs=(-7, -6, -5, -4, -3, -2, -1))

    # verification code
    # tensor_list = [dx, dhx, dcx, dw_ih, dw_hh, db_ih, db_hh]
    # expected_tensor_list = lstm_backward_data_h(np_input, hx, cx, w_ih, w_hh, b_ih, b_hh, gradh)

    # assert_res = True

    # for input_id in range(0, 8):
    #     act_output = tensor_list[input_id]
    #     print("act_output =\n", act_output)
    #     print("expect_output =\n", expected_tensor_list[input_id])

    #     assert_res = compare_tensor(act_output, expected_tensor_list[input_id], rtol = 5e-02, atol = 1e-4, equal_nan=True)
    #     print("LSTM_cell_c_grad input_id = ", input_id, "; assert_res = ", assert_res)
    #     print("Max error = " , np.max(np.abs(act_output - expected_tensor_list[input_id])))
    #     input("Press ENTER...")

    return None, None, None, True


def lstmcell_h_ad_run(shape, dtype, kernel_name="lstmcell_h_ad", attrs={}):

    batch_size, input_size, hidden_size = shape
    shapes = init_lstmcell_shapes(shape)
    print("lstmcell_h_ad - shapes:", shapes)

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        for input_id in range(0, 7):
            mod = utils.op_build_test(lstmcell_h_ad,
                                      shapes[0:8], [dtype, dtype, dtype, dtype, dtype, dtype, dtype, dtype],
                                      op_attrs=[input_id], kernel_name=kernel_name, attrs=attrs, tuning=t)
            if t:
                b_hh, b_ih, cx, expected_tensor_list, gradh, hx, np_input, tensor_list, w_hh, w_ih = gen_lstmcell_h_ad_data(
                    dtype, shapes)
                act_output = tensor_list[input_id]
                return mod, expected_tensor_list, (np_input, hx, cx, w_ih, w_hh, b_ih, b_hh, gradh, act_output)
            else:
                return mod
    else:
        assert_res = True
        for input_id in range(0, 7):
            mod = utils.op_build_test(lstmcell_h_ad,
                                      shapes[0:8], [dtype, dtype, dtype, dtype, dtype, dtype, dtype, dtype],
                                      op_attrs=[input_id], kernel_name='lstmcell_h_ad', attrs=attrs)
            # print(mod.imported_modules[0].get_source())

            b_hh, b_ih, cx, expected_tensor_list, gradh, hx, np_input, tensor_list, w_hh, w_ih = gen_lstmcell_h_ad_data(
                dtype, shapes)
            act_output = tensor_list[input_id]
            act_output = utils.mod_launch(mod, (np_input, hx, cx, w_ih, w_hh, b_ih, b_hh, gradh, act_output),
                                          expect=expected_tensor_list[input_id])
            print("act_output =\n", act_output)
            print("expect_output =\n", expected_tensor_list[input_id])

            assert_res = compare_tensor(act_output, expected_tensor_list[input_id], rtol=5e-02, atol=5e-3, equal_nan=True)
            print("LSTM_cell_h input_id = ", input_id, "; assert_res = ", assert_res)
            print("Max error = ", np.max(np.abs(act_output - expected_tensor_list[input_id])))
            # input("Press ENTER...")

        return None, None, None, True


def gen_lstmcell_h_ad_data(dtype, shapes):
    np_input, cx, hx, w_ih, w_hh, b_ih, b_hh, gradh, gradc, dw_ih, dw_hh, db_ih, db_hh, dcx, dhx, dx = \
        init_lstmcell_data(shapes, dtype)
    # lstmcell(input, hx, cx, w_ih, w_hh, b_ih, b_hh)
    tensor_list = [dx, dhx, dcx, dw_ih, dw_hh, db_ih, db_hh]
    expected_tensor_list = lstm_backward_data_h(np_input, hx, cx, w_ih, w_hh, b_ih, b_hh, gradh)
    return b_hh, b_ih, cx, expected_tensor_list, gradh, hx, np_input, tensor_list, w_hh, w_ih


def lstmcell_c_ad_run(shape, dtype, kernel_name="lstmcell_c_ad", attrs={}):

    batch_size, input_size, hidden_size = shape
    shapes = init_lstmcell_shapes(shape)
    print("lstmcell_c_ad - shapes:", shapes)

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        for input_id in range(0, 7):
            mod = utils.op_build_test(lstmcell_c_ad,
                                      shapes[0:8], [dtype, dtype, dtype, dtype, dtype, dtype, dtype, dtype],
                                      op_attrs=[input_id], kernel_name=kernel_name, attrs=attrs, tuning=t)
            if t:
                b_hh, b_ih, cx, expected_tensor_list, gradc, hx, np_input, tensor_list, w_hh, w_ih = \
                    gen_lstmcell_c_ad_data(dtype, shapes)
                act_output = tensor_list[input_id]
                return mod, expected_tensor_list, (np_input, hx, cx, w_ih, w_hh, b_ih, b_hh, gradc, act_output)
            else:
                return mod
    else:
        assert_res = True
        for input_id in range(0, 7):
            mod = utils.op_build_test(lstmcell_c_ad,
                                      shapes[0:8], [dtype, dtype, dtype, dtype, dtype, dtype, dtype, dtype],
                                      op_attrs=[input_id], kernel_name='lstmcell_c_ad', attrs=attrs)

            b_hh, b_ih, cx, expected_tensor_list, gradc, hx, np_input, tensor_list, w_hh, w_ih = gen_lstmcell_c_ad_data(
                dtype, shapes)
            act_output = tensor_list[input_id]
            act_output = utils.mod_launch(mod, (np_input, hx, cx, w_ih, w_hh, b_ih, b_hh, gradc, act_output))
            print("act_output =\n", act_output)
            print("expect_output =\n", expected_tensor_list[input_id])

            assert_res &= compare_tensor(act_output, expected_tensor_list[input_id], rtol=1e-02, atol=1e-2, equal_nan=True)
            print("LSTM_cell_c input_id = ", input_id, "; assert_res = ", assert_res)
            print("Max error = ", np.max(np.abs(act_output - expected_tensor_list[input_id])))
            # input("Press ENTER...")

        return None, None, None, True


def gen_lstmcell_c_ad_data(dtype, shapes):
    np_input, cx, hx, w_ih, w_hh, b_ih, b_hh, gradh, gradc, dw_ih, dw_hh, db_ih, db_hh, dcx, dhx, dx = \
        init_lstmcell_data(shapes, dtype)
    # lstmcell(input, hx, cx, w_ih, w_hh, b_ih, b_hh)
    tensor_list = [dx, dhx, dcx, dw_ih, dw_hh, db_ih, db_hh]
    expected_tensor_list = lstm_backward_data_c(np_input, hx, cx, w_ih, w_hh, b_ih, b_hh, gradc)
    return b_hh, b_ih, cx, expected_tensor_list, gradc, hx, np_input, tensor_list, w_hh, w_ih


def init_rnncell_shapes(shape):
    # shape: batch_size, input_size, hidden_states_size
    batch_size, input_size, hidden_states_size = shape

    input_shape = (batch_size, input_size)
    hidden_shape = (batch_size, hidden_states_size)
    w_ih_shape = (hidden_states_size, input_size)
    w_hh_shape = (hidden_states_size, hidden_states_size)
    b_ih_shape = (hidden_states_size,)
    b_hh_shape = (hidden_states_size,)

    grad_shape = (batch_size, hidden_states_size,)

    return [input_shape, hidden_shape, w_ih_shape, w_hh_shape, b_ih_shape, b_hh_shape, grad_shape]


def init_rnncell_data(shapes, dtype):
    input = random_gaussian(shapes[0], miu=0.1, sigma=0.1).astype(dtype)
    hidden = random_gaussian(shapes[1], miu=0.1, sigma=0.1).astype(dtype)
    w_ih = random_gaussian(shapes[2], miu=0.1, sigma=0.1).astype(dtype)
    w_hh = random_gaussian(shapes[3], miu=0.1, sigma=0.1).astype(dtype)
    b_ih = random_gaussian(shapes[4], miu=0.1, sigma=0.1).astype(dtype)
    b_hh = random_gaussian(shapes[5], miu=0.1, sigma=0.1).astype(dtype)
    grad = random_gaussian(shapes[6], miu=0.1, sigma=0.1).astype(dtype)

    dinput = np.full(input.shape, np.nan, dtype)
    dhidden = np.full(hidden.shape, np.nan, dtype)
    dw_ih = np.full(w_ih.shape, np.nan, dtype)
    dw_hh = np.full(w_hh.shape, np.nan, dtype)
    db_ih = np.full(b_ih.shape, np.nan, dtype)
    db_hh = np.full(b_hh.shape, np.nan, dtype)

    return input, hidden, w_ih, w_hh, b_ih, b_hh, grad, dinput, dhidden, dw_ih, dw_hh, db_ih, db_hh


def rnn_tanh_cell_ad_run(shape, dtype, kernel_name="rnncell_tanh_ad", attrs={}):
    shapes = init_rnncell_shapes(shape)
    print("rnncell_tanh_ad - shapes:", shapes)
    assert_res = True

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        for input_id in range(0, 6):
            mod = utils.op_build_test(rnncell_tanh_ad,
                                      shapes[0:7], [dtype, dtype, dtype, dtype, dtype, dtype, dtype],
                                      op_attrs=[input_id], kernel_name=kernel_name, attrs=attrs, tuning=t)
            if t:
                b_hh, b_ih, expected_tensor_list, grad, hidden, np_input, tensor_list, w_hh, w_ih = gen_rnn_tanh_cell_ad_data(
                    dtype, shapes)
                act_output = tensor_list[input_id]
                return mod, expected_tensor_list, (np_input, hidden, w_ih, w_hh, b_ih, b_hh, grad, act_output)
            else:
                return mod
    else:
        for input_id in range(0, 6):
            mod = utils.op_build_test(rnncell_tanh_ad,
                                      shapes[0:7], [dtype, dtype, dtype, dtype, dtype, dtype, dtype],
                                      op_attrs=[input_id], kernel_name='rnn_tanh_cell_ad', attrs=attrs)

            b_hh, b_ih, expected_tensor_list, grad, hidden, np_input, tensor_list, w_hh, w_ih = gen_rnn_tanh_cell_ad_data(
                dtype, shapes)
            act_output = tensor_list[input_id]
            act_output = utils.mod_launch(mod, (np_input, hidden, w_ih, w_hh, b_ih, b_hh, grad, act_output))
            compare_result_tensor = compare_tensor(act_output, expected_tensor_list[input_id], rtol=5e-02, atol=1e-4, equal_nan=True)
            print("RNN_cell input_id = ", input_id, "; assert_res = ", compare_result_tensor)
            assert_res = assert_res & compare_result_tensor

        return None, None, None, assert_res


def gen_rnn_tanh_cell_ad_data(dtype, shapes):
    np_input, hidden, w_ih, w_hh, b_ih, b_hh, grad, dinput, dhidden, dw_ih, dw_hh, db_ih, db_hh = init_rnncell_data(
        shapes, dtype)
    tensor_list = [dinput, dhidden, dw_ih, dw_hh, db_ih, db_hh]
    np_igates = np.dot(np_input, w_ih.transpose(1, 0)) + b_ih
    np_hgates = np.dot(hidden, w_hh.transpose(1, 0)) + b_hh
    np_h = np.tanh(np_igates + np_hgates)
    expect_dgates = grad * (1.0 - np_h * np_h)
    expect_dw_ih = np.dot(expect_dgates.transpose(1, 0), np_input)
    expect_db_ih = np.sum(expect_dgates, axis=0)
    expect_dw_hh = np.dot(expect_dgates.transpose(1, 0), hidden)
    expect_db_hh = np.sum(expect_dgates, axis=0)
    expect_dinput = np.dot(expect_dgates, w_ih)
    expect_dhidden = np.dot(expect_dgates, w_hh)
    expected_tensor_list = [expect_dinput, expect_dhidden, expect_dw_ih, expect_dw_hh, expect_db_ih, expect_db_hh]
    return b_hh, b_ih, expected_tensor_list, grad, hidden, np_input, tensor_list, w_hh, w_ih


def rnn_tanh_cell_grad_run(shape, dtype, kernel_name="rnn_tanh_cell_grad", attrs={}):
    shapes = init_rnncell_shapes(shape)
    print("rnn_tanh_cell_grad - shapes:", shapes)
    mod = utils.op_build_test(rnn_tanh_cell_grad, shapes, [dtype, dtype, dtype, dtype, dtype, dtype, dtype],
                              op_attrs=[], kernel_name='rnn_tanh_cell_grad', attrs=attrs)

    # verification code
    return None, None, None, True


def rnn_relu_cell_grad_run(shape, dtype, kernel_name="rnn_relu_cell_grad", attrs={}):
    shapes = init_rnncell_shapes(shape)
    print("rnn_relu_cell_grad - shapes:", shapes)

    input, hidden, w_ih, w_hh, b_ih, b_hh, grad, dinput, dhidden, dw_ih, dw_hh, db_ih, db_hh = init_rnncell_data(shapes, dtype)

    mod = utils.op_build_test(rnn_relu_cell_grad, shapes, [dtype, dtype, dtype, dtype, dtype, dtype, dtype],
                              op_attrs=[], kernel_name='rnn_relu_cell_grad', attrs=attrs)

    # verification code
    return None, None, None, True


def rnn_relu_cell_ad_run(shape, dtype, kernel_name="rnncell_tanh_ad", attrs={}):
    shapes = init_rnncell_shapes(shape)
    print("rnncell_tanh_ad - shapes:", shapes)
    assert_res = True

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        for input_id in range(0, 6):
            mod = utils.op_build_test(rnncell_relu_ad,
                                      shapes[0:7], [dtype, dtype, dtype, dtype, dtype, dtype, dtype],
                                      op_attrs=[input_id], kernel_name=kernel_name, attrs=attrs, tuning=t)
            if t:
                b_hh, b_ih, expected_tensor_list, grad, np_hidden, np_input, tensor_list, w_hh, w_ih = gen_rnn_relu_cell_ad_data(
                    dtype, shapes)
                act_output = tensor_list[input_id]
                return mod, expected_tensor_list, (np_input, np_hidden, w_ih, w_hh, b_ih, b_hh, grad, act_output)
            else:
                return mod
    else:
        for input_id in range(0, 6):
            mod = utils.op_build_test(rnncell_relu_ad,
                                      shapes[0:7], [dtype, dtype, dtype, dtype, dtype, dtype, dtype],
                                      op_attrs=[input_id], kernel_name='rnncell_relu_ad', attrs=attrs)
            # print(mod.imported_modules[0].get_source())

            b_hh, b_ih, expected_tensor_list, grad, np_hidden, np_input, tensor_list, w_hh, w_ih = gen_rnn_relu_cell_ad_data(
                dtype, shapes)
            act_output = tensor_list[input_id]
            act_output = utils.mod_launch(mod, (np_input, np_hidden, w_ih, w_hh, b_ih, b_hh, grad, act_output))
            print("act_output = ", act_output)
            print("expect_output = ", expected_tensor_list[input_id])

            assert_res &= compare_tensor(act_output, expected_tensor_list[input_id], rtol=5e-02, atol=1e-4, equal_nan=True)
            print("RNN_cell input_id = ", input_id, "; assert_res = ", assert_res)

        return None, None, None, True


def gen_rnn_relu_cell_ad_data(dtype, shapes):
    np_input, np_hidden, w_ih, w_hh, b_ih, b_hh, grad, dinput, dhidden, dw_ih, dw_hh, db_ih, db_hh = init_rnncell_data(
        shapes, dtype)
    tensor_list = [dinput, dhidden, dw_ih, dw_hh, db_ih, db_hh]
    # igates = dense(input, w_ih, b_ih, use_bias)
    # hgates = dense(hidden, w_hh, b_hh, use_bias)
    # h = relu6(igates + hgates)
    np_igates = np.dot(np_input, w_ih.transpose(1, 0)) + b_ih
    np_hgates = np.dot(np_hidden, w_hh.transpose(1, 0)) + b_hh
    np_h = np_igates + np_hgates
    np_h[np_h < 0.0] = 0.0
    np_h[np_h > 6.0] = 6.0
    expect_dh = np.ones_like(np_h)
    expect_dh[expect_dh < 0.0] = 0.0
    expect_dh[expect_dh > 6.0] = 0.0
    expect_dgates = grad * expect_dh
    expect_dw_ih = np.dot(expect_dgates.transpose(1, 0), np_input)
    expect_db_ih = np.sum(expect_dgates, axis=0)
    expect_dw_hh = np.dot(expect_dgates.transpose(1, 0), np_hidden)
    expect_db_hh = np.sum(expect_dgates, axis=0)
    expect_dinput = np.dot(expect_dgates, w_ih)
    expect_dhidden = np.dot(expect_dgates, w_hh)
    expected_tensor_list = [expect_dinput, expect_dhidden, expect_dw_ih, expect_dw_hh, expect_db_ih, expect_db_hh]
    return b_hh, b_ih, expected_tensor_list, grad, np_hidden, np_input, tensor_list, w_hh, w_ih
