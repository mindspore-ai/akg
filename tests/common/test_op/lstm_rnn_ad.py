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

"""operator dsl function: lstm_rnn_ad"""
import akg
from tests.common.test_op.lstm_rnn import lstmcell, rnn_tanh_cell, rnn_relu_cell


def lstmcell_h_ad(_input, hx, cx, w_ih, w_hh, b_ih, b_hh, Head, input_id):
    forward_h_op, _ = lstmcell(_input, hx, cx, w_ih, w_hh, b_ih, b_hh)
    tensor_list = [_input, hx, cx, w_ih, w_hh, b_ih, b_hh]
    _jacs = list(akg.differentiate(forward_h_op, [tensor_list[input_id]], Head))

    ###################################################
    # Need to disable CSE due to stmt dense() + dense()
    attrs = dict()
    attrs['disable_cse'] = True
    attrs['to_three_address_reuse'] = True
    attrs['to_three_min_split'] = 10
    ###################################################

    return _jacs[0], attrs


def lstmcell_c_ad(_input, hx, cx, w_ih, w_hh, b_ih, b_hh, Head, input_id):
    _, forward_c_op = lstmcell(_input, hx, cx, w_ih, w_hh, b_ih, b_hh)

    tensor_list = [_input, hx, cx, w_ih, w_hh, b_ih, b_hh]
    _jacs = list(akg.differentiate(forward_c_op, [tensor_list[input_id]], Head))

    ###################################################
    # Need to disable CSE due to stmt dense() + dense()
    attrs = dict()
    attrs['disable_cse'] = True
    attrs['to_three_address_reuse'] = True
    attrs['to_three_address_min_split'] = 10
    ###################################################

    return _jacs[0], attrs


def rnncell_tanh_ad(inputs, hidden, w_ih, w_hh, b_ih, b_hh, Head, input_id):
    forward_op = rnn_tanh_cell(inputs, hidden, w_ih, w_hh, b_ih, b_hh)
    tensor_list = [inputs, hidden, w_ih, w_hh, b_ih, b_hh]
    _jacs = list(akg.differentiate(forward_op, [tensor_list[input_id]], Head))

    return _jacs[0]


def rnncell_relu_ad(inputs, hidden, w_ih, w_hh, b_ih, b_hh, Head, input_id):
    forward_op = rnn_relu_cell(inputs, hidden, w_ih, w_hh, b_ih, b_hh)

    tensor_list = [inputs, hidden, w_ih, w_hh, b_ih, b_hh]
    _jacs = list(akg.differentiate(forward_op, [tensor_list[input_id]], Head))

    return _jacs[0]
