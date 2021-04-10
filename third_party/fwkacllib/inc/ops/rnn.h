/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file rnn.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_RNN_H_
#define OPS_BUILT_IN_OP_PROTO_INC_RNN_H_

#include "graph/operator_reg.h"

namespace ge {
/**
*@brief: Basic LSTM Cell forward calculation.
*@par Inputs:
*five inputs:
*@li x:A 4D Tensor. Must be one of the following types: float16. The format must be FRACTAL_NZ.
*@li h:A 4D Tensor. Must be one of the following types: float16. The format must be FRACTAL_NZ.
*@li c:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li w:A 4D Tensor. Must be one of the following types: float16. The format must be FRACTAL_Z.
*@li b:A 1D Tensor. Must be one of the following types: float16. The format must be ND . \n

*@par Attributes:
*@li keep_prob:An integer identifying the keep prob in the op. Default to 1.
*@li forget_bias:An integer identifying the forget bias in the op. Default to 1.
*@li state_is_tuple:An bool identifying if the hidden state and cell state is tuple. Default to true.
*@li activation:An string identifying the type of activation function in the op. Default to "tanh". Only tanh is currently supported . \n

*@par Outputs:
*seven outputs:
*@li mask:A 1D Tensor. Must be one of the following types: uint8.
*@li ct:A 4D Tensor. Must be one of the following types: float16, float32.
*@li ht:A 4D Tensor. Must be one of the following types: float16.
*@li it:A 4D Tensor. Must be one of the following types: float16, float32.
*@li jt:A 4D Tensor. Must be one of the following types: float16, float32.
*@li ft:A 4D Tensor. Must be one of the following types: float16, float32.
*@li ot:A 4D Tensor. Must be one of the following types: float16, float32.
*@li tanhct:A 4D Tensor. Must be one of the following types: float16, float32.
*/
REG_OP(BasicLSTMCell)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .INPUT(h, TensorType({DT_FLOAT16}))
    .INPUT(c, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(w, TensorType({DT_FLOAT16}))
    .INPUT(b, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(mask, TensorType({DT_UINT8}))
    .OUTPUT(ct, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(ht, TensorType({DT_FLOAT16}))
    .OUTPUT(it, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(jt, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(ft, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(ot, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(tanhct, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(keep_prob, Float, 1.0)
    .ATTR(forget_bias, Float, 1.0)
    .ATTR(state_is_tuple, Bool, true)
    .ATTR(activation, String, "tanh")
    .OP_END_FACTORY_REG(BasicLSTMCell)

/**
*@brief: Dynamic LSTM forward calculation . \n

*@par Inputs:
*@li x:A 4D Tensor. Must be the type float32. The format must be FRACTAL_NZ.
*@li w:A 4D Tensor. Must be the type float32. The format must be FRACTAL_Z.
*@li b:A 1D Tensor. Must be the type float32. The format must be ND . \n

*@par Outputs:
*output_h:A Tensor of output. Must be the type float32. The format must be FRACTAL_Z.

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(DynamicLSTM)
    .INPUT(x, TensorType({DT_FLOAT32}))
    .INPUT(w, TensorType({DT_FLOAT32}))
    .INPUT(b, TensorType({DT_FLOAT32}))
    .OUTPUT(output_h, TensorType({DT_FLOAT32}))
    .OP_END_FACTORY_REG(DynamicLSTM)

/**
*@brief: DynamicRNNGrad calculation.
*@par Inputs:
*ten inputs: \n
*@li x:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li w:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li b:A 1D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li y:A 1D Tensor. Must be one of the following types: int32. The format must be FRACTAL_NZ.
*@li init_h:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li init_c:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li h:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li c:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li dy:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li dh:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li dc:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li i:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li j:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li f:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li o:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li tanhct:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li seq_length:A 1D Tensor. Must be one of the following types: int32.
*@li mask:A 1D Tensor. Must be one of the following types: int8.
*@li wci:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li wcf:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li wco:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.

*@par Attributes:
*@li cell_type:An string identifying the cell type in the op. Default to "LSTM". Only LSTM is currently supported.
*@li direction:An string identifying the direction in the op. Default to "UNIDIRECTIONAL". Only UNIDIRECTIONAL is currently supported.
*@li cell_depth:An integer identifying the cell depth in the op. Default to 1.
*@li use_peephole:An bool identifying if use peephole in the op. Default to false.
*@li keep_prob:An float identifying the keep prob in the op. Default to 1.
*@li cell_clip:An float identifying the cell clip in the op. Default to -1.
*@li num_proj:An integer identifying the num projection in the op. Default to 0.
*@li time_major:An bool identifying the time major in the op. Default to false.
*@li activation:An string identifying the type of activation function in the op. Default to "tanh". Only tanh is currently supported.
*@li forget_bias:An float identifying the forget bias in the op. Default to 0.
*@li is_training:An bool identifying is training in the op. Default to true.

*@par Outputs:
*eight outputs: \n
*@li dw:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li db:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li dx:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li dh_prev:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li dc_prev:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li dwci:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li dwcf:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li dwco:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*/
REG_OP(DynamicRNNGrad)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(w, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(b, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(init_h, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(init_c, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(h, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(c, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(dy, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(dh, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(dc, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(i, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(j, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(f, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(o, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(tanhct, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(seq_length, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(mask, TensorType({DT_UINT8}))
    .OPTIONAL_INPUT(wci, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(wcf, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(wco, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(dw, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(db, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(dx, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(dh_prev, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(dc_prev, TensorType({DT_FLOAT16, DT_FLOAT}))
    .DYNAMIC_OUTPUT(dwci, TensorType({DT_FLOAT16, DT_FLOAT}))
    .DYNAMIC_OUTPUT(dwcf, TensorType({DT_FLOAT16, DT_FLOAT}))
    .DYNAMIC_OUTPUT(dwco, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(cell_type, String, "LSTM")
    .ATTR(direction, String, "UNIDIRECTIONAL")
    .ATTR(cell_depth, Int, 0)
    .ATTR(use_peephole, Bool, false)
    .ATTR(keep_prob, Float, -1.0)
    .ATTR(cell_clip, Float, -1.0)
    .ATTR(num_proj, Int, 0)
    .ATTR(time_major, Bool, true)
    .ATTR(forget_bias, Float, 0.0)
    .OP_END_FACTORY_REG(DynamicRNNGrad)

/**
*@brief: DynamicRNN calculation.
*@par Inputs:
*ten inputs:
*@li x:A required 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li w:A required 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_ZN_LSTM.
*@li b:A required 1D Tensor. Must be one of the following types: float16, float32. The format must be ND.
*@li seq_length:A optional 1D Tensor. Must be one of the following types: int32. The format must be ND.
*@li init_h:A optional 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li init_c:A optional 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li wci:A optional 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_ZN_LSTM.
*@li wcf:A optional 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_ZN_LSTM.
*@li wco:A optional 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_ZN_LSTM.
*@li mask:A optional 1D Tensor. Must be one of the following types: uint8. The format must be ND . \n

*@par Attributes:
*@li cell_type:An string identifying the cell type in the op. Default to "LSTM". Only LSTM is currently supported.
*@li direction:An string identifying the direction in the op. Default to "UNIDIRECTIONAL". Only UNIDIRECTIONAL is currently supported.
*@li cell_depth:An integer identifying the cell depth in the op. Default to 1.
*@li use_peephole:An bool identifying if use peephole in the op. Default to false.
*@li keep_prob:An float identifying the keep prob in the op. Default to 1.
*@li cell_clip:An float identifying the cell clip in the op. Default to -1.
*@li num_proj:An integer identifying the num projection in the op. Default to 0.
*@li time_major:An bool identifying the time major in the op. Default to true.
*@li activation:An string identifying the type of activation function in the op. Default to "tanh". Only tanh is currently supported.
*@li forget_bias:An float identifying the forget bias in the op. Default to 0.
*@li is_training:An bool identifying is training in the op. Default to true . \n

*@par Outputs:
*eight outputs:
*@li y:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li output_h:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li output_c:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li i:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li j:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li f:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li o:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li tanhct:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@par Third-party framework compatibility:
* Compatible with the TF operator LSTM.
*/
REG_OP(DynamicRNN)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(w, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(b, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(seq_length, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(init_h, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(init_c, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(wci, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(wcf, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(wco, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(mask, TensorType({DT_UINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(output_h, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(output_c, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(i, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(j, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(f, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(o, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(tanhc, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(cell_type, String, "LSTM")
    .ATTR(direction, String, "UNIDIRECTIONAL")
    .ATTR(cell_depth, Int, 1)
    .ATTR(use_peephole, Bool, false)
    .ATTR(keep_prob, Float, 1.0)
    .ATTR(cell_clip, Float, -1.0)
    .ATTR(num_proj, Int, 0)
    .ATTR(time_major, Bool, true)
    .ATTR(activation, String, "tanh")
    .ATTR(forget_bias, Float, 0.0)
    .ATTR(is_training, Bool, true)
    .OP_END_FACTORY_REG(DynamicRNN)

/**
*@brief: DynamicLSTMV2 calculation.
*@par Inputs:
*ten inputs:
*@li x:A required 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li w:A required 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_ZN_LSTM.
*@li b:A required 1D Tensor. Must be one of the following types: float16, float32. The format must be ND.
*@li cont:A required 2D Tensor. Must be one of the following types: float16, float32. The format must be ND.
*@li w_xc_x_static:A optional 2D Tensor. Must be one of the following types: float16, float32. The format must be ND.
*@li h0:A optional 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li c0:A optional 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li wci:A optional 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_ZN_LSTM.
*@li wcf:A optional 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_ZN_LSTM.
*@li wco:A optional 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_ZN_LSTM.
*@li mask:A optional 1D Tensor. Must be one of the following types: uint8. The format must be ND .

*@par Attributes:
*@li num_output:An integer identifying the num projection in the op. Default to 0.
*@li expose_hidden:An bool identifying the expose_hidden in the op. Default to flase.
*@li need_output_last:An bool identifying the time major in the op. Default to true.
*@li forget_bias:An float identifying the forget bias in the op. Default to 0.

*@par Outputs:
*eight outputs:
*@li y:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li output_h:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li output_c:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li last_output_h:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li last_output_c:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@par Third-party framework compatibility:
* Compatible with the Caffe operator LSTM.
*@par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(DynamicLSTMV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(w, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(b, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(cont, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(w_xc_x_static, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(h0, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(c0, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(wci, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(wcf, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(wco, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(mask, TensorType({DT_UINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(output_h, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(output_c, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(last_output_h, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(last_output_c, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(num_output, Int, 0)
    .ATTR(expose_hidden, Bool, false)
    .ATTR(need_output_last, Bool, false)
    .ATTR(forget_bias, Float, 0.0)
    .OP_END_FACTORY_REG(DynamicLSTMV2)

/**
*@brief: LSTMInputGrad calculation.
*@par Inputs:
*ten inputs: \n
*@li w:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li init_c:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li h:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li c:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li dy:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li dh:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li dc:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li i:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li j:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li f:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li o:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li tanhct:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.


*@par Outputs:
*eight outputs: \n
*@li dx:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li dh_prev:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li dc_prev:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*/
REG_OP(LSTMInputGrad)
    .INPUT(w, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(init_c, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(c, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(dy, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(dh, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(dc, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(i, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(j, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(f, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(o, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(tanhct, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(dx, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(dh_prev, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(dc_prev, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(dgate, TensorType({DT_FLOAT16}))
    .OP_END_FACTORY_REG(LSTMInputGrad)


/**
*@brief: Basic LSTM Cell backward calculation.Calculate the gradient of input and hidden state.
*@par Inputs:
*three inputs:
*@li dgate:A 4D Tensor. Must be one of the following types: float16. The format must be FRACTAL_NZ.
*@li w:A 4D Tensor. Must be one of the following types: float16. The format must be FRACTAL_Z.
*@li dropout_mask:A 1D Tensor. Must be one of the following types: uint8. The format must be ND . \n

*@par Attributes:
*keep_prob:An integer identifying the keep prob in the op. Default to 1 . \n

*@par Outputs:
*two outputs:
*@li dxt:A 4D Tensor. Must be one of the following types: float16, float32.
*@li dht:A 4D Tensor. Must be one of the following types: float16, float32.

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(BasicLSTMCellInputGrad)
    .INPUT(dgate, TensorType({DT_FLOAT16}))
    .INPUT(w, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(dropout_mask, TensorType({DT_UINT8}))
    .OUTPUT(dxt, TensorType({DT_FLOAT16, DT_FLOAT32}))
    .OUTPUT(dht, TensorType({DT_FLOAT16, DT_FLOAT32}))
    .ATTR(keep_prob, Float, 1.0)
    .OP_END_FACTORY_REG(BasicLSTMCellInputGrad)

/**
*@brief: Basic LSTM Cell backward calculation.Calculate the gradient of weight and bias.
*@par Inputs:
*three inputs:
*@li x:A 4D Tensor. Must be one of the following types: float16. The format must be FRACTAL_NZ.
*@li h:A 4D Tensor. Must be one of the following types: float16. The format must be FRACTAL_NZ.
*@li dgate:A 4D Tensor. Must be one of the following types: uint8. The format must be FRACTAL_NZ . \n

*@par Outputs:
*two outputs:
*@li dw:A 4D Tensor. Must be one of the following types: float16.
*@li db:A 4D Tensor. Must be one of the following types: float16, float32.

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(BasicLSTMCellWeightGrad)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .INPUT(h, TensorType({DT_FLOAT16}))
    .INPUT(dgate, TensorType({DT_FLOAT16}))
    .OUTPUT(dw, TensorType({DT_FLOAT16}))
    .OUTPUT(db, TensorType({DT_FLOAT16, DT_FLOAT32}))
    .OP_END_FACTORY_REG(BasicLSTMCellWeightGrad)

/**
*@brief: Basic LSTM Cell backward calculation.Calculate the gradient of gates and cell state.
*@par Inputs:
*eight inputs:
*@li c:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li dht:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li dct:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li it:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li jt:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li ft:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li ot:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li tanhct:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ . \n

*@par Attributes:
*@li forget_bias:An integer identifying the forget bias in the op. Default to 1.
*@li activation:An string identifying the type of activation function in the op. Default to "tanh". Only tanh is currently supported . \n

*@par Outputs:
*two outputs:
*@li dgate:A 4D Tensor. Must be one of the following types: float16.
*@li dct_1:A 4D Tensor. Must be one of the following types: float16, float32.

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(BasicLSTMCellCStateGrad)
    .INPUT(c, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(dht, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(dct, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(it, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(jt, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(ft, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(ot, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(tanhct, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(dgate, TensorType({DT_FLOAT16}))
    .OUTPUT(dct_1, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(forget_bias, Float, 1.0)
    .ATTR(activation, String, "tanh")
    .OP_END_FACTORY_REG(BasicLSTMCellCStateGrad)

/**
*@brief: RNN operator.
*@par Inputs:
*eight inputs:
*@li x:A 4D Tensor. Must be one of the following types: float16. The format must be FRACTAL_NZ.
*@li cont:A 1D Tensor. Must be one of the following types: float16. The format must be ND.
*@li x_static:A 4D Tensor. Must be one of the following types: float16. The format must be FRACTAL_NZ.
*@li h_0:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li w_xh:A 4D Tensor. Must be one of the following types: float16. The format must be FRACTAL_Z.
*@li w_sh:A 4D Tensor. Must be one of the following types: float16. The format must be FRACTAL_Z.
*@li w_hh:A 4D Tensor. Must be one of the following types: float16. The format must be FRACTAL_Z.
*@li w_ho:A 4D Tensor. Must be one of the following types: float16. The format must be FRACTAL_Z.
*@li bias_h:A 1D Tensor. Must be one of the following types: float16, float32. The format must be ND.
*@li bias_o:A 1D Tensor. Must be one of the following types: float16, float32. The format must be ND . \n

*@par Attributes:
*@li expose_hidden:An bool identifying if expose the hidden state of last time step. Default to false.
*@li num_output:An integer identifying the number of output features. Default to 0 . \n

*@par Outputs:
*two outputs:
*@li o:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li h_t:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(RNN)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .INPUT(cont, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(x_static, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(h_0, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(w_xh, TensorType({DT_FLOAT16}))
    .INPUT(bias_h, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(w_sh, TensorType({DT_FLOAT16}))
    .INPUT(w_hh, TensorType({DT_FLOAT16}))
    .INPUT(w_ho, TensorType({DT_FLOAT16}))
    .INPUT(bias_o, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(o, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(h_t, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(num_output, Int, 0)
    .ATTR(expose_hidden, Bool, false)
    .OP_END_FACTORY_REG(RNN)

/**
*@brief: BasicRNNCell operator.
*@par Inputs:
*eight inputs:
*@li x:A 4D Tensor. Must be one of the following types: float16. The format must be FRACTAL_NZ.
*@li cont:A 1D Tensor. Must be one of the following types: float16. The format must be ND.
*@li w_xh_x_static:A 4D Tensor. Must be one of the following types: float16. The format must be FRACTAL_NZ.
*@li h_0:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li w_xh:A 4D Tensor. Must be one of the following types: float16. The format must be FRACTAL_Z.
*@li w_hh:A 4D Tensor. Must be one of the following types: float16. The format must be FRACTAL_Z.
*@li w_ho:A 4D Tensor. Must be one of the following types: float16. The format must be FRACTAL_Z.
*@li bias_h:A 1D Tensor. Must be one of the following types: float16, float32. The format must be ND.
*@li bias_o:A 1D Tensor. Must be one of the following types: float16, float32. The format must be ND . \n

*@par Attributes:
*@li expose_hidden:An bool identifying if expose the hidden state of last time step. Default to false.
*@li num_output:An integer identifying the number of output features. Default to 0 . \n

*@par Outputs:
*two outputs:
*@li o_t:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li h_t:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(BasicRNNCell)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(cont, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(w_xh_x_static, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(h_0, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(w_xh, TensorType({DT_FLOAT16}))
    .INPUT(bias_h, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(w_hh, TensorType({DT_FLOAT16}))
    .INPUT(w_ho, TensorType({DT_FLOAT16}))
    .INPUT(bias_o, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(o_t, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(h_t, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(expose_hidden, Bool, false)
    .ATTR(num_output, Int, 0)
    .OP_END_FACTORY_REG(BasicRNNCell)

/**
*@brief DynamicGRU calculation.
*@par Inputs:
*seven inputs: 
*@li x:Must be one of the following types: float16. The format must be FRACTAL_NZ.
*@li w:Must be one of the following types: float16. The format must be FRACTAL_Z.
*@li b:Must be one of the following types: float16, float32. The format must be ND.
*@li cw:Must be one of the following types: float16. The format must be FRACTAL_Z.
*@li cb:Must be one of the following types: float16, float32. The format must be ND.
*@li seq_length:Must be one of the following types: int32. The format must be ND.
*@li init_h:Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.

*@par Attributes:
*@li direction:An string identifying the direction in the op. Default to "UNIDIRECTIONAL". Only UNIDIRECTIONAL is currently supported.
*@li cell_depth:An integer identifying the cell depth in the op. Default to 1.
*@li keep_prob:An float identifying the keep prob in the op. Default to 1.
*@li cell_clip:An float identifying the cell clip in the op. Default to -1.
*@li num_proj:An integer identifying the num projection in the op. Default to 0.
*@li time_major:An bool identifying the time major in the op. Default to true.
*@li activation:An string identifying the type of activation function in the op. Default to "tanh". Only tanh is currently supported.
*@li is_training:An bool identifying is training in the op. Default to true.

*@par Outputs:
*five outputs: 
*@li y:Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li output_h:Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li r:Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li i:Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li n:Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(DynamicGRU)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .INPUT(w, TensorType({DT_FLOAT16}))
    .INPUT(b, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(cw, TensorType({DT_FLOAT16}))
    .INPUT(cb, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(seq_length, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(init_h, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(output_h, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(r, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(i, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(n, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(direction, String, "UNIDIRECTIONAL")
    .ATTR(cell_depth, Int, 1)
    .ATTR(keep_prob, Float, 1.0)
    .ATTR(cell_clip, Float, -1.0)
    .ATTR(num_proj, Int, 0)
    .ATTR(time_major, Bool, true)
    .ATTR(activation, String, "tanh")
    .ATTR(is_training, Bool, true)
    .OP_END_FACTORY_REG(DynamicGRU)

/**
*@brief DynamicGRUV2 calculation.
*@par Inputs:
*seven inputs: 
*@li x:Must be one of the following types: float16. The format must be FRACTAL_NZ.
*@li weight_input:Must be one of the following types: float16. The format must be FRACTAL_Z.
*@li weight_hidden:Must be one of the following types: float16. The format must be FRACTAL_Z.
*@li bias_input:Must be one of the following types: float16, float32. The format must be ND.
*@li bias_hidden:Must be one of the following types: float16, float32. The format must be ND.
*@li seq_length:Must be one of the following types: int32. The format must be ND.
*@li init_h:Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.

*@par Attributes:
*@li direction:An string identifying the direction in the op. Default to "UNIDIRECTIONAL". Only UNIDIRECTIONAL is currently supported.
*@li cell_depth:An integer identifying the cell depth in the op. Default to 1.
*@li keep_prob:An float identifying the keep prob in the op. Default to 1.
*@li cell_clip:An float identifying the cell clip in the op. Default to -1.
*@li num_proj:An integer identifying the num projection in the op. Default to 0.
*@li time_major:An bool identifying the time major in the op. Default to true.
*@li activation:An string identifying the type of activation function in the op. Default to "tanh". Only tanh is currently supported.
*@li gate_order:An string identifying the gate order in weight and bias. Default to "zrh". "rzh" is another option.
*@li reset_after:An bool identifying whether to apply reset gate after matrix multiplication. Default to true.
*@li is_training:An bool identifying is training in the op. Default to true.

*@par Outputs:
*six outputs: 
*@li y:Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li output_h:Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li update:Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li reset:Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li new:Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li hidden_new:Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(DynamicGRUV2)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .INPUT(weight_input, TensorType({DT_FLOAT16}))
    .INPUT(weight_hidden, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(bias_input, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(bias_hidden, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(seq_length, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(init_h, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(output_h, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(update, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(reset, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(new, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(hidden_new, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(direction, String, "UNIDIRECTIONAL")
    .ATTR(cell_depth, Int, 1)
    .ATTR(keep_prob, Float, 1.0)
    .ATTR(cell_clip, Float, -1.0)
    .ATTR(num_proj, Int, 0)
    .ATTR(time_major, Bool, true)
    .ATTR(activation, String, "tanh")
    .ATTR(gate_order, String, "zrh")
    .ATTR(reset_after, Bool, true)
    .ATTR(is_training, Bool, true)
    .OP_END_FACTORY_REG(DynamicGRUV2)


/**
*@brief DynamicGRUV2Hidden calculation.
*@par Inputs:
*five inputs: 
*@li x_weight_input:Must be one of the following types: float32. The format must be FRACTAL_NZ.
*@li weight_hidden:Must be one of the following types: float16. The format must be FRACTAL_Z.
*@li bias_hidden:Must be one of the following types: float16, float32. The format must be ND.
*@li seq_length:Must be one of the following types: int32. The format must be ND.
*@li init_h:Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.

*@par Attributes:
*@li direction:An string identifying the direction in the op. Default to "UNIDIRECTIONAL". 
Only UNIDIRECTIONAL is currently supported.
*@li cell_depth:An integer identifying the cell depth in the op. Default to 1.
*@li keep_prob:An float identifying the keep prob in the op. Default to 1.
*@li cell_clip:An float identifying the cell clip in the op. Default to -1.
*@li num_proj:An integer identifying the num projection in the op. Default to 0.
*@li time_major:An bool identifying the time major in the op. Default to true.
*@li activation:An string identifying the type of activation function in the op. Default to "tanh". 
Only tanh is currently supported.
*@li gate_order:An string identifying the gate order in weight and bias. Default to "zrh". "rzh" is another option.
*@li reset_after:An bool identifying whether to apply reset gate after matrix multiplication. Default to true.
*@li is_training:An bool identifying is training in the op. Default to true.

*@par Outputs:
*six outputs: 
*@li y:Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li output_h:Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li update:Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li reset:Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li new:Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li hidden_new:Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(DynamicGRUV2Hidden)
    .INPUT(x_weight_input, TensorType({DT_FLOAT32}))
    .INPUT(weight_hidden, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(bias_hidden, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(seq_length, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(init_h, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(output_h, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(update, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(reset, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(new, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(hidden_new, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(direction, String, "UNIDIRECTIONAL")
    .ATTR(cell_depth, Int, 1)
    .ATTR(keep_prob, Float, 1.0)
    .ATTR(cell_clip, Float, -1.0)
    .ATTR(num_proj, Int, 0)
    .ATTR(time_major, Bool, true)
    .ATTR(activation, String, "tanh")
    .ATTR(gate_order, String, "zrh")
    .ATTR(reset_after, Bool, true)
    .ATTR(is_training, Bool, true)
    .OP_END_FACTORY_REG(DynamicGRUV2Hidden)


/**
*@brief: DynamicGRUV2Grad calculation.
*@par Inputs:
*fourteen inputs: \n
*@li x:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li weight_input:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li weight_hidden:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li y:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li init_h:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li h:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li dy:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li dh:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li update:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li reset:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li new:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li hidden_new:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li seq_length:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li mask:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.

*@par Attributes:
*@li direction:An string identifying the direction in the op. Default to "UNIDIRECTIONAL". Only UNIDIRECTIONAL is currently supported.
*@li cell_depth:An integer identifying the cell depth in the op. Default to 1.
*@li keep_prob:An float identifying the keep prob in the op. Default to 1.
*@li cell_clip:An float identifying the cell clip in the op. Default to -1.
*@li num_proj:An integer identifying the num projection in the op. Default to 0.
*@li time_major:An bool identifying the time major in the op. Default to true.
*@li gate_order:An string identifying the gate order in weight and bias. Default to "zrh". "rzh" is another option.
*@li reset_after:An bool identifying whether to apply reset gate after matrix multiplication. Default to true.

*@par Outputs:
*six outputs: \n
*@li dw_input:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li dw_hidden:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li db_input:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li db_hidden:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li dx:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li dh_prev:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(DynamicGRUV2Grad)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(weight_input, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(weight_hidden, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(init_h, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(h, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(dy, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(dh, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(update, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(reset, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(new, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(hidden_new, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(seq_length, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(mask, TensorType({DT_UINT8}))
    .OUTPUT(dw_input, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(dw_hidden, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(db_input, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(db_hidden, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(dx, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(dh_prev, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(direction, String, "UNIDIRECTIONAL")
    .ATTR(cell_depth, Int, 0)
    .ATTR(keep_prob, Float, -1.0)
    .ATTR(cell_clip, Float, -1.0)
    .ATTR(num_proj, Int, 0)
    .ATTR(time_major, Bool, true)
    .ATTR(gate_order, String, "zrh")
    .ATTR(reset_after, Bool, true)
    .OP_END_FACTORY_REG(DynamicGRUV2Grad)

/**
*@brief: GRUV2HiddenGrad calculation.
*@par Inputs:
*nine inputs: \n
*@li dh_pre_t:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li init_h:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li h:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li dy:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li dh:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li update:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li reset:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li new:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li hidden_new:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.

*@par Attributes:
*@li t_state:An Int identifying the current t state. Default to [0, 4].
*@li gate_order:An string identifying the gate order in weight and bias. Default to "zrh". "rzh" is another option.

*@par Outputs:
*three outputs: \n
*@li dh_prev:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li dgate_h:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li dnt_x:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(GRUV2HiddenGradCell)
    .INPUT(dh_pre_t, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(h, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(dy, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(dh, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(update, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(reset, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(new, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(hidden_new, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(dh_prev, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(dgate_h, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(dnt_x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(t_state, Int, 0)
    .ATTR(gate_order, String, "zrh")
    .OP_END_FACTORY_REG(GRUV2HiddenGradCell)

/**
* @brief Calculates the reversed outputs of the function "embedding". \n

* @par Inputs:
* Two inputs, including:
* @li grad: A mutable Tensor of word grad. Must be one of the following types:
*     float32.
* @li indices: A mutable word index Tensor of the int32 type.\n

* @par Attributes:
* @li num_weights: An int attr which use to judge how many words in dict. \n

* @li padding_idx: An int attr judge which word to fill zeros. Defaults to "-1". \n

* @li scale_grad_by_freq: An optional bool. Defaults to "False".
*     If "True", "grad_weight" will be scale by word_frequency.
*     If "False", "grad_weight" will not be scale by word_frequency. \n

* @par Outputs:
* @li grad_weight: A mutable output Tensor of new word grad has the same type as "grads". \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator EmbeddingDenseGrad.
*/
REG_OP(EmbeddingDenseGrad)
    .INPUT(grad, TensorType({ DT_FLOAT32 }))  /* "First operand." */
    .INPUT(indices, TensorType({ DT_INT32 })) /* "Second operand." */
    .OUTPUT(y, TensorType({ DT_FLOAT32 }))    /* "Result, has same element type as two inputs" */
    .REQUIRED_ATTR(num_weights, Int)
    .ATTR(padding_idx, Int, -1)
    .ATTR(scale_grad_by_freq, Bool, false)
    .OP_END_FACTORY_REG(EmbeddingDenseGrad)

/**
*@brief CommonLSTM calculation.
*@par Inputs:
*eight inputs: \n
*@li x:Each time step is a 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li w:Each direction is a 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_ZN_LSTM.
*@li r:Each direction is a 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_ZN_LSTM.
*@li b:An optional input. Each direction is a 1D Tensor. Must be one of the following types: float16, float32. The format must be ND.
*@li sequence_lens:An optional input. A 1D Tensor.Must be one of the following types: int32. The format must be ND.
*@li initial_h:An optional input. Each direction is a 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li initial_c:An optional input. Each direction is a 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li p:An optional input. Each direction is a 1D Tensor.Must be one of the following types: float16, float32. The format must be ND.

*@par Attributes:
*@li activation_alpha:Optional scaling values used by some activation functions. Empty is currently supported.
*@li activation_beta:Optional scaling values used by some activation functions. Empty is currently supported.
*@li activations:The list of activation functions. Empty is currently supported.
*@li clip:An float identifying the cell clip in the op. Default to -1.
*@li direction:Specify if the RNN is forward, reverse, or bidirectional. Must be one of forward(default), reverse, or bidirectional.
*@li hidden_size:Number of neurons in the hidden layer. Reserved.
*@li input_forget:Couple the input and forget gates if 1. Reserved.

*@par Outputs:
*three outputs: \n
*@li y:First dimension is time step, second dimension is direction, others is a 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li y_h:Each direction is a 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li y_c:Each direction is a 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*/

REG_OP(CommonLSTM)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(w, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(r, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(b, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(sequence_lens, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(initial_h, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(initial_c, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(p, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y_h, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y_c, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(activation_alpha, ListFloat, {})
    .ATTR(activation_beta, ListFloat, {})
    .ATTR(activations, ListString, {})
    .ATTR(clip, Float, -1.0)
    .ATTR(direction, String, "forward")
    .REQUIRED_ATTR(hidden_size, Int)
    .ATTR(input_forget, Int, 0)
    .OP_END_FACTORY_REG(CommonLSTM)
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_RNN_H_
