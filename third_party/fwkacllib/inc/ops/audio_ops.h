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
 * \file audio_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_AUDIO_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_AUDIO_OPS_H_

#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Mel-Frequency Cepstral Coefficient (MFCC) calculation consists of
taking the DCT-II of a log-magnitude mel-scale spectrogram . \n

*@par Inputs:
*Input "spectrogram" is a 3D tensor. Input "sample_rate" is a scalar.
* @li spectrogram: A 3D float tensor.
* @li sample_rate: The MFCC sample rate . \n

*@par Attributes:
*@li upper_frequency_limit: The highest frequency for calculation.
*@li lower_frequency_limit: The lowest frequency for calculation.
*@li filterbank_channel_count: Resolution of the Mel bank.
*@li dct_coefficient_count: Number of output channels to produce
per time slice . \n

*@par Outputs:
*y: A Tensor of type float32 . \n

*@attention Constraints:
*Mfcc runs on the Ascend AI CPU, which delivers poor performance.

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Mfcc . \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(Mfcc)
    .INPUT(spectrogram, TensorType({DT_FLOAT}))
    .INPUT(sample_rate, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .ATTR(upper_frequency_limit, Float, 4000)
    .ATTR(lower_frequency_limit, Float, 20)
    .ATTR(filterbank_channel_count, Int, 40)
    .ATTR(dct_coefficient_count, Int, 13)
    .OP_END_FACTORY_REG(Mfcc)

/**
*@brief Decodes and generates spectrogram using wav float tensor . \n

*@par Inputs:
*Input "x" is a 2D matrix.
* x: A float tensor. Float representation of audio data . \n

*@par Attributes:
*@li window_size: Size of the spectrogram window.
*@li stride: Size of the spectrogram stride.
*@li magnitude_squared: If true, uses squared magnitude . \n

*@par Outputs:
*spectrogram: A 3D float Tensor . \n

*@attention Constraints:
*AudioSpectrogram runs on the Ascend AI CPU, which delivers
poor performance . \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator AudioSpectrogram . \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/

REG_OP(AudioSpectrogram)
    .INPUT(x, TensorType({DT_FLOAT}))
    .OUTPUT(spectrogram, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(window_size, Int)
    .REQUIRED_ATTR(stride, Int)
    .ATTR(magnitude_squared, Bool, false)
    .OP_END_FACTORY_REG(AudioSpectrogram)

/**
*@brief Decodes a 16-bit WAV file into a float tensor . \n

*@par Inputs:
*contents: A Tensor of type string. The WAV-encoded audio, usually from a file . \n

*@par Attributes:
*@li desired_channels: An optional int. Defaults to "-1".
Number of sample channels wanted.
*@li desired_samples: An optional int. Defaults to "-1".
Length of audio requested . \n

*@par Outputs:
*@li *audio: A Tensor of type float32.
*@li *sample_rate: A Tensor of type int32 . \n

*@attention Constraints:
*DecodeWav runs on the Ascend AI CPU, which delivers poor performance.

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator DecodeWav . \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/

REG_OP(DecodeWav)
    .INPUT(contents, TensorType({DT_STRING}))
    .OUTPUT(audio, TensorType({DT_FLOAT}))
    .OUTPUT(sample_rate, TensorType({DT_INT32}))
    .ATTR(desired_channels, Int, -1)
    .ATTR(desired_samples, Int, -1)
    .OP_END_FACTORY_REG(DecodeWav)

/**
*@brief Encode audio data using the WAV file format . \n

*@par Inputs:
*Including:
* @li audio: A Tensor of type DT_FLOAT.
* @li sample_rate: A Tensor of type DT_INT32 . \n

*@par Outputs:
*contents: A Tensor of type DT_STRING . \n

*@attention Constraints:
*EncodeWav runs on the Ascend AI CPU, which delivers poor performance.

*@par Third-party framework compatibility
*Compatible with tensorflow Operator EncodeWav . \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/

REG_OP(EncodeWav)
    .INPUT(audio, TensorType({DT_FLOAT}))
    .INPUT(sample_rate, TensorType({DT_INT32}))
    .OUTPUT(contents, TensorType({DT_STRING}))
    .OP_END_FACTORY_REG(EncodeWav)
}   // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_AUDIO_OPS_H_
