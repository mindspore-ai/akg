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

#ifndef GE_OP_AUDIO_OPS_H_
#define GE_OP_AUDIO_OPS_H_

#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Mel-Frequency Cepstral Coefficient (MFCC) calculation consists of \n
taking the DCT-II of a log-magnitude mel-scale spectrogram.

*@par Inputs: 
*Input "spectrogram" is a 3D tensor. Input "sample_rate" is a scalar. \n
* @li spectrogram: A 3D float tensor.
* @li sample_rate: The MFCC sample rate.

*@par Attributes: 
*@li upper_frequency_limit: The highest frequency for calculation.
*@li lower_frequency_limit: The lowest frequency for calculation.
*@li filterbank_channel_count: Resolution of the Mel bank.
*@li dct_coefficient_count: Number of output channels to produce \n
per time slice.

*@par Outputs: 
*y: A Tensor of type float32.

*@attention Constraints: \n
*Mfcc runs on the Ascend AI CPU, which delivers poor performance. \n
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
*@brief Decodes and generates spectrogram using wav float tensor.

*@par Inputs: 
*Input "x" is a 2D matrix. \n
* x: A float tensor. Float representation of audio data.

*@par Attributes: 
*@li window_size: Size of the spectrogram window.
*@li stride: Size of the spectrogram stride.
*@li magnitude_squared: If true, uses squared magnitude.

*@par Outputs: 
*spectrogram: A 3D float Tensor.

*@attention Constraints: \n
*AudioSpectrogram runs on the Ascend AI CPU, which delivers \n
poor performance.
*/

REG_OP(AudioSpectrogram)
    .INPUT(x, TensorType({DT_FLOAT}))
    .OUTPUT(spectrogram, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(window_size, Int)
    .REQUIRED_ATTR(stride, Int)
    .ATTR(magnitude_squared, Bool, false)
    .OP_END_FACTORY_REG(AudioSpectrogram)

/**
*@brief Decodes a 16-bit WAV file into a float tensor.

*@par Inputs: 
*contents: A Tensor of type string. The WAV-encoded audio, usually from a file.

*@par Attributes: 
*@li desired_channels: An optional int. Defaults to "-1". \n
Number of sample channels wanted.
*@li desired_samples: An optional int. Defaults to "-1". \n
Length of audio requested.

*@par Outputs: 
*@li *audio: A Tensor of type float32.
*@li *sample_rate: A Tensor of type int32.

*@attention Constraints: \n
*DecodeWav runs on the Ascend AI CPU, which delivers poor performance. \n
*/

REG_OP(DecodeWav)
    .INPUT(contents, TensorType({DT_STRING}))
    .OUTPUT(audio, TensorType({DT_FLOAT}))
    .OUTPUT(sample_rate, TensorType({DT_INT32}))
    .ATTR(desired_channels, Int, -1)
    .ATTR(desired_samples, Int, -1)
    .OP_END_FACTORY_REG(DecodeWav)

/**
*@brief Encode audio data using the WAV file format.

*@par Inputs:
*Including: \n
* @li audio: A Tensor of type DT_FLOAT.
* @li sample_rate: A Tensor of type DT_INT32.

*@par Outputs:
*contents: A Tensor of type DT_STRING.

*@attention Constraints:\n
*EncodeWav runs on the Ascend AI CPU, which delivers poor performance.\n

*/

REG_OP(EncodeWav)
    .INPUT(audio, TensorType({DT_FLOAT}))
    .INPUT(sample_rate, TensorType({DT_INT32}))
    .OUTPUT(contents, TensorType({DT_STRING}))
    .OP_END_FACTORY_REG(EncodeWav)
}   // namespace ge

#endif  // GE_OP_AUDIO_OPS_H_
