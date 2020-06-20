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

#ifndef GE_OP_AIPP_H
#define GE_OP_AIPP_H

#include "graph/operator_reg.h"

namespace ge {
/**
*@brief Performs AI pre-processing (AIPP) on images including color space conversion (CSC), image normalization (by subtracting the mean value or multiplying a factor), image cropping (by specifying the crop start and cropping the image to the size required by the neural network), and much more.

*@par Inputs:
*@li images: An NCHW or NHWC tensor of type uint8, specifying the input to the data layer.
*@li params: Dynamic AIPP configuration parameters of type uint8.

*@par Attributes:
*aipp_config_path: A required string, specifying the path of the AIPP configuration file

*@par Outputs:
*features: The AIPP-processed output tensor of type float16 or uint8.
*/
REG_OP(Aipp)
    .INPUT(images, TensorType{DT_UINT8})
    .OPTIONAL_INPUT(params, TensorType{DT_UINT8})
    .OUTPUT(features, TensorType({DT_FLOAT16, DT_UINT8}))
    .ATTR(aipp_config_path, String, "./aipp.cfg")
    .OP_END_FACTORY_REG(Aipp)
} // namespace ge

/**
*@brief Performs This op is for dynamic aipp.If you set aipp-mode to dynamic in aipp config file, framework will auto add one input node to graph at last.

*@par Attributes:
*index: specify aipp serial num

*@par Outputs:
*features: The AIPP-processed output tensor of all types.
*/
namespace ge {
REG_OP(AippData)
    .INPUT(data, TensorType::ALL())
    .OUTPUT(out, TensorType::ALL())
    .ATTR(index, Int, 0)
    .OP_END_FACTORY_REG(AippData)
}

#endif // GE_OP_AIPP_H
