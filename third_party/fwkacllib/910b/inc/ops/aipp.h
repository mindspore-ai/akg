/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file aipp.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_AIPP_H_
#define OPS_BUILT_IN_OP_PROTO_INC_AIPP_H_

#include "graph/operator_reg.h"

namespace ge {
/**
*@brief Performs AI pre-processing (AIPP) on images including color space conversion (CSC),
image normalization (by subtracting the mean value or multiplying a factor), image cropping
(by specifying the crop start and cropping the image to the size required by the neural network), and much more. \n

*@par Inputs:
*@li images: An NCHW or NHWC tensor of type uint8, specifying the input to the data layer.
*@li params: Dynamic AIPP configuration parameters of type uint8. \n

*@par Attributes:
*aipp_config_path: A required string, specifying the path of the AIPP configuration file. \n

*@par Outputs:
*features: The AIPP-processed output tensor of type float16 or uint8.
*@par Third-party framework compatibility
* It is a custom operator. It has no corresponding operator in Caffe.
*@par Restrictions:
*Warning: This operator can be integrated only by configuring INSERT_OP_FILE of aclgrphBuildModel. Please do not use it directly.
*/
REG_OP(Aipp)
    .INPUT(images, TensorType{DT_UINT8})
    .OPTIONAL_INPUT(params, TensorType{DT_UINT8})
    .OUTPUT(features, TensorType({DT_FLOAT16, DT_UINT8}))
    .ATTR(aipp_config_path, String, "./aipp.cfg")
    .OP_END_FACTORY_REG(Aipp)

/**
*@brief Performs this op is for dynamic aipp.If you set aipp-mode to dynamic
in aipp config file, framework will auto add one input node to graph at last. \n

*@par Inputs:
*data: An NCHW or NHWC tensor of type uint8, specifying the input to the data layer. \n

*@par Attributes:
*index: specify aipp serial num \n

*@par Outputs:
*out: The AIPP-processed output tensor of all types. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator AippData.
*@par Restrictions:
*Warning: This operator can be integrated only by configuring INSERT_OP_FILE of aclgrphBuildModel. Please do not use it directly.
*/
REG_OP(AippData)
    .INPUT(data, TensorType::ALL())
    .OUTPUT(out, TensorType::ALL())
    .ATTR(index, Int, 0)
    .OP_END_FACTORY_REG(AippData)
} // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_AIPP_H_
