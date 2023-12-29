/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
 * \file image.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_IMAGE_H_
#define OPS_BUILT_IN_OP_PROTO_INC_IMAGE_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief LUT3D
* calculating: img, lut_table, lut_img
*    c = img.shape[- 1]
*    n_h_w = reduce(lambda x, y: x * y, img.shape[:-1])
*    lut_table_n = lut_table.shape[0]
*    tensor_img = img.reshape(n_h_w, c)
*    tensor_lut = np.cast["float32"](lut_table.reshape(lut_table_n * lut_table_n * lut_table_n, 3))
*    img_fp32 = np.cast["float32"](tensor_img)
*    img_fp32 = img_fp32 * (lut_table_n - 1) / 255.
*
*    b_floor = np.cast["int32"](np.floor(img_fp32[:, 0]))
*    b_ceil = np.cast["int32"](np.ceil(img_fp32[:, 0]))
*    g_floor = np.cast["int32"](np.floor(img_fp32[:, 1]))
*    g_ceil = np.cast["int32"](np.ceil(img_fp32[:, 1]))
*    r_floor = np.cast["int32"](np.floor(img_fp32[:, 2]))
*    r_ceil = np.cast["int32"](np.ceil(img_fp32[:, 2]))
*
*    b_fl_idx = b_floor * lut_table_n * lut_table_n
*    b_cl_idx = b_ceil * lut_table_n * lut_table_n
*    g_fl_idx = g_floor * lut_table_n
*    g_cl_idx = g_ceil * lut_table_n
*    r_fl_idx = r_floor
*    r_cl_idx = r_ceil
*
*    tensor_index1 = b_fl_idx + g_fl_idx + r_fl_idx
*    tensor_index2 = b_cl_idx + g_fl_idx + r_fl_idx
*    tensor_index3 = b_fl_idx + g_cl_idx + r_fl_idx
*    tensor_index4 = b_fl_idx + g_fl_idx + r_cl_idx
*    tensor_index5 = b_fl_idx + g_cl_idx + r_cl_idx
*    tensor_index6 = b_cl_idx + g_fl_idx + r_cl_idx
*    tensor_index7 = b_cl_idx + g_cl_idx + r_fl_idx
*    tensor_index8 = b_cl_idx + g_cl_idx + r_cl_idx
*
*    lut_tensor1 = tensor_lut[tensor_index1]
*    lut_tensor2 = tensor_lut[tensor_index2]
*    lut_tensor3 = tensor_lut[tensor_index3]
*    lut_tensor4 = tensor_lut[tensor_index4]
*    lut_tensor5 = tensor_lut[tensor_index5]
*    lut_tensor6 = tensor_lut[tensor_index6]
*    lut_tensor7 = tensor_lut[tensor_index7]
*    lut_tensor8 = tensor_lut[tensor_index8]
*
*    fract_b = np.repeat((img_fp32[:, 0] - b_floor)[:, None], 3, 1)
*    fract_b_1 = 1 - fract_b
*    fract_g = np.repeat((img_fp32[:, 1] - g_floor)[:, None], 3, 1)
*    fract_g_1 = 1 - fract_g
*    fract_r = np.repeat((img_fp32[:, 2] - r_floor)[:, None], 3, 1)
*    fract_r_1 = 1 - fract_r
*
*    lut_img = ((lut_tensor1 * fract_r_1 + lut_tensor4 * fract_r) * fract_g_1 \
*              + (lut_tensor3 * fract_r_1 + lut_tensor5 * fract_r) * fract_g) * fract_b_1 \
*              + ((lut_tensor2 * fract_r_1 + lut_tensor6 * fract_r) * fract_g_1 \
*              + (lut_tensor7 * fract_r_1 + lut_tensor8 * fract_r) * fract_g) * fract_b
*
* @par Inputs:
* Two inputs, including:
* @li img: A Tensor. Must be one of the following types: uint8, float32.
* @li lut_table: A Tensor. Has the same type as "img" . \n

* @par Outputs:
* lut_img: A Tensor of type float32. Has the same shape as "img" . \n
*/
REG_OP(LUT3D)
    .INPUT(img, TensorType({DT_UINT8, DT_FLOAT}))
    .INPUT(lut_table, TensorType({DT_UINT8, DT_FLOAT}))
    .OUTPUT(lut_img, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(LUT3D)

/**
* @Adjust the brightness of one or more images . \n

* @Inputs:
* Input images is a tensor of at least 3 dimensions. The last dimension is
interpretted as channels, and must be three. Inputs include:
* @ images:A Tensor of type float32 or float16. Images to adjust. At least 3-D.
* @ delta:A Tensor of type float32. Add delta to all components of the tensor image . \n

* @ Outputs:
* y:A Tensor of type float32 or float16 . \n

* @attention Constraints:
* Input images is a tensor of at least 3 dimensions. The last dimension is
interpretted as channels, and must be three . \n

* @ Third-party framework compatibility
* Compatible with tensorflow AdjustBrightness operator.
*/
REG_OP(AdjustBrightness)
    .INPUT(images, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(delta, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(AdjustBrightness)

/**
* @ Adjust the brightnessV2 of one or more images . \n

* @ Inputs:
* Input images is a tensor of at least 3 dimensions. The last dimension is
interpretted as channels, and must be three. Inputs include:
* @ images:A Tensor of type uint8. Images to adjust. At least 3-D.
* @ factor:A Tensor of type float. Multiply factor to all components of the tensor image . \n

* @ Outputs:
* y:A Tensor of type uint8 . \n

* @attention Constraints:
* Input images is a tensor of at least 3 dimensions. The last dimension is
interpretted as channels, and must be three . \n

* @ Third-party framework compatibility
* Compatible with tensorflow AdjustBrightness operator.
*/
REG_OP(AdjustBrightnessV2)
    .INPUT(images, TensorType({DT_UINT8}))
    .INPUT(factor, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_UINT8}))
    .OP_END_FACTORY_REG(AdjustBrightnessV2)

/**
* @brief Adjust the contrast of images for DVPP with mean, need mean input. \n

* @par Inputs:
* Input images is a tensor of at least 3 dimensions. The last 3 dimensions are
interpreted as '[height, width, channels]'. Inputs include:
* @li images: A Tensor of type float. Images to adjust. At least 3-D. The format
must be NHWC.
* @li mean: A Tensor of type float.Indicates the average value of each channel. \n
* @li contrast_factor: A Tensor of type float. A float multiplier for adjusting contrast . \n

* @par Outputs:
* y: A Tensor of type float. The format must be NHWC. \n

* @par Attributes:
* @li data_format: An optional string. Could be "HWC" or "CHW". Defaults to "HWC".
Value used for inferring real format of images.
* @attention Constraints:
* Input images is a tensor of at least 3 dimensions. The last dimension is
interpretted as channels, and must be three . \n

* @par Third-party framework compatibility
* Compatible with tensorflow AdjustContrast operator.
*/
REG_OP(AdjustContrastWithMean)
    .INPUT(images, TensorType({DT_FLOAT16,DT_FLOAT,DT_UINT8}))
    .INPUT(mean, TensorType({DT_FLOAT16,DT_FLOAT,DT_UINT8}))
    .INPUT(contrast_factor, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT,DT_UINT8}))
    .ATTR(data_format, String, "HWC")
    .OP_END_FACTORY_REG(AdjustContrastWithMean)

/**
* @brief Convert an RGB image to a gray image. \n

* @par Inputs:
* Input images is a tensor of at least 3 dimensions. The last 3 dimensions are
interpreted as '[height, width, channels]'. Inputs include:
* @li images: A Tensor of type T. Images to adjust. At least 3-D. The format
must be NHWC or NCHW.

* @par Outputs:
* y: A Tensor of type T. The format must be NHWC or NCHW. \n

* @par Attributes:
* @li output_channels: An optional int. Could be 1 or 3. Defaults to 1.
Value used for different mean mode.
* @li data_format: An optional string. Could be "HWC" or "CHW". Defaults to "HWC".
Value used for inferring real format of images.

* @par DataType:
* @li T: type of uint8 or float32. \n

* @attention Constraints:
* Input images is a tensor of at least 3 dimensions. The last dimension is
interpretted as channels, and must be three. \n
*/
REG_OP(RgbToGrayscale)
    .INPUT(images, "T")
    .ATTR(data_format, String, "HWC")
    .ATTR(output_channels, Int, 1)
    .OUTPUT(y, "T")
    .DATATYPE(T, TensorType({DT_UINT8, DT_FLOAT}))
    .OP_END_FACTORY_REG(RgbToGrayscale)

/**
* @ Another method to adjust the saturation of one or more images. \n

* @ Inputs:
* Input images is a tensor of at least 3 dimensions. The last dimension is
interpretted as channels, and must be three. Inputs include:
* @ images:A Tensor of type uint8. Images to adjust. At least 3-D. The format
could be NHWC or NCHW.
* @ scale:A Tensor of type float. A float scale used in blending operation to adjust the saturation . \n

* @ Outputs:
* y:A Tensor of type uint8. The format could be NHWC or NCHW. \n

* @par Attributes:
* @li data_format: An optional string. Could be "HWC" or "CHW". Defaults to "CHW".
Value used for inferring real format of images.

* @attention Constraints:
* Input images is a tensor of at least 3 dimensions. The last dimension is
interpretted as channels, and must be three . \n

* @ Third-party framework compatibility
* Compatible with Pytorch AdjustSaturation operator.
*/

REG_OP(AdjustSaturationV2)
    .INPUT(images, TensorType({DT_UINT8}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_UINT8}))
    .ATTR(data_format, String, "CHW")
    .OP_END_FACTORY_REG(AdjustSaturationV2)

/**
* @brief Resize images to size using trilinear interpolation . \n

* @par Inputs:
* Input images must be a 5-D tensor. Inputs include:
* @li x: 5-D tensor. Must set the format, supported format list ["NCDHW, NDHWC"]
* @li size: A 1-D int32 Tensor of 3 elements: new_depth, new_height, new_width. The new
size for the images . \n

* @par Attributes:
* @li align_corners: If true, the centers of the 8 corner pixels of the input and
output tensors are aligned, preserving the values at the corner pixels.
Defaults to false .
* @li half_pixel_centers: An optional bool. Defaults to false . \n
* @par Outputs:
* y: 5-D with shape [batch, channels, new_depth, new_height, new_width] . \n

* @par Third-party framework compatibility
* Compatible with onnx Resize operator using trilinear interpolation.
*/

REG_OP(ResizeTrilinear)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(size, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(align_corners, Bool, false)
    .ATTR(half_pixel_centers, Bool, false)
    .OP_END_FACTORY_REG(ResizeTrilinear)

/**
* @brief Applies a affine transformation to an image. \n

* @par Inputs:
* @li x: An tensor of at least 3 dimensions, type must be float32 or uint8.
* @li matrix: transformation matrix, format ND , shape must be (2, 3), type must be float32. \n

* @par Attributes:
* @li dst_size: Required int32 and int64, shape must be (1, 2), specifying the size of the output image.
* Must be greater than "0".
* @li interpolation: An optional string. Used to select interpolation type.
only support "bilinear"/"nearest"/"cubic"/"area", default "bilinear".
* @li border_type: An optional string. Pixel extension method, currently only support "constant", default "constant".
* @li border_value: An optional float. Used when border_type is "constant".
Data type is the same as that of the original picture. The number of data is the same as that of the original
picture channels. Default value is 0.
* @li data_format: An optional string. Could be "HWC" or "CHW". Defaults to "HWC".
                   Value used for inferring real format of images. \n

* @par Outputs:
* y: output tensor of at least 3 dimensions, type must be float32 or uint8.
*/
REG_OP(WarpAffineV2)
    .INPUT(x, TensorType({DT_FLOAT, DT_UINT8}))
    .INPUT(matrix, TensorType({DT_FLOAT}))
    .INPUT(dst_size, TensorType({DT_INT32,DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_UINT8}))
    .ATTR(interpolation, String, "bilinear")
    .ATTR(border_type, String, "constant")
    .ATTR(border_value, Float, 0.0)
    .ATTR(data_format, String, "HWC")
    .OP_END_FACTORY_REG(WarpAffineV2)

/**
* @brief change an image size. \n

* @par Inputs:
* @li x: An tensor of at least 3 dimensions, type must be float32 or uint8.
* @li dst_size: Required int32 and int64, shape must be (1, 2), specifying the size of the output image. \n

* @par Attributes:
* @li interpolation: An optional string. Interpolation type, only support "bilinear"/"nearest"/"cubic"/"area",
default "nearest".
* @li data_format: An optional string. Could be "HWC" or "CHW". Defaults to "HWC".
                   Value used for inferring real format of images. \n

* @par Outputs:
* y: output tensor of at least 3 dimensions, type must be float32 or uint8.
*/
REG_OP(ResizeV2)
    .INPUT(x, TensorType({DT_FLOAT, DT_UINT8}))
    .INPUT(dst_size, TensorType({DT_INT32,DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_UINT8}))
    .ATTR(interpolation, String, "nearest")
    .ATTR(data_format, String, "HWC")
    .OP_END_FACTORY_REG(ResizeV2)

/**
* @brief Applies a gaussian blur to an image. \n

* @par Inputs:
* @li x: An NCHW or NHWC tensor of type T. \n
* @li matrix: transformation matrix, format ND , shape must be (2, 3), type must be float32. \n

* @par Attributes:
* @li kernel_size: A required ListInt.
* contain 2 elements: [size_width, size_height].
* every element must be 1 or 3 or 5.
* @li sigma: A required ListFloat.
* contain 2 elements: [sigma_x, sigma_y].
* @li padding_mode: An optional string. padding mode, only support "constant" and "reflect", default "constant". \n

* @par DataType:
* @li T: type of uint8 or float32. \n

* @par Outputs:
* y: output tensor, has the same type and shape as input x.
*/
REG_OP(GaussianBlur)
    .INPUT(x, "T")
    .OUTPUT(y, "T")
    .REQUIRED_ATTR(kernel_size, ListInt)
    .REQUIRED_ATTR(sigma, ListFloat)
    .ATTR(padding_mode, String, "constant")
    .DATATYPE(T, TensorType({DT_UINT8, DT_FLOAT}))
    .OP_END_FACTORY_REG(GaussianBlur)

/**
* @brief Applies a rotate to an image. \n

* @par Inputs:
* @li x: An NHWC or NCHW tensor of type T. \n

* @par Attributes:
* @li angle: An required float attr. In degress counter clockwise.
* @li center: An optional ListInt, center of rotation. Origin is the upper left corner.
Default is the center of the image.
* @li expand: An optional Bool, expansion flag. If true, expands the output image to make it large enough to hold the
              entire rotated image. If false or omitted, make the output image the same size as the input image.
              Note that the expand flag assumes rotation around the center and no translation.
* @li interpolation: An optional string. Interpolation type, only support "bilinear"/"nearest", default "nearest".
* @li padding_mode: An optional string. Pixel extension method, only support "constant" and "edge", default "constant".
* @li padding_value: An optional float. Used when padding_mode is "constant". Data type is the same as that of the
original picture. The number of data is the same as that of the original picture channels. Deatulat value is 0 . \n
* @li data_format: An optional string. Could be "HWC" or "CHW". Defaults to "HWC".
Value used for inferring real format of images.

* @par Outputs:
* y: output tensor, NHWC or NCHW, type must be T.

* @par DataType:
* @li T: type of uint8 or float32. \n
*/
REG_OP(Rotate)
    .INPUT(x, "T")
    .OUTPUT(y, "T")
    .REQUIRED_ATTR(angle, Float)
    .ATTR(center, ListInt, {})
    .ATTR(expand, Bool, false)
    .ATTR(interpolation, String, "nearest")
    .ATTR(padding_mode, String, "constant")
    .ATTR(padding_value, Float, 0.0)
    .ATTR(data_format, String, "HWC")
    .DATATYPE(T, TensorType({DT_UINT8, DT_FLOAT}))
    .OP_END_FACTORY_REG(Rotate)

/**
* @brief Applies crop to image. \n

* @par Inputs:
* @li x: An tensor of at least 3 dimensions, type must be float32 or uint8. \n
* @li boxes: A Tensor of type uint32 or int32. A 2-D tensor of shape
[num_boxes, 4], 4 numbers represent [left, top, left+width, top+height].
* @li box_index: A Tensor of type uint32 or int32. A 1-D tensor of shape
[num_boxes] with int32 values in [0, batch).\n

* @par Attributes:
* @li data_format: An optional string. Could be "HWC" or "CHW". Defaults to
"CHW". Value used for inferring real format of images.

* @par Outputs:
* y: output tensor, NHWC or NCHW, type must be float32 or uint8. \n
*/
REG_OP(ImgCrop)
    .INPUT(x, TensorType({DT_FLOAT, DT_UINT8}))
    .INPUT(boxes, TensorType({DT_UINT32, DT_INT32}))
    .INPUT(box_index, TensorType({DT_UINT32, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_UINT8}))
    .ATTR(data_format, String, "CHW")
    .OP_END_FACTORY_REG(ImgCrop)
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_IMAGE_H_