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
 * \file image_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_IMAGE_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_IMAGE_OPS_H_

#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Adjust the hue of one or more images . \n

*@par Inputs:
*Input images is a tensor of at least 3 dimensions. The last dimension is
interpretted as channels, and must be three. Inputs include:
*@li images:A Tensor of type float. Images to adjust. At least 3-D. The format
must be NHWC.
*@li delta:A Tensor of type float. A float delta to add to the hue . \n

*@par Outputs:
*y:A Tensor of type float. The format must be NHWC. \n

*@attention Constraints:
*Input images is a tensor of at least 3 dimensions. The last dimension is
interpretted as channels, and must be three . \n

*@par Third-party framework compatibility
*Compatible with tensorflow AdjustHue operator.
*/

REG_OP(AdjustHue)
    .INPUT(images, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(delta, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OP_END_FACTORY_REG(AdjustHue)

/**
*@brief Adjust the saturation of one or more images . \n

*@par Inputs:
*Input images is a tensor of at least 3 dimensions. The last dimension is
interpretted as channels, and must be three. Inputs include:
*@li images:A Tensor of type float. Images to adjust. At least 3-D. The format
must be NHWC.
*@li scale:A Tensor of type float. A float scale to add to the saturation . \n

*@par Outputs:
*y:A Tensor of type float. The format must be NHWC. \n

*@attention Constraints:
*Input images is a tensor of at least 3 dimensions. The last dimension is
interpretted as channels, and must be three . \n

*@par Third-party framework compatibility
*Compatible with tensorflow AdjustSaturation operator.
*/

REG_OP(AdjustSaturation)
    .INPUT(images, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OP_END_FACTORY_REG(AdjustSaturation)

/**
*@brief Adjust the contrast of one or more images . \n

*@par Inputs:
*Input images is a tensor of at least 3 dimensions. The last 3 dimensions are
interpreted as '[height, width, channels]'. Inputs include:
*@li images:A Tensor of type float. Images to adjust. At least 3-D. The format
must be NHWC.
*@li scale:A Tensor of type float. A float multiplier for adjusting contrast . \n

*@par Outputs:
*y:A Tensor of type float. The format must be NHWC. \n

*@attention Constraints:
*Input images is a tensor of at least 3 dimensions. The last dimension is
interpretted as channels, and must be three . \n

*@par Third-party framework compatibility
*Compatible with tensorflow AdjustContrast operator.
*/

REG_OP(AdjustContrast)
    .INPUT(images, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(contrast_factor, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OP_END_FACTORY_REG(AdjustContrast)

/**
*@brief Extracts crops from the input image tensor and resizes them. Extracts
crops from the input image tensor and resizes them using bilinear sampling or
nearest neighbor sampling to a common output size specified by crop_size . \n

*@par Inputs:
*Input images must be a 4-D tensor. Inputs include:
*@li images:A Tensor. Must be one of the following types:uint8, uint16, int8,
int16, int32, int64, float16, float, double. A 4-D tensor of shape
[batch, image_height, image_width, depth]. The format must be NHWC.
*@li boxes: A Tensor of type float. A 2-D tensor of shape [num_boxes, 4].
*@li box_index: A Tensor of type int32. A 1-D tensor of shape [num_boxes] with
int32 values in [0, batch).
*@li crop_size: A Tensor of type int32. A 1-D tensor of 2 elements, crop_size
= [crop_height, crop_width]. All cropped image patches are resized to this size . \n

*@par Attributes:
*@li extrapolation_value: An optional float. Defaults to 0. Value used for
extrapolation, when applicable.
*@li method: An optional string from: '"bilinear", "nearest"'. Defaults to
"bilinear". Currently two sampling methods are supported: Bilinear and
NearestNeighbor . \n

*@par Outputs:
*y:A Tensor of type float. The format must be NHWC. \n

*@attention Constraints:
*Input images must be a 4-D tensor . \n

*@par Third-party framework compatibility
*Compatible with tensorflow CropAndResize operator.
*/

REG_OP(CropAndResize)
    .INPUT(x, TensorType({DT_UINT8, DT_UINT16, DT_INT8, \
        DT_INT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(boxes, TensorType({DT_FLOAT}))
    .INPUT(box_index, TensorType({DT_INT32}))
    .INPUT(crop_size, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .ATTR(extrapolation_value, Float, 0)
    .ATTR(method, String, "bilinear")
    .OP_END_FACTORY_REG(CropAndResize)

/**
*@brief Extracts crops from the input image tensor and resizes them.
* Extracts crops from the input image tensor and resizes them using bilinear sampling or
* nearest neighbor sampling to a common output size specified by crop_size . \n

*@par Inputs:
*Input images must be a 5HD tensor. Inputs include:
*@li x:A Tensor. Must be one of the following types:float16, float. A 5HD tensor of shape
* [batch, C1, image_height, image_width, C0].
*@li boxes: A Tensor of type float. A 2-D tensor of shape [num_boxes, 4].
*@li box_index: A Tensor of type int32. A 1-D tensor of shape [num_boxes] with int32 values in [0, batch) . \n

*@par Attributes:
*@li crop_size: list int. [crop_height, crop_width]. All cropped image patches are resized to this size.
*@li extrapolation_value: An optional float. Defaults to 0. Value used for extrapolation, when applicable.
*@li method: An optional string from: '"bilinear"'. Defaults to "bilinear" . \n

*@par Outputs:
*y:A Tensor of type float . \n

*@attention Constraints:
*Input images must be a 5HD tensor . \n

*@par Third-party framework compatibility
*Compatible with tensorflow CropAndResize operator.

* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use CropAndResize instead.
*/
REG_OP(CropAndResizeD)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(boxes, TensorType({DT_FLOAT}))
    .INPUT(box_index, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(crop_size, ListInt)
    .ATTR(extrapolation_value, Float, 0)
    .ATTR(method, String, "bilinear")
    .OP_END_FACTORY_REG(CropAndResizeD)

/**
*@brief Computes the gradient of the crop_and_resize op wrt the input
boxes tensor . \n

*@par Inputs:
*Input images and grads must be a 4-D tensor. Inputs include:
*@li grads: A 4-D tensor of shape [num_boxes, crop_height, crop_width, depth].
The format must be NHWC.
*@li images: A 4-D tensor of shape [batch, image_height, image_width, depth].
The format must be NHWC.
Both image_height and image_width need to be positive.
*@li boxes: A 2-D tensor of shape [num_boxes, 4]. The i-th row of the tensor
specifies the coordinates of a box in the box_ind[i] image and is specified in
normalized coordinates [y1, x1, y2, x2].
*@li box_index: A 1-D tensor of shape [num_boxes] with int32 values in
[0, batch). The value of box_ind[i] specifies the image that the i-th box
refers to . \n

*@par Attributes:
method: A string specifying the interpolation method. Only 'bilinear' is
supported for now . \n

*@par Outputs:
*y:A 2-D tensor of shape [num_boxes, 4] . \n

*@attention Constraints:
*Input images and grads must be a 4-D tensor . \n

*@par Third-party framework compatibility
*Compatible with tensorflow CropAndResizeGradBoxes operator.
*/

REG_OP(CropAndResizeGradBoxes)
    .INPUT(grads, TensorType({DT_FLOAT}))
    .INPUT(images, TensorType({DT_UINT8, DT_UINT16, DT_INT8, DT_INT16, \
        DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(boxes, TensorType({DT_FLOAT}))
    .INPUT(box_index, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .ATTR(method, String, "bilinear")
    .OP_END_FACTORY_REG(CropAndResizeGradBoxes)

/**
*@brief Computes the gradient of the crop_and_resize op wrt the input
images tensor . \n

*@par Inputs:
*Input grads must be a 4-D tensor. Inputs include:
*@li grads: A 4-D tensor of shape [num_boxes, crop_height, crop_width, depth].
The format must be NHWC.
*@li boxes: A 2-D tensor of shape [num_boxes, 4]. The i-th row of the tensor
specifies the coordinates of a box in the box_ind[i] image and is specified
in normalized coordinates [y1, x1, y2, x2].
*@li box_index: A 1-D tensor of shape [num_boxes] with int32 values in
[0, batch). The value of box_ind[i] specifies the image that the i-th box
refers to.
*@li image_size: A 1-D tensor with value [batch, image_height, image_width,
depth] containing the original image size. Both image_height and image_width
need to be positive . \n

*@par Attributes:
method: A string specifying the interpolation method. Only 'bilinear' is
supported for now . \n

*@par Outputs:
*y:A 4-D tensor of shape [batch, image_height, image_width, depth]. The format
must be NHWC. \n

*@attention Constraints:
*Input grads must be a 4-D tensor . \n

*@par Third-party framework compatibility
*Compatible with tensorflow CropAndResizeGradImage operator.
*/

REG_OP(CropAndResizeGradImage)
    .INPUT(grads, TensorType({DT_FLOAT}))
    .INPUT(boxes, TensorType({DT_FLOAT}))
    .INPUT(box_index, TensorType({DT_INT32}))
    .INPUT(image_size, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .ATTR(method, String, "bilinear")
    .REQUIRED_ATTR(T, Type)
    .OP_END_FACTORY_REG(CropAndResizeGradImage)

/**
*@brief Extracts a glimpse from the input tensor . \n

*@par Inputs:
*Input x must be a 4-D tensor. Inputs include:
*@li x: A 4-D float tensor of shape [batch_size, height, width, channels].
The format must be NHWC.
*@li size: A 1-D tensor of 2 elements containing the size of the glimpses to
extract. The glimpse height must be specified first, following by the glimpse
width.
*@li offsets: A 2-D integer tensor of shape [batch_size, 2] containing the y,
x locations of the center of each window . \n

*@par Attributes:
*@li centered: indicates if the offset coordinates are centered relative to
the image, in which case the (0, 0) offset is relative to the center of the
input images. If false, the (0,0) offset corresponds to the upper left corner
of the input images.
*@li normalized: indicates if the offset coordinates are normalized.
*@li uniform_noise: indicates if the noise should be generated using a
uniform distribution or a Gaussian distribution.
*@li noise: indicates if the noise should uniform, gaussian, or zero.
The default is uniform which means the the noise type will be decided by
uniform_noise . \n

*@par Outputs:
*y:A tensor representing the glimpses [batch_size, glimpse_height,
glimpse_width, channels]. The format must be NHWC. \n

*@attention Constraints:
*Input x must be a 4-D tensor . \n

*@par Third-party framework compatibility
*Compatible with tensorflow CropAndResizeGradImage operator.
*/

REG_OP(ExtractGlimpse)
    .INPUT(x, TensorType({DT_FLOAT}))
    .INPUT(size, TensorType({DT_INT32}))
    .INPUT(offsets, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .ATTR(centered, Bool, true)
    .ATTR(normalized, Bool, true)
    .ATTR(uniform_noise, Bool, true)
    .ATTR(noise, String, "uniform")
    .OP_END_FACTORY_REG(ExtractGlimpse)

/**
*@brief Convert one or more images from HSV to RGB . \n

*@par Inputs:
*Last dimension of input x must be size 3. Inputs include:
*images: 1-D or higher rank. HSV data to convert. Last dimension must be size 3 . \n

*@par Outputs:
*y:images converted to RGB . \n

*@attention Constraints:
*Last dimension of input x must be size 3 . \n

*@par Third-party framework compatibility
*Compatible with tensorflow HSVToRGB operator.
*/

REG_OP(HSVToRGB)
    .INPUT(images, TensorType({DT_FLOAT16,DT_FLOAT,DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT,DT_DOUBLE}))
    .OP_END_FACTORY_REG(HSVToRGB)

/**
*@brief Resize quantized images to size using quantized bilinear interpolation . \n

*@par Inputs:
*Input images must be a 4-D tensor. Inputs include:
*@li images: 4-D with shape [batch, height, width, channels]. The format must
be NHWC.
*@li size: A 1-D int32 Tensor of 2 elements: new_height, new_width. The new
size for the images.
*@li min: A Tensor of type float.
*@li max: A Tensor of type float . \n

*@par Attributes:
*@li align_corners: An optional bool. Defaults to False. If true, the centers
of the 4 corner pixels of the input and output tensors are aligned, preserving
the values at the corner pixels. Defaults to false.
*@li half_pixel_centers: indicates if the offset coordinates are normalized . \n

*@par Outputs:
*@li resized_images: 4-D with shape [batch, new_height, new_width, channels].
The format must be NHWC.
*@li y_min: A Tensor of type float.
*@li y_max: A Tensor of type float . \n

*@attention Constraints:
*Input images and output images must be quantized types . \n

*@par Third-party framework compatibility
*Compatible with tensorflow QuantizedResizeBilinear operator.
*/

REG_OP(QuantizedResizeBilinear)
    .INPUT(images, TensorType({DT_QUINT8,DT_QINT32,DT_FLOAT}))
    .INPUT(size, TensorType({ DT_INT32 }))
    .INPUT(min, TensorType({ DT_FLOAT }))
    .INPUT(max, TensorType({ DT_FLOAT }))
    .OUTPUT(resized_images, TensorType({DT_QUINT8,DT_QINT32,DT_FLOAT }))
    .OUTPUT(y_min, TensorType({ DT_FLOAT }))
    .OUTPUT(y_max, TensorType({ DT_FLOAT }))
    .ATTR(align_corners, Bool, false)
    .ATTR(half_pixel_centers, Bool, false)
    .OP_END_FACTORY_REG(QuantizedResizeBilinear)

/**
*@brief Resize images to size using area interpolation . \n

*@par Inputs:
*Input images must be a 4-D tensor. Inputs include:
*@li images: 4-D with shape [batch, height, width, channels]. The format must
be NHWC.
*@li size: A 1-D int32 Tensor of 2 elements: new_height, new_width.
The new size for the images . \n

*@par Attributes:
*align_corners: If true, the centers of the 4 corner pixels of the input and
output tensors are aligned, preserving the values at the corner pixels.
Defaults to false . \n

*@par Outputs:
*y: 4-D with shape [batch, new_height, new_width, channels]. The format must
be NHWC. \n

*@attention Constraints:
*Input images can be of different types but output images are always float . \n

*@par Third-party framework compatibility
*Compatible with tensorflow ResizeArea operator.
*/

REG_OP(ResizeArea)
    .INPUT(images, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
        DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(size, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .ATTR(align_corners, Bool, false)
    .OP_END_FACTORY_REG(ResizeArea)

/**
*@brief Computes the gradient of bicubic interpolation . \n

*@par Inputs:
*Input grads must be a 4-D tensor. Inputs include:
*@li grads: A Tensor of type float. 4-D with shape [batch, height, width,
channels]. The format must be NHWC.
*@li original_image: A Tensor. Must be one of the following types: float,
double. 4-D with shape [batch, orig_height, orig_width, channels], The image
tensor that was resized. The format must be NHWC. \n

*@par Attributes:
*@li align_corners: An optional bool. Defaults to False. If true, the centers
of the 4 corner pixels of the input and grad tensors are aligned. Defaults to
false.
*@li half_pixel_centers: An optional bool. Defaults to False . \n

*@par Outputs:
*y: A Tensor. Has the same type as original_image. The format must be NHWC. \n

*@attention Constraints:
*Input images can be of different types but output images are always float .

*@par Third-party framework compatibility
*Compatible with tensorflow ResizeBicubicGrad operator.
*/

REG_OP(ResizeBicubicGrad)
    .INPUT(grads, TensorType({DT_FLOAT}))
    .INPUT(original_image, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE}))
    .ATTR(align_corners, Bool, false)
    .ATTR(half_pixel_centers, Bool, false)
    .OP_END_FACTORY_REG(ResizeBicubicGrad)

/**
*@brief Resize images to size using bicubic interpolation . \n

*@par Inputs:
*Input images must be a 4-D tensor. Inputs include:
*@li images: 4-D with shape [batch, height, width, channels]. The format
must be NHWC.
*@li size: A 1-D int32 Tensor of 2 elements: new_height, new_width. The new
size for the images . \n

*@par Attributes:
*@li align_corners: If true, the centers of the 4 corner pixels of the input
and output tensors are aligned, preserving the values at the corner pixels.
Defaults to false.
*@li half_pixel_centers: An optional bool. Defaults to False . \n

*@par Outputs:
*y: 4-D with shape [batch, new_height, new_width, channels]. The format
must be NHWC. \n

*@attention Constraints:
*Input images can be of different types but output images are always float .

*@par Third-party framework compatibility
*Compatible with tensorflow ResizeBicubic operator.
*/

REG_OP(ResizeBicubic)
    .INPUT(images, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
        DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(size, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .ATTR(align_corners, Bool, false)
    .ATTR(half_pixel_centers, Bool, false)
    .OP_END_FACTORY_REG(ResizeBicubic)

/**
*@brief Computes the gradient of nearest neighbor interpolation . \n

*@par Inputs:
*Input grads must be a 4-D tensor. Inputs include:
*@li grads: A Tensor. Must be one of the following types: uint8, int8, int32,
float16, float, double. Must set the format, supported format list ["NCHW, NHWC"]
*@li size: A 1-D int32 Tensor of 2 elements: orig_height, orig_width.
The original input size . \n

*@par Attributes:
*@li align_corners: An optional bool. Defaults to False. If true, the centers
of the 4 corner pixels of the input and grad tensors are aligned. Defaults to
false.
*@li half_pixel_centers: An optional bool. Defaults to False . \n

*@par Outputs:
*y: A Tensor. Has the same type as grads . \n

*@attention Constraints:
*Input grads must be a 4-D tensor . \n

*@par Third-party framework compatibility
*Compatible with tensorflow ResizeNearestNeighborV2Grad operator.
*/

REG_OP(ResizeNearestNeighborV2Grad)
    .INPUT(grads, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32,
                              DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(size, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32,
                           DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .ATTR(align_corners, Bool, false)
    .ATTR(half_pixel_centers, Bool, false)
    .OP_END_FACTORY_REG(ResizeNearestNeighborV2Grad)

/**
*@brief Computes the gradient of nearest neighbor interpolation . \n

*@par Inputs:
*Input grads must be a 4-D tensor. Inputs include:
*grads: A Tensor. 4-D with shape [batch, height, width, channels].


*@par Attributes:
*@li align_corners: An optional bool. Defaults to False. If true, the centers
of the 4 corner pixels of the input and grad tensors are aligned. Defaults to
false.
*@li size: An list type. Specify the images size . \n

*@par Outputs:
*y: A Tensor. Has the same type as grads . \n

*@par Third-party framework compatibility
*Compatible with tensorflow ResizeNearestNeighborV2GradD operator.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use ResizeNearestNeighborV2Grad instead.
*/

REG_OP(ResizeNearestNeighborV2GradD)
    .INPUT(grads, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(size, ListInt)
    .ATTR(align_corners, Bool, false)
    .ATTR(half_pixel_centers, Bool, false)
    .OP_END_FACTORY_REG(ResizeNearestNeighborV2GradD)

/**
*@brief Computes the gradient of bilinear interpolation . \n

*@par Inputs:
*Input grads must be a 4-D tensor. Inputs include:
*@li grads: A Tensor of type float32. Must set the format, supported format list ["NCHW, NHWC"]
*@li original_image: A Tensor. 4-D shape. Must set the format, supported format list ["NCHW, NHWC"]
channels], The image tensor that was resized . \n

*@par Attributes:
*align_corners: An optional bool. Defaults to False. If true, the centers of
the 4 corner pixels of the input and grad tensors are aligned. Defaults to
false . \n

*@par Outputs:
*y: A Tensor. Has the same type as original_image . \n

*@attention Constraints:
*Input grads must be a 4-D tensor . \n

*@par Third-party framework compatibility
*Compatible with tensorflow ResizeBilinearV2Grad operator.
*/

REG_OP(ResizeBilinearV2Grad)
    .INPUT(grads, TensorType({DT_FLOAT}))
    .INPUT(original_image, TensorType::FloatingDataType())
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .ATTR(align_corners, Bool, false)
    .ATTR(half_pixel_centers, Bool, false)
    .OP_END_FACTORY_REG(ResizeBilinearV2Grad)

/**
*@brief Resize images to size using bilinear interpolation . \n

*@par Inputs:
*Input images must be a 4-D tensor. Inputs include:
*@li x: 4-D tensor. Must set the format, supported format list ["NCHW, NHWC"]
*@li size: A 1-D int32 Tensor of 2 elements: new_height, new_width. The new
size for the images . \n

*@par Attributes:
*align_corners: If true, the centers of the 4 corner pixels of the input and
output tensors are aligned, preserving the values at the corner pixels.
Defaults to false . \n

*@par Outputs:
*y: 4-D with shape [batch, new_height, new_width, channels] . \n

*@attention Constraints:
*Input images can be of different types but output images are always float . \n

*@par Third-party framework compatibility
*Compatible with tensorflow ResizeBilinearV2 operator.
*/

REG_OP(ResizeBilinearV2)
    .INPUT(x, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16,
                               DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(size, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .ATTR(align_corners, Bool, false)
    .ATTR(half_pixel_centers, Bool, false)
    .OP_END_FACTORY_REG(ResizeBilinearV2)

/**
*@brief Converts one or more images from RGB to HSV . \n

*@par Inputs:
*Last dimension of input images must be size 3. Inputs include:
*images: A Tensor. Must be one of the following types: float, double. 1-D or
higher rank. RGB data to convert. Last dimension must be size 3 . \n

*@par Outputs:
*y: A Tensor. Has the same type as images . \n

*@attention Constraints:
*Outputs a tensor of the same shape as the images tensor, containing the HSV
value of the pixels. The output is only well defined if the value in images
are in [0,1] . \n

*@par Third-party framework compatibility
*Compatible with tensorflow RGBToHSV operator.
*/

REG_OP(RGBToHSV)
    .INPUT(images, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE }))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE }))
    .OP_END_FACTORY_REG(RGBToHSV)

/**
*@brief Generate a single randomly distorted bounding box for an image . \n

*@par Inputs:
*Input images must be a 4-D tensor. Inputs include:
*@li image_size: 1-D, containing [height, width, channels].
*@li bounding_boxes: 3-D with shape [batch, N, 4] describing the N bounding
boxes associated with the image.
*@li min_object_covered: The cropped area of the image must contain at least
this fraction of any bounding box supplied. The value of this parameter should
be non-negative. In the case of 0, the cropped area does not need to overlap
any of the bounding boxes supplied . \n

*@par Attributes:
*@li seed: If either seed or seed2 are set to non-zero, the random number
generator is seeded by the given seed. Otherwise, it is seeded by a random seed.
*@li seed2: A second seed to avoid seed collision.
*@li aspect_ratio_range: The cropped area of the image must have an aspect
ratio = width / height within this range.
*@li max_attempts: Number of attempts at generating a cropped region of the
image of the specified constraints. After max_attempts failures, return the
entire image.
*@li use_image_if_no_bounding_boxes: Controls behavior if no bounding boxes
supplied. If true, assume an implicit bounding box covering the whole input.
If false, raise an error . \n

*@par Outputs:
*@li begin: 1-D, containing [offset_height, offset_width, 0].
*@li size: 1-D, containing [target_height, target_width, -1].
*@li bboxes: 3-D with shape [1, 1, 4] containing the distorted bounding box . \n

*@attention Constraints:
*Input images can be of different types but output images are always float . \n

*@par Third-party framework compatibility
*Compatible with tensorflow SampleDistortedBoundingBoxExt2 operator.
*/

REG_OP(SampleDistortedBoundingBoxExt2)
    .INPUT(image_size, TensorType({ DT_UINT8, DT_INT8, DT_INT16, \
        DT_INT32, DT_INT64 }))
    .INPUT(bounding_boxes, TensorType({ DT_FLOAT }))
    .INPUT(min_object_covered, TensorType({ DT_FLOAT }))
    .OUTPUT(begin, TensorType({ DT_UINT8, DT_INT8, DT_INT16, \
        DT_INT32, DT_INT64 }))
    .OUTPUT(size, TensorType({ DT_UINT8, DT_INT8, DT_INT16, \
        DT_INT32, DT_INT64 }))
    .OUTPUT(bboxes, TensorType({ DT_FLOAT }))
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .ATTR(aspect_ratio_range, ListFloat, { 0.75f, 1.33f })
    .ATTR(area_range, ListFloat, { 0.05f, 1.0f })
    .ATTR(max_attempts, Int, 100)
    .ATTR(use_image_if_no_bounding_boxes, Bool, false)
    .OP_END_FACTORY_REG(SampleDistortedBoundingBoxExt2)

/**
*@brief Resize images to size using nearest neighbor interpolation . \n

*@par Inputs:
*Input x must be a 4-D tensor. Inputs include:
*@li x: 4-D tensor. Must set the format, supported format list ["NCHW, NHWC"].
*@li size: A 1-D int32 Tensor of 2 elements: new_height, new_width.
The new size for the images . \n

*@par Attributes:
*align_corners: If true, the centers of the 4 corner pixels of the input and
output tensors are aligned, preserving the values at the corner pixels.
Defaults to false . \n

*@par Outputs:
*y: 4-D with shape [batch, new_height, new_width, channels] . \n

*@par Third-party framework compatibility
*Compatible with tensorflow ResizeNearestNeighborV2 operator.
*/

REG_OP(ResizeNearestNeighborV2)
    .INPUT(x, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32,
                               DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(size, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32,
                           DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .ATTR(align_corners, Bool, false)
    .ATTR(half_pixel_centers, Bool, false)
    .OP_END_FACTORY_REG(ResizeNearestNeighborV2)

/**
*@brief Draw bounding boxes on a batch of images . \n

*@par Inputs:
*Input images must be a 4-D tensor. Inputs include:
*@li images: A Tensor. Must be one of the following types: float. 4-D with
shape [batch, height, width, depth]. A batch of images. The format must be NHWC.
*@li boxes: A Tensor of type float32. 3-D with shape [batch,
num_bounding_boxes, 4] containing bounding boxes . \n

*@par Outputs:
*A Tensor. Has the same type as images. The format must be NHWC. \n

*@attention Constraints:
*Input images must be a 4-D tensor . \n

*@par Third-party framework compatibility
*Compatible with tensorflow DrawBoundingBoxes operator.
*/

REG_OP(DrawBoundingBoxes)
    .INPUT(images, TensorType({DT_FLOAT}))
    .INPUT(boxes, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(DrawBoundingBoxes)

/**
*@brief Greedily selects a subset of bounding boxes in descending order of
score . \n

*@par Inputs:
*Input boxes and  scores must be float type. Inputs include:
*@li boxes: A 2-D float tensor of shape [num_boxes, 4].
*@li scores: A 1-D float tensor of shape [num_boxes] representing a single
score corresponding to each box (each row of boxes).
*@li max_output_size: A scalar integer tensor representing the maximum number
of boxes to be selected by non max suppression . \n

*@par Attributes:
*iou_threshold: A float representing the threshold for deciding whether boxes
overlap too much with respect to IOU . \n

*@par Outputs:
*selected_indices: A 1-D integer tensor of shape [M] representing the selected
indices from the boxes tensor, where M <= max_output_size . \n

*@attention Constraints:
*Input boxes and  scores must be float type . \n

*@par Third-party framework compatibility
*Compatible with tensorflow NonMaxSuppression operator.
*/

REG_OP(NonMaxSuppression)
    .INPUT(boxes, TensorType({DT_FLOAT}))
    .INPUT(scores, TensorType({DT_FLOAT}))
    .INPUT(max_output_size, TensorType({DT_INT32}))
    .OUTPUT(selected_indices, TensorType({DT_INT32}))
    .ATTR(iou_threshold, Float, 0.5f)
    .OP_END_FACTORY_REG(NonMaxSuppression)

/**
*@brief Greedily selects a subset of bounding boxes in descending order of
score . \n

*@par Inputs:
*Input boxes and  scores must be float type. Inputs include:
*@li boxes: A 2-D float tensor of shape [num_boxes, 4].
*@li scores: A 1-D float tensor of shape [num_boxes] representing a single
score corresponding to each box (each row of boxes).
*@li max_output_size: A scalar integer tensor representing the maximum number
of boxes to be selected by non max suppression.
*@li iou_threshold: A 0-D float tensor representing the threshold for deciding
whether boxes overlap too much with respect to IOU . \n

*@par Outputs:
*selected_indices: A 1-D integer tensor of shape [M] representing the selected
indices from the boxes tensor, where M <= max_output_size . \n

*@attention Constraints:
*Input boxes and  scores must be float type . \n

*@par Third-party framework compatibility
*Compatible with tensorflow NonMaxSuppressionV2 operator.
*/

REG_OP(NonMaxSuppressionV2)
    .INPUT(boxes, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(scores, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(max_output_size, TensorType({DT_INT32}))
    .INPUT(iou_threshold, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(selected_indices, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(NonMaxSuppressionV2)

/**
*@brief Greedily selects a subset of bounding boxes in descending order of
score . \n

*@par Inputs:
*Input boxes and  scores must be float type. Inputs include:
*@li boxes: A 2-D float tensor of shape [num_boxes, 4].
*@li scores: A 1-D float tensor of shape [num_boxes] representing a single
score corresponding to each box (each row of boxes).
*@li max_output_size: A scalar integer tensor representing the maximum number
of boxes to be selected by non max suppression.
*@li iou_threshold: A 0-D float tensor representing the threshold for deciding
whether boxes overlap too much with respect to IOU.
*@li score_threshold: A 0-D float tensor representing the threshold for
deciding when to remove boxes based on score . \n

*@par Outputs:
*selected_indices: A 1-D integer tensor of shape [M] representing the selected
indices from the boxes tensor, where M <= max_output_size . \n

*@attention Constraints:
*Input boxes and  scores must be float type . \n

*@par Third-party framework compatibility
*Compatible with tensorflow NonMaxSuppressionV3 operator.
*/

REG_OP(NonMaxSuppressionV3)
    .INPUT(boxes, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(scores, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(max_output_size, TensorType({DT_INT32}))
    .INPUT(iou_threshold, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(score_threshold, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(selected_indices, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(NonMaxSuppressionV3)

/**
*@brief Greedily selects a subset of bounding boxes in descending order of
score . \n

*@par Inputs:
*Input boxes and  scores must be float type. Inputs include:
*@li boxes: A 2-D float tensor of shape [num_boxes, 4].
*@li scores: A 1-D float tensor of shape [num_boxes] representing a single
score corresponding to each box (each row of boxes).
*@li max_output_size: A scalar integer tensor representing the maximum number
of boxes to be selected by non max suppression.
*@li iou_threshold: A 0-D float tensor representing the threshold for deciding
whether boxes overlap too much with respect to IOU.
*@li score_threshold: A 0-D float tensor representing the threshold for
deciding when to remove boxes based on score . \n

*@par Attributes:
*pad_to_max_output_size: If true, the output selected_indices is padded
to be of length max_output_size. Defaults to false . \n

*@par Outputs:
*@li selected_indices: A 1-D integer tensor of shape [M] representing the
selected indices from the boxes tensor, where M <= max_output_size.
*@li valid_outputs: A 0-D integer tensor representing the number of valid
elements in selected_indices, with the valid elements appearing first . \n

*@attention Constraints:
*Input boxes and  scores must be float type . \n

*@par Third-party framework compatibility
*Compatible with tensorflow NonMaxSuppressionV4 operator.
*/

REG_OP(NonMaxSuppressionV4)
    .INPUT(boxes, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(scores, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(max_output_size, TensorType({DT_INT32}))
    .INPUT(iou_threshold, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(score_threshold, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(selected_indices, TensorType({DT_INT32}))
    .OUTPUT(valid_outputs, TensorType({DT_INT32}))
    .ATTR(pad_to_max_output_size, Bool, false)
    .OP_END_FACTORY_REG(NonMaxSuppressionV4)

/**
*@brief Greedily selects a subset of bounding boxes in descending order of
score . \n

*@par Inputs:
*Input overlaps and  scores must be float type. Inputs include:
*@li overlaps: A 2-D float tensor of shape [num_boxes, num_boxes]
representing the n-by-n box overlap values.
*@li scores: A 1-D float tensor of shape [num_boxes] representing a single
score corresponding to each box (each row of boxes).
*@li max_output_size: A scalar integer tensor representing the maximum number
of boxes to be selected by non max suppression.
*@li overlap_threshold: A 0-D float tensor representing the threshold for
deciding whether boxes overlap too.
*@li score_threshold: A 0-D float tensor representing the threshold for
deciding when to remove boxes based on score . \n

*@par Attributes:
*pad_to_max_output_size: If true, the output selected_indices is padded
to be of length max_output_size. Defaults to false . \n

*@par Outputs:
*selected_indices: A 1-D integer tensor of shape [M] representing the
selected indices from the boxes tensor, where M <= max_output_size . \n

*@par Third-party framework compatibility
*Compatible with tensorflow NonMaxSuppressionWithOverlaps operator.
*/

REG_OP(NonMaxSuppressionWithOverlaps)
    .INPUT(overlaps, TensorType({DT_FLOAT}))
    .INPUT(scores, TensorType({DT_FLOAT}))
    .INPUT(max_output_size, TensorType({DT_INT32}))
    .INPUT(overlap_threshold, TensorType({DT_FLOAT}))
    .INPUT(score_threshold, TensorType({DT_FLOAT}))
    .OUTPUT(selected_indices, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(NonMaxSuppressionWithOverlaps)

/**
*@brief JPEG-encode an image . \n

*@par Inputs:
*Input image must be unit8 type. Inputs include:
*image: A 3-D uint8 Tensor of shape [height, width, channels] . \n

*@par Attributes:
*@li format: Per pixel image format.
*@li quality: Quality of the compression from 0 to 100 (higher is better
and slower).
*@li progressive: If True, create a JPEG that loads progressively (coarse
to fine).
*@li optimize_size: If True, spend CPU/RAM to reduce size with no quality
change.
*@li chroma_downsampling: A boolean, default is true.
*@li density_unit: Unit used to specify x_density and y_density: pixels per
inch ('in') or centimeter ('cm').
*@li x_density: Horizontal pixels per density unit.
*@li y_density: Vertical pixels per density unit.
*@li xmp_metadata: If not empty, embed this XMP metadata in the image header . \n

*@par Outputs:
*contents: 0-D. JPEG-encoded image . \n

*@par Third-party framework compatibility
*Compatible with tensorflow EncodeJpeg operator.
*/

REG_OP(EncodeJpeg)
    .INPUT(image, TensorType({DT_UINT8}))
    .OUTPUT(contents, TensorType({DT_STRING}))
    .ATTR(format, String, "")
    .ATTR(quality, Int, 95)
    .ATTR(progressive, Bool, false)
    .ATTR(optimize_size, Bool, false)
    .ATTR(chroma_downsampling, Bool, true)
    .ATTR(density_unit, String, "in")
    .ATTR(x_density, Int, 300)
    .ATTR(y_density, Int, 300)
    .ATTR(xmp_metadata, String, "")
    .OP_END_FACTORY_REG(EncodeJpeg)

/**
*@brief PNG-encode an image.
*@par Inputs:
*Input image must be unit8 or uint16 type. Inputs include:
*image: is a 3-D uint8 or uint16 Tensor of shape [height, width, channels]
where channels is: 1: for grayscale; 2: for grayscale + alpha; 3: for RGB;
4: for RGBA . \n

*@par Attributes:
*compression: Compression level . \n

*@par Outputs:
*contents: 0-D. PNG-encoded image . \n

*@par Third-party framework compatibility
*Compatible with tensorflow EncodePng operator.
*/

REG_OP(EncodePng)
    .INPUT(image, TensorType({DT_UINT8, DT_UINT16}))
    .OUTPUT(contents, TensorType({DT_STRING}))
    .ATTR(compression, Int, -1)
    .OP_END_FACTORY_REG(EncodePng)

/**
*@brief Resizes "images" to "size" using bilinear interpolation . \n

*@par Inputs:
* One input:
*x: An NC1HWC0 Tensor.
* Must be one of the following types: float16, float32 . \n

*@par Attributes:
*@li size: A required int32 Tensor specifying the new size for the images.
No default value.
*@li align_corners: An optional bool. If "true", the centers of the corner
pixels of the input and output tensors are aligned. Defaults to "false" . \n

*@par Outputs:
*y: A Tensor with type float32 and the same format as input "images" . \n

*@attention Constraints:
*@li The input "size" must be a tensor of 2 elements: size[0] <= 2048,
size[1] <= 2048.
*@li The input "images" must be a tensor of 5 elements: images[2] <= 2048,
images[3] <= 2048 . \n

*@par Third-party framework compatibility
* Compatible with TensorFlow operator ResizeBilinearV2D.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use ResizeBilinearV2 instead.
*/
REG_OP(ResizeBilinearV2D)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .ATTR(align_corners, Bool, false)
    .ATTR(half_pixel_centers, Bool, false)
    .REQUIRED_ATTR(size, ListInt)
    .OP_END_FACTORY_REG(ResizeBilinearV2D)

/**
*@brief Resizes "images" to "size" using bilinear interpolation and keep ratio at the time. \n

*@par Inputs:
* One input:
*images: An NC1HWC0 Tensor.
* Must be one of the following types: float16, float32 . \n

*@par Attributes:
*@li min_dimension: A required int32 attribute for the min dimension for the images.
* No default value.
*@li max_dimension: A required int32 attribute for the max dimension for the images.
* No default value.
*@li align_corners: An optional bool. If "true", the centers of the corner
* pixels of the input and output tensors are aligned. Defaults to "false".
*@li half_pixel_centers: indicates if the offset coordinates are normalized
* Defaults to "false" . \n

*@par Outputs:
*y: A Tensor with type float32 and the same format as input "images" . \n

*@attention Constraints:
* The input "images" must be a tensor of 5 elements: images[2] <= 2048,
images[3] <= 2048.
*/
REG_OP(KeepRatioResizeBilinear)
    .INPUT(images, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(min_dimension, Int)
    .REQUIRED_ATTR(max_dimension, Int)
    .ATTR(align_corners, Bool, false)
    .ATTR(half_pixel_centers, Bool, false)
    .OP_END_FACTORY_REG(KeepRatioResizeBilinear)

/**
*@brief Resizes "images" to "size" using nearest neighbor interpolation . \n

*@par Inputs:
* One input:
*x: An NC1HWC0 Tensor.
* Must be one of the following types: float16, float32, int32, int8, uint8

*@par Attributes:
*@li size: A required int32 Tensor specifying the new size for the images.
No default value.
*@li align_corners: An optional bool. If "true", the centers of the corner
pixels of the input and output tensors are aligned. Defaults to "false" . \n

*@par Outputs:
*y: A Tensor with the same type and format as input "images" . \n

*@attention Constraints:
* The input "size" must be a tensor of 2 elements: size[0] <= 7680,
size[1] <= 4320

*@par Third-party framework compatibility
* Compatible with TensorFlow operator ResizeNearestNeighborV2.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use ResizeNearestNeighborV2 instead.
*/
REG_OP(ResizeNearestNeighborV2D)
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .REQUIRED_ATTR(size, ListInt)
    .ATTR(align_corners, Bool, false)
    .ATTR(half_pixel_centers, Bool, false)
    .OP_END_FACTORY_REG(ResizeNearestNeighborV2D)

/**
*@brief Extract the shape information of a JPEG-encoded image . \n

*@par Inputs:
*Input contents must be 0-D. Inputs include:
*contents: 0-D. The JPEG-encoded image . \n

*@par Attributes:
*output_type: The output type of the operation (int32 or int64). Defaults
to int32 . \n

*@par Outputs:
*image_shape: 1-D. The image shape with format [height, width, channels] . \n

*@par Third-party framework compatibility
*Compatible with tensorflow ExtractJpegShape operator.
*/

REG_OP(ExtractJpegShape)
    .INPUT(contents, TensorType({DT_STRING}))
    .OUTPUT(image_shape, TensorType({DT_INT32, DT_INT64}))
    .REQUIRED_ATTR(output_type, Type)
    .OP_END_FACTORY_REG(ExtractJpegShape)

/**
*@brief Draw bounding boxes on a batch of images . \n

*@par Inputs:
*@li images: 4-D with shape `[batch, height, width, depth]`.
A batch of images.
*@li boxes: 3-D with shape `[batch, num_bounding_boxes, 4]`
containing bounding boxes.
*@li colors: 2-D. A list of RGBA colors to cycle through for the boxes . \n

*@par Outputs:
*y: Returns 4-D with the same shape as `images`.
The batch of input images with bounding boxes drawn on the images . \n

*@par Third-party framework compatibility
* Compatible with tensorflow DrawBoundingBoxesV2 operator.
*/

REG_OP(DrawBoundingBoxesV2)
    .INPUT(images, TensorType({DT_FLOAT}))
    .INPUT(boxes, TensorType({DT_FLOAT}))
    .INPUT(colors, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(DrawBoundingBoxesV2)

/**
*@brief Greedily selects a subset of bounding boxes in descending order of score,
pruning away boxes that have high intersection-over-union (IOU) overlap
with previously selected boxes . \n

*@par Inputs:
*@li boxes: A 2-D float tensor of shape `[num_boxes, 4]`.
*@li scores: A 1-D float tensor of shape `[num_boxes]` representing a single
score corresponding to each box (each row of boxes).
*@li max_output_size: A scalar integer tensor representing the maximum number of
boxes to be selected by non max suppression.
*@li iou_threshold: A 0-D float tensor representing the threshold for deciding whether
boxes overlap too much with respect to IOU.
*@li score_threshold: A 0-D float tensor representing the threshold for deciding when to
remove boxes based on score.
*@li soft_nms_sigma: A 0-D float tensor representing the sigma parameter for Soft NMS . \n

*@par Attributes:
pad_to_max_output_size: If true, the output `selected_indices` is padded to be of length
`max_output_size`. Defaults to false. If not specified, defaults to false . \n

*@par Outputs:
*@li selected_indices: A 1-D integer tensor of shape [M] representing the
selected indices from the boxes tensor, where M <= max_output_size.
*@li selected_scores: A 1-D float tensor of shape `[M]` representing the corresponding
scores for each selected box, where `M <= max_output_size`.
*@li valid_outputs: A 0-D integer tensor representing the number of valid
elements in selected_indices, with the valid elements appearing first . \n

*@par Third-party framework compatibility
* Compatible with tensorflow NonMaxSuppressionV5 operator.
*/

REG_OP(NonMaxSuppressionV5)
    .INPUT(boxes, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(scores, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(max_output_size, TensorType({DT_INT32}))
    .INPUT(iou_threshold, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(score_threshold, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(soft_nms_sigma, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(selected_indices, TensorType({DT_INT32}))
    .OUTPUT(selected_scores, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(valid_outputs, TensorType({DT_INT32}))
    .ATTR(pad_to_max_output_size, Bool, false)
    .REQUIRED_ATTR(T, Type)
    .OP_END_FACTORY_REG(NonMaxSuppressionV5)

/**
*@brief Resizes "images" to "size" by scale and translate . \n

*@par Inputs:
*@li images: A `Tensor`. Must be one of the following types: `int8`, `uint8`,
`int16`, `uint16`, `int32`, `int64`, `bfloat16`, `float32`, `float64`.
*@li size: A `Tensor` of type `int32`.
*@li scale: A `Tensor` of type `float32`.
*@li translation: A `Tensor` of type `float32` . \n

*@li kernel_type: type is string, default  lanczos3
*@li antialias: type is bool, default true \n

*@par Outputs:
*y: A Tensor with type float32 . \n

*@par Third-party framework compatibility
* Compatible with TensorFlow ScaleAndTranslate operator.
*/

REG_OP(ScaleAndTranslate)
    .INPUT(images, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16,
                               DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(size, TensorType({DT_INT32}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(translation, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .ATTR(kernel_type, String, "lanczos3")
    .ATTR(antialias, Bool, true)
    .OP_END_FACTORY_REG(ScaleAndTranslate)

/**
*@brief Computes the gradient by scale and translate . \n

*@par Inputs:
*@li grads: A `Tensor`. Must be one of the following types: `float32`.
*@li original_image: A `Tensor`. Must have the same type as `grads`.
*@li scale: A `Tensor` of type `float32`.
*@li translation: A `Tensor` of type `float32` . \n

*@li kernel_type: type is string, default  lanczos3
*@li antialias: type is bool, default true

*@par Outputs:
*y: A `Tensor`. Has the same type as `grads` . \n

*@par Third-party framework compatibility
* Compatible with TensorFlow ScaleAndTranslateGrad operator.
*/

REG_OP(ScaleAndTranslateGrad)
    .INPUT(grads, TensorType({DT_FLOAT}))
    .INPUT(original_image, TensorType({DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(translation, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .ATTR(kernel_type, String, "lanczos3")
    .ATTR(antialias, Bool, true)
    .OP_END_FACTORY_REG(ScaleAndTranslateGrad)

/**
*@brief Greedily selects a subset of bounding boxes in descending order of score,
This operation performs non_max_suppression on the inputs per batch, across all classes . \n

*@par Inputs:
*@li boxes: A 4-D float tensor of shape `[batch_size, num_boxes, q, 4]`. If `q` is 1 then
same boxes are used for all classes otherwise, if `q` is equal to number of
classes, class-specific boxes are used.
*@li scores: A 3-D float tensor of shape `[batch_size, num_boxes, num_classes]`
representing a single score corresponding to each box (each row of boxes).
*@li max_output_size_per_class: A scalar integer tensor representing the maximum number of
boxes to be selected by non max suppression per class.
*@li max_total_size: A scalar representing maximum number of boxes retained over all classes.
*@li iou_threshold: A 0-D float tensor representing the threshold for deciding whether
boxes overlap too much with respect to IOU.
*@li score_threshold: A 0-D float tensor representing the threshold for deciding when to remove
boxes based on score . \n

*@par Attributes:
*@li pad_per_class: If false, the output nmsed boxes, scores and classes
are padded/clipped to `max_total_size`. If true, the
output nmsed boxes, scores and classes are padded to be of length
`max_size_per_class`*`num_classes`, unless it exceeds `max_total_size` in
which case it is clipped to `max_total_size`. Defaults to false.
*@li clip_boxes: If true, assume the box coordinates are between [0, 1] and clip the output boxes
if they fall beyond [0, 1]. If false, do not do clipping and output the box
coordinates as it is. If not specified, defaults to true . \n

*@par Outputs:
*nmsed_boxes:type is float
*nmsed_scores:type is float
*nmsed_classes:type is float  \n

*@par Third-party framework compatibility
* Compatible with tensorflow CombinedNonMaxSuppression operator.
*/

REG_OP(CombinedNonMaxSuppression)
    .INPUT(boxes, TensorType({DT_FLOAT}))
    .INPUT(scores, TensorType({DT_FLOAT}))
    .INPUT(max_output_size_per_class, TensorType({DT_INT32}))
    .INPUT(max_total_size, TensorType({DT_INT32}))
    .INPUT(iou_threshold, TensorType({DT_FLOAT}))
    .INPUT(score_threshold, TensorType({DT_FLOAT}))
    .OUTPUT(nmsed_boxes, TensorType({DT_FLOAT}))
    .OUTPUT(nmsed_scores, TensorType({DT_FLOAT}))
    .OUTPUT(nmsed_classes, TensorType({DT_FLOAT}))
    .OUTPUT(valid_detections, TensorType({DT_INT32}))
    .ATTR(pad_per_class, Bool, false)
    .ATTR(clip_boxes, Bool, true)
    .OP_END_FACTORY_REG(CombinedNonMaxSuppression)

/**
*@brief Function spatial transformer . \n

*@par Inputs:
*@li x: A Tensor dtype of float16, float32.
*@li theta: A Tensor dtype of float16, float32, auxiliary coefficients . \n

*@par Attributes:
*@li output_size: A tuple output size.
*@li default_theta: A tuple default theta
*@li use_default_theta: List use default theta
*@li align_corners: Align corners

*@par Outputs:
*y: A Tensor dtype of float16, float32, should be same shape and type as x.
*/
REG_OP(SpatialTransformerD)
    .INPUT(x, TensorType({DT_FLOAT,DT_FLOAT16}))
    .OPTIONAL_INPUT(theta, TensorType({DT_FLOAT,DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT,DT_FLOAT16}))
    .ATTR(output_size, ListInt, {-1, -1})
    .ATTR(default_theta, ListFloat, {})
    .ATTR(align_corners, Bool, false)
    .ATTR(use_default_theta, ListBool, {})
    .OP_END_FACTORY_REG(SpatialTransformerD)

/**
* @brief Resize the input tensor. \n
currently, only support resize image tensor using nearest neighbor and linear interpolation.

* @par Inputs:
* Input x must be a 4-D tensor. Inputs include: \n
* @li x: A Tensor. Must be one of the following types: uint8, int8, int16, \n
int32, int64, float16, float, double. 4-D with shape [batch, height, width, channels] \n
or shape [batch, channels, height, width].
* @li roi: A 1-D float Tensor. only takes effect when attr coordinate_transformation_mode \n
is "tf_crop_and_resize"
* @li scales: A 1-D float Tensor, the scale array along each dimension, Only one of \n
'scales' and 'sizes' can be specified.
* @li sizes: A 1-D int64 Tensor, The size of the output tensor. nly one of \n
'scales' and 'sizes' can be specified.  If 'size' is specified, then set scales \n
to empty data (zero shape) in this operator's input list.

* @par Attributes:
* @li coordinate_transformation_mode: String. Defaults to half_pixel. how to transform \n
the coordinate in the resized tensor to the coordinate in the original tensor. \n
other optional: pytorch_half_pixel, align_corners, asymmetric, tf_half_pixel_for_nn, \n
tf_crop_and_resize.
* @li cubic_coeff_a: Float. Defaults to -0.75, only used in cubic interpolation. \n
other optional: -0.5
* @li exclude_outside: Int. Defaults to 0, If set to 1, the weight of sampling \n
locations outside the tensor will be set to 0 and the weight will be renormalized \n
so that their sum is 1.0.
* @li extrapolation_value: Float. Defaults to 0.0f. When coordinate_transformation_mode \n
is "tf_crop_and_resize" and x_original is outside the range [0, length_original - 1], \n
this value is used as the corresponding output value.
* @li mode: String. Defaults to nearest. Three interpolation modes: nearest (default), \n
linear and cubic.
* @li nearest_mode: String. Defaults to round_prefer_floor. Four modes: round_prefer_floor, \n
round_prefer_ceil, floor, ceil. Only used by nearest interpolation.

* @par Outputs:
* y: A Tensor. Has the same type as x.

* @attention Constraints: \n
* Input x must be a 4-D tensor.

* @par Third-party framework compatibility
* Compatible with tensorflow ResizeNearestNeighborV2 operator.
*/

REG_OP(Resize)
    .INPUT(x, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32,
                                DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(roi, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(scales, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(sizes, TensorType({DT_INT64}))
    .OUTPUT(y, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32,
                                DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .ATTR(coordinate_transformation_mode, String, "half_pixel")
    .ATTR(cubic_coeff_a, Float, -0.75)
    .ATTR(exclude_outside, Int, 0)
    .ATTR(extrapolation_value, Float, 0)
    .ATTR(mode, String, "nearest")
    .ATTR(nearest_mode, String, "round_prefer_floor")
    .OP_END_FACTORY_REG(Resize)
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_IMAGE_OPS_H_
