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

#ifndef GE_OP_MAGE_OPS_H_
#define GE_OP_MAGE_OPS_H_

#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Adjust the hue of one or more images.

*@par Inputs:
*Input images is a tensor of at least 3 dimensions. The last dimension is \n
interpretted as channels, and must be three. Inputs include: \n
*@li images:A Tensor of type float. Images to adjust. At least 3-D.
*@li delta:A Tensor of type float. A float delta to add to the hue.

*@par Outputs:
*y:A Tensor of type float.

*@attention Constraints: \n
*Input images is a tensor of at least 3 dimensions. The last dimension is \n
interpretted as channels, and must be three.

*/

REG_OP(AdjustHue)
    .INPUT(images, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(delta, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OP_END_FACTORY_REG(AdjustHue)

/**
*@brief Adjust the saturation of one or more images.

*@par Inputs:
*Input images is a tensor of at least 3 dimensions. The last dimension is \n
interpretted as channels, and must be three. Inputs include: \n
*@li images:A Tensor of type float. Images to adjust. At least 3-D.
*@li scale:A Tensor of type float. A float scale to add to the saturation.

*@par Outputs:
*y:A Tensor of type float.

*@attention Constraints: \n
*Input images is a tensor of at least 3 dimensions. The last dimension is \n
interpretted as channels, and must be three.

*/

REG_OP(AdjustSaturation)
    .INPUT(images, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OP_END_FACTORY_REG(AdjustSaturation)

/**
*@brief Adjust the contrast of one or more images.

*@par Inputs:
*Input images is a tensor of at least 3 dimensions. The last 3 dimensions are \n
interpreted as '[height, width, channels]'. Inputs include: \n
*@li images:A Tensor of type float. Images to adjust. At least 3-D.
*@li scale:A Tensor of type float. A float multiplier for adjusting contrast.

*@par Outputs:
*y:A Tensor of type float.

*@attention Constraints: \n
*Input images is a tensor of at least 3 dimensions. The last dimension is \n
interpretted as channels, and must be three.

*/

REG_OP(AdjustContrast)
    .INPUT(images, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(contrast_factor, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OP_END_FACTORY_REG(AdjustContrast)

/**
*@brief Extracts crops from the input image tensor and resizes them. Extracts \n
crops from the input image tensor and resizes them using bilinear sampling or \n
nearest neighbor sampling to a common output size specified by crop_size.

*@par Inputs:
*Input images must be a 4-D tensor. Inputs include: \n
*@li images:A Tensor. Must be one of the following types:uint8, uint16, int8, \n
int16, int32, int64, float16, float, double. A 4-D tensor of shape \n
[batch, image_height, image_width, depth].
*@li boxes: A Tensor of type float. A 2-D tensor of shape [num_boxes, 4].
*@li box_index: A Tensor of type int32. A 1-D tensor of shape [num_boxes] with \n
int32 values in [0, batch).
*@li crop_size: A Tensor of type int32. A 1-D tensor of 2 elements, crop_size \n
= [crop_height, crop_width]. All cropped image patches are resized to this size.

*@par Attributes:
*@li extrapolation_value: An optional float. Defaults to 0. Value used for \n
extrapolation, when applicable.
*@li method: An optional string from: '"bilinear", "nearest"'. Defaults to \n
"bilinear". Currently two sampling methods are supported: Bilinear and \n
NearestNeighbor.

*@par Outputs:
*y:A Tensor of type float.

*@attention Constraints: \n
*Input images must be a 4-D tensor.

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
*@brief Computes the gradient of the crop_and_resize op wrt the input \n
boxes tensor.

*@par Inputs:
*Input images and grads must be a 4-D tensor. Inputs include: \n
*@li grads: A 4-D tensor of shape [num_boxes, crop_height, crop_width, depth].
*@li images: A 4-D tensor of shape [batch, image_height, image_width, depth]. \n
Both image_height and image_width need to be positive.
*@li boxes: A 2-D tensor of shape [num_boxes, 4]. The i-th row of the tensor \n
specifies the coordinates of a box in the box_ind[i] image and is specified in \n
normalized coordinates [y1, x1, y2, x2].
*@li box_index: A 1-D tensor of shape [num_boxes] with int32 values in \n
[0, batch). The value of box_ind[i] specifies the image that the i-th box \n
refers to.

*@par Attributes:
method: A string specifying the interpolation method. Only 'bilinear' is \n
supported for now.

*@par Outputs:
*y:A 2-D tensor of shape [num_boxes, 4].

*@attention Constraints: \n
*Input images and grads must be a 4-D tensor.

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
*@brief Computes the gradient of the crop_and_resize op wrt the input \n
images tensor.

*@par Inputs:
*Input grads must be a 4-D tensor. Inputs include: \n
*@li grads: A 4-D tensor of shape [num_boxes, crop_height, crop_width, depth].
*@li boxes: A 2-D tensor of shape [num_boxes, 4]. The i-th row of the tensor \n
specifies the coordinates of a box in the box_ind[i] image and is specified \n
in normalized coordinates [y1, x1, y2, x2].
*@li box_index: A 1-D tensor of shape [num_boxes] with int32 values in \n
[0, batch). The value of box_ind[i] specifies the image that the i-th box \n
refers to.
*@li image_size: A 1-D tensor with value [batch, image_height, image_width, \n
depth] containing the original image size. Both image_height and image_width \n
need to be positive.

*@par Attributes:
method: A string specifying the interpolation method. Only 'bilinear' is \n
supported for now.

*@par Outputs:
*y:A 4-D tensor of shape [batch, image_height, image_width, depth].

*@attention Constraints: \n
*Input grads must be a 4-D tensor.

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
*@brief Extracts a glimpse from the input tensor.

*@par Inputs:
*Input x must be a 4-D tensor. Inputs include: \n
*@li x: A 4-D float tensor of shape [batch_size, height, width, channels].
*@li size: A 1-D tensor of 2 elements containing the size of the glimpses to \n
extract. The glimpse height must be specified first, following by the glimpse \n
width.
*@li offsets: A 2-D integer tensor of shape [batch_size, 2] containing the y, \n
x locations of the center of each window.

*@par Attributes:
*@li centered: indicates if the offset coordinates are centered relative to \n
the image, in which case the (0, 0) offset is relative to the center of the \n
input images. If false, the (0,0) offset corresponds to the upper left corner \n
of the input images.
*@li normalized: indicates if the offset coordinates are normalized.
*@li uniform_noise: indicates if the noise should be generated using a \n
uniform distribution or a Gaussian distribution.
*@li noise: indicates if the noise should uniform, gaussian, or zero. \n
The default is uniform which means the the noise type will be decided by \n
uniform_noise.

*@par Outputs:
*y:A tensor representing the glimpses [batch_size, glimpse_height, \n
glimpse_width, channels].

*@attention Constraints: \n
*Input x must be a 4-D tensor.

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
*@brief Convert one or more images from HSV to RGB.

*@par Inputs:
*Last dimension of input x must be size 3. Inputs include: \n
*images: 1-D or higher rank. HSV data to convert. Last dimension must be size 3.

*@par Outputs:
*y:images converted to RGB.

*@attention Constraints: \n
*Last dimension of input x must be size 3.

*/

REG_OP(HSVToRGB)
    .INPUT(images, TensorType({DT_FLOAT16,DT_FLOAT,DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT,DT_DOUBLE}))
    .OP_END_FACTORY_REG(HSVToRGB)

/**
*@brief Resize quantized images to size using quantized bilinear interpolation.

*@par Inputs:
*Input images must be a 4-D tensor. Inputs include: \n
*@li images: 4-D with shape [batch, height, width, channels].
*@li size: A 1-D int32 Tensor of 2 elements: new_height, new_width. The new \n
size for the images.
*@li min: A Tensor of type float.
*@li max: A Tensor of type float.

*@par Attributes:
*@li align_corners: An optional bool. Defaults to False. If true, the centers \n
of the 4 corner pixels of the input and output tensors are aligned, preserving \n
the values at the corner pixels. Defaults to false.
*@li half_pixel_centers: indicates if the offset coordinates are normalized.

*@par Outputs:
*@li resized_images: 4-D with shape [batch, new_height, new_width, channels].
*@li y_min: A Tensor of type float.
*@li y_max: A Tensor of type float.

*@attention Constraints: \n
*Input images and output images must be quantized types.

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
*@brief Resize images to size using area interpolation.

*@par Inputs:
*Input images must be a 4-D tensor. Inputs include: \n
*@li images: 4-D with shape [batch, height, width, channels].
*@li size: A 1-D int32 Tensor of 2 elements: new_height, new_width. \n
The new size for the images.

*@par Attributes:
*align_corners: If true, the centers of the 4 corner pixels of the input and \n
output tensors are aligned, preserving the values at the corner pixels. \n
Defaults to false.

*@par Outputs:
*y: 4-D with shape [batch, new_height, new_width, channels].

*@attention Constraints: \n
*Input images can be of different types but output images are always float.

*/

REG_OP(ResizeArea)
    .INPUT(images, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
        DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(size, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .ATTR(align_corners, Bool, false)
    .OP_END_FACTORY_REG(ResizeArea)

/**
*@brief Computes the gradient of bicubic interpolation.

*@par Inputs:
*Input grads must be a 4-D tensor. Inputs include: \n
*@li grads: A Tensor of type float. 4-D with shape [batch, height, width, \n
channels].
*@li original_image: A Tensor. Must be one of the following types: float, \n
double. 4-D with shape [batch, orig_height, orig_width, channels], The image \n
tensor that was resized.

*@par Attributes:
*@li align_corners: An optional bool. Defaults to False. If true, the centers \n
of the 4 corner pixels of the input and grad tensors are aligned. Defaults to \n
false.
*@li half_pixel_centers: An optional bool. Defaults to False.

*@par Outputs:
*y: A Tensor. Has the same type as original_image.

*@attention Constraints: \n
*Input images can be of different types but output images are always float.

*/

REG_OP(ResizeBicubicGrad)
    .INPUT(grads, TensorType({DT_FLOAT}))
    .INPUT(original_image, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE}))
    .ATTR(align_corners, Bool, false)
    .ATTR(half_pixel_centers, Bool, false)
    .OP_END_FACTORY_REG(ResizeBicubicGrad)

/**
*@brief Resize images to size using bicubic interpolation.

*@par Inputs:
*Input images must be a 4-D tensor. Inputs include: \n
*@li images: 4-D with shape [batch, height, width, channels].
*@li size: A 1-D int32 Tensor of 2 elements: new_height, new_width. The new \n
size for the images.

*@par Attributes:
*@li align_corners: If true, the centers of the 4 corner pixels of the input \n
and output tensors are aligned, preserving the values at the corner pixels. \n
Defaults to false.
*@li half_pixel_centers: An optional bool. Defaults to False.

*@par Outputs:
*y: 4-D with shape [batch, new_height, new_width, channels].

*@attention Constraints: \n
*Input images can be of different types but output images are always float.

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
*@brief Computes the gradient of nearest neighbor interpolation.

*@par Inputs:
*Input grads must be a 4-D tensor. Inputs include: \n
*@li grads: A Tensor. Must be one of the following types: uint8, int8, int32, \n
float16, float, double. 4-D with shape [batch, height, width, channels].
*@li size: A 1-D int32 Tensor of 2 elements: orig_height, orig_width. \n
The original input size.

*@par Attributes:
*@li align_corners: An optional bool. Defaults to False. If true, the centers \n
of the 4 corner pixels of the input and grad tensors are aligned. Defaults to \n
false.
*@li half_pixel_centers: An optional bool. Defaults to False.

*@par Outputs:
*y: A Tensor. Has the same type as grads.

*@attention Constraints: \n
*Input grads must be a 4-D tensor.
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
*@brief Computes the gradient of nearest neighbor interpolation.

*@par Inputs:
*Input grads must be a 4-D tensor. Inputs include: \n
*grads: A Tensor. 4-D with shape [batch, height, width, channels].


*@par Attributes:
*@li align_corners: An optional bool. Defaults to False. If true, the centers \n
of the 4 corner pixels of the input and grad tensors are aligned. Defaults to \n
false.
*@li size: An list type. Specify the images size.

*@par Outputs:
*y: A Tensor. Has the same type as grads.

*/

REG_OP(ResizeNearestNeighborV2GradD)
    .INPUT(grads, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(size, ListInt)
    .ATTR(align_corners, Bool, false)
    .ATTR(half_pixel_centers, Bool, false)
    .OP_END_FACTORY_REG(ResizeNearestNeighborV2GradD)

/**
*@brief Computes the gradient of bilinear interpolation.

*@par Inputs:
*Input grads must be a 4-D tensor. Inputs include: \n
*@li grads: A Tensor of type float32. 4-D with shape [batch, height, width, \n
channels].
*@li original_image: A Tensor. 4-D with shape [batch, orig_height, orig_width, \n
channels], The image tensor that was resized.

*@par Attributes:
*align_corners: An optional bool. Defaults to False. If true, the centers of \n
the 4 corner pixels of the input and grad tensors are aligned. Defaults to \n
false.

*@par Outputs:
*y: A Tensor. Has the same type as original_image.

*@attention Constraints: \n
*Input grads must be a 4-D tensor.
*/

REG_OP(ResizeBilinearV2Grad)
    .INPUT(grads, TensorType({DT_FLOAT}))
    .INPUT(original_image, TensorType::FloatingDataType())
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .ATTR(align_corners, Bool, false)
    .ATTR(half_pixel_centers, Bool, false)
    .OP_END_FACTORY_REG(ResizeBilinearV2Grad)

/**
*@brief Resize images to size using bilinear interpolation.

*@par Inputs:
*Input images must be a 4-D tensor. Inputs include: \n
*@li x: 4-D with shape [batch, height, width, channels].
*@li size: A 1-D int32 Tensor of 2 elements: new_height, new_width. The new \n
size for the images.

*@par Attributes:
*align_corners: If true, the centers of the 4 corner pixels of the input and \n
output tensors are aligned, preserving the values at the corner pixels. \n
Defaults to false.

*@par Outputs:
*y: 4-D with shape [batch, new_height, new_width, channels].

*@attention Constraints: \n
*Input images can be of different types but output images are always float.
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
*@brief Converts one or more images from RGB to HSV.

*@par Inputs:
*Last dimension of input images must be size 3. Inputs include: \n
*images: A Tensor. Must be one of the following types: float, double. 1-D or \n
higher rank. RGB data to convert. Last dimension must be size 3.

*@par Outputs:
*y: A Tensor. Has the same type as images.

*@attention Constraints: \n
*Outputs a tensor of the same shape as the images tensor, containing the HSV \n
value of the pixels. The output is only well defined if the value in images \n
are in [0,1].

*/

REG_OP(RGBToHSV)
    .INPUT(images, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE }))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE }))
    .OP_END_FACTORY_REG(RGBToHSV)

/**
*@brief Generate a single randomly distorted bounding box for an image.

*@par Inputs:
*Input images must be a 4-D tensor. Inputs include: \n
*@li image_size: 1-D, containing [height, width, channels].
*@li bounding_boxes: 3-D with shape [batch, N, 4] describing the N bounding \n
boxes associated with the image.
*@li min_object_covered: The cropped area of the image must contain at least \n
this fraction of any bounding box supplied. The value of this parameter should \n
be non-negative. In the case of 0, the cropped area does not need to overlap \n
any of the bounding boxes supplied.

*@par Attributes:
*@li seed: If either seed or seed2 are set to non-zero, the random number \n
generator is seeded by the given seed. Otherwise, it is seeded by a random seed.
*@li seed2: A second seed to avoid seed collision.
*@li aspect_ratio_range: The cropped area of the image must have an aspect \n
ratio = width / height within this range.
*@li max_attempts: Number of attempts at generating a cropped region of the \n
image of the specified constraints. After max_attempts failures, return the \n
entire image.
*@li use_image_if_no_bounding_boxes: Controls behavior if no bounding boxes \n
supplied. If true, assume an implicit bounding box covering the whole input. \n
If false, raise an error.

*@par Outputs:
*@li begin: 1-D, containing [offset_height, offset_width, 0].
*@li size: 1-D, containing [target_height, target_width, -1].
*@li bboxes: 3-D with shape [1, 1, 4] containing the distorted bounding box.

*@attention Constraints: \n
*Input images can be of different types but output images are always float.

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
*@brief Resize images to size using nearest neighbor interpolation.

*@par Inputs:
*Input x must be a 4-D tensor. Inputs include: \n
*@li x: 4-D with shape [batch, height, width, channels].
*@li size: A 1-D int32 Tensor of 2 elements: new_height, new_width. \n
The new size for the images.

*@par Attributes:
*align_corners: If true, the centers of the 4 corner pixels of the input and \n
output tensors are aligned, preserving the values at the corner pixels. \n
Defaults to false.

*@par Outputs:
*y: 4-D with shape [batch, new_height, new_width, channels].
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
*@brief Draw bounding boxes on a batch of images.

*@par Inputs:
*Input images must be a 4-D tensor. Inputs include: \n
*@li images: A Tensor. Must be one of the following types: float. 4-D with \n
shape [batch, height, width, depth]. A batch of images.
*@li boxes: A Tensor of type float32. 3-D with shape [batch, \n
num_bounding_boxes, 4] containing bounding boxes.

*@par Outputs:
*A Tensor. Has the same type as images.

*@attention Constraints: \n
*Input images must be a 4-D tensor.

*/

REG_OP(DrawBoundingBoxes)
    .INPUT(images, TensorType({DT_FLOAT}))
    .INPUT(boxes, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(DrawBoundingBoxes)

/**
*@brief Greedily selects a subset of bounding boxes in descending order of \n
score.

*@par Inputs:
*Input boxes and  scores must be float type. Inputs include: \n
*@li boxes: A 2-D float tensor of shape [num_boxes, 4].
*@li scores: A 1-D float tensor of shape [num_boxes] representing a single \n
score corresponding to each box (each row of boxes).
*@li max_output_size: A scalar integer tensor representing the maximum number \n
of boxes to be selected by non max suppression.

*@par Attributes:
*iou_threshold: A float representing the threshold for deciding whether boxes \n
overlap too much with respect to IOU.

*@par Outputs:
*selected_indices: A 1-D integer tensor of shape [M] representing the selected \n
indices from the boxes tensor, where M <= max_output_size.

*@attention Constraints: \n
*Input boxes and  scores must be float type.

*/

REG_OP(NonMaxSuppression)
    .INPUT(boxes, TensorType({DT_FLOAT}))
    .INPUT(scores, TensorType({DT_FLOAT}))
    .INPUT(max_output_size, TensorType({DT_INT32}))
    .OUTPUT(selected_indices, TensorType({DT_INT32}))
    .ATTR(iou_threshold, Float, 0.5f)
    .OP_END_FACTORY_REG(NonMaxSuppression)

/**
*@brief Greedily selects a subset of bounding boxes in descending order of \n
score.

*@par Inputs:
*Input boxes and  scores must be float type. Inputs include: \n
*@li boxes: A 2-D float tensor of shape [num_boxes, 4].
*@li scores: A 1-D float tensor of shape [num_boxes] representing a single \n
score corresponding to each box (each row of boxes).
*@li max_output_size: A scalar integer tensor representing the maximum number \n
of boxes to be selected by non max suppression.
*@li iou_threshold: A 0-D float tensor representing the threshold for deciding \n
whether boxes overlap too much with respect to IOU.

*@par Outputs:
*selected_indices: A 1-D integer tensor of shape [M] representing the selected \n
indices from the boxes tensor, where M <= max_output_size.

*@attention Constraints: \n
*Input boxes and  scores must be float type.

*/

REG_OP(NonMaxSuppressionV2)
    .INPUT(boxes, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(scores, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(max_output_size, TensorType({DT_INT32}))
    .INPUT(iou_threshold, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(selected_indices, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(NonMaxSuppressionV2)

/**
*@brief Greedily selects a subset of bounding boxes in descending order of \n
score.

*@par Inputs:
*Input boxes and  scores must be float type. Inputs include: \n
*@li boxes: A 2-D float tensor of shape [num_boxes, 4].
*@li scores: A 1-D float tensor of shape [num_boxes] representing a single \n
score corresponding to each box (each row of boxes).
*@li max_output_size: A scalar integer tensor representing the maximum number \n
of boxes to be selected by non max suppression.
*@li iou_threshold: A 0-D float tensor representing the threshold for deciding \n
whether boxes overlap too much with respect to IOU.
*@li score_threshold: A 0-D float tensor representing the threshold for \n
deciding when to remove boxes based on score.

*@par Outputs:
*selected_indices: A 1-D integer tensor of shape [M] representing the selected \n
indices from the boxes tensor, where M <= max_output_size.

*@attention Constraints: \n
*Input boxes and  scores must be float type.

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
*@brief Greedily selects a subset of bounding boxes in descending order of \n
score.

*@par Inputs:
*Input boxes and  scores must be float type. Inputs include: \n
*@li boxes: A 2-D float tensor of shape [num_boxes, 4].
*@li scores: A 1-D float tensor of shape [num_boxes] representing a single \n
score corresponding to each box (each row of boxes).
*@li max_output_size: A scalar integer tensor representing the maximum number \n
of boxes to be selected by non max suppression.
*@li iou_threshold: A 0-D float tensor representing the threshold for deciding \n
whether boxes overlap too much with respect to IOU.
*@li score_threshold: A 0-D float tensor representing the threshold for \n
deciding when to remove boxes based on score.

*@par Attributes:
*pad_to_max_output_size: If true, the output selected_indices is padded \n
to be of length max_output_size. Defaults to false.

*@par Outputs:
*@li selected_indices: A 1-D integer tensor of shape [M] representing the \n
selected indices from the boxes tensor, where M <= max_output_size.
*@li valid_outputs: A 0-D integer tensor representing the number of valid \n
elements in selected_indices, with the valid elements appearing first.

*@attention Constraints: \n
*Input boxes and  scores must be float type.

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
*@brief Greedily selects a subset of bounding boxes in descending order of \n
score.

*@par Inputs:
*Input overlaps and  scores must be float type. Inputs include: \n
*@li overlaps: A 2-D float tensor of shape [num_boxes, num_boxes] \n
representing the n-by-n box overlap values.
*@li scores: A 1-D float tensor of shape [num_boxes] representing a single \n
score corresponding to each box (each row of boxes).
*@li max_output_size: A scalar integer tensor representing the maximum number \n
of boxes to be selected by non max suppression.
*@li overlap_threshold: A 0-D float tensor representing the threshold for \n
deciding whether boxes overlap too.
*@li score_threshold: A 0-D float tensor representing the threshold for \n
deciding when to remove boxes based on score.

*@par Attributes:
*pad_to_max_output_size: If true, the output selected_indices is padded \n
to be of length max_output_size. Defaults to false.

*@par Outputs:
*selected_indices: A 1-D integer tensor of shape [M] representing the \n
selected indices from the boxes tensor, where M <= max_output_size.

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
*@brief JPEG-encode an image.

*@par Inputs:
*Input image must be unit8 type. Inputs include: \n
*image: A 3-D uint8 Tensor of shape [height, width, channels].

*@par Attributes:
*@li format: Per pixel image format.
*@li quality: Quality of the compression from 0 to 100 (higher is better \n
and slower).
*@li progressive: If True, create a JPEG that loads progressively (coarse \n
to fine).
*@li optimize_size: If True, spend CPU/RAM to reduce size with no quality \n
change.
*@li chroma_downsampling: A boolean, default is true.
*@li density_unit: Unit used to specify x_density and y_density: pixels per \n
inch ('in') or centimeter ('cm').
*@li x_density: Horizontal pixels per density unit.
*@li y_density: Vertical pixels per density unit.
*@li xmp_metadata: If not empty, embed this XMP metadata in the image header.

*@par Outputs:
*contents: 0-D. JPEG-encoded image.

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
*Input image must be unit8 or uint16 type. Inputs include: \n
*image: is a 3-D uint8 or uint16 Tensor of shape [height, width, channels] \n
where channels is: 1: for grayscale; 2: for grayscale + alpha; 3: for RGB; \n
4: for RGBA.

*@par Attributes:
*compression: Compression level.

*@par Outputs:
*contents: 0-D. PNG-encoded image.

*/

REG_OP(EncodePng)
    .INPUT(image, TensorType({DT_UINT8, DT_UINT16}))
    .OUTPUT(contents, TensorType({DT_STRING}))
    .ATTR(compression, Int, -1)
    .OP_END_FACTORY_REG(EncodePng)

/**
*@brief Resizes "images" to "size" using bilinear interpolation.

*@par Inputs:
* One input:
*x: An NC1HWC0 Tensor. \n
* Must be one of the following types: float16, float32.

*@par Attributes:
*@li size: A required int32 Tensor specifying the new size for the images. \n
No default value.
*@li align_corners: An optional bool. If "true", the centers of the corner \n
pixels of the input and output tensors are aligned. Defaults to "false".

*@par Outputs:
*y: A Tensor with type float32 and the same format as input "images".

*@attention Constraints:
*@li The input "size" must be a tensor of 2 elements: size[0] <= 2048, \n
size[1] <= 2048.
*@li The input "images" must be a tensor of 5 elements: images[2] <= 2048, \n
images[3] <= 2048.
*/
REG_OP(ResizeBilinearV2D)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .ATTR(align_corners, Bool, false)
    .ATTR(half_pixel_centers, Bool, false)
    .REQUIRED_ATTR(size, ListInt)
    .OP_END_FACTORY_REG(ResizeBilinearV2D)

/**
*@brief Resizes "images" to "size" using nearest neighbor interpolation.

*@par Inputs:
* One input:
*x: An NC1HWC0 Tensor. \n
* Must be one of the following types: float16, float32, int32, int8, uint8

*@par Attributes:
*@li size: A required int32 Tensor specifying the new size for the images. \n
No default value.
*@li align_corners: An optional bool. If "true", the centers of the corner \n
pixels of the input and output tensors are aligned. Defaults to "false".

*@par Outputs:
*y: A Tensor with the same type and format as input "images".

*@attention Constraints:
* The input "size" must be a tensor of 2 elements: size[0] <= 7680, \n
size[1] <= 4320
*/
REG_OP(ResizeNearestNeighborV2D)
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .REQUIRED_ATTR(size, ListInt)
    .ATTR(align_corners, Bool, false)
    .ATTR(half_pixel_centers, Bool, false)
    .OP_END_FACTORY_REG(ResizeNearestNeighborV2D)

/**
*@brief Extract the shape information of a JPEG-encoded image.

*@par Inputs:
*Input contents must be 0-D. Inputs include: \n
*contents: 0-D. The JPEG-encoded image.

*@par Attributes:
*output_type: The output type of the operation (int32 or int64). Defaults \n
to int32.

*@par Outputs:
*image_shape: 1-D. The image shape with format [height, width, channels].
*/

REG_OP(ExtractJpegShape)
    .INPUT(contents, TensorType({DT_STRING}))
    .OUTPUT(image_shape, TensorType({DT_INT32, DT_INT64}))
    .REQUIRED_ATTR(output_type, Type)
    .OP_END_FACTORY_REG(ExtractJpegShape)

/**
*@brief Draw bounding boxes on a batch of images.

*@par Inputs:
*@li images: 4-D with shape `[batch, height, width, depth]`. \n
A batch of images.
*@li boxes: 3-D with shape `[batch, num_bounding_boxes, 4]` \n
containing bounding boxes.
*@li colors: 2-D. A list of RGBA colors to cycle through for the boxes.

*@par Outputs:
*y: Returns 4-D with the same shape as `images`. \n
The batch of input images with bounding boxes drawn on the images.
*/

REG_OP(DrawBoundingBoxesV2)
    .INPUT(images, TensorType({DT_FLOAT}))
    .INPUT(boxes, TensorType({DT_FLOAT}))
    .INPUT(colors, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(DrawBoundingBoxesV2)

/**
*@brief Greedily selects a subset of bounding boxes in descending order of score, \n
pruning away boxes that have high intersection-over-union (IOU) overlap \n
with previously selected boxes.

*@par Inputs:
*@li boxes: A 2-D float tensor of shape `[num_boxes, 4]`.
*@li scores: A 1-D float tensor of shape `[num_boxes]` representing a single \n
score corresponding to each box (each row of boxes).
*@li max_output_size: A scalar integer tensor representing the maximum number of \n
boxes to be selected by non max suppression.
*@li iou_threshold: A 0-D float tensor representing the threshold for deciding whether \n
boxes overlap too much with respect to IOU.
*@li score_threshold: A 0-D float tensor representing the threshold for deciding when to \n 
remove boxes based on score.
*@li soft_nms_sigma: A 0-D float tensor representing the sigma parameter for Soft NMS.

*@par Attributes:
pad_to_max_output_size: If true, the output `selected_indices` is padded to be of length \n
`max_output_size`. Defaults to false. If not specified, defaults to false.

*@par Outputs:
*@li selected_indices: A 1-D integer tensor of shape [M] representing the \n
selected indices from the boxes tensor, where M <= max_output_size.
*@li selected_scores: A 1-D float tensor of shape `[M]` representing the corresponding \n
scores for each selected box, where `M <= max_output_size`.
*@li valid_outputs: A 0-D integer tensor representing the number of valid \n
elements in selected_indices, with the valid elements appearing first.
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
*@brief Resizes "images" to "size" by scale and translate.

*@par Inputs:
*@li images: A `Tensor`. Must be one of the following types: `int8`, `uint8`, \n
`int16`, `uint16`, `int32`, `int64`, `bfloat16`, `half`, `float32`, `float64`.
*@li size: A `Tensor` of type `int32`.
*@li scale: A `Tensor` of type `float32`.
*@li translation: A `Tensor` of type `float32`.

*@par Outputs:
*y: A Tensor with type float32.
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
*@brief Computes the gradient by scale and translate.

*@par Inputs:
*@li grads: A `Tensor`. Must be one of the following types: `float32`.
*@li original_image: A `Tensor`. Must have the same type as `grads`.
*@li scale: A `Tensor` of type `float32`.
*@li translation: A `Tensor` of type `float32`.

*@par Outputs:
*y: A `Tensor`. Has the same type as `grads`.
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
*@brief Greedily selects a subset of bounding boxes in descending order of score, \n
This operation performs non_max_suppression on the inputs per batch, across all classes.

*@par Inputs:
*@li boxes: A 4-D float tensor of shape `[batch_size, num_boxes, q, 4]`. If `q` is 1 then \n
same boxes are used for all classes otherwise, if `q` is equal to number of \n
classes, class-specific boxes are used.
*@li scores: A 3-D float tensor of shape `[batch_size, num_boxes, num_classes]` \n
representing a single score corresponding to each box (each row of boxes).
*@li max_output_size_per_class: A scalar integer tensor representing the maximum number of \n
boxes to be selected by non max suppression per class.
*@li max_total_size: A scalar representing maximum number of boxes retained over all classes. \n
*@li iou_threshold: A 0-D float tensor representing the threshold for deciding whether \n
boxes overlap too much with respect to IOU.
*@li score_threshold: A 0-D float tensor representing the threshold for deciding when to remove \n
boxes based on score.

*@par Attributes:
*@li pad_per_class: If false, the output nmsed boxes, scores and classes \n
are padded/clipped to `max_total_size`. If true, the \n
output nmsed boxes, scores and classes are padded to be of length \n
`max_size_per_class`*`num_classes`, unless it exceeds `max_total_size` in \n
which case it is clipped to `max_total_size`. Defaults to false.
*@li clip_boxes: If true, assume the box coordinates are between [0, 1] and clip the output boxes \n
if they fall beyond [0, 1]. If false, do not do clipping and output the box \n
coordinates as it is. If not specified, defaults to true.

*@par Outputs:
*y: A 1-D integer tensor of shape `[M]` representing the selected \n
indices from the boxes tensor, where `M <= max_output_size`.

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

}  // namespace ge

#endif  // GE_OP_MAGE_OPS_H_
