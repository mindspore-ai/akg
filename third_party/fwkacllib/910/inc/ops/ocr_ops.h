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
 * \file ocr_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_OCR_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_OCR_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
*@brief batch input x acording to attr batch_size and enqueue.
*@par Inputs:
*@li x: A Tensor need to batch of type float16/float32/float64/int8/int32/int64/uint8/uint32/uint64. \n
*@li queue_id:A Tensor of type uint32, queue id.

*@par Outputs:
*enqueue_count: A Tensor of type int64, enqueue tensor number.

*@par Attributes:
*@li batch_size: An optional int. Batch size.
*@li queue_name: An optional string. Queue name.
*@li queue_depth: An optional int. Queue depth.
*@li pad_mode: An optional string from: '"REPLICATE", "ZERO"'. Defaults to
"REPLICATE". Pad mode.
*/
REG_OP(BatchEnqueue)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, \
        DT_INT8, DT_INT32, DT_INT64, DT_UINT8, DT_UINT32, DT_UINT64}))
    .OPTIONAL_INPUT(queue_id, TensorType({DT_UINT32}))
    .OUTPUT(enqueue_count, TensorType({DT_INT32}))
    .ATTR(batch_size, Int, 8)
    .ATTR(queue_name, String, "")
    .ATTR(queue_depth, Int, 100)
    .ATTR(pad_mode, String, "REPLICATE")
    .OP_END_FACTORY_REG(BatchEnqueue)

/**
*@brief batch input x acording to attr batch_size and enqueue.
*@par Inputs:
*@li imgs_data: A Tensor of type uint8. Multi img data value. \n
*@li imgs_offset:A Tensor of type int32. Offset of every img data in input imgs_data. \n
*@li imgs_size:A Tensor of type int32. Shape of every img data. \n
*@li langs:A Tensor of type int32. Lang of every img data. \n
*@li langs_score:A Tensor of type int32. Lang score of every img data. \n

*@par Outputs:
*@liimgs: A Tensor of type uint8. Multi imgs data after reconition pre handle.
*@liimgs_relation: A Tensor of type int32. Output imgs orders in input imgs.
*@liimgs_lang: A Tensor of type int32. Output batch imgs langs.

*@par Attributes:
*@li batch_size: An optional int. Batch size.
*@li data_format: An optional string from: '"NHWC", "NCHW"'. Defaults to
"NHWC". Data format.
*@li pad_mode: An optional string from: '"REPLICATE", "ZERO"'. Defaults to
"REPLICATE". Pad mode.
*/
REG_OP(OCRRecognitionPreHandle)
    .INPUT(imgs_data, TensorType({DT_UINT8}))
    .INPUT(imgs_offset, TensorType({DT_INT32}))
    .INPUT(imgs_size, TensorType({DT_INT32}))
    .INPUT(langs, TensorType({DT_INT32}))
    .INPUT(langs_score, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(imgs, TensorType({DT_UINT8}))
    .OUTPUT(imgs_relation, TensorType({DT_INT32}))
    .OUTPUT(imgs_lang, TensorType({DT_INT32}))
    .OUTPUT(imgs_piece_fillers, TensorType({DT_INT32}))
    .ATTR(batch_size, Int, 8)
    .ATTR(data_format, String, "NHWC")
    .ATTR(pad_mode, String, "REPLICATE")
    .OP_END_FACTORY_REG(OCRRecognitionPreHandle)

/**
*@brief ocr detection pre handle.
*@par Inputs:
*img: A Tensor of type uint8. img data value. \n

*@par Outputs:
*@li resized_img: A Tensor of type uint8. Img after detection pre handle.
*@li h_scale: A Tensor of type float. H scale.
*@li w_scale: A Tensor of type float. W scale.

*@par Attributes:
*data_format: An optional string from: '"NHWC", "NCHW"'. Defaults to
"NHWC". Data format.
*/
REG_OP(OCRDetectionPreHandle)
    .INPUT(img, TensorType({DT_UINT8}))
    .OUTPUT(resized_img, TensorType({DT_UINT8}))
    .OUTPUT(h_scale, TensorType({DT_FLOAT}))
    .OUTPUT(w_scale, TensorType({DT_FLOAT}))
    .ATTR(data_format, String, "NHWC")
    .OP_END_FACTORY_REG(OCRDetectionPreHandle)

/**
*@brief ocr identify prehandle.
*@par Inputs:
*@li imgs_data: A Tensor of type uint8. Multi img data value. \n
*@li imgs_offset:A Tensor of type int32. Offset of every img data in input imgs_data. \n
*@li imgs_size:A Tensor of type int32. Shape of every img data. \n

*@par Outputs:
*resized_imgs: A Tensor of type uint8. Multi imgs after identify pre handle.

*@par Attributes:
*@li size: An optional int. Size.
*@li data_format: An optional string from: '"NHWC", "NCHW"'. Defaults to
"NHWC". Data format.
*/
REG_OP(OCRIdentifyPreHandle)
    .INPUT(imgs_data, TensorType({DT_UINT8}))
    .INPUT(imgs_offset, TensorType({DT_INT32}))
    .INPUT(imgs_size, TensorType({DT_INT32}))
    .OUTPUT(resized_imgs, TensorType({DT_UINT8}))
    .REQUIRED_ATTR(size, ListInt)
    .ATTR(data_format, String, "NHWC")
    .OP_END_FACTORY_REG(OCRIdentifyPreHandle)

/**
*@brief batch dilate polygons according to expand_scale.
*@par Inputs:
*@li polys_data: A Tensor of type int32. point data of every polygon. \n
*@li polys_offset:A Tensor of type int32. Offset of every polygon . \n
*@li polys_size:A Tensor of type int32. Size of every polygon. \n
*@li score:A Tensor of type float. Score of every point in image. \n
*@li min_border:A Tensor of type int32. Minimum width of each polygon. \n
*@li min_area_thr:A Tensor of type int32. Minimum area of each polygon. \n
*@li score_thr:A Tensor of type float. Minimum confidence score of each polygon. \n
*@li expands_cale:A Tensor of type float. Polygon expansion multiple. \n

*@par Outputs:
*@li dilated_polys_data: A Tensor of type int32. Point data of every dilated polygon. \n
*@li dilated_polys_offset: A Tensor of type int32. Offset of every dilated polygon . \n
*@li dilated_polys_size: A Tensor of type int32. Size of every dilated polygon. \n
*/
REG_OP(BatchDilatePolys)
    .INPUT(polys_data, TensorType({DT_INT32}))
    .INPUT(polys_offset, TensorType({DT_INT32}))
    .INPUT(polys_size, TensorType({DT_INT32}))
    .INPUT(score, TensorType({DT_FLOAT}))
    .INPUT(min_border, TensorType({DT_INT32}))
    .INPUT(min_area_thr, TensorType({DT_INT32}))
    .INPUT(score_thr, TensorType({DT_FLOAT}))
    .INPUT(expands_cale, TensorType({DT_FLOAT}))
    .OUTPUT(dilated_polys_data, TensorType({DT_INT32}))
    .OUTPUT(dilated_polys_offset, TensorType({DT_INT32}))
    .OUTPUT(dilated_polys_size, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(BatchDilatePolys)

/**
*@brief find contours acording to img.
*@par Inputs:
*@li img: A Tensor of type uint8. Img data value. \n

*@par Outputs:
*@li polys_data: A Tensor of type int32. Point data of every contours. \n
*@li polys_offset:A Tensor of type int32. Offset of every contours . \n
*@li polys_size:A Tensor of type int32. Size of every contours. \n
*/
REG_OP(OCRFindContours)
    .INPUT(img, TensorType({DT_UINT8}))
    .OUTPUT(polys_data, TensorType({DT_INT32}))
    .OUTPUT(polys_offset, TensorType({DT_INT32}))
    .OUTPUT(polys_size, TensorType({DT_INT32}))
    .ATTR(value_mode, Int, 0)
    .OP_END_FACTORY_REG(OCRFindContours)

/**
*@brief dequeue data acording to queue_id and queue_name.
*@par Inputs:
*@li queue_id:An Tensor of type uint32, queue id. \n

*@par Outputs:
*data: A Tensor of type RealNumberType, dequeue tensor. \n

*@par Attributes:
*@li output_type: A required type. dequeue data type.
*@li output_shape: A required listint. dequeue data shape.
*@li queue_name: An optional string. Queue name.   \n
*/
REG_OP(Dequeue)
    .OPTIONAL_INPUT(queue_id, TensorType({DT_UINT32}))
    .OUTPUT(data, TensorType::RealNumberType())
    .REQUIRED_ATTR(output_type, Type)
    .REQUIRED_ATTR(output_shape, ListInt)
    .ATTR(queue_name, String, "")
    .OP_END_FACTORY_REG(Dequeue);

/**
*@brief ocr detection post handle.
*@par Inputs:
*@li img: A Tensor of type uint8. original image data.
*@li polys_data: A Tensor of type int32. point data of every poly.
*@li polys_offset:A Tensor of type int32. Offset of every poly.
*@li polys_size:A Tensor of type int32. Size of every poly. \n

*@par Outputs:
*@li imgs_data: A Tensor of type int32. imgs_data of original image.
*@li imgs_offset: A Tensor of type int32. Offset of every imgs data.
*@li imgs_size: A Tensor of type int32. Shape of every imgs data.
*@li rect_points: A Tensor of type int32. Rect points of every imgs. \n

*@par Attributes:
*@li data_format: An optional string from: '"NHWC", "NCHW"'. Defaults to
"NHWC". Data format.
*/
REG_OP(OCRDetectionPostHandle)
    .INPUT(img, TensorType({DT_UINT8}))
    .INPUT(polys_data, TensorType({DT_INT32}))
    .INPUT(polys_offset, TensorType({DT_INT32}))
    .INPUT(polys_size, TensorType({DT_INT32}))
    .OUTPUT(imgs_data, TensorType({DT_UINT8}))
    .OUTPUT(imgs_offset, TensorType({DT_INT32}))
    .OUTPUT(imgs_size, TensorType({DT_INT32}))
    .OUTPUT(rect_points, TensorType({DT_INT32}))
    .ATTR(data_format, String, "NHWC")
    .OP_END_FACTORY_REG(OCRDetectionPostHandle);

/**
*@brief resize and clip polys.
*@par Inputs:
*@li polys_data: A Tensor of type int32. point data of every poly.
*@li polys_offset:A Tensor of type int32. Offset of every poly .
*@li polys_size:A Tensor of type int32. Size of every poly.
*@li h_scale:A Tensor of type float. Expand scale of height.
*@li w_scale:A Tensor of type float. Expand scale of width.
*@li img_h:A Tensor of type int32. Height of original image.
*@li img_w:A Tensor of type int32. Width of original image. \n

*@par Outputs:
*@li clipped_polys_data: A Tensor of type int32. point data of every clipped poly. \n
*@li clipped_polys_offset: A Tensor of type int32. Offset of every clipped poly . \n
*@li clipped_polys_size: A Tensor of type int32. Size of every clipped poly. \n
*@li clipped_polys_num: A Tensor of type int32. Number of clipped polys. \n
*/
REG_OP(ResizeAndClipPolys)
    .INPUT(polys_data, TensorType({DT_INT32}))
    .INPUT(polys_offset, TensorType({DT_INT32}))
    .INPUT(polys_size, TensorType({DT_INT32}))
    .INPUT(h_scale, TensorType({DT_FLOAT}))
    .INPUT(w_scale, TensorType({DT_FLOAT}))
    .INPUT(img_h, TensorType({DT_INT32}))
    .INPUT(img_w, TensorType({DT_INT32}))
    .OUTPUT(clipped_polys_data, TensorType({DT_INT32}))
    .OUTPUT(clipped_polys_offset, TensorType({DT_INT32}))
    .OUTPUT(clipped_polys_size, TensorType({DT_INT32}))
    .OUTPUT(clipped_polys_num, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(ResizeAndClipPolys);


} // namespace ge


#endif  // OPS_BUILT_IN_OP_PROTO_INC_OCR_OPS_H_
