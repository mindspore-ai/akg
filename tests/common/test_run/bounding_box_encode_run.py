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

import numpy as np
from tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from test_op import bounding_box_encode
import copy
from gen_random import random_gaussian

COORDINATES_LEN = 4
COORDINATES_PAD_LEN = 8


# def encode_data(anchor_box, groundtruth_box, anchor_samples, dtype, scale_factors, epsilon):
#     def _process_data_for_cordinates(data):
#         '''
#
#         :param data: box (ymin, xmin, ymax, xmax)
#         :return:
#         '''
#         data = np.abs(data)
#         y_0, x_0, y_1, x_1 = [np.squeeze(i) for i in np.split(data, 4, -1)]
#         y_max = np.maximum(y_0, y_1)
#         y_min = np.minimum(y_0, y_1)
#         x_max = np.maximum(x_0, x_1)
#         x_min = np.minimum(x_0, x_1)
#         return np.stack([y_min, x_min, y_max, x_max], -1)
#
#     def _get_center_coordinates_and_sizes(box_corners):
#         """Computes the center coordinates, height and width of the boxes.
#
#         Args:
#           scope: name scope of the function.
#
#         Returns:
#           a list of 4 1-D tensors [ycenter, xcenter, height, width].
#         """
#         ymin, xmin, ymax, xmax = box_corners
#         width = xmax - xmin
#         height = ymax - ymin
#         ycenter = ymin + height / 2.
#         xcenter = xmin + width / 2.
#         return ycenter, xcenter, height, width
#
#     def _encode(anchors, boxes, scale_factors, epsilon):
#         ycenter_a, xcenter_a, ha, wa = _get_center_coordinates_and_sizes(anchors)
#         ycenter, xcenter, h, w = _get_center_coordinates_and_sizes(boxes)
#
#         ha += epsilon
#         wa += epsilon
#         h += epsilon
#         w += epsilon
#
#         tx = (xcenter - xcenter_a) / wa
#         ty = (ycenter - ycenter_a) / ha
#         tw = np.log(w / wa)
#         th = np.log(h / ha)
#         # Scales location targets as used in paper for joint training.
#         if scale_factors:
#             ty *= scale_factors[0]
#             tx *= scale_factors[1]
#             th *= scale_factors[2]
#             tw *= scale_factors[3]
#         return ty, tx, th, tw
#
#     anchor_box_data = random_gaussian(anchor_box, miu=1, sigma=0.1).astype(dtype)
#     anchor_box_data = _process_data_for_cordinates(anchor_box_data)
#     groundtruth_box_data = random_gaussian(groundtruth_box, miu=1, sigma=0.2).astype(dtype)
#     groundtruth_box_data = _process_data_for_cordinates(groundtruth_box_data)
#     anchor_samples_data = random_gaussian(anchor_samples, miu=1, sigma=0.3).astype("int32")
#
#     num_groundtruths = groundtruth_box_data.shape[1]
#     batchsize = anchor_samples_data.shape[0]
#     num_anchors = anchor_box_data.shape[0]
#     output_shape = (batchsize, num_anchors, 4)
#     expect = np.full(output_shape, 0, dtype)
#
#     for i in range(batchsize):
#         for j in range(num_anchors):
#             sample_value = anchor_samples_data[i,j]
#             if sample_value < num_groundtruths:
#                 expect[i,j] = _encode(anchor_box_data[j], groundtruth_box_data[i, sample_value], scale_factors, epsilon)
#     return anchor_box_data, groundtruth_box_data, anchor_samples_data, expect

def encode_data_vector(anchor_box, groundtruth_box, anchor_samples, dtype, scale_factors, epsilon):
    def _process_data_for_cordinates_vector(data):
        '''

        :param data: box (ymin, xmin, ymax, xmax)
        :return:
        '''
        data = np.abs(data)
        y_0, x_0, y_1, x_1, _, _, _, _ = [np.squeeze(i) for i in np.split(data, COORDINATES_PAD_LEN, -1)]
        y_max = np.maximum(y_0, y_1)
        y_min = np.minimum(y_0, y_1)
        x_max = np.maximum(x_0, x_1)
        x_min = np.minimum(x_0, x_1)
        data_cordinates = np.stack([y_min, x_min, y_max, x_max], -1)
        data[..., :COORDINATES_LEN] = data_cordinates
        return data

    def _encode_vector(ycenter_a, xcenter_a, ha, wa, ycenter, xcenter, h, w, scale_factors, epsilon):
        ha = np.add(ha, epsilon)
        wa = np.add(wa, epsilon)
        h = np.add(h, epsilon)
        w = np.add(w, epsilon)

        tx = np.divide(np.subtract(xcenter, xcenter_a), wa)
        ty = np.divide(np.subtract(ycenter, ycenter_a), ha)
        tw = np.log(np.divide(w, wa))
        th = np.log(np.divide(h, ha))
        # Scales location targets as used in paper for joint training.
        if scale_factors:
            ty = np.multiply(ty, scale_factors[0])
            tx = np.multiply(tx, scale_factors[1])
            th = np.multiply(th, scale_factors[2])
            tw = np.multiply(tw, scale_factors[3])
        return ty, tx, th, tw

    def _get_center_coordinates_and_sizes_vector(box_data):
        """Computes the center coordinates, height and width of the boxes.

        Returns:
          a list of 4 1-D tensors [ycenter, xcenter, height, width].
        """
        ymin, xmin, ymax, xmax = [np.squeeze(i) for i in np.split(box_data, 4, 0)]
        width = np.subtract(xmax, xmin)
        height = np.subtract(ymax, ymin)
        ycenter = np.add(ymin, np.multiply(height, 0.5))
        xcenter = np.add(xmin, np.multiply(width, 0.5))
        return ycenter, xcenter, height, width

    anchor_box_data = random_gaussian(anchor_box, miu=1, sigma=0.1).astype(dtype)
    anchor_box_data = _process_data_for_cordinates_vector(anchor_box_data)
    groundtruth_box_data = random_gaussian(groundtruth_box, miu=1, sigma=0.2).astype(dtype)
    groundtruth_box_data = _process_data_for_cordinates_vector(groundtruth_box_data)

    num_groundtruths = groundtruth_box_data.shape[1]
    num_anchors = anchor_box_data.shape[0]
    np.random.seed(1)
    anchor_samples_data = np.random.randint(0, num_anchors, anchor_samples, dtype="int32")
    limmit_data = np.full(anchor_samples, num_groundtruths, dtype="int32")
    limmit_index = np.where(anchor_samples_data > num_groundtruths)
    anchor_samples_data[limmit_index] = limmit_data[limmit_index]

    batchsize = anchor_samples_data.shape[0]

    anchor_box_data_broadcast = np.broadcast_to(anchor_box_data, (batchsize,) + tuple(anchor_box))
    anchor_box_data_broadcast = np.transpose(anchor_box_data_broadcast, axes=[2, 0, 1])
    anchor_samples_box_data = np.full((COORDINATES_LEN, batchsize, num_anchors), 0, dtype)
    for i in range(batchsize):
        for j in range(num_anchors):
            for m in range(COORDINATES_LEN):
                for k in range(num_groundtruths):
                    if anchor_samples_data[i, j] == k:
                        anchor_samples_box_data[m, i, j] = groundtruth_box_data[i, k, m]

    ycenter_a, xcenter_a, height_a, width_a = _get_center_coordinates_and_sizes_vector(
        anchor_box_data_broadcast[:COORDINATES_LEN, ...])
    ycenter, xcenter, height, width = _get_center_coordinates_and_sizes_vector(
        anchor_samples_box_data[:COORDINATES_LEN, ...])

    ty, tx, th, tw = _encode_vector(ycenter_a, xcenter_a, height_a, width_a, ycenter, xcenter, height, width,
                                    scale_factors, epsilon)
    expect = np.stack([ty, tx, th, tw], axis=-1)
    return anchor_box_data, groundtruth_box_data, anchor_samples_data, expect


def bounding_box_encode_run(anchor_box_shape, groundtruth_box_shape, anchor_samples_shape, dtype, scale_factors,
                            epsilon, kernel_name, attrs={}):
    # check_shape:
    # bachsize
    assert (groundtruth_box_shape[0] == anchor_samples_shape[0])
    # num_archors
    assert (anchor_box_shape[0] == anchor_samples_shape[1])
    assert (not scale_factors or len(scale_factors) == COORDINATES_LEN) and (
        anchor_box_shape[-1] == COORDINATES_PAD_LEN) and (
        groundtruth_box_shape[-1] == COORDINATES_PAD_LEN)

    op_attrs = [scale_factors, epsilon]

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(bounding_box_encode.bouding_box_encode,
                                  [anchor_box_shape, groundtruth_box_shape, anchor_samples_shape],
                                  [dtype, dtype, "int32"],
                                  op_attrs, kernel_name=kernel_name, attrs=attrs, dump_code=True, tuning=t)
        if t:
            anchor_box_data, anchor_samples_data, expect, groundtruth_box_data, output_data = gen_data(anchor_box_shape,
                                                                                                       anchor_samples_shape,
                                                                                                       dtype, epsilon,
                                                                                                       groundtruth_box_shape,
                                                                                                       scale_factors)
            return mod, expect, (anchor_box_data, groundtruth_box_data, anchor_samples_data, output_data)
        else:
            return mod
    else:
        mod = utils.op_build_test(bounding_box_encode.bouding_box_encode,
                                  [anchor_box_shape, groundtruth_box_shape, anchor_samples_shape],
                                  [dtype, dtype, "int32"],
                                  op_attrs, kernel_name=kernel_name, attrs=attrs, dump_code=True)
        anchor_box_data, anchor_samples_data, expect, groundtruth_box_data, output_data = gen_data(anchor_box_shape,
                                                                                                   anchor_samples_shape,
                                                                                                   dtype, epsilon,
                                                                                                   groundtruth_box_shape,
                                                                                                   scale_factors)
        output = utils.mod_launch(mod, (anchor_box_data, groundtruth_box_data, anchor_samples_data, output_data),
                                  expect=expect)

        # compare result
        compare_result = compare_tensor(output, expect, rtol=5e-3, equal_nan=True)
        return (anchor_box_data, groundtruth_box_data, anchor_samples_data), output, expect, compare_result


def gen_data(anchor_box_shape, anchor_samples_shape, dtype, epsilon, groundtruth_box_shape, scale_factors):
    # Generate data
    anchor_box_data, groundtruth_box_data, anchor_samples_data, expect = encode_data_vector(anchor_box_shape,
                                                                                            groundtruth_box_shape,
                                                                                            anchor_samples_shape, dtype,
                                                                                            scale_factors, epsilon)
    # mod launch
    out_shape = expect.shape
    output_data = np.full(out_shape, np.nan, dtype)
    return anchor_box_data, anchor_samples_data, expect, groundtruth_box_data, output_data
