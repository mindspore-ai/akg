# Copyright 2020 Huawei Technologies Co., Ltd
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

"""operator dsl function: pooling"""
import akg
from akg import topi
from akg.utils.format_transform import get_shape
from akg.utils import validation_check as vc_util, custom_tiling as ct_util
from akg.utils.dsl_create import cal_pad_shapes_by_strategy
from akg.ops.nn.maxpool import maxpool_with_argmax


def quantize_chk_cfg_and_gen_outdtype(
        quant_algo, scale_mode, scale_sqrt, qdrtensors):
    """check all the params is valid, and general output dtype"""
    # check quantize algorithm and quantize scale type
    if quant_algo is None:
        # quantize switch off
        if scale_mode is not None or scale_sqrt is not None \
            or qdrtensors is not None:
            raise RuntimeError("Invalid Quantize Config.")
        out_dtype = "float16"
        return out_dtype

    # quantize switch on, all quantize params should not be None
    if scale_mode is None or scale_sqrt is None \
        or qdrtensors is None:
        raise RuntimeError("Invalid Quantize Config!")

    if len(quant_algo) != 2 or any([i not in [0, 1] for i in quant_algo]):
        raise RuntimeError("Invalid Quantize Config!!!")

    # check quantize algorithm
    if quant_algo[0] not in (0, 1):
        raise RuntimeError("Quantize algorithm just support 0 for non "
                           "offset and 1 for half offset, but get {}."
                           "".format(quant_algo[0]))

    # check quantize scale type
    if quant_algo[1] != 0:
        raise RuntimeError("Quantize scale only support SCALAR now.")

    # non offset get int8, half offset get uint8
    out_dtype = "int8" if quant_algo[0] == 0 else "uint8"

    if scale_mode not in (0, 1, 2):
        raise ValueError("Invalid scale mode, just support '0,1,2' but get "
                         "{}!".format(scale_mode))
    # now scale mode limit
    if scale_mode in (0, 1):
        raise RuntimeError("quantized_avg_pool just support requantize now!")

    # check scale method
    if scale_sqrt not in (0, 1):
        raise RuntimeError("Invalid scale moethod!")

    # scalar scale type for now
    if len(qdrtensors) != 2:
        raise RuntimeError("qdrtensors should contain two tensors for scale "
                           "and offset!")
    if get_shape(qdrtensors[0]) != [1] or get_shape(qdrtensors[1]) != [1]:
        raise RuntimeError("Scale for dequantize or requantize only "
                           "support scalar tensor.")
    vc_util.ops_dtype_check(qdrtensors[0].dtype,
                            vc_util.DtypeForDavinci.FLOAT16)
    vc_util.ops_dtype_check(qdrtensors[1].dtype,
                            vc_util.DtypeForDavinci.FLOAT16)
    #  vc_util.ops_dtype_check(qdrtensors[0].dtype,
    #                          vc_util.DtypeForDavinci.ALL_FLOAT)
    #  vc_util.ops_dtype_check(qdrtensors[1].dtype,
    #                          vc_util.DtypeForDavinci.ALL_FLOAT)

    return out_dtype


def quantized_chk_and_gen_pool_params(in_tensor, ksize, strides,
                                      padding, data_format):
    """check paramaters"""
    vc_util.check_shape(get_shape(in_tensor))
    #  vc_util.ops_dtype_check(in_tensor.dtype, vc_util.DtypeForDavinci.ALL_FLOAT)
    vc_util.ops_dtype_check(in_tensor.dtype, vc_util.DtypeForDavinci.FLOAT16)

    if data_format in ("NHWC",):
        hw_indices = [1, 2]
    elif data_format in ("NC1HWC0", "NCHW"):
        hw_indices = [2, 3]
    else:
        raise RuntimeError("Data format {} is not supported."
                           "".format(data_format))

    if data_format in ("NHWC", "NCHW"):
        if len(ksize) != 4 or len(strides) != 4:
            raise RuntimeError("Invalid params, ksize and strides dim must all "
                               "be 4!!")
    else:
        if len(ksize) != 5 or len(strides) != 5:
            raise RuntimeError("Invalid params, ksize and strides dim must all "
                               "be 5!!")

    if any([ksize[i] != 1 for i, _ in enumerate(ksize) if i not in hw_indices]):
        raise RuntimeError("Only supports pooling across width/height, "
                           "and other ksize dimension should be one.")
    if any([strides[i] != 1 for i, _ in enumerate(strides) \
             if i not in hw_indices]):
        raise RuntimeError("Only supports pooling across width/height, "
                           "and other strides dimension should be one.")

    window = [ksize[i] for i in hw_indices]
    stride = [strides[i] for i in hw_indices]

    shape = get_shape(in_tensor)
    in_h, in_w = [shape[i] for i in hw_indices]

    if window[0] >= in_h and window[1] >= in_w:
        window[0] = in_h
        window[1] = in_w
        padding = "VALID"
        stride = [1, 1]

    return window, stride, padding


def get_attrs():
    """Get default attrs for maxpool."""
    default_attr_map = {
        "pragma_reschedule": 1,
        "pragma_opt_for_davinci": 1,
        "pragma_reorder_schedule": True,
        "enable_pre_poly_loop_partition": False,
        "enable_post_poly_loop_partition": False,
        "disable_cse": True,
        "enable_bk_optimize": False,
    }
    return default_attr_map

def quantized_maxpool_tiling_strategy(data, kernel, stride, pad, quant_algo):
    """Custom tiling for quantized maxpool."""
    batch, c_1, fm_h, fm_w, c_0 = get_shape(data)
    _, [out_h, out_w] = \
        cal_pad_shapes_by_strategy(get_shape(data), kernel, stride, pad)


    strategy = list()
    if c_0 == 16:
        h_cut = out_h
        if fm_h >= 50 and fm_w >= 50:
            h_cut = 3
        dim_ind = 0
        tiling_params = list()
        if batch > 1:
            tiling_params.append([1, ct_util.TileConstraint.FACTOR, dim_ind])
            dim_ind = dim_ind + 1
        if c_1 > 1:
            tiling_params.append([1, ct_util.TileConstraint.FACTOR, dim_ind])
            dim_ind = dim_ind + 1
        tiling_params.append([h_cut, ct_util.TileConstraint.FACTOR, dim_ind])
        tiling_params.append(["H", ct_util.TileConstraint.SET_AXIS_INFO, dim_ind])
        tiling_params.append([out_w, ct_util.TileConstraint.FACTOR, dim_ind + 1])

        if quant_algo is not None:
            tiling_params.append([kernel[0], ct_util.TileConstraint.FACTOR, dim_ind + 2])
            tiling_params.append([kernel[1], ct_util.TileConstraint.FACTOR, dim_ind + 3])
            tiling_params.append([16, ct_util.TileConstraint.FACTOR, dim_ind + 4])
        else:
            tiling_params.append([kernel[0], ct_util.TileConstraint.FACTOR, dim_ind + 3])
            tiling_params.append([kernel[1], ct_util.TileConstraint.FACTOR, dim_ind + 4])
            tiling_params.append([16, ct_util.TileConstraint.FACTOR, dim_ind + 2])

        for para in tiling_params:
            strategy += ct_util.create_constraint_on_axis(
                values=para[0], constraints=para[1], axis=para[2])

        #  if batch > 1:
        #      strategy += ct_util.create_constraint_on_axis(
        #          values=1,
        #          constraints=ct_util.TileConstraint.FACTOR,
        #          axis=dim_ind)
        #      dim_ind = dim_ind + 1
        #  if c_1 > 1:
        #      strategy += ct_util.create_constraint_on_axis(
        #          values=1,
        #          constraints=ct_util.TileConstraint.FACTOR,
        #          axis=dim_ind)
        #      dim_ind = dim_ind + 1
        #  strategy += ct_util.create_constraint_on_axis(
        #      values=h_cut,
        #      constraints=ct_util.TileConstraint.FACTOR,
        #      axis=dim_ind)
        #  strategy += ct_util.create_constraint_on_axis(
        #      values="H",
        #      constraints=ct_util.TileConstraint.SET_AXIS_INFO,
        #      axis=dim_ind)
        #  strategy += ct_util.create_constraint_on_axis(
        #      values=out_w,
        #      constraints=ct_util.TileConstraint.FACTOR,
        #      axis=dim_ind+1)
        #  strategy += ct_util.create_constraint_on_axis(
        #      values=kernel[0],
        #      constraints=ct_util.TileConstraint.FACTOR,
        #      axis=dim_ind+2)
        #  strategy += ct_util.create_constraint_on_axis(
        #      values=kernel[1],
        #      constraints=ct_util.TileConstraint.FACTOR,
        #      axis=dim_ind+3)
        #  strategy += ct_util.create_constraint_on_axis(
        #      values=16,
        #      constraints=ct_util.TileConstraint.FACTOR,
        #      axis=dim_ind+4)
    return strategy

def _quantized_max_pool_compute(x, window, stride, qdrtensors,
                                out_dtype, padding, quant_algo, _):
    """compute for quantized avgpool"""
    res, _, _ = maxpool_with_argmax(x, window, stride, padding)

    if quant_algo is not None:
        scale_req, offset_req = qdrtensors
        # scale
        res = topi.multiply(res, scale_req[0])
        if quant_algo[0] == 1:
            # offset
            res = topi.add(res, offset_req[0])

    res = topi.cast(res, out_dtype)
    return res


@vc_util.check_input_type(akg.tvm.tensor.Tensor,
                          (list, tuple, type(None)),
                          (list, tuple), (list, tuple),
                          (str, type(None)), (str, type(None)),
                          (list, tuple, type(None)),
                          (int, type(None)), (int, type(None)))
def quantized_max_pool(x, qdrtensors,
                       ksize, strides, padding="VALID", data_format="NHWC",
                       quant_algo=None, scale_mode=None, scale_sqrt=None):
    """
    Quantized MaxPool.

    Args:
        x (tvm.tensor.Tensor): Input tensor, only support float16 dtype,
            and NC1HWC0 format.
        qdrtensors (Union[List[tvm.tensor.Tensor], Tuple[tvm.tensor.Tensor]]):
            [scale_q, offset_q] for quantize scale mode(Not supported now),
            [scale_deq_req] for dequantize scale mode(just for requantize now),
            [scale_req, offset_req] for requantize half offset mode.
        ksize (Union[list, tuple]): Pooling window, only support pooling in H or W.
        strides (Union[list, tuple]): Pooling stride, only support pooling in H or W.
        padding (str): Padding method, VALID or SAME.
        data_format (str): Data format just for ksize and strides, support
                           NHWC, NCHW, NC1HWC0.
        quant_algo (Union[list, tuple, None]): Two ints for quantize algorithm and
            quantize scale type(dequantize/requantize, quantize always scalar).
            quant_algo[0] - quantize algorithm, 0:non offset, 1:half offset.
            quant_algo[1] - quantize scale type, 0:scalar, 1:vector(Not supported now).
        scale_mode (Union[int, None]): Scale mode, 0: quantize(Not supported now),
            1: dequantize(Not supported now), 2: requantize.
        scale_sqrt (Union[int, None]): Scale method, 0: non sqrt, 1: sqrt.

    Returns:
        A tvm.tensor.Tensor. If quant_algo is None, dtype is float16, otherwise
        int8 for non offset quantize algorithm and uint8 for half offset.
    """
    window, stride, padding = \
        quantized_chk_and_gen_pool_params(x, ksize, strides, padding, data_format)
    out_dtype = quantize_chk_cfg_and_gen_outdtype(quant_algo, scale_mode,
                                                  scale_sqrt, qdrtensors)
    #  out_dtype = "float16"

    attrs = get_attrs()
    attrs["custom_tiling"] = quantized_maxpool_tiling_strategy(
        x, window, stride, padding, quant_algo)

    return _quantized_max_pool_compute(x, window, stride, qdrtensors, out_dtype,
                                       padding, quant_algo, scale_sqrt), attrs
