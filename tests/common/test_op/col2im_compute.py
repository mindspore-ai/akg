import akg
import tvm
from akg.utils import kernel_exec as utils
from tests.common.test_op.col2im import intrin_col2im


def col2im_manual_schedule(shape, kernel, stride, pad, dtype, output_H_W, polyhedral=True, attrs=None):
    """
    Col2im operation with manual schedule.

     Args:
        shape (Union[list, tuple]): seven int numbers for the input's image size.
        kernel (Union[list, tuple]): two int numbers for the sliding window's size.
        stride (Union[list, tuple]): two int numbers for the sliding window's stride.
        pad: (Union[list, tuple]): four int numbers for padding's sizes: top, bottom, left, and right
        dtype (str): parameters' type.
        output_H_W (Union[list, tuple]): two int numbers for the output's height and width.
        polyhedral (bool): If True, use auto-schedule, else use manual-schedule, default value is True.
        attrs (dict): Specifies parameters used in manual-schedule.

    Returns:
        tvm.tensor.Tensor as result for col2im operation.
    """

    N, C1, KH, KW, OH, OW, C0 = shape
    H, W = output_H_W
    output_shape = (N, C1, H, W, C0)
    kernel_h, kernel_w = kernel
    stride_h, stride_w = stride
    pad_t, pad_b, pad_l, pad_r = pad

    assert H == (OH - 1) * stride_h + kernel_h - (pad_t + pad_b), "Height of input and output do not match"
    assert W == (OW - 1) * stride_w + kernel_w - (pad_l + pad_r), "Width of input and output do not match"

    col2im = intrin_col2im(shape, output_shape, kernel, stride, pad, dtype)

    # tensor for the input data
    data = tvm.placeholder(shape, dtype, name="input_data")

    # assume we need the whole width of A
    # choose a section of the rows of A that encompasses all of the windows in the current window-batch
    res = tvm.compute(
        output_shape,
        lambda b, c1, h, w, c0:
            data(b, c1, h % KH, w % KW, h % OH, w % OW, c0),
        name="col2im_intrinsic"
    )

    # schedule for differetiation operation
    s = tvm.create_schedule([res.op])

    res_ub = s.cache_write(res, "local.UB")
    data_ub = s.cache_read(data, "local.UB", [res_ub])

    b, c1, h, w, c0 = res.op.axis

    s[data_ub].compute_at(s[res], c1)
    s[res_ub].compute_at(s[res], c1)

    s[res_ub].tensorize(res_ub.op.axis[0], col2im)

    with akg.build_config(add_lower_pass=utils.debug_mode(0), dump_pass_ir=True):
        mod = akg.build(s, [data, res], "cce", name="col2im_manual_schedule", attrs=attrs, polyhedral=polyhedral)
        source_code = mod.imported_modules[0].get_source()
        kernel_name = "col2im_manual_schedule"
        utils.create_code(kernel_name, "./", source_code)
    return mod
