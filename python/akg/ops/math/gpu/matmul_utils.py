import numpy as np
import akg.topi as topi


def auto_out_transpose(expect, layout_out="NHDT"):
    if len(expect.shape) == 3:
        layout_out = layout_out[1:]
    if len(expect.shape) == 2:
        layout_out = layout_out[2:]
    layout_out_int = layout_out.replace('N', '0').replace(
        'H', '1').replace('D', '2').replace('T', '3')
    layout_out_list = list(layout_out_int)
    layout_out_axis = np.argsort(layout_out_list)
    expect = topi.transpose(expect, axes=tuple(layout_out_axis))

    return expect
