from akg.tvm.hybrid import script
import numpy as np
from akg.utils import kernel_exec as utils


@script(capture=locals())
def hpl_trsm(a, b):
    out = output_tensor(b.shape, b.dtype)
    inverse_0 = allocate(b.shape, b.dtype, "local")
    row = b.shape[0]
    col = b.shape[1]
    for l in range(col // 16):
        for i in range(row):
            for j in range(i):
                for k in range(16):
                    inverse_0[i, l*16+k] = a[i, j] * out[j, l*16+k]
                    out[i, l*16+k] = out[i, l*16+k] - inverse_0[i, l*16+k]
            for k in range(col):
                out[i, l*16+k] = out[i, l*16+k] / a[i, i]
    return out
