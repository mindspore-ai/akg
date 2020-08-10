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
from gen_random import random_gaussian
from akg.utils import kernel_exec as utils
from akg import backend as cce
from test_op import matmul_mansch


def matmul_run_mansch(MatrixShape, l1_tiling, l0_tiling, kernel_name, attrs=None):
    mShape = MatrixShape[0]
    kShape = MatrixShape[1]
    nShape = MatrixShape[2]

    mBurstSize = cce.BLOCK_IN
    kBurstSize = cce.BLOCK_REDUCE
    nBurstSize = cce.BLOCK_OUT

    res_dtype = "float%d" % cce.OUT_WIDTH
    A_dtype = "float%d" % cce.INP_WIDTH
    B_dtype = "float%d" % cce.WGT_WIDTH

    # compute matrix shape as cube
    AShape = (mShape // mBurstSize, kShape // kBurstSize, mBurstSize, kBurstSize)
    BShape = (kShape // kBurstSize, nShape // nBurstSize, nBurstSize, kBurstSize)
    CShape = (nShape // nBurstSize, mShape // mBurstSize, mBurstSize, nBurstSize)

    # generate data
    A = random_gaussian(AShape, miu=0.5, sigma=0.01).astype(A_dtype)
    B = random_gaussian(BShape, miu=0.5, sigma=0.01).astype(B_dtype)
    out_data = np.zeros(CShape).astype(res_dtype)

    # launch the kernel
    mod = matmul_mansch.gemm_dsl(MatrixShape, l1_tiling, l0_tiling, kernel_name)
    source_code = mod.imported_modules[0].get_source()
    utils.create_code(kernel_name, ".", source_code)
    res = utils.mod_launch(mod, [A, B, out_data])

    # transform numpy data to compute benchMark
    A = A.swapaxes(1, 2)
    B = B.swapaxes(1, 3)
    B = B.swapaxes(2, 3)
    A = A.reshape((mShape, kShape))
    B = B.reshape((kShape, nShape))
    C = np.zeros((mShape, nShape)).astype(np.float16)
    np.matmul(A, B, C)

    # transform CCE output to (m, n) form
    res = res.swapaxes(0, 2)
    res = res.swapaxes(0, 1)
    res = res.reshape((mShape, nShape))
    assert_res = True
    # compare with numpy
    try:
        np.testing.assert_allclose(res, C, rtol=1e-2, equal_nan=True, verbose=True)
    except:
        assert_res = False
    return (A, B), out_data, C, assert_res
