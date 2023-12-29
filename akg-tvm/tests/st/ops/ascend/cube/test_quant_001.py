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

import akg
from akg.utils import kernel_exec as utils
import akg.tvm
from akg import dim

block_size = 32


def test_quant(fmap_shape):
    # input shape(NCHW -> NC1HWC0)
    in_n, in_c, in_h, in_w = fmap_shape
    assert in_c % 32 == 0
    input_shape_nc1hwc0 = (in_n, in_c // 16, in_h, in_w, 16)
    in_n, in_c1, in_h, in_w, in_c0 = input_shape_nc1hwc0

    # placeholder (NC1HWC0)
    FMap = akg.tvm.placeholder(input_shape_nc1hwc0, dtype='float16', name='FMap')

    ScaleQ = akg.tvm.placeholder((16, ), dtype='float16', name='ScaleQ')
    OffsetQ = akg.tvm.placeholder((16, ), dtype='float16', name='OffsetQ')

    out_shape_nc1hwc0 = (in_n, in_c // 32, in_h, in_w, 32)
    print(out_shape_nc1hwc0)
    out_n, out_c1, out_h, out_w, out_c0 = out_shape_nc1hwc0

    # quantize
    Quant = akg.tvm.compute(out_shape_nc1hwc0,
                        lambda n, c1, h, w, c0: (FMap[n, c1 + c0 // 16, h, w, c0 % 16] * ScaleQ[0] + OffsetQ[0]).astype('int8'), name='output')

    info = dim.Dim()
    info.setdim(index=0, axis=0, tilel1=2, tilel0=0)
    info.setdim(index=0, axis=0, tilel1=32, tilel0=0)
    info.setdim(index=0, axis=0, tilel1=32, tilel0=0)
    info.setdim(index=0, axis=0, tilel1=16, tilel0=0)

    # schedule
    s = akg.tvm.create_schedule(Quant.op)
    with akg.build_config(add_lower_pass=utils.debug_mode(0), dump_pass_ir=True):
        mod = akg.build(s, [FMap, ScaleQ, OffsetQ, Quant], 'cce', name='cce_quant', attrs={'dim': str(info)}, polyhedral=True)

    source_code = mod.imported_modules[0].get_source()
    print(source_code)


if __name__ == '__main__':
    # test_quant((1, 32, 214, 214))
    test_quant((1, 64, 32, 32))
