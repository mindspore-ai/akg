# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
# limitations under the License

import numpy as np
from akg.utils import kernel_exec as utils
from tests.common.gen_random import random_gaussian
from tests.common.test_op.resnet.fused_is_finite import fused_is_finite


def gen_data(shape, dtype, layout='NHWC'):
    support_list = {"float16": np.float16, "float32": np.float32}
    data = random_gaussian(shape, miu=1, sigma=0.1).astype(dtype)
    if layout == "NCHW":
        data = np.transpose(data, axes=(0, 2, 3, 1))
    elif layout != "NHWC":
        raise NotImplementedError('Layout not supported {} '.format(layout))

    data_isfinite = np.isfinite(data)
    n, h, w, c = np.shape(data_isfinite)
    expect = np.all(data_isfinite, axis=(0, 1, 2, 3))
    expect = np.broadcast_to(expect, (1,))
    output = expect
    return data, expect, output


def test_fused_is_finite(shape, layout='NHWC', poly_sch=False):
    dtype = "float32"
    if poly_sch:
        mod = utils.op_build_test(fused_is_finite, [shape], [dtype], op_attrs=[layout], kernel_name="fused_is_finite",
                                  attrs={"target": "cuda",
                                         "enable_akg_reduce_lib": True, "enable_atomic_add": True})

    data, expect, output = gen_data(shape, dtype, layout)
    args = (data, output)
    output = utils.mod_launch(mod, args, expect=expect)
    res = np.allclose(output, expect, rtol=5e-03, atol=1e-8)
    print("Test {}".format("Pass" if res else "Fail"))
    if not res:
        print("Error cuda:========================")
        print(mod.imported_modules[0].get_source())
        raise AssertionError("Test fail")

    return True
