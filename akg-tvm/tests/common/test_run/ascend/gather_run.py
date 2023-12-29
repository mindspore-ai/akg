# Copyright 2019-2021 Huawei Technologies Co., Ltd
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
from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from akg.ops.array.ascend import Gather


def gather_data(params_shape, indices_shape, params_dtype, indices_dtype):
    params_0, _ = params_shape
    indices_0, = indices_shape

    params_half = params_0 // 2
    params_last = params_0 - 1

    indices_last = indices_0 - 1

    params = np.full(params_shape, 1.0, dtype=params_dtype)
    params[2][0] = 2.0
    params[5][0] = 5.0
    params[params_half][0] = 9.0
    params[params_last][0] = 8.0

    indices = np.full(indices_shape, 1, dtype=indices_dtype)
    indices[0] = 2
    indices[1] = 5
    indices[2] = params_last
    indices[indices_last] = params_half

    output_shape = (indices_shape[0], params_shape[1])
    bench_mark = np.full(output_shape, 1.0, dtype=params_dtype)
    bench_mark[0][0] = 2.0
    bench_mark[1][0] = 5.0
    bench_mark[2][0] = 8.0
    bench_mark[indices_last][0] = 9.0
    return params, indices, bench_mark, output_shape


def gather_run(params_shape, indices_shape, params_dtype, indices_dtype, axis, attrs):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(Gather, [params_shape, indices_shape], [params_dtype, indices_dtype], [axis],
                                  kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            params, indices, bench_mark, output_shape = gather_data(params_shape, indices_shape, params_dtype,
                                                                    indices_dtype)
            output = np.full(output_shape, np.nan, params_dtype)
            return mod, bench_mark, (params, indices, output)
        else:
            return mod
    else:
        mod = Gather(params_shape, indices_shape, params_dtype, indices_dtype, axis, "gather_cce", "./")
        params, indices, bench_mark, output_shape = gather_data(params_shape, indices_shape, params_dtype, indices_dtype)

        # mod launch
        output = np.full(output_shape, np.nan, params_dtype)
        output = utils.mod_launch(mod, (params, indices, output), expect=bench_mark)
        compare_res = compare_tensor(output, bench_mark, rtol=5e-03, equal_nan=True)
        print(" ========== PASS ============")
        return (params, indices), output, bench_mark, compare_res
