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
from akg.utils import kernel_exec as utils
from akg.ops.array_gpu.one_hot import one_hot
from tests.common.tensorio import compare_tensor
from tests.common.test_run.one_hot_run import gen_data


def test_ms_one_hot(shape, depth, dtype, on_value, off_value, axis, poly_sch=False):
    if poly_sch:
        mod = utils.op_build_test(one_hot, [shape], [dtype], kernel_name="one_hot", op_attrs=[on_value, off_value, depth, axis, dtype], attrs={"target": "cuda"})

    # gen data
    expect, data_tmp, on_value_tensor, off_value_tensor, output = gen_data(axis, depth, dtype, shape, on_value, off_value)
    data = data_tmp.astype(dtype)
    output = utils.mod_launch(mod, (data, output), expect = expect)
    ret = compare_tensor(output, expect, rtol=5e-03, atol=1.e-8, equal_nan=True)
    print("Test {}".format("Pass" if ret else "Failed"))
    if not ret:
        print("Error cuda:========================")
        print(mod.imported_modules[0].get_source())
        raise AssertionError("Test fail")
    return True