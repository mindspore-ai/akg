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

"""unittest for micro-tuning"""
from akg.utils import kernel_exec
from akg.ops.array.ascend import four2five


def test_four2five_without_custom_tiling(build_shape, dtype, op_attrs):
    """This test case will fail without cunstom tiling and micro-tuning will automatically adjust tile sizes."""
    build_attr = op_attrs + [False]
    return kernel_exec.op_build_test(four2five.four2five, [build_shape], [dtype], build_attr, kernel_name="four2five", attrs={}, tuning=False)


if __name__ == "__main__":
    test_four2five_without_custom_tiling(
        [32, 1001, 1, 1], "float16", ['NCHW', 'float16'])
