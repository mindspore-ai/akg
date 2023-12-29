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

import logging
from akg.utils import kernel_exec as utils

LEVEL1_SUCC = "Help tiling level 1 exit successfully"
LEVEL2_SUCC = "Help tiling level 2 exit successfully"
LEVEL3_SUCC = "Help tiling level 3 exit successfully"


def build_five2four(shape_5d, dtype, op_attrs, attrs, kernel_name='five2four', tuning=False):
    from akg.ops.array.ascend import five2four
    utils.op_build_test(five2four.five2four, [shape_5d], [dtype], op_attrs, kernel_name=kernel_name, attrs=attrs, tuning=tuning)


def test_five2four():
    shape_5d = [1, 1, 1088, 1, 16]
    dtype = "float32"
    op_attrs = [[1, 1088, 1, 16], "float32", 'NHWC']
    try:
        attrs = {"help_tiling": 1}
        build_five2four(shape_5d, dtype, op_attrs, attrs)
    except SystemExit:
        logging.info(LEVEL1_SUCC)

    try:
        attrs = {"help_tiling": 2}
        build_five2four(shape_5d, dtype, op_attrs, attrs)
    except SystemExit:
        logging.info(LEVEL2_SUCC)

    try:
        attrs = {"help_tiling": 3}
        build_five2four(shape_5d, dtype, op_attrs, attrs)
    except SystemExit:
        logging.info(LEVEL3_SUCC)


if __name__ == "__main__":
    test_five2four()
