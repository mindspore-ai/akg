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
import akg
import akg.tvm
from akg.utils import kernel_exec as utils
from tests.common.test_run.proposal_sort_run import np_proposal_sort
from tests.common.test_op import proposal_sort
from tests.common.tensorio import compare_tensor


def test_001():
    shape = (1, 256, 8)
    topk = int(32)
    score_threshold = float(0)
    dtype = "float16"
    kernel_name = "cce_proposal_sort_fp16"
    attrs = None

    data_np, expect = np_proposal_sort(shape, topk, score_threshold)
    output = np.full(expect.shape, 0, dtype)

    data = akg.tvm.placeholder(shape, dtype, "input_1")
    out = proposal_sort.proposal_sort(data, topk, score_threshold)

    s = akg.tvm.create_schedule(out.op)
    with akg.build_config(add_lower_pass=[(0, akg.tvm.ParseHalideIRFromCode)], dump_pass_ir=False):
        mod = akg.build(s, [data, out], "cce", name="proposal_sort", polyhedral=True)
    output = utils.mod_launch(mod, (data_np, output))
    test_case_result = compare_tensor(output, expect, rtol=5e-03, equal_nan=True)
    assert(test_case_result)
    print(" ========== PARSER PASSED ============")


if __name__ == "__main__":
    test_001()
