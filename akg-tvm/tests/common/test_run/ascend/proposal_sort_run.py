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
from tests.common.test_op.ascend import proposal_sort



def np_proposal_sort(shape, topk, dtype):
    bs, box_number, _ = shape
    a = np.zeros([bs, box_number, 8], dtype=np.float16)
    indice = np.arange(box_number, dtype=np.float16)
    item = np.arange(box_number, dtype=np.int16)
    item = np.frombuffer(item.tobytes(), np.float16)
    np.random.shuffle(item)
    out = np.zeros([bs, topk, 8], dtype=np.float16)
    for i in range(bs):
        order = np.argsort(item)[::-1]
        for j in range(box_number):
            a[i, j, 0] = indice[j]
            a[i, j, 4] = item[j]
        for j in range(topk):
            out[i, j] = a[i, order[j]]
    output = np.full(out.shape, 0, dtype)
    return a, out, output


def proposal_sort_run(shape, topk: int, dtype, kernel_name, attrs):
    op_attrs = [topk]

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(proposal_sort.proposal_sort, [shape], [dtype], op_attrs=op_attrs,
                                  kernel_name=kernel_name, attrs=attrs, log_code=False, tuning=t)
        if t:
            data, expect, output = np_proposal_sort(shape, topk,  dtype)
            return mod, expect, (data, output)
        else:
            return mod
    else:
        mod = utils.op_build_test(proposal_sort.proposal_sort, [shape], [dtype], op_attrs=op_attrs,
                                  kernel_name=kernel_name, attrs=attrs, log_code=False)
        data, expect, output = np_proposal_sort(shape, topk, dtype)
        output = utils.mod_launch(mod, (data, output), expect=expect)
        test_case_result = compare_tensor(output, expect, rtol=5e-03, equal_nan=True)
        print(" ========== PASS ============")
        return data, output, expect, test_case_result
