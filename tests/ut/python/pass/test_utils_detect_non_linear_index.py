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

import akg.tvm


def DetectNonLinearIndex(index, constVars=[]):
    f = akg.tvm.get_global_func("cce_util.DetectNonLinearIndex")
    indexInfo = f(index, constVars)
    if (len(indexInfo) == 0):
        return [], []
    indexVars = indexInfo[0]
    indexStrides = indexInfo[1]

    return indexVars, indexStrides


def IsEqualExprList(exprList1, exprList2):
    assert len(exprList1) == len(exprList2)
    for i in range(len(exprList1)):
        if (akg.tvm.ir_pass.Simplify(exprList1[i] - exprList2[i]).value != 0):
            return False
    return True


def test_basic():
    a = akg.tvm.var("a")
    b = akg.tvm.var("b")

    indexVars, indexStrides = DetectNonLinearIndex(a * 4 + b * 6 + 7, [])
    assert IsEqualExprList(indexVars, [a, b])
    assert IsEqualExprList(indexStrides, [4, 6, 7])

    indexVars, indexStrides = DetectNonLinearIndex(a * 4 + b * 6 + 7, [a])
    assert IsEqualExprList(indexVars, [b])
    assert IsEqualExprList(indexStrides, [6, a * 4 + 7])

    indexVars, indexStrides = DetectNonLinearIndex(a * 4 + b * 6 + 7, [b])
    assert IsEqualExprList(indexVars, [a])
    assert IsEqualExprList(indexStrides, [4, b * 6 + 7])

    indexVars, indexStrides = DetectNonLinearIndex(a * 4 + b * 6 + 7, [a, b])
    assert IsEqualExprList(indexVars, [])
    assert IsEqualExprList([indexStrides[-1]], [a * 4 + b * 6 + 7])

    indexVars, indexStrides = DetectNonLinearIndex(a * 4 * (a + 1) + b * 6 + 7, [])
    assert indexVars == []
    assert indexStrides == []

    indexVars, indexStrides = DetectNonLinearIndex(a * 4 * (a + 1) + b * 6 + 7, [a])
    assert IsEqualExprList(indexVars, [b])
    assert IsEqualExprList(indexStrides, [6, a * 4 * (a + 1) + 7])

    indexVars, indexStrides = DetectNonLinearIndex(a * 4 * (a + 1) + b * 6 + 7, [b])
    assert indexVars == []
    assert indexStrides == []

    indexVars, indexStrides = DetectNonLinearIndex(a * 4 * (a + 1) + b * 6 + 7, [a, b])
    assert IsEqualExprList(indexVars, [])
    assert IsEqualExprList([indexStrides[-1]], [a * 4 * (a + 1) + b * 6 + 7])

    indexVars, indexStrides = DetectNonLinearIndex(a * 4 + (a + 1) + b * 6 + 7, [])
    assert IsEqualExprList(indexVars, [a, b])
    assert IsEqualExprList(indexStrides, [5, 6, 8])

    indexVars, indexStrides = DetectNonLinearIndex(a % 3 * 4 + (a + 1) + b * 6 + 7, [])
    assert IsEqualExprList(indexVars, [a % 3, a, b])
    assert IsEqualExprList(indexStrides, [4, 1, 6, 8])

    indexVars, indexStrides = DetectNonLinearIndex(a % 3 * 4 + (a + 1) + b * 6 + 7, [a])
    assert IsEqualExprList(indexVars, [a % 3, b])
    assert IsEqualExprList(indexStrides, [4, 6, ((a + 1) + 7)])

    indexVars, indexStrides = DetectNonLinearIndex(a % 3 * 4 + (a + 1) + b * 6 + 7, [a % 3])
    assert IsEqualExprList(indexVars, [a, b])
    assert IsEqualExprList(indexStrides, [1, 6, ((a % 3) * 4 + 8)])

    indexVars, indexStrides = DetectNonLinearIndex(a % b % 3 * 4 + (a + 1) + b * 6 + 7, [a % b % 3])
    assert IsEqualExprList(indexVars, [a, b])
    assert IsEqualExprList(indexStrides, [1, 6, ((((a % b) % 3) * 4) + 8)])

    indexVars, indexStrides = DetectNonLinearIndex(a % 3 * 4 + (a % 3 + 1) + b * 6 + 7, [])
    assert IsEqualExprList(indexVars, [a % 3, b])
    assert IsEqualExprList(indexStrides, [5, 6, 8])

    indexVars, indexStrides = DetectNonLinearIndex(a * b * 6 + 7, [])
    assert IsEqualExprList(indexVars, [])
    assert IsEqualExprList(indexStrides, [])

    indexVars, indexStrides = DetectNonLinearIndex(a * b * 6 + 7, [a])
    assert IsEqualExprList(indexVars, [b])
    assert IsEqualExprList(indexStrides, [a * 6, 7])

    indexVars, indexStrides = DetectNonLinearIndex(a * b * 6 + 7, [a, b])
    assert IsEqualExprList(indexVars, [])
    assert IsEqualExprList([indexStrides[-1]], [(((a * b) * 6) + 7)])

    indexVars, indexStrides = DetectNonLinearIndex(a % 3 * b * 6 + 7, [a % 3])
    assert IsEqualExprList(indexVars, [b])
    assert IsEqualExprList(indexStrides, [((a % 3) * 6), 7])

    indexVars, indexStrides = DetectNonLinearIndex(a % 3 * b * 6 + 7, [b])
    assert IsEqualExprList(indexVars, [a % 3])
    assert IsEqualExprList(indexStrides, [(b * 6), 7])

    indexVars, indexStrides = DetectNonLinearIndex(a % 3 * b * 6 + 7, [a % 3, b])
    assert IsEqualExprList(indexVars, [])
    assert IsEqualExprList([indexStrides[-1]], [((((a % 3) * b) * 6) + 7)])

    indexVars, indexStrides = DetectNonLinearIndex(a % 3 * b * 6 + 7, [])
    assert IsEqualExprList(indexVars, [])
    assert IsEqualExprList(indexStrides, [])

    indexVars, indexStrides = DetectNonLinearIndex(a % 3 * b * 6 + 7, [a])
    assert IsEqualExprList(indexVars, [])
    assert IsEqualExprList(indexStrides, [])

    indexVars, indexStrides = DetectNonLinearIndex(a % 3 * 6 + 7, [a])
    assert IsEqualExprList(indexVars, [a % 3])
    assert IsEqualExprList(indexStrides, [6, 7])

    indexVars, indexStrides = DetectNonLinearIndex(a % 3 * 6 + 7, [b])
    assert IsEqualExprList(indexVars, [a % 3])
    assert IsEqualExprList(indexStrides, [6, 7])

    indexVars, indexStrides = DetectNonLinearIndex(a % 3 * 6 + 7, [])
    assert IsEqualExprList(indexVars, [a % 3])
    assert IsEqualExprList(indexStrides, [6, 7])

    indexVars, indexStrides = DetectNonLinearIndex(a % 3 * 6, [])
    assert IsEqualExprList(indexVars, [a % 3])
    assert IsEqualExprList(indexStrides, [6, 0])

    cc5 = akg.tvm.var("cc5")
    cc6 = akg.tvm.var("cc6")
    cc4 = akg.tvm.var("cc4")
    cc7 = akg.tvm.var("cc7")
    indexVars, indexStrides = DetectNonLinearIndex(((((((cc5 % 2)*1440) + (cc6*288)) + ((cc5//6)*63)) + (cc7*9)) + ((cc5//2)*3)) + cc4, [])
    assert IsEqualExprList(indexVars, [cc5%2, cc6, cc5//6, cc7, cc5//2, cc4])
    assert IsEqualExprList(indexStrides, [1440, 288, 63, 9, 3, 1, 0])

if __name__ == "__main__":
    test_basic()
