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

import akg
import akg.tvm


def test1():
    T1 = akg.tvm.var("T1")
    T2 = akg.tvm.var("T2")
    H = akg.tvm.var("H")
    cc2 = akg.tvm.var("cc2")
    e = T1 <= ((((T1*-1)*cc2) + 1) + akg.tvm.expr.Div((((1 + 1) - 3) + H), 2))
    print("=========TEST==========")
    print("constraints:")
    print("Res of infer bound of ", T1)
    stmt = akg.tvm.ir_pass.TestReduceInequality(e, cc2)
    print(stmt)
    print("=========END==========")



def test2():
    T1 = akg.tvm.var("T1")
    T2 = akg.tvm.var("T2")
    H = akg.tvm.var("H")
    cc2 = akg.tvm.var("cc2")
    e = ((((T1 - 1)*2) + 3) - 1)  <= ((H + (1 - 1)) - (cc2*(T1*2)))
    print("=========TEST==========")
    print("constraints:")
    print("Res of infer bound of ", T1)
    stmt = akg.tvm.ir_pass.TestReduceInequality(e, cc2)
    print(stmt)
    print("=========END==========")


def test_simplify():
    T1 = akg.tvm.var("T1")
    e = akg.tvm.expr.Div(T1, T1)
    print("=========TEST==========")
    stmt = akg.tvm.ir_pass.TestSimplify(e)
    print(stmt)
    print("=========END==========")

def test_gcd():
    T1 = akg.tvm.var("T1")
    print("=========TEST==========")
    stmt = akg.tvm.ir_pass.TestGcd(T1, 0)
    print(stmt)
    print("=========END==========")


if __name__ == "__main__":
    test1()
    test_simplify()
