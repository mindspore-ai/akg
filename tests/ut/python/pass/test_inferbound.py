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


def run(cons, e):
    print("=========TEST==========")
    print("constraints:")
    for c in cons:
        print(c)
    print("Res of infer bound of ", e)
    stmt = akg.tvm.ir_pass.TestInferBoundWithCond(e, cons)
    print(stmt)
    print("=========END==========")


def run1(cons, e, var_set):
    print("=========TEST==========")
    print("constraints:")
    for c in cons:
        print(c)
    print("Res of infer bound of ", e)
    stmt = akg.tvm.ir_pass.TestInferBoundWithCond(e, cons, var_set)
    print(stmt)
    print("=========END==========")




def test1():
    T1 = akg.tvm.var("T1")
    e = akg.tvm.expr.Sub(T1, 20)
    cons = list()
    cons.append(T1 > 0)
    cons.append(T1 <= 128)
    run(cons, e)
    # // (-19, 108)

def test_floordiv1():
    T1 = akg.tvm.var("T1")
    e = akg.tvm.expr.Div(T1 + 15, 16) * 16
    cons = list()
    cons.append(T1 > 0)
    fd1 = akg.tvm.expr.FloorDiv(T1 + 15, 16) * 16
    fd2 = akg.tvm.expr.FloorDiv(T1 + 7, 8) * 8
    cons.append(T1 > 0)
    cons.append(fd1 + fd2 <= 3968)
    run(cons, e)
    # // (16, 1984)

def test_maxpool():
    a = akg.tvm.var("a")
    b = akg.tvm.var("b")
    e = b * 9 + 144
    cons = list()
    cons.append(a > 0)
    cons.append(b > 0)
    cons.append(b <= 1819)
    cons.append(b*384 + (1*3+3)*(b+1)*32 <= 126592)
    run(cons, e)
    # // (153, 2119)


def test_scale_inequality():
    a = akg.tvm.var("a")
    b = akg.tvm.var("b")

    e = akg.tvm.expr.FloorDiv((b - 1), 16384) + 1
    #e = akg.tvm.expr.FloorDiv((a - 1), 128) + 1
    cons = list()
    cons.append(akg.tvm.expr.FloorDiv(a+15, 16) * 96 + akg.tvm.expr.FloorDiv(a+7, 8) * 64 <= 126968)
    cons.append(b < a)
    run(cons, e)
    # (0, 1)

def test_polynominial():
    a = akg.tvm.var("a")
    b = akg.tvm.var("b")
    cons = list()
    cons.append(a>0)
    cons.append(b>0)
    cons.append(b<=1819)
    cons.append(b*384+((a*3)+3)*(b+1)*32<=126592)
    e = (a*3+3)*(b+1)*16
    run(cons, e)
    #(192, 63072)

def test_conv():
    a = akg.tvm.var("a")
    b = akg.tvm.var("b")
    c = akg.tvm.var("c")
    cons = list()
    cons.append(a>0)
    cons.append(b>0)
    cons.append(c>0)
    cons.append((b*16)*a <= 2047)
    cons.append((a*16)*c <= 1023)
    cons.append((b*16)*c <= 1023)
    e = 256*a
    run(cons, e)
    # (256, 16128)


def test_min():
    a = akg.tvm.var("a")
    b = akg.tvm.var("b")
    CI1 =akg.tvm.var("CI1")
    H = akg.tvm.var("H")
    cc2 = akg.tvm.var("CC2")
    cc3 = akg.tvm.var("cc3")
    W = akg.tvm.var("W")
    e = (((CI1*((akg.tvm.expr.Min(2, (H - cc2)) + 1) - akg.tvm.expr.Max(0, (1 - cc2))))*((akg.tvm.expr.Min(29, (W - (cc3*28))) + 1) - akg.tvm.expr.Max(0, (1 - (cc3*28)))))*16)
    cond = [(cc3 >= 0), (cc3 < (akg.tvm.div((W - 1),28) + 1)), (cc2 >= 0), (cc2 < H), (H > 0), (W > 0)]
    run(cond, e)

def test_poly2():
    a = akg.tvm.var("a")
    b = akg.tvm.var("b")
    conds = list()
    conds.append(a*b*480 + a*64 + 4192 <= 131072)
    conds.append(b < 40)
    conds.append(b > 0)
    run(conds, a)




if __name__ == "__main__":
    test1()
    test_floordiv1()
    test_maxpool()
    test_scale_inequality()
    test_polynominial()
    test_conv()
    test_min()
    test_poly2()
