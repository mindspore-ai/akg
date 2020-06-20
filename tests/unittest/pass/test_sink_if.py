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
from akg.tvm.hybrid import script


@script
def test_001(x):
    ''' TEST_CASE_01
    for (i, 0, 16) {
      if((i * 2) > 8) {
        for(j, 0, 10) {
          for(k, 0, 16) {
            out(i, j) = in(i, j)
          }
        }
      }
    }
    '''
    y = output_tensor(x.shape, x.dtype)
    for i in range(x.shape[0]):
        if i * 2 > 8:
            for k in range(10):
                for j in range(x.shape[1]):
                    y[i, j] = x[i, j]
    return y


ans_001 = '\
for (i, 0, 16) {\n\
  for (k, 0, 10) {\n\
    for (j, 0, 16) {\n\
      if (((i*2) > 8)) {\n\
        test_001(i, j) = in(i, j)\n\
      }\n\
    }\n\
  }\n\
}\n'


@script
def test_002_helper(x):
    y0 = output_tensor(x.shape, x.dtype)
    y1 = output_tensor(x.shape, x.dtype)
    for i in range(x.shape[0]):
        if i * 2 > 8:
            for k in range(10):
                for j in range(x.shape[1]):
                    y0[i, j] = x[i, j]
            for j in range(x.shape[1]):
                for k in range(11):
                    y1[i, j] = x[i, j]
    return y0, y1


@script
def test_002(x):
    ''' TEST_CASE_02
    for (i, 0, 16) {
      if((i * 2) > 8) {
        for(k, 0, 10) {
          for(j, 0, 16) {
            out1(i, j) = in(i, j)
          }
        }
        for(j, 0, 16) {
          for(k, 0, 11) {
            out2(i, j) = in(i, j)
          }
        }
      }
    }
    out(0, 0) = out1(0, 0) + out2(1, 1)
    '''
    y0, y1 = test_002_helper(x)
    y = output_tensor(x.shape, x.dtype)
    y[0, 0] = y0[0, 0] + y1[1, 1]
    return y


ans_002 = '\
for (i, 0, 16) {\n\
  for (k, 0, 10) {\n\
    for (j, 0, 16) {\n\
      if (((i*2) > 8)) {\n\
        test_002_helper(i, j).value[0] = in(i, j)\n\
      }\n\
    }\n\
  }\n\
  for (j, 0, 16) {\n\
    for (k, 0, 11) {\n\
      if (((i*2) > 8)) {\n\
        test_002_helper(i, j).value[1] = in(i, j)\n\
      }\n\
    }\n\
  }\n\
}\n'


@script
def test_003(x):
    ''' TEST_CASE_03
    for (i, 0, 16) {
      if((i * 2) > 8) {
        for(k, 0, 10) {
          for(j, 0, 16) {
            out(i, j) = 0
          }
        }
      } else {
        for(j, 0, 16) {
          for(k, 0, 11) {
            out(i, j) = in(i, j)
          }
        }
      }
    }
    '''
    y = output_tensor(x.shape, x.dtype)
    for i in range(x.shape[0]):
        if i * 2 > 8:
            for k in range(10):
                for j in range(x.shape[1]):
                    y[i, j] = 0
        else:
            for j in range(x.shape[1]):
                for k in range(11):
                    y[i, j] = x[i, j]
    return y


ans_003 = '\
for (i, 0, 16) {\n\
  for (k, 0, 10) {\n\
    for (j, 0, 16) {\n\
      if (((i*2) > 8)) {\n\
        test_003(i, j) = 0\n\
      }\n\
    }\n\
  }\n\
  for (j, 0, 16) {\n\
    for (k, 0, 11) {\n\
      if (!((i*2) > 8)) {\n\
        test_003(i, j) = in(i, j)\n\
      }\n\
    }\n\
  }\n\
}\n'


@script
def test_004(x):
    ''' TEST_CASE_04
    for (i, 0, 16) {
      if((i * 2) > 8) {
        for(k, 0, 10) {
          for(j, 0, 16) {
            out(i, j) = 0
          }
        }
      } else {
        out(i, i) = in(i, i)
      }
    }
    '''
    y = output_tensor(x.shape, x.dtype)
    for i in range(x.shape[0]):
        if i * 2 > 8:
            for k in range(10):
                for j in range(x.shape[1]):
                    y[i, j] = 0
        else:
            y[i, i] = x[i, i]
    return y


ans_004 = '\
for (i, 0, 16) {\n\
  for (k, 0, 10) {\n\
    for (j, 0, 16) {\n\
      if (((i*2) > 8)) {\n\
        test_004(i, j) = 0\n\
      }\n\
    }\n\
  }\n\
  if (!((i*2) > 8)) {\n\
    test_004(i, i) = in(i, i)\n\
  }\n\
}\n'


@script
def test_005(x):
    ''' TEST_CASE_05
    for (i, 0, 16) {
      if((i * 2) > 8) {
        out(i, i) = in(i, 0)
        for(k, 0, 10) {
          for(j, 0, 16) {
            out(i, j) = 0
          }
        }
      } else {
        out(i, i) = in(i, i)
      }
    }
    '''
    y = output_tensor(x.shape, x.dtype)
    for i in range(x.shape[0]):
        if i * 2 > 8:
            y[i, i] = x[i, 0]
            for k in range(10):
                for j in range(x.shape[1]):
                    y[i, j] = 0
        else:
            y[i, i] = x[i, i]
    return y


ans_005 = '\
for (i, 0, 16) {\n\
  if (((i*2) > 8)) {\n\
    test_005(i, i) = in(i, 0)\n\
  }\n\
  for (k, 0, 10) {\n\
    for (j, 0, 16) {\n\
      if (((i*2) > 8)) {\n\
        test_005(i, j) = 0\n\
      }\n\
    }\n\
  }\n\
  if (!((i*2) > 8)) {\n\
    test_005(i, i) = in(i, i)\n\
  }\n\
}\n'


@script
def test_006(x):
    ''' TEST_CASE_06
    for (i, 0, 16) {
      if((i * 2) > 8) {
        for(k, 0, 10) {
          if (k > 5) {
            for(j, 0, 16) {
              out(i, j) = 0
            }
          }
        }
        out(i, i) = in(i, 0)
      } else {
        out(i, i) = in(i, i)
      }
    }
    '''
    y = output_tensor(x.shape, x.dtype)
    for i in range(x.shape[0]):
        if i * 2 > 8:
            for k in range(10):
                if k > 5:
                    for j in range(x.shape[1]):
                        y[i, j] = 0
            y[i, i] = x[i, 0]
        else:
            y[i, i] = x[i, i]
    return y


ans_006 = '\
for (i, 0, 16) {\n\
  for (k, 0, 10) {\n\
    for (j, 0, 16) {\n\
      if (((i*2) > 8)) {\n\
        if ((k > 5)) {\n\
          test_006(i, j) = 0\n\
        }\n\
      }\n\
    }\n\
  }\n\
  if (((i*2) > 8)) {\n\
    test_006(i, i) = in(i, 0)\n\
  }\n\
  if (!((i*2) > 8)) {\n\
    test_006(i, i) = in(i, i)\n\
  }\n\
}\n'


def test(func, ans):
    shape = (16, 16)
    dtype = 'float16'
    x = akg.tvm.placeholder(shape, name='in', dtype=dtype)

    res = func(x)

    s = akg.tvm.create_schedule(res.op)
    bounds = akg.tvm.schedule.InferBound(s)
    stmt = akg.tvm.schedule.ScheduleOps(s, bounds)
    print('---------------BEFORE------------------')
    print(stmt)

    stmt = akg.tvm.ir_pass.SinkIfStmt(stmt)
    print('---------------AFTER-------------------')
    print(stmt)

    for_node = stmt

    def GetNode(op):
        nonlocal for_node
        if isinstance(op, akg.tvm.stmt.For):
            for_node = op
    akg.tvm.ir_pass.PostOrderVisit(stmt, GetNode)
    assert(str(for_node) == ans)


if __name__ == "__main__":
    test(test_001, ans_001)
    test(test_002, ans_002)
    test(test_003, ans_003)
    test(test_004, ans_004)
    test(test_005, ans_005)
    test(test_006, ans_006)
