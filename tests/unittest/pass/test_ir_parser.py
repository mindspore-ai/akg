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
from akg.backend import build_module

@script
def test_001(x):
    ''' TEST_CASE_01
    for (i, 0, 16) {
      for(k, 0, 1) {
        if(i > 0) {
          for(j, 0, 1) {
            out(i, k) = 0
          }
        }
        out(i, i) = in(i, 0) * in(0, i)
      }
    }
    '''
    y = output_tensor(x.shape, x.dtype)
    for i in range(x.shape[0]):
        for k in range(1):
            if k > 0:
                for j in range(1):
                    y[i, k] = 0
            y[i, i] = x[i, 0] * x[0, i]
    return y

ans_001 = '\
realize test_001<float16>([0, 16], [0, 16]) {\n\
  produce test_001 {\n\
    for (i, 0, 16) {\n\
      for (k, 0, 1) {\n\
        if ((k > 0)) {\n\
          for (j, 0, 1) {\n\
            test_001(i, k) = 0\n\
          }\n\
        }\n\
        test_001(i, i) = (input(i, 0)*input(0, i))\n\
      }\n\
    }\n\
  }\n\
}\n'

def test(func, ans):
    shape = (16, 16)
    dtype = 'float16'
    x = akg.tvm.placeholder(shape, name='input', dtype=dtype)

    res = func(x)

    s = akg.tvm.create_schedule(res.op)
    bounds = akg.tvm.schedule.InferBound(s)
    stmt = akg.tvm.schedule.ScheduleOps(s, bounds)
    print('---------------BEFORE------------------')
    print(stmt)

    binds, _ = build_module.get_binds([x, res])
    stmt = akg.tvm.ParseHalideIRFromCode(str(stmt), binds)
    print('---------------AFTER-------------------')
    print(stmt)

    assert(str(stmt) == ans)

if __name__ == "__main__":
    test(test_001, ans_001)
