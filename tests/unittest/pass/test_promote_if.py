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
      for(j, 0, 10) {
        for(k, 0, 16) {
          if((i * 2) > 8) {
            out(i, j) = in(i, j)
          }
        }
      }
    }
    '''
    y = output_tensor(x.shape, x.dtype)
    for i in range(x.shape[0]):
        for k in range(10):
            for j in range(x.shape[1]):
                if i * 2 > 8:
                    y[i, j] = x[i, j]
    return y


ans_001 = '\
for (i, 0, 16) {\n\
  if ((4 < i)) {\n\
    for (k, 0, 10) {\n\
      for (j, 0, 16) {\n\
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
        for k in range(10):
            for j in range(x.shape[1]):
                if i * 2 > 8:
                    y0[i, j] = x[i, j]
        for j in range(x.shape[1]):
            for k in range(11):
                if i * 2 > 8:
                    y1[i, j] = x[i, j]
    return y0, y1


@script
def test_002(x):
    ''' TEST_CASE_02
    for (i, 0, 16) {
      for (k, 0, 10) {
        for (j, 0, 16) {
          if (((i*2) > 8)) {
            test_002_helper(i, j).value[0] =in(i, j)
          }
        }
      }
      for (j, 0, 16) {
        for (k, 0, 11) {
          if (((i*2) > 8)) {
            test_002_helper(i, j).value[1] =in(i, j)
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
  if ((4 < i)) {\n\
    for (k, 0, 10) {\n\
      for (j, 0, 16) {\n\
        test_002_helper(i, j).value[0] = in(i, j)\n\
      }\n\
    }\n\
    for (j, 0, 16) {\n\
      for (k, 0, 11) {\n\
        test_002_helper(i, j).value[1] = in(i, j)\n\
      }\n\
    }\n\
  }\n\
}\n'


@script
def test_003(x):
    ''' TEST_CASE_03
    for (i, 0, 16) {
      for(k, 0, 10) {
        for(j, 0, 16) {
          if((i * 2) > 8) {
            out(i, j) = 0
          }
        }
      for(j, 0, 16) {
        for(k, 0, 11) {
          if((i * 2) <= 8) {
            out(i, j) = in(i, j)
        }
      }
    }
    '''
    y = output_tensor(x.shape, x.dtype)
    for i in range(x.shape[0]):
        for k in range(10):
            for j in range(x.shape[1]):
                if i * 2 > 8:
                    y[i, j] = 0
        for j in range(x.shape[1]):
            for k in range(11):
                if i * 2 <= 8:
                    y[i, j] = x[i, j]
    return y


ans_003 = '\
for (i, 0, 16) {\n\
  if ((4 < i)) {\n\
    for (k, 0, 10) {\n\
      for (j, 0, 16) {\n\
        test_003(i, j) = 0\n\
      }\n\
    }\n\
  } else {\n\
    for (j, 0, 16) {\n\
      for (k, 0, 11) {\n\
        test_003(i, j) = in(i, j)\n\
      }\n\
    }\n\
  }\n\
}\n'


@script
def test_004(x):
    ''' TEST_CASE_04
    for (i, 0, 16) {
      for(k, 0, 10) {
        for(j, 0, 16) {
          if((i * 2) > 8) {
            out(i, j) = 0
          }
        }
      }
      if((i * 2) <= 8) {
        out(i, i) = in(i, i)
      }
    }
    '''
    y = output_tensor(x.shape, x.dtype)
    for i in range(x.shape[0]):
        for k in range(10):
            for j in range(x.shape[1]):
                if i * 2 > 8:
                    y[i, j] = 0
        if i * 2 <= 8:
            y[i, i] = x[i, i]
    return y


ans_004 = '\
for (i, 0, 16) {\n\
  if ((4 < i)) {\n\
    for (k, 0, 10) {\n\
      for (j, 0, 16) {\n\
        test_004(i, j) = 0\n\
      }\n\
    }\n\
  } else {\n\
    test_004(i, i) = in(i, i)\n\
  }\n\
}\n'


@script
def test_005(x):
    ''' TEST_CASE_05
    for (i, 0, 16) {
      if((i * 2) > 8) {
        out(i, i) = in(i, 0)
      }
      for(k, 0, 10) {
        for(j, 0, 16) {
          if((i * 2) > 8) {
            out(i, j) = 0
          }
        }
      }
      if((i * 2) <= 8) {
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
                if i * 2 > 8:
                    y[i, j] = 0
        if i * 2 <= 8:
            y[i, i] = x[i, i]
    return y


ans_005 = '\
for (i, 0, 16) {\n\
  if ((4 < i)) {\n\
    test_005(i, i) = in(i, 0)\n\
    for (k, 0, 10) {\n\
      for (j, 0, 16) {\n\
        test_005(i, j) = 0\n\
      }\n\
    }\n\
  } else {\n\
    test_005(i, i) = in(i, i)\n\
  }\n\
}\n'


@script
def test_006(x):
    ''' TEST_CASE_06
    for (i, 0, 16) {
      for(k, 0, 10) {
        for(j, 0, 16) {
          if((i * 2) > 8) {
            if (k > 5) {
              out(i, j) = 0
            } else {
              out(i, j) = 1
            }
          }
        }
      }
      if((i * 2) > 8) {
        out(i, i) = in(i, 0)
      }
      if((i * 2) <= 8) {
        out(i, i) = in(i, i)
      }
    }
    '''
    y = output_tensor(x.shape, x.dtype)
    for i in range(x.shape[0]):
        for k in range(10):
            for j in range(x.shape[1]):
                if i * 2 > 8:
                    if k > 5:
                        y[i, j] = 0
                    else:
                        y[i, j] = 1
        if i * 2 > 8:
            y[i, i] = x[i, 0]
        if i * 2 <= 8:
            y[i, i] = x[i, i]
    return y


ans_006 = '\
for (i, 0, 16) {\n\
  if ((4 < i)) {\n\
    for (k, 0, 10) {\n\
      if ((5 < k)) {\n\
        for (j, 0, 16) {\n\
          test_006(i, j) = 0\n\
        }\n\
      } else {\n\
        for (j, 0, 16) {\n\
          test_006(i, j) = 1\n\
        }\n\
      }\n\
    }\n\
    test_006(i, i) = in(i, 0)\n\
  } else {\n\
    test_006(i, i) = in(i, i)\n\
  }\n\
}\n'


@script
def test_007(x):
    ''' TEST_CASE_07
    for (i, 0, 16) {
      for(k, 0, 10) {
        for(j, 0, 16) {
          if (k > 5) {
            if((i * 2) > 8) {
              out(i, j) = 0
            }
          } else {
            if((i * 2) > 8) {
              out(i, j) = 1
            }
          }
        }
      }
      if((i * 2) > 8) {
        out(i, i) = in(i, 0)
      }
      if((i * 2) <= 8) {
        out(i, i) = in(i, i)
      }
    }
    '''
    y = output_tensor(x.shape, x.dtype)
    for i in range(x.shape[0]):
        for k in range(10):
            for j in range(x.shape[1]):
                if k > 5:
                    if i * 2 > 8:
                        y[i, j] = 0
                else:
                    if i * 2 > 8:
                        y[i, j] = 1
        if i * 2 > 8:
            y[i, i] = x[i, 0]
        if i * 2 <= 8:
            y[i, i] = x[i, i]
    return y


ans_007 = '\
for (i, 0, 16) {\n\
  if ((4 < i)) {\n\
    for (k, 0, 10) {\n\
      if ((5 < k)) {\n\
        for (j, 0, 16) {\n\
          test_007(i, j) = 0\n\
        }\n\
      } else {\n\
        for (j, 0, 16) {\n\
          test_007(i, j) = 1\n\
        }\n\
      }\n\
    }\n\
    test_007(i, i) = in(i, 0)\n\
  } else {\n\
    test_007(i, i) = in(i, i)\n\
  }\n\
}\n'


@script
def test_008(x):
    ''' TEST_CASE_08
    for (i, 0, 16) {
      for(k, 0, 10) {
        if (k > 5) {
          for(j, 0, 16) {
            if (j > 3) {
              if (i < 7) {
                out(0, 1) = 0
                if (k >= 9) {
                  out(i, j) = 1
                }
              }
            }
          }
        }
        if (i < 7) {
          for(j, 0, 16) {
            if (j > 3) {
              if (k > 5) {
                if (k < 9) {
                  out(i, j) = 2
                } else {
                  out(j, i) = 3
                }
              }
            }
          }
        } else {
          if (k > 5) {
            for(j, 0, 16) {
              if (j > 3) {
                if (k < 9) {
                  out(i, j) = 4
                } else {
                  out(j, i) = 5
                }
              }
            }
          }
        }
      }
    }
    '''
    y = output_tensor(x.shape, x.dtype)
    for i in range(x.shape[0]):
        for k in range(10):
            if k > 5:
                for j in range(x.shape[1]):
                    if j > 3:
                        if i < 7:
                            y[0, 1] = 0
                            if k >= 9:
                                y[i, j] = 1
            if i < 7:
                for j in range(x.shape[1]):
                    if j > 3:
                        if k > 5:
                            if k < 9:
                                y[i, j] = 2
                            else:
                                y[j, i] = 3
            else:
                if k > 5:
                    for j in range(x.shape[1]):
                        if j > 3:
                            if k < 9:
                                y[i, j] = 4
                            else:
                                y[j, i] = 5
    return y


ans_008 = '\
for (i, 0, 16) {\n\
  if ((i < 7)) {\n\
    for (k, 0, 10) {\n\
      if ((5 < k)) {\n\
        for (j, 0, 16) {\n\
          if ((3 < j)) {\n\
            test_008(0, 1) = 0\n\
            if ((9 <= k)) {\n\
              test_008(i, j) = 1\n\
            }\n\
          }\n\
        }\n\
        if ((k < 9)) {\n\
          for (j, 0, 16) {\n\
            if ((3 < j)) {\n\
              test_008(i, j) = 2\n\
            }\n\
          }\n\
        } else {\n\
          for (j, 0, 16) {\n\
            if ((3 < j)) {\n\
              test_008(j, i) = 3\n\
            }\n\
          }\n\
        }\n\
      }\n\
    }\n\
  } else {\n\
    for (k, 0, 10) {\n\
      if ((5 < k)) {\n\
        if ((k < 9)) {\n\
          for (j, 0, 16) {\n\
            if ((3 < j)) {\n\
              test_008(i, j) = 4\n\
            }\n\
          }\n\
        } else {\n\
          for (j, 0, 16) {\n\
            if ((3 < j)) {\n\
              test_008(j, i) = 5\n\
            }\n\
          }\n\
        }\n\
      }\n\
    }\n\
  }\n\
}\n'


@script
def test_009(x):
    ''' TEST_CASE_09
    for (i, 0, 16) {
      for(k, 0, 10) {
        for(j, 0, 16) {
          if (k > 5) {
            if (j > 3) {
              if (i < 7) {
                out(0, 1) = 0
                if(k >= 9) {
                  out(i, j) = 1
                }
              }
            }
          }
          if (i < 7) {
            if (j > 3) {
              if (k > 5) {
                if(k < 9) {
                  out(i, j) = 2
                } else {
                  out(j, i) = 3
                }
              }
            }
          } else {
            if (k > 5) {
              if (j > 3) {
                if(k < 9) {
                  out(i, j) = 4
                } else {
                  out(j, i) = 5
                }
              }
            }
          }
        }
      }
    }
    '''
    y = output_tensor(x.shape, x.dtype)
    for i in range(x.shape[0]):
        for k in range(10):
            for j in range(x.shape[1]):
                if k > 5:
                    if j > 3:
                        if i < 7:
                            y[0, 1] = 0
                            if k >= 9:
                                y[i, j] = 1
                if i < 7:
                    if j > 3:
                        if k > 5:
                            if k < 9:
                                y[i, j] = 2
                            else:
                                y[j, i] = 3
                else:
                    if k > 5:
                        if j > 3:
                            if k < 9:
                                y[i, j] = 4
                            else:
                                y[j, i] = 5
    return y


ans_009 = '\
for (i, 0, 16) {\n\
  if ((i < 7)) {\n\
    for (k, 0, 10) {\n\
      if ((5 < k)) {\n\
        for (j, 0, 16) {\n\
          if ((3 < j)) {\n\
            test_009(0, 1) = 0\n\
            if ((9 <= k)) {\n\
              test_009(i, j) = 1\n\
              test_009(j, i) = 3\n\
            } else {\n\
              test_009(i, j) = 2\n\
            }\n\
          }\n\
        }\n\
      }\n\
    }\n\
  } else {\n\
    for (k, 0, 10) {\n\
      if ((5 < k)) {\n\
        if ((k < 9)) {\n\
          for (j, 0, 16) {\n\
            if ((3 < j)) {\n\
              test_009(i, j) = 4\n\
            }\n\
          }\n\
        } else {\n\
          for (j, 0, 16) {\n\
            if ((3 < j)) {\n\
              test_009(j, i) = 5\n\
            }\n\
          }\n\
        }\n\
      }\n\
    }\n\
  }\n\
}\n'


@script
def test_010(x):
    ''' TEST_CASE_10
    for (l, 0, 12) {
      for (i, 0, 16) {
        for(k, 0, 10) {
          for(j, 0, 16) {
            if (k > 5) {
              if (j > 3) {
                if (i > 2) {
                  if (i < 11) {
                    if (k < 9) {
                      if (j < 9) {
                        if (l < 7) {
                          if (j > 5) {
                            out(i, j) = 0
                          } else {
                            out(j, i) = 1
                          }
                        }
                      }
                    }
                  } else {
                    if (j < 8) {
                      if (l < 7) {
                        if (k < 9) {
                          out(i, j) = 2
                        } else {
                          out(j, i) = 3
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    '''
    y = output_tensor(x.shape, x.dtype)
    for l in range(12):
        for i in range(x.shape[0]):
            for k in range(10):
                for j in range(x.shape[1]):
                    if k > 5:
                        if j > 3:
                            if i > 2:
                                if i < 11:
                                    if k < 9:
                                        if j < 9:
                                            if l < 7:
                                                if j > 5:
                                                    y[i, j] = 0
                                                else:
                                                    y[j, i] = 1
                                else:
                                    if j < 8:
                                        if l < 7:
                                            if k < 9:
                                                y[i, j] = 2
                                            else:
                                                y[j, i] = 3
    return y


ans_010 = '\
for (l, 0, 12) {\n\
  if ((l < 7)) {\n\
    for (i, 0, 16) {\n\
      if ((2 < i)) {\n\
        if ((i < 11)) {\n\
          for (k, 0, 10) {\n\
            if ((5 < k)) {\n\
              if ((k < 9)) {\n\
                for (j, 0, 16) {\n\
                  if ((3 < j)) {\n\
                    if ((j < 9)) {\n\
                      if ((5 < j)) {\n\
                        test_010(i, j) = 0\n\
                      } else {\n\
                        test_010(j, i) = 1\n\
                      }\n\
                    }\n\
                  }\n\
                }\n\
              }\n\
            }\n\
          }\n\
        } else {\n\
          for (k, 0, 10) {\n\
            if ((5 < k)) {\n\
              if ((k < 9)) {\n\
                for (j, 0, 16) {\n\
                  if ((3 < j)) {\n\
                    if ((j < 8)) {\n\
                      test_010(i, j) = 2\n\
                    }\n\
                  }\n\
                }\n\
              } else {\n\
                for (j, 0, 16) {\n\
                  if ((3 < j)) {\n\
                    if ((j < 8)) {\n\
                      test_010(j, i) = 3\n\
                    }\n\
                  }\n\
                }\n\
              }\n\
            }\n\
          }\n\
        }\n\
      }\n\
    }\n\
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

    stmt = akg.tvm.ir_pass.PromoteIfStmt(stmt, False)
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
    test(test_007, ans_007)
    test(test_008, ans_008)
    test(test_009, ans_009)
    test(test_010, ans_010)
