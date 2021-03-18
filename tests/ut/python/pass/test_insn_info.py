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
from akg.tvm._ffi.node import NodeBase, register_node
import akg.backend as cce


@register_node
class StmtStoreInfo(NodeBase):
    def __init__(self, comInfo):

        strides = comInfo['strides']
        shape = comInfo['shape']
        var = comInfo['var']
        name = comInfo['name']
        index = comInfo['index']
        dtype = comInfo['dtype']

        scope = ""
        if 'scope' in comInfo:
            scope = comInfo['scope']
        elem_offset = akg.tvm.const(0, 'int32')
        if 'elem_offset' in comInfo:
            elem_offset = comInfo['elem_offset']
        insn_offset = akg.tvm.const(0, 'int32')
        if 'insn_offset' in comInfo:
            insn_offset = comInfo['insn_offset']
        data_alignment = -2
        if 'data_alignment' in comInfo:
            data_alignment = comInfo['data_alignment']
        data = None
        if 'data' in comInfo:
            data = comInfo['data']

        f = akg.tvm.get_global_func("cce_util.create_storeinfo")
        self.__init_handle_by_constructor__(f, strides, shape, var, scope, name, index, elem_offset, insn_offset, dtype, data_alignment, data)


def DicttoStmtStoreInfo(comInfo):
    stmtInfo = StmtStoreInfo(comInfo)

    return stmtInfo


def createComInfo(name, strides, shape, var, index, dtype):
    comInfo = {}
    comInfo['name'] = name
    comInfo['strides'] = strides
    comInfo['shape'] = shape
    comInfo['var'] = var
    comInfo['index'] = index
    comInfo['dtype'] = dtype

    return DicttoStmtStoreInfo(comInfo)


cc1 = akg.tvm.expr.Var("cc1", 'int32')
cc2 = akg.tvm.expr.Var("cc2", 'int32')
cc3 = akg.tvm.expr.Var("cc3", 'int32')
cc4 = akg.tvm.expr.Var("cc4", 'int32')
test_comInfo_list = [createComInfo("input_0_local_UB", [1], [32], [cc2], cc3, 'float16'),
                     createComInfo("input_1_local_UB", [1], [32], [cc3], cc3, 'float16'),
                     createComInfo("input_2_local_UB", [32, 1], [16, 32], [cc1, cc2], cc1 * 32 + cc2, 'float16'),
                     createComInfo("input_3_local_UB", [32, 1], [16, 32], [cc1, cc2], cc1 * 32 + cc2, 'float16'),
                     createComInfo("input_4_local_UB", [16, 1], [64, 16], [cc2, cc3], cc2 * 16 + cc3, 'float16'),
                     createComInfo("input_5_local_UB", [512, 32, 1], [16, 16, 32], [cc1, cc2, cc3], cc1 * 512 + cc2 * 32 + cc3, 'float16'),
                     createComInfo("input_6_local_UB", [256, 32, 1], [16, 16, 16], [cc1, cc2, cc3], cc1 * 256 + cc2 * 16 + cc3, 'float16')]


def testEliminateVarInExpr():
    # expr, elimVars, resultExpr
    test_suite = [[cc3 * 16 + cc4, [cc4], cc3 * 16],
                  [cc1 * 256 + cc2 * 16 + cc3, [cc1, cc3], cc2 * 16],
                  [cc1 * 256 + cc2 * 16 + cc3, [], cc1 * 256 + cc2 * 16 + cc3],
                  [(cc1 * 256 + cc3 * 16) * 32 + cc2 * 32 + cc4, [cc2, cc3], cc1 * 256 * 32 + cc4]]
    idx = 0
    f = akg.tvm.get_global_func("cce_util.EliminateVarInExpr")
    for suite in test_suite:
        result = f(suite[0], suite[1])
        assert str(akg.tvm.ir_pass.Simplify(result)) == str(akg.tvm.ir_pass.Simplify(suite[2])), "case " + str(idx) + " failed"
        idx += 1


def testGetBufScope():
    test_suite = {'input_1_local_UB': cce.scope_ubuf,
                  'input_1_local_L1': cce.scope_cbuf,
                  'input_1_local_L0A': cce.scope_ca,
                  'input_1_local_L0B': cce.scope_cb,
                  'input_1_local_L0C': cce.scope_cc,
                  'input_1_local_UB_dst_tmp': cce.scope_ubuf,
                  'input_1': cce.dma_copy_global}
    idx = 0
    f = akg.tvm.get_global_func("cce_util.GetBufScope")
    for key, value in test_suite.items():
        result = f(key)
        assert result == value, "case " + str(idx) + " failed"
        idx += 1


def testGetVarsInExpr():
    test_suite = [[cc3 * 16 + cc4, [cc3, cc4]],
                  [(cc1 * 256 + cc3 * 16) * 32 + cc2 * 32 + cc4, [cc1, cc3, cc2, cc4]],
                  [cc1 * 256 + cc2 * 16 + cc3, [cc1, cc2, cc3]]]
    f = akg.tvm.get_global_func("cce_util.GetVarsInExpr")
    idx = 0
    for suite in test_suite:
        result = f(suite[0])
        assert set(str(result)) == set(str(suite[1])), "case " + str(idx) + " failed"
        idx += 1


def testIsElementwise():
    test_suite = [[[test_comInfo_list[2]], [test_comInfo_list[2], test_comInfo_list[3]], True],
                  [[test_comInfo_list[0]], [test_comInfo_list[2], test_comInfo_list[0]], False],
                  [[test_comInfo_list[4]], [test_comInfo_list[0], test_comInfo_list[4]], False]]
    f = akg.tvm.get_global_func("cce_util.IsElementwise")
    idx = 0
    for suite in test_suite:
        result = f(suite[0], suite[1])
        assert result == suite[2], "case " + str(idx) + " failed"
        idx += 1


def testIsBroadcast():
    test_suite = [[[test_comInfo_list[2]], [test_comInfo_list[2], test_comInfo_list[3]], False],
                  [[test_comInfo_list[5]], [test_comInfo_list[4], test_comInfo_list[5]], True],
                  [[test_comInfo_list[4]], [test_comInfo_list[0], test_comInfo_list[4]], True]]
    f = akg.tvm.get_global_func("cce_util.IsBroadcast")
    idx = 0
    for suite in test_suite:
        result = f(suite[0], suite[1])
        assert result == suite[2], "case " + str(idx) + " failed"
        idx += 1


def testIsLastAxisReduction():
    test_suite = [[[test_comInfo_list[3]], [test_comInfo_list[3], test_comInfo_list[5]], True],
                  [[test_comInfo_list[2]], [test_comInfo_list[2], test_comInfo_list[3]], False],
                  [[test_comInfo_list[4]], [test_comInfo_list[6], test_comInfo_list[4]], False]]
    f = akg.tvm.get_global_func("cce_util.IsLastAxisReduction")
    idx = 0
    for suite in test_suite:
        result = f(suite[0], suite[1])
        assert result == suite[2], "case " + str(idx) + " failed"
        idx += 1


if __name__ == "__main__":
    testEliminateVarInExpr()
    testGetBufScope()
    testGetVarsInExpr()
    testIsBroadcast()
    testIsElementwise()
