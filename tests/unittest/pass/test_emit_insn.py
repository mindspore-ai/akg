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
import os

test_suite_comInfo = {
}


def StmtStoreInfotoDict(stmtInfo):
    comInfo = {}
    comInfo['name'] = stmtInfo.name
    comInfo['strides'] = list(stmtInfo.strides)
    comInfo['shape'] = list(stmtInfo.shape)
    comInfo['var'] = list(stmtInfo.var)
    comInfo['scope'] = stmtInfo.scope
    comInfo['index'] = stmtInfo.index
    comInfo['elem_offset'] = stmtInfo.elemOffset
    comInfo['insn_offset'] = stmtInfo.insnOffset
    comInfo['insn_offset_as_src'] = stmtInfo.insnOffsetAsSrc
    comInfo['dtype'] = stmtInfo.dtype
    comInfo['data_alignment'] = stmtInfo.dataAlignment
    comInfo['data'] = stmtInfo.data

    return comInfo


def getJson(file):
    if os.path.exists(file):
        f = open(file)
        str = f.read()
        jStr = str.split('*/')[1].strip()
        stmt = akg.tvm.load_json(jStr)
        print(stmt)
        return stmt
    return None


def testComInfo(stmt, result):
    f = akg.tvm.get_global_func("cce_util.GetCompactComputationInfo")
    testInfo = f(stmt, True)
    dstInfoList = [StmtStoreInfotoDict(info) for info in testInfo.dstInfoList]
    srcInfoList = [StmtStoreInfotoDict(info) for info in testInfo.srcInfoList]

    for key, value in result['dst'].items():
        assert dstInfoList[0][key] == value
    for idx, srcDict in enumerate(result['src']):
        for key, value in srcDict.items():
            assert srcInfoList[idx][key] == value


def testEmitInsn(stmt, result):
    stmt = akg.tvm.ir_pass.EmitInsn(stmt)
    assert str(stmt) == result


if __name__ == "__main__":
    debugPath = "../debug_info/"
    for key, result in test_suite_comInfo.items():
        stmt = getJson(debugPath + key)
        if stmt != None:
            testComInfo(stmt, result)
