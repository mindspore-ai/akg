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

import akg
import akg.tvm
from akg import backend as cce
from akg.backend.cce_build import build_config

# a simple copy loop from Tensor B to Tensor A
A = akg.tvm.placeholder((33,), name="A")
B = akg.tvm.compute((33,), lambda i: A[i] * A[i], name="B")

s = akg.tvm.create_schedule(B.op)
A_ubuf = s.cache_read(A, cce.scope_ubuf, [B])
B_ubuf = s.cache_write(B, cce.scope_ubuf)

s[B_ubuf].emit_insn(B_ubuf.op.axis[0], "vec_binary_mul")
s[A_ubuf].emit_insn(A_ubuf.op.axis[0], "dma_copy")
s[B].emit_insn(B.op.axis[0], "dma_copy")
s[A_ubuf].buffer_align((1, 16))

with build_config:
    mod = akg.build(s, [B, A], "cce", name="test_kernel")

source_code = mod.imported_modules[0].get_source()
print(source_code)
