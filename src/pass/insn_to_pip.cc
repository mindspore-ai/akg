/**
 * Copyright 2019 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "pass/common.h"

namespace akg {
namespace ir {
// {inst name , pipe}
std::map<std::string, int> inst_map = {{"mad", PIPE_M},
                                       {"vcmax", PIPE_V},
                                       {"vcgmax", PIPE_V},
                                       {"vcmin", PIPE_V},
                                       {"vcgmin", PIPE_V},
                                       {"vcmp_gt", PIPE_V},
                                       {"vcmpv_gt", PIPE_V},
                                       {"vcmp_ge", PIPE_V},
                                       {"vcmpv_ge", PIPE_V},
                                       {"vcmp_lt", PIPE_V},
                                       {"vcmpv_lt", PIPE_V},
                                       {"vcmp_le", PIPE_V},
                                       {"vcmpv_le", PIPE_V},
                                       {"vcmp_eq", PIPE_V},
                                       {"vcmpv_eq", PIPE_V},
                                       {"get_cmpmask", PIPE_V},
                                       {"vmax", PIPE_V},
                                       {"vmin", PIPE_V},
                                       {"vector_dup", PIPE_V},
                                       {"vadd", PIPE_V},
                                       {"vand", PIPE_V},
                                       {"vor", PIPE_V},
                                       {"vsub", PIPE_V},
                                       {"vmul", PIPE_V},
                                       {"vdiv", PIPE_V},
                                       {"vmla", PIPE_V},
                                       {"vaxpy", PIPE_V},
                                       {"vln", PIPE_V},
                                       {"vexp", PIPE_V},
                                       {"vrec", PIPE_V},
                                       {"vsqrt", PIPE_V},
                                       {"vrsqrt", PIPE_V},
                                       {"vabs", PIPE_V},
                                       {"vadds", PIPE_V},
                                       {"vaadd", PIPE_V},
                                       {"vextract", PIPE_V},
                                       {"viou", PIPE_V},
                                       {"vrpac", PIPE_V},
                                       {"vcadd", PIPE_V},
                                       {"vcgadd", PIPE_V},
                                       {"vmuls", PIPE_V},
                                       {"vconv", PIPE_V},
                                       {"vrelu", PIPE_V},
                                       {"vnot", PIPE_V},
                                       {"vsel", PIPE_V},
                                       {"vmadd", PIPE_V},
                                       {"set_cmpmask", PIPE_V},
                                       {"set_vector_mask", PIPE_V},
                                       {"vbitsort", PIPE_V},
                                       {"vmrgsort4", PIPE_V},
                                       {"rpn_cor", PIPE_V},
                                       {"rpn_cor_diag", PIPE_V},
                                       {"vtranspose", PIPE_V},
                                       {"copy_gm_to_ubuf", PIPE_MTE2},
                                       {"copy_ubuf_to_ubuf", PIPE_V},
                                       {"copy_ubuf_to_gm", PIPE_MTE3},
                                       {"copy_ubuf_to_cbuf", PIPE_MTE3},
                                       {"reg_mov", PIPE_S},
                                       {"argmax_cast", PIPE_S},
                                       {"printer_cast", PIPE_S},
                                       {"scatter_vnchwconv_b16", PIPE_V},
                                       {"scatter_vadd", PIPE_V},
                                       {"copy_matrix_cc_to_ubuf", PIPE_V},
                                       {"load_cbuf_to_ca", PIPE_MTE1},
                                       {"load_cbuf_to_cb", PIPE_MTE1},
                                       {"load_gm_to_ca", PIPE_MTE2},
                                       {"load_gm_to_cb", PIPE_MTE2},
                                       {"copy_gm_to_cbuf", PIPE_MTE2},
                                       {"img2col_cbuf_to_ub", PIPE_MTE1},
                                       {"img2col_cbuf_to_ca", PIPE_MTE1},
                                       {"img2col_cbuf_to_cb", PIPE_MTE1},
                                       {"col2img", PIPE_V}};

int GetIntrinPipe(std::string insn) {
  if (insn.find("vconv") == 0) {
    insn = "vconv";
  }

  if (inst_map.count(insn) > 0) {
    return inst_map[insn];
  }

  return 0;
}
}  // namespace ir
}  // namespace akg
