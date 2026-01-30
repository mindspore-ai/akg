
from core import *
from api import *


@sub_kernel(core_num=8)
def exp_adds_op(gm_input, gm_output):
    ub_tmp = move_to_ub(gm_input)
    ub_exp = vexp(ub_tmp)
    ub_add = vadds(ub_exp, 1)
    gm_output.load(ub_add)


if __name__ == '__main__':
    set_context("310P")
    gm_input = Tensor(
        "GM", "FP16", [40, 256], format="ND", multi_core=False)
    gm_output = Tensor("GM", "FP16", [40, 256], format="ND", multi_core=False)

    exp_adds_op(gm_input, gm_output)
    compile_kernel("./swft.cce")
