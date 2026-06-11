import os

from swft.api import *
from swft.core import *


OP_NAME = "vector_add"
LENGTH = 4096


@sub_kernel(core_num=1)
def vector_add_kernel(input0, input1, output0):
    x_ub = move_to_ub(input0)
    y_ub = move_to_ub(input1)
    out_ub = vadd(x_ub, y_ub)
    output0.load(out_ub)


class ModelNew:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x, y):
        del x, y
        set_context("310P")
        input0 = Tensor("GM", "FP16", [LENGTH], "ND", False)
        input1 = Tensor("GM", "FP16", [LENGTH], "ND", False)
        output0 = Tensor("GM", "FP16", [LENGTH], "ND", False)
        vector_add_kernel(input0, input1, output0)

        current_dir = os.path.dirname(__file__)
        cce_path = os.path.join(current_dir, OP_NAME, OP_NAME + ".cce")
        compile_kernel(cce_path, OP_NAME)
        exec_kernel(
            OP_NAME,
            locals(),
            inputs=["input0", "input1"],
            outputs=["output0"],
            device_id=0,
        )
