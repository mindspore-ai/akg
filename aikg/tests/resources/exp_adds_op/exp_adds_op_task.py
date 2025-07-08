import numpy as np
from save_data_utils import save_data_and_json


def exp_adds_impl(input_np):
    tmp = np.exp(input_np)
    expected = tmp + 1.0
    return expected


def exp_adds_host_run_both():
    input_np = np.random.normal(0.0, 0.5, size=(8, 256)).astype(np.float16)
    expected = exp_adds_impl(input_np)

    tiling = list()
    ouput_np = np.zeros_like(expected).astype(np.float16)
    exp_adds_op_impl_tiling(input_np, ouput_np, tiling)
    exp_adds_op_impl_npu(input_np, ouput_np, tiling)
    save_data_and_json(input_list=[input_np], output_list=[ouput_np], expect_list=[expected], op_name="exp_adds_op")
    return expected


def exp_adds_host_run():
    input_np = np.random.normal(0.0, 0.5, size=(8, 256)).astype(np.float16)
    expected = exp_adds_impl(input_np)
    save_data_and_json(input_list=[input_np], output_list=[], expect_list=[expected], op_name="exp_adds_op")
    return expected


if __name__ == "__main__":
    import sys
    mode = "expect_only"
    if len(sys.argv) > 1:
        mode = sys.argv[1]

    if mode == "both":
        exp_adds_host_run_both()
    else:
        exp_adds_host_run()
