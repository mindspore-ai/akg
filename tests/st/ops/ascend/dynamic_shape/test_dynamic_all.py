import sys
from collections.abc import Iterable
from tests.common import boot
from tests.common.test_run.ascend.batchmatmul_run import batchmatmul_execute
from tests.common.test_run import reduce_sum_run

def run_case(*args, **kwargs):
    if len(sys.argv) >= 3:
        case_name = sys.argv[2]
        if len(args) < 2:
            return
        if not (isinstance(args[0], Iterable) and case_name in args[0]):
            if not (isinstance(args[1], Iterable) and case_name in args[1]):
                return
    boot.run(*args, **kwargs)


def run_conv_case(*args, **kwargs):
    if len(sys.argv) >= 3:
        case_name = sys.argv[2]
        if len(args) < 2:
            return
        if not (isinstance(args[0], Iterable) and case_name in args[0]):
            if not (isinstance(args[1], Iterable) and case_name in args[1]):
                return
    boot.run_conv(*args, **kwargs)


def test_dynamic_manual():
    run_case("002_cast_test_case_1_23_dim_2", "cast_run", ((1, 23), "float16", "float32"), (((1, 0), (23, 0)),"dynamic"))
    run_case("relu_001_gx", "relu_run", ((1, 128), "float16", 1e-5), ((16, 0), (1, 0)),"dynamic")
    run_case("test_squeeze_16_1__1", "squeeze_run", [(16, 1), 1, "int32", "squeeze"], [(16, 16), (1, 1)],"dynamic")
    run_case("argmax_001", "argmax_run", ((3, 1020), "float32", 1),"dynamic")
    run_case("reshape_010", "reshape_run", [(32, 2048, 1, 1), (32, 2048), "float16"],"dynamic")
    run_case("001_equal_count", "equal_count_run", (((32,), (32,)), "int32", "equal_count"),"dynamic")

    run_case("softmax_01", "softmax_run", ((16, 1024), "float16", -1, "cce_softmax_fp16"),"dynamic")
    run_case("bias_add_fp16_002", "bias_add_run", ([32, 1001], "DefaultFormat", "float16"), [(1, 1), (1, 1)],"dynamic")
    run_case("001_sum", reduce_sum_run, ((2, 3, 5), (0, 1), False, "float32"),"dynamic")
    run_case("mean_01", "mean_run", ((8,), "float16", (0,), False, "cce_mean_1_64_fp16"),"dynamic")
    run_case("five2four_009", "five2four_run", ([32, 2048, 1, 1], "float16", 'NCHW', "float16"),"dynamic")
    run_case("four2five_016", "four2five_run", ([1, 1024, 14, 14], "float16", 'NCHW', 'float16'),"dynamic")
    run_case("resnet50_maxpool_fp16_c", "maxpool_with_argmax_run", ((32, 4, 112, 112, 16), (3, 3), (2, 2), (0, 1, 0, 1), True, "float16"),"dynamic")
    run_case("resnet50_Bn5dFp16Ref01", "fused_batch_norm_run", ((32, 4, 112, 112, 16), "float32", 0.99, 1e-5, True, "NC1HWC0", None, "resnet50_Bn5dFp16Ref01"),"dynamic")


def test_dynamic_auto():
    run_case("cast_test_case", "cast_run", ((1, 8192), "float16", "float32"),"dynamic")
    run_case("test_resnet50_relu_002", "relu_run", ((32, 128, 7, 7, 16), "float32", 1e-5), "dynamic")
    run_case("test_squeeze_16_1__1", "squeeze_run", [(16, 1), 1, "int32", "squeeze"],"dynamic")
    run_case("argmax_001", "argmax_run", ((3, 1020), "float32", 1),"dynamic")
    run_case("reshape_010", "reshape_run", [(32, 2048, 1, 1), (32, 2048), "float16"],"dynamic")
    run_case("001_equal_count", "equal_count_run", (((32,), (32,)), "int32", "equal_count"),"dynamic")

    run_case("softmax_01", "softmax_run", ((16, 1024), "float16", -1, "cce_softmax_fp16"),"dynamic")
    run_case("bias_add_fp16_002", "bias_add_run", ([32, 1001], "DefaultFormat", "float16"),"dynamic")
    run_case("001_sum", reduce_sum_run, ((2, 3, 5), (0, 1), False, "float32"),"dynamic")
    run_case("mean_01", "mean_run", ((8,), "float16", (0,), False, "cce_mean_1_64_fp16"),"dynamic")
    run_case("five2four_009", "five2four_run", ([32, 2048, 1, 1], "float16", 'NCHW', "float16"),"dynamic")
    run_case("four2five_016", "four2five_run", ([1, 1024, 14, 14], "float16", 'NCHW', 'float16'),"dynamic")
    run_case("resnet50_maxpool_fp16_c", "maxpool_run", ((32, 4, 112, 112, 16), (3, 3), (2, 2), (1, 1, 1, 1), True, "float16"),"dynamic")
    run_case("resnet50_Bn5dFp16Ref01", "fused_batch_norm_run", ((32, 4, 112, 112, 16), "float32", 0.99, 1e-5, True, "NC1HWC0", None, "resnet50_Bn5dFp16Ref01"),"dynamic")


def test_static_shape():
    run_case("002_cast_test_case_1_23_dim_2", "cast_run", ((1, 23), "float16", "float32"))
    run_case("relu_001_gx", "relu_run", ((1, 128), "float16", 1e-5))
    run_case("test_squeeze_16_1__1", "squeeze_run", [(16, 1), 1, "int32", "squeeze"])
    run_case("argmax_001", "argmax_run", ((3, 1020), "float32", 1),)
    run_case("reshape_010", "reshape_run", [(32, 2048, 1, 1), (32, 2048), "float16"],)
    run_case("001_equal_count", "equal_count_run", (((32,), (32,)), "int32", "equal_count"))
    run_case("softmax_01", "softmax_run", ((16, 1024), "float16", -1, "cce_softmax_fp16"))
    run_case("bias_add_fp16_002", "bias_add_run", ([32, 1001], "DefaultFormat", "float16"))
    run_case("001_sum", reduce_sum_run, ((2, 3, 5), (0, 1), False, "float32"))
    run_case("mean_01", "mean_run", ((8,), "float16", (0,), False, "cce_mean_1_64_fp16"))
    run_case("five2four_009", "five2four_run", ([32, 2048, 1, 1], "float16", 'NCHW', "float16"))
    run_case("four2five_016", "four2five_run", ([1, 1024, 14, 14], "float16", 'NCHW', 'float16'))
    run_case("resnet50_maxpool_fp16_c", "maxpool_run", ((32, 4, 112, 112, 16), (3, 3), (2, 2), (1, 1, 1, 1), True, "float16"))
    run_case("resnet50_Bn5dFp16Ref01", "fused_batch_norm_run", ((32, 4, 112, 112, 16), "float32", 0.99, 1e-5, True, "NC1HWC0", None, "resnet50_Bn5dFp16Ref01"))
    run_conv_case("conv_run006", "conv_run", ((1, 2048, 7, 7), (512, 2048, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False))


def test_all_dynamic_conv():
    run_conv_case("conv_run006", "conv_run", ((1, 2048, 7, 7), (512, 2048, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False) , [1, 16, 16, 16, 16, 7, 128], "dynamic")

def test_partial_dynamic_conv():
    run_conv_case("conv_run006", "conv_run", ((1, 2048, 7, 7), (512, 2048, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False) , [1, 16, 16, 16, 16, 7, 128], "partial_dynamic")

def test_partial_dynamic_conv_perf():
    run_conv_case("conv_run001", "conv_run", ((1, 1024, 14, 14), (2048, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False), [14, 2048, 64, 128, 128, 14, 64], "partial_dynamic", "bypassL1")
    run_conv_case("conv_run002", "conv_run", ((1, 1024, 14, 14), (256, 1024, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False), [14, 256, 208, 64, 128, 14, 64], "partial_dynamic")
    run_conv_case("conv_run003", "conv_run", ((1, 1024, 14, 14), (512, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False), [14, 512, 64, 32, 512, 14, 64], "partial_dynamic", "bypassL1")
    run_conv_case("conv_run004", "conv_run", ((1, 128, 28, 28), (128, 128, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False), [30, 128, 112, 32, 128, 30, 8], "partial_dynamic")
    run_conv_case("conv_run005", "conv_run", ((1, 128, 28, 28), (512, 128, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False), [28, 512, 784, 16, 32, 28, 8], "partial_dynamic")
    run_conv_case("conv_run006", "conv_run", ((1, 2048, 7, 7), (512, 2048, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False), [7, 512, 64, 32, 512, 7, 128], "partial_dynamic", "bypassL1")
    run_conv_case("conv_run007", "conv_run", ((1, 256, 14, 14), (1024, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False), [7, 1024, 112, 32, 256, 14, 16], "partial_dynamic")
    run_conv_case("conv_run008", "conv_run", ((1, 256, 14, 14), (256, 256, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False), [16, 256, 208, 64, 128, 16, 16], "partial_dynamic", "bypassL1")
    run_conv_case("conv_run009", "conv_run", ((1, 256, 56, 56), (128, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False), [7, 128, 252, 64, 128, 56, 16], "partial_dynamic")
    run_conv_case("conv_run010", "conv_run", ((1, 256, 56, 56), (64, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False), [8, 64, 224, 16, 64, 56, 16], "partial_dynamic")
    run_conv_case("conv_run011", "conv_run", ((1, 3, 224, 224), (64, 3, 7, 7), (2, 3, 2, 3), (2, 2), (1, 1), False), [61, 64, 448, 16, 64, 230, 1], "partial_dynamic", "bypassL1"),
    run_conv_case("conv_run012", "conv_run", ((1, 512, 28, 28), (128, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False), [14, 128, 448, 16, 64, 28, 32], "partial_dynamic")
    run_conv_case("conv_run013", "conv_run", ((1, 512, 28, 28), (256, 512, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False), [13, 256, 112, 64, 256, 28, 32], "partial_dynamic")
    run_conv_case("conv_run014", "conv_run", ((1, 512, 7, 7), (2048, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False), [7, 2048, 64, 16, 512, 7, 32], "partial_dynamic", "bypassL1")
    run_conv_case("conv_run015", "conv_run", ((1, 512, 7, 7), (512, 512, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False), [9, 512, 49, 32, 512, 9, 32], "partial_dynamic", "bypassL1")
    run_conv_case("conv_run016", "conv_run", ((1, 64, 56, 56), (256, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False), [56, 256, 784, 16, 32, 56, 4], "partial_dynamic", "bypassL1")
    run_conv_case("conv_run017", "conv_run", ((1, 64, 56, 56), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False), [56, 64, 784, 16, 32, 56, 4], "partial_dynamic")
    run_conv_case("conv_run018", "conv_run", ((1, 64, 56, 56), (64, 64, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False), [58, 64, 448, 16, 64, 58, 4], "partial_dynamic")
    run_conv_case("conv_run019", "conv_run", ((1, 256, 56, 56), (512, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False), [7, 512, 196, 64, 256, 56, 16], "partial_dynamic")
    run_conv_case("conv_run020", "conv_run", ((1, 512, 28, 28), (1024, 512, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False), [13, 1024, 112, 32, 256, 28, 32], "partial_dynamic", "bypassL1")

def test_partial_dynamic_conv_autotiling():
    run_conv_case("conv_run006", "conv_run", ((1, 2048, 7, 7), (512, 2048, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False) ,"partial_dynamic")

def test_static_conv():
    run_conv_case("conv_run001", "conv_run", ((1, 1024, 14, 14), (2048, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False)),
    # run_conv_case("conv_run002", "conv_run", ((1, 1024, 14, 14), (256, 1024, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)),
    # run_conv_case("conv_run003", "conv_run", ((1, 1024, 14, 14), (512, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False)),
    # run_conv_case("conv_run004", "conv_run", ((1, 128, 28, 28), (128, 128, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False)),
    # run_conv_case("conv_run005", "conv_run", ((1, 128, 28, 28), (512, 128, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)),
    # run_conv_case("conv_run006", "conv_run", ((1, 2048, 7, 7), (512, 2048, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)),
    # run_conv_case("conv_run007", "conv_run", ((1, 256, 14, 14), (1024, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)),
    # run_conv_case("conv_run008", "conv_run", ((1, 256, 14, 14), (256, 256, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False)),
    # run_conv_case("conv_run009", "conv_run", ((1, 256, 56, 56), (128, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False)),
    # run_conv_case("conv_run010", "conv_run", ((1, 256, 56, 56), (64, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)),
    # run_conv_case("conv_run011", "conv_run", ((1, 3, 224, 224), (64, 3, 7, 7), (2, 3, 2, 3), (2, 2), (1, 1), False)),
    # run_conv_case("conv_run012", "conv_run", ((1, 512, 28, 28), (128, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)),
    # run_conv_case("conv_run013", "conv_run", ((1, 512, 28, 28), (256, 512, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False)),
    # run_conv_case("conv_run014", "conv_run", ((1, 512, 7, 7), (2048, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)),
    # run_conv_case("conv_run015", "conv_run", ((1, 512, 7, 7), (512, 512, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False)),
    # run_conv_case("conv_run016", "conv_run", ((1, 64, 56, 56), (256, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)),
    # run_conv_case("conv_run017", "conv_run", ((1, 64, 56, 56), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)),
    # run_conv_case("conv_run018", "conv_run", ((1, 64, 56, 56), (64, 64, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False)),
    # run_conv_case("conv_run019", "conv_run", ((1, 256, 56, 56), (512, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False)),
    # run_conv_case("conv_run020", "conv_run", ((1, 512, 28, 28), (1024, 512, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False)),
    # run_conv_case("conv_run021", "conv_run", ((1, 256, 56, 56), ( 128,  256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)),#1.5
    # run_conv_case("conv_run022", "conv_run", ((1, 512, 28, 28), ( 256,  512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)),#1.5
    # run_conv_case("conv_run023", "conv_run", ((1,1024, 14, 14), ( 512, 1024, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)),#1.5
    # run_conv_case("conv_run024", "conv_run", ((1, 128, 56, 56), ( 128,  128, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1), False)),#1.5
    # run_conv_case("conv_run025", "conv_run", ((1, 256, 28, 28), ( 256,  256, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1), False)),#1.5
    # run_conv_case("conv_run026", "conv_run", ((1, 512, 14, 14), ( 512,  512, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1), False)),#1.5


def mini_ci_conv():
    run_conv_case("conv_run006", "conv_run", ((1, 2048, 7, 7), (512, 2048, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False))
    # resnet50 conv layer
    run_conv_case("conv_run001", "conv_run", ((1, 1024, 14, 14), (2048, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False), [1, 16, 16, 16, 16, 14, 64], "partial_dynamic")
    run_conv_case("conv_run002", "conv_run", ((1, 1024, 14, 14), (256, 1024, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False), [1, 16, 16, 16, 16, 14, 64], "partial_dynamic")
    run_conv_case("conv_run003", "conv_run", ((1, 1024, 14, 14), (512, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False), [1, 16, 16, 16, 16, 14, 64], "partial_dynamic")
    run_conv_case("conv_run004", "conv_run", ((1, 128, 28, 28), (128, 128, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False), [3, 16, 16, 16, 16, 30, 8], "partial_dynamic"),
    run_conv_case("conv_run005", "conv_run", ((1, 128, 28, 28), (512, 128, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False), [1, 16, 16, 16, 16, 28, 8], "partial_dynamic")
    run_conv_case("conv_run006", "conv_run", ((1, 2048, 7, 7), (512, 2048, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False), [1, 16, 16, 16, 16, 7, 128], "partial_dynamic")
    run_conv_case("conv_run007", "conv_run", ((1, 256, 14, 14), (1024, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False), [1, 16, 16, 16, 16, 14, 16], "partial_dynamic")
    run_conv_case("conv_run008", "conv_run", ((1, 256, 14, 14), (256, 256, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False), [3, 16, 16, 16, 16, 16, 16], "partial_dynamic"),
    run_conv_case("conv_run009", "conv_run", ((1, 256, 56, 56), (128, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False), [1, 16, 16, 16, 16, 56, 16], "partial_dynamic")
    run_conv_case("conv_run010", "conv_run", ((1, 256, 56, 56), (64, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False), [1, 16, 16, 16, 16, 56, 16], "partial_dynamic")
    run_conv_case("conv_run011", "conv_run", ((1, 3, 224, 224), (64, 3, 7, 7), (2, 3, 2, 3), (2, 2), (1, 1), False), [13, 16, 16, 16, 16, 230, 1], "partial_dynamic"),
    run_conv_case("conv_run012", "conv_run", ((1, 512, 28, 28), (128, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False), [1, 16, 16, 16, 16, 28, 32], "partial_dynamic")
    run_conv_case("conv_run013", "conv_run", ((1, 512, 28, 28), (256, 512, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False), [1, 16, 16, 16, 16, 28, 32], "partial_dynamic")
    run_conv_case("conv_run014", "conv_run", ((1, 512, 7, 7), (2048, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False), [1, 16, 16, 16, 16, 7, 32], "partial_dynamic")
    run_conv_case("conv_run015", "conv_run", ((1, 512, 7, 7), (512, 512, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False), [3, 16, 16, 16, 16, 9, 32], "partial_dynamic"),
    run_conv_case("conv_run016", "conv_run", ((1, 64, 56, 56), (256, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False), [1, 16, 16, 16, 16, 56, 4], "partial_dynamic")
    run_conv_case("conv_run017", "conv_run", ((1, 64, 56, 56), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False), [1, 16, 16, 16, 16, 56, 4], "partial_dynamic")
    run_conv_case("conv_run018", "conv_run", ((1, 64, 56, 56), (64, 64, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False), [3, 16, 16, 16, 16, 58, 4], "partial_dynamic"),
    run_conv_case("conv_run019", "conv_run", ((1, 256, 56, 56), (512, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False), [1, 16, 16, 16, 16, 56, 16], "partial_dynamic")
    run_conv_case("conv_run020", "conv_run", ((1, 512, 28, 28), (1024, 512, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False), [1, 16, 16, 16, 16, 28, 32], "partial_dynamic")

    run_conv_case("conv_run001", "conv_run", ((1, 1024, 14, 14), (2048, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False), [1, 16, 16, 16, 16, 14, 64], "dynamic")
    run_conv_case("conv_run002", "conv_run", ((1, 1024, 14, 14), (256, 1024, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False), [1, 16, 16, 16, 16, 14, 64], "dynamic")
    run_conv_case("conv_run003", "conv_run", ((1, 1024, 14, 14), (512, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False), [1, 16, 16, 16, 16, 14, 64], "dynamic")
    run_conv_case("conv_run004", "conv_run", ((1, 128, 28, 28), (128, 128, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False), [3, 16, 16, 16, 16, 30, 8], "dynamic"),
    run_conv_case("conv_run005", "conv_run", ((1, 128, 28, 28), (512, 128, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False), [1, 16, 16, 16, 16, 28, 8], "dynamic")
    run_conv_case("conv_run006", "conv_run", ((1, 2048, 7, 7), (512, 2048, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False), [1, 16, 16, 16, 16, 7, 128], "dynamic")
    run_conv_case("conv_run007", "conv_run", ((1, 256, 14, 14), (1024, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False), [1, 16, 16, 16, 16, 14, 16], "dynamic")
    run_conv_case("conv_run008", "conv_run", ((1, 256, 14, 14), (256, 256, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False), [3, 16, 16, 16, 16, 16, 16], "dynamic"),
    run_conv_case("conv_run009", "conv_run", ((1, 256, 56, 56), (128, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False), [1, 16, 16, 16, 16, 56, 16], "dynamic")
    run_conv_case("conv_run010", "conv_run", ((1, 256, 56, 56), (64, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False), [1, 16, 16, 16, 16, 56, 16], "dynamic")
    run_conv_case("conv_run011", "conv_run", ((1, 3, 224, 224), (64, 3, 7, 7), (2, 3, 2, 3), (2, 2), (1, 1), False), [13, 16, 16, 16, 16, 230, 1], "dynamic"),
    run_conv_case("conv_run012", "conv_run", ((1, 512, 28, 28), (128, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False), [1, 16, 16, 16, 16, 28, 32], "dynamic")
    run_conv_case("conv_run013", "conv_run", ((1, 512, 28, 28), (256, 512, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False), [1, 16, 16, 16, 16, 28, 32], "dynamic")
    run_conv_case("conv_run014", "conv_run", ((1, 512, 7, 7), (2048, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False), [1, 16, 16, 16, 16, 7, 32], "dynamic")
    run_conv_case("conv_run015", "conv_run", ((1, 512, 7, 7), (512, 512, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False), [3, 16, 16, 16, 16, 9, 32], "dynamic"),
    run_conv_case("conv_run016", "conv_run", ((1, 64, 56, 56), (256, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False), [1, 16, 16, 16, 16, 56, 4], "dynamic")
    run_conv_case("conv_run017", "conv_run", ((1, 64, 56, 56), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False), [1, 16, 16, 16, 16, 56, 4], "dynamic")
    run_conv_case("conv_run018", "conv_run", ((1, 64, 56, 56), (64, 64, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False), [3, 16, 16, 16, 16, 58, 4], "dynamic"),
    run_conv_case("conv_run019", "conv_run", ((1, 256, 56, 56), (512, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False), [1, 16, 16, 16, 16, 56, 16], "dynamic")
    run_conv_case("conv_run020", "conv_run", ((1, 512, 28, 28), (1024, 512, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False), [1, 16, 16, 16, 16, 28, 32], "dynamic")

def mini_ci():
    # all these cases are passed in earlier version, please make sure they are still passed after your commit
    # DYNAMIC SHAPE #
    run_case("002_cast_test_case_1_23_dim_2", "cast_run", ((1, 23), "float16", "float32"),"dynamic")
    run_case("resnet50_maxpool_fp16_c", "maxpool_with_argmax_run", ((2, 2, 15, 15, 16), (3, 3), (2, 2), (0, 0, 0, 0), True, "float16"),"dynamic")
    run_case("relu_001_gx", "relu_run", ((1, 128), "float16", 1e-5),"dynamic")
    run_case("test_squeeze_16_1__1", "squeeze_run", [(16, 1), 1, "int32", "squeeze"],"dynamic")
    run_case("reshape_010", "reshape_run", [(32, 2048, 1, 1), (32, 2048), "float16"],"dynamic")

    mini_ci_conv()

    # # STATIC SHAPE #
    run_case("002_cast_test_case_1_23_dim_2", "cast_run", ((1, 23), "float16", "float32"))
    run_case("001_equal_count", "equal_count_run", (((32,), (32,)), "int32", "equal_count"))
    run_case("mean_01", "mean_run", ((8,), "float16", (0,), False, "cce_mean_1_64_fp16"))
    run_case("relu_001_gx", "relu_run", ((1, 128), "float16", 1e-5))
    run_case("test_squeeze_16_1__1", "squeeze_run", [(16, 1), 1, "int32", "squeeze"])
    run_case("reshape_010", "reshape_run", [(32, 2048, 1, 1), (32, 2048), "float16"],)
    run_case("001_equal_count", "equal_count_run", (((32,), (32,)), "int32", "equal_count"))
    run_case("softmax_01", "softmax_run", ((16, 1024), "float16", -1, "cce_softmax_fp16"))
    run_case("bias_add_fp16_002", "bias_add_run", ([32, 1001], "DefaultFormat", "float16"))
    # run_case("001_sum", reduce_sum_run, ((2, 3, 5), (0, 1), False, "float32"))  # output nan
    run_case("mean_01", "mean_run", ((8,), "float16", (0,), False, "cce_mean_1_64_fp16"))
    run_case("five2four_009", "five2four_run", ([32, 2048, 1, 1], "float16", 'NCHW', "float16"))


def test_dynamic_bn():
    run_case("test_resnet50_bn_5D_reference000", "fused_batch_norm_run",
     ((32, 128, 7, 7, 16), "float32", 0.1, 1e-4, False, "NC1HWC0", None, "resnet50_bn_5D_reference01"), "dynamic")
    run_case("test_resnet50_bn_5D_reference001", "fused_batch_norm_run",
     ((32, 16, 14, 14, 16), "float32", 0.1, 1e-4, False, "NC1HWC0", None, "resnet50_bn_5D_reference01"), "dynamic")
    run_case("test_resnet50_bn_5D_reference002", "fused_batch_norm_run",
     ((32, 16, 56, 56, 16), "float32", 0.1, 1e-4, False, "NC1HWC0", None, "resnet50_bn_5D_reference01"), "dynamic")
    run_case("test_resnet50_bn_5D_reference003", "fused_batch_norm_run",
     ((32, 32, 28, 28, 16), "float32", 0.1, 1e-4, False, "NC1HWC0", None, "resnet50_bn_5D_reference01"), "dynamic")
    run_case("test_resnet50_bn_5D_reference004", "fused_batch_norm_run",
     ((32, 32, 7, 7, 16), "float32", 0.1, 1e-4, False, "NC1HWC0", None, "resnet50_bn_5D_reference01"), "dynamic")
    run_case("test_resnet50_bn_5D_reference005", "fused_batch_norm_run",
     ((32, 4, 112, 112, 16), "float32", 0.1, 1e-4, False, "NC1HWC0", None, "resnet50_bn_5D_reference01"), "dynamic")
    run_case("test_resnet50_bn_5D_reference006", "fused_batch_norm_run",
     ((32, 4, 56, 56, 16), "float32", 0.1, 1e-4, False, "NC1HWC0", None, "resnet50_bn_5D_reference01"), "dynamic")
    run_case("test_resnet50_bn_5D_reference007", "fused_batch_norm_run",
     ((32, 64, 14, 14, 16), "float32", 0.1, 1e-4, False, "NC1HWC0", None, "resnet50_bn_5D_reference01"), "dynamic")
    run_case("test_resnet50_bn_5D_reference008", "fused_batch_norm_run",
     ((32, 8, 28, 28, 16), "float32", 0.1, 1e-4, False, "NC1HWC0", None, "resnet50_bn_5D_reference01"), "dynamic")
    run_case("test_resnet50_bn_5D_reference009", "fused_batch_norm_run",
     ((32, 8, 56, 56, 16), "float32", 0.1, 1e-4, False, "NC1HWC0", None, "resnet50_bn_5D_reference010"), "dynamic")
    run_case("test_resnet50_bn_5D_reference010", "fused_batch_norm_run",
     ((32, 16, 28, 28, 16), "float32", 0.1, 1e-4, False, "NC1HWC0", None, "resnet50_bn_5D_reference011"), "dynamic")
    run_case("test_resnet50_bn_5D_reference011", "fused_batch_norm_run",
     ((32, 32, 14, 14, 16), "float32", 0.1, 1e-4, False, "NC1HWC0", None, "resnet50_bn_5D_reference012"), "dynamic")


def test_dynamic_matmul():
    #run_case("test_resnet50_matmul_000", batchmatmul_execute, ((), 32, 10, 2048, (10,), "float32", False, True, "batchmatmul_output"), "dynamic")
    run_case("test_resnet50_matmul_001", batchmatmul_execute, ((), 2048, 10, 32, (), "float32", True, False, "batchmatmul_output"),    "dynamic")
    run_case("test_resnet50_matmul_002", batchmatmul_execute, ((), 32, 2048, 10, (), "float32", False, False, "batchmatmul_output"),   "dynamic")
    run_case("test_resnet50_matmul_003", batchmatmul_execute, ((), 2048, 1001, 32, (), "float32", True, False, "batchmatmul_output"),  "dynamic")
    run_case("test_resnet50_matmul_004", batchmatmul_execute, ((), 32, 2048, 1001, (), "float32", False, False, "batchmatmul_output"), "dynamic")
    #run_case("test_resnet50_matmul_005", batchmatmul_execute, ((), 32, 1001, 2048, (1001,), "float32", False, True, "batchmatmul_output"), "dynamic")



def test_dynamic_resnet50():
    mini_ci_conv()
    test_dynamic_bn()
    test_dynamic_matmul()

    # mean
    run_case("test_resnet50_mean_000", "mean_run", ((32, 128, 7, 7, 16), "float32", (2, 3), True, "cce_mean"), "dynamic")
    run_case("test_resnet50_mean_001", "mean_run", ((32, 128, 7, 7, 16), "float16", (2, 3), True, "cce_mean"), "dynamic")

    # relu
    run_case("test_resnet50_relu_000", "relu_run", ((32, 128, 7, 7, 16), "float32", 1e-5), "dynamic")
    run_case("test_resnet50_relu_001", "relu_run", ((32, 16, 14, 14, 16), "float32", 1e-5), "dynamic")
    run_case("test_resnet50_relu_002", "relu_run", ((32, 16, 56, 56, 16), "float32", 1e-5), "dynamic")
    run_case("test_resnet50_relu_003", "relu_run", ((32, 32, 28, 28, 16), "float32", 1e-5), "dynamic")
    run_case("test_resnet50_relu_004", "relu_run", ((32, 32, 7, 7, 16), "float32", 1e-5), "dynamic")
    run_case("test_resnet50_relu_005", "relu_run", ((32, 4, 112, 112, 16), "float32", 1e-5), "dynamic")
    run_case("test_resnet50_relu_006", "relu_run", ((32, 4, 56, 56, 16), "float32", 1e-5), "dynamic")
    run_case("test_resnet50_relu_007", "relu_run", ((32, 64, 14, 14, 16), "float32", 1e-5), "dynamic")
    run_case("test_resnet50_relu_008", "relu_run", ((32, 8, 28, 28, 16), "float32", 1e-5), "dynamic")
    run_case("test_resnet50_relu_009", "relu_run", ((32, 8, 56, 56, 16), "float32", 1e-5), "dynamic")
    run_case("test_resnet50_relu_010", "relu_run", ((32, 16, 28, 28, 16), "float32", 1e-5), "dynamic")
    run_case("test_resnet50_relu_011", "relu_run", ((32, 32, 14, 14, 16), "float32", 1e-5), "dynamic")
    run_case("test_resnet50_relu_012", "relu_run", ((32, 128, 7, 7, 16), "float16", 1e-5), "dynamic")
    run_case("test_resnet50_relu_013", "relu_run", ((32, 16, 14, 14, 16), "float16", 1e-5), "dynamic")
    run_case("test_resnet50_relu_014", "relu_run", ((32, 16, 56, 56, 16), "float16", 1e-5), "dynamic")
    run_case("test_resnet50_relu_015", "relu_run", ((32, 32, 28, 28, 16), "float16", 1e-5), "dynamic")
    run_case("test_resnet50_relu_016", "relu_run", ((32, 32, 7, 7, 16), "float16", 1e-5), "dynamic")
    run_case("test_resnet50_relu_017", "relu_run", ((32, 4, 112, 112, 16), "float16", 1e-5), "dynamic")
    run_case("test_resnet50_relu_018", "relu_run", ((32, 4, 56, 56, 16), "float16", 1e-5), "dynamic")
    run_case("test_resnet50_relu_019", "relu_run", ((32, 64, 14, 14, 16), "float16", 1e-5), "dynamic")
    run_case("test_resnet50_relu_020", "relu_run", ((32, 8, 28, 28, 16), "float16", 1e-5), "dynamic")
    run_case("test_resnet50_relu_021", "relu_run", ((32, 8, 56, 56, 16), "float16", 1e-5), "dynamic")
    run_case("test_resnet50_relu_022", "relu_run", ((32, 16, 28, 28, 16), "float16", 1e-5), "dynamic")
    run_case("test_resnet50_relu_023", "relu_run", ((32, 32, 14, 14, 16), "float16", 1e-5), "dynamic")

    # Add
    run_case("test_resnet50_add_000", "add_run", ([32, 128, 7, 7, 16], [32, 128, 7, 7, 16], "float32", "cce_add_fp32"),   "dynamic")
    run_case("test_resnet50_add_001", "add_run", ([32, 16, 56, 56, 16], [32, 16, 56, 56, 16], "float32", "cce_add_fp32"), "dynamic")
    run_case("test_resnet50_add_002", "add_run", ([32, 32, 28, 28, 16], [32, 32, 28, 28, 16], "float32", "cce_add_fp32"), "dynamic")
    run_case("test_resnet50_add_003", "add_run", ([32, 64, 14, 14, 16], [32, 64, 14, 14, 16], "float32", "cce_add_fp32"), "dynamic")
    run_case("test_resnet50_add_004", "add_run", ([32, 128, 7, 7, 16], [32, 128, 7, 7, 16], "float16", "cce_add_fp16"),   "dynamic")
    run_case("test_resnet50_add_005", "add_run", ([32, 16, 56, 56, 16], [32, 16, 56, 56, 16], "float16", "cce_add_fp16"), "dynamic")
    run_case("test_resnet50_add_006", "add_run", ([32, 32, 28, 28, 16], [32, 32, 28, 28, 16], "float16", "cce_add_fp16"), "dynamic")
    run_case("test_resnet50_add_007", "add_run", ([32, 64, 14, 14, 16], [32, 64, 14, 14, 16], "float16", "cce_add_fp16"), "dynamic")

    # bias_add
    run_case("test_resnet50_bias_add_000", "bias_add_run", ([32, 10], "DefaultFormat", "float32"), "dynamic")
    run_case("test_resnet50_bias_add_001", "bias_add_run", ([32, 1001], "DefaultFormat", "float32"), "dynamic")
    run_case("test_resnet50_bias_add_002", "bias_add_run", ([32, 10], "DefaultFormat", "float16"), "dynamic")
    run_case("test_resnet50_bias_add_003", "bias_add_run", ([32, 1001], "DefaultFormat", "float16"), "dynamic")

    # reshape
    run_case("test_resnet50_reshape_000", "reshape_run", [(32, 2048, 1, 1), (32, 2048), "float32"], "dynamic")
    run_case("test_resnet50_reshape_001", "reshape_run", [(32, 2048), (32, 2048, 1, 1), "float32"], "dynamic")
    run_case("test_resnet50_reshape_002", "reshape_run", [(32, 2048, 1, 1), (32, 2048), "float16"], "dynamic")
    run_case("test_resnet50_reshape_003", "reshape_run", [(32, 2048), (32, 2048, 1, 1), "float16"], "dynamic")

    # cast
    run_case("test_resnet50_cast_000", "cast_run", ((64, 128, 16, 16), "float32", "float16"), "dynamic")
    run_case("test_resnet50_cast_001", "cast_run", ((32, 64, 16, 16), "float32", "float16"), "dynamic")
    run_case("test_resnet50_cast_002", "cast_run", ((16, 32, 16, 16), "float32", "float16"), "dynamic")
    run_case("test_resnet50_cast_003", "cast_run", ((4, 16, 16, 16), "float32", "float16"), "dynamic")
    run_case("test_resnet50_cast_004", "cast_run", ((49, 4, 16, 16), "float32", "float16"), "dynamic")
    run_case("test_resnet50_cast_005", "cast_run", ((32, 4, 112, 112, 16), "float16", "float32"), "dynamic")
    run_case("test_resnet50_cast_006", "cast_run", ((32, 4, 56, 56, 16), "float32", "float16"), "dynamic")
    run_case("test_resnet50_cast_007", "cast_run", ((32, 16, 56, 56, 16), "float16", "float32"), "dynamic")
    run_case("test_resnet50_cast_008", "cast_run", ((36, 4, 16, 16), "float32", "float16"), "dynamic")
    run_case("test_resnet50_cast_009", "cast_run", ((4, 4, 16, 16), "float32", "float16"), "dynamic")
    run_case("test_resnet50_cast_010", "cast_run", ((32, 4, 56, 56, 16), "float16", "float32"), "dynamic")
    run_case("test_resnet50_cast_011", "cast_run", ((16, 4, 16, 16), "float32", "float16"), "dynamic")
    run_case("test_resnet50_cast_012", "cast_run", ((32, 16, 56, 56, 16), "float32", "float16"), "dynamic")
    run_case("test_resnet50_cast_013", "cast_run", ((32, 32, 28, 28, 16), "float16", "float32"), "dynamic")
    run_case("test_resnet50_cast_014", "cast_run", ((8, 32, 16, 16), "float32", "float16"), "dynamic")
    run_case("test_resnet50_cast_015", "cast_run", ((72, 8, 16, 16), "float32", "float16"), "dynamic")
    run_case("test_resnet50_cast_016", "cast_run", ((16, 8, 16, 16), "float32", "float16"), "dynamic")
    run_case("test_resnet50_cast_017", "cast_run", ((32, 8, 56, 56, 16), "float16", "float32"), "dynamic")
    run_case("test_resnet50_cast_018", "cast_run", ((32, 8, 56, 56, 16), "float32", "float16"), "dynamic")
    run_case("test_resnet50_cast_019", "cast_run", ((32, 8, 28, 28, 16), "float16", "float32"), "dynamic")
    run_case("test_resnet50_cast_020", "cast_run", ((32, 8, 28, 28, 16), "float32", "float16"), "dynamic")
    run_case("test_resnet50_cast_021", "cast_run", ((32, 8, 16, 16), "float32", "float16"), "dynamic")
    run_case("test_resnet50_cast_022", "cast_run", ((32, 32, 28, 28, 16), "float32", "float16"), "dynamic")
    run_case("test_resnet50_cast_023", "cast_run", ((32, 64, 14, 14, 16), "float16", "float32"), "dynamic")
    run_case("test_resnet50_cast_024", "cast_run", ((16, 64, 16, 16), "float32", "float16"), "dynamic")
    run_case("test_resnet50_cast_025", "cast_run", ((144, 16, 16, 16), "float32", "float16"), "dynamic")
    run_case("test_resnet50_cast_026", "cast_run", ((32, 16, 16, 16), "float32", "float16"), "dynamic")
    run_case("test_resnet50_cast_027", "cast_run", ((32, 16, 28, 28, 16), "float16", "float32"), "dynamic")
    run_case("test_resnet50_cast_028", "cast_run", ((32, 16, 28, 28, 16), "float32", "float16"), "dynamic")
    run_case("test_resnet50_cast_029", "cast_run", ((32, 16, 14, 14, 16), "float16", "float32"), "dynamic")
    run_case("test_resnet50_cast_030", "cast_run", ((32, 16, 14, 14, 16), "float32", "float16"), "dynamic")
    run_case("test_resnet50_cast_031", "cast_run", ((64, 16, 16, 16), "float32", "float16"), "dynamic")
    run_case("test_resnet50_cast_032", "cast_run", ((32, 64, 14, 14, 16), "float32", "float16"), "dynamic")
    run_case("test_resnet50_cast_033", "cast_run", ((32, 128, 7, 7, 16), "float16", "float32"), "dynamic")
    run_case("test_resnet50_cast_034", "cast_run", ((32, 128, 16, 16), "float32", "float16"), "dynamic")
    run_case("test_resnet50_cast_035", "cast_run", ((288, 32, 16, 16), "float32", "float16"), "dynamic")
    run_case("test_resnet50_cast_036", "cast_run", ((64, 32, 16, 16), "float32", "float16"), "dynamic")
    run_case("test_resnet50_cast_037", "cast_run", ((32, 32, 14, 14, 16), "float16", "float32"), "dynamic")
    run_case("test_resnet50_cast_038", "cast_run", ((32, 32, 14, 14, 16), "float32", "float16"), "dynamic")
    run_case("test_resnet50_cast_039", "cast_run", ((32, 32, 7, 7, 16), "float16", "float32"), "dynamic")
    run_case("test_resnet50_cast_040", "cast_run", ((32, 32, 7, 7, 16), "float32", "float16"), "dynamic")
    run_case("test_resnet50_cast_041", "cast_run", ((128, 32, 16, 16), "float32", "float16"), "dynamic")
    run_case("test_resnet50_cast_042", "cast_run", ((32, 128, 7, 7, 16), "float32", "float16"), "dynamic")
    run_case("test_resnet50_cast_043", "cast_run", ((32, 4, 112, 112, 16), "float32", "float16"), "dynamic")
    run_case("test_resnet50_cast_044", "cast_run", ((32, 128, 1, 1, 16), "float32", "float16"), "dynamic")
    run_case("test_resnet50_cast_045", "cast_run", ((32, 2048, 1, 1), "float16", "float32"), "dynamic")
    run_case("test_resnet50_cast_048", "cast_run", ((64, 128, 16, 16), "float16", "float32"), "dynamic")
    run_case("test_resnet50_cast_049", "cast_run", ((32, 64, 16, 16), "float16", "float32"), "dynamic")
    run_case("test_resnet50_cast_050", "cast_run", ((16, 32, 16, 16), "float16", "float32"), "dynamic")
    run_case("test_resnet50_cast_051", "cast_run", ((4, 16, 16, 16), "float16", "float32"), "dynamic")
    run_case("test_resnet50_cast_052", "cast_run", ((49, 4, 16, 16), "float16", "float32"), "dynamic")
    run_case("test_resnet50_cast_053", "cast_run", ((36, 4, 16, 16), "float16", "float32"), "dynamic")
    run_case("test_resnet50_cast_054", "cast_run", ((4, 4, 16, 16), "float16", "float32"), "dynamic")
    run_case("test_resnet50_cast_055", "cast_run", ((16, 4, 16, 16), "float16", "float32"), "dynamic")
    run_case("test_resnet50_cast_056", "cast_run", ((8, 32, 16, 16), "float16", "float32"), "dynamic")
    run_case("test_resnet50_cast_057", "cast_run", ((72, 8, 16, 16), "float16", "float32"), "dynamic")
    run_case("test_resnet50_cast_058", "cast_run", ((16, 8, 16, 16), "float16", "float32"), "dynamic")
    run_case("test_resnet50_cast_059", "cast_run", ((32, 8, 56, 56, 16), "float32", "float16"), "dynamic")
    run_case("test_resnet50_cast_060", "cast_run", ((32, 8, 56, 56, 16), "float16", "float32"), "dynamic")
    run_case("test_resnet50_cast_061", "cast_run", ((32, 8, 16, 16), "float16", "float32"), "dynamic")
    run_case("test_resnet50_cast_062", "cast_run", ((16, 64, 16, 16), "float16", "float32"), "dynamic")
    run_case("test_resnet50_cast_063", "cast_run", ((144, 16, 16, 16), "float16", "float32"), "dynamic")
    run_case("test_resnet50_cast_064", "cast_run", ((32, 16, 16, 16), "float16", "float32"), "dynamic")
    run_case("test_resnet50_cast_065", "cast_run", ((32, 16, 28, 28, 16), "float16", "float32"), "dynamic")
    run_case("test_resnet50_cast_066", "cast_run", ((32, 16, 28, 28, 16), "float32", "float16"), "dynamic")
    run_case("test_resnet50_cast_067", "cast_run", ((64, 16, 16, 16), "float16", "float32"), "dynamic")
    run_case("test_resnet50_cast_068", "cast_run", ((32, 128, 16, 16), "float16", "float32"), "dynamic")
    run_case("test_resnet50_cast_069", "cast_run", ((288, 32, 16, 16), "float16", "float32"), "dynamic")
    run_case("test_resnet50_cast_070", "cast_run", ((64, 32, 16, 16), "float16", "float32"), "dynamic")
    run_case("test_resnet50_cast_071", "cast_run", ((32, 32, 14, 14, 16), "float16", "float32"), "dynamic")
    run_case("test_resnet50_cast_072", "cast_run", ((32, 32, 14, 14, 16), "float32", "float16"), "dynamic")
    run_case("test_resnet50_cast_073", "cast_run", ((128, 32, 16, 16), "float16", "float32"), "dynamic")
    run_case("test_resnet50_cast_074", "cast_run", ((32, 2048, 1, 1), "float32", "float16"), "dynamic")
    run_case("test_resnet50_cast_075", "cast_run", ((32, 128, 1, 1, 16), "float16", "float32"), "dynamic")
    run_case("test_resnet50_cast_080", "cast_run", ((64, 128, 16, 16), "bool", "int32"), "dynamic")


    # four2five
    run_case("test_resnet50_four2five_000", "four2five_run", ([32, 3, 224, 224], "float32", "NCHW", "float16"), "dynamic")
    run_case("test_resnet50_four2five_001", "four2five_run", ([32, 2048, 7, 7], "float32", "NCHW", "float16"), "dynamic")
    run_case("test_resnet50_four2five_002", "four2five_run", ([32, 224, 224, 3], "float32", 'NHWC', "float16"), "dynamic")
    run_case("test_resnet50_four2five_003", "four2five_run", ([32, 3, 224, 224], "float16", "NCHW", "float16"), "dynamic")
    run_case("test_resnet50_four2five_004", "four2five_run", ([32, 2048, 7, 7], "float16", "NCHW", "float16"), "dynamic")
    run_case("test_resnet50_four2five_005", "four2five_run", ([32, 224, 224, 3], "float16", 'NHWC', "float16"), "dynamic")

    # five2four
    run_case("test_resnet50_five2four_000", "five2four_run", ([32, 2048, 1, 1], "float16", "NCHW", "float16"), "dynamic")
    run_case("test_resnet50_five2four_001", "five2four_run", ([32, 2048, 1, 1], "float32", "NCHW", "float16"), "dynamic")

    # softmax
    run_case("test_resnet50_softmax_001", "softmax_run", ((32, 10), "float16", -1, "softmax_16"), "dynamic")
    run_case("test_resnet50_softmax_002", "softmax_run", ((32, 10), "float32", -1, "softmax_32"), "dynamic")
    run_case("test_resnet50_softmax_003", "softmax_run", ((32, 1001), "float16", -1, "softmax_16"), "dynamic")
    run_case("test_resnet50_softmax_004", "softmax_run", ((32, 1001), "float32", -1, "softmax_32"), "dynamic")

    # argmax
    run_case("test_resnet50_argmax_001", "argmax_run", ((32, 10), "float16", -1), "dynamic")
    run_case("test_resnet50_argmax_002", "argmax_run", ((32, 10), "float32", -1), "dynamic")
    run_case("test_resnet50_argmax_003", "argmax_run", ((32, 1001), "float16", -1), "dynamic")
    run_case("test_resnet50_argmax_004", "argmax_run", ((32, 1001), "float32", -1), "dynamic")

    # EqualCount
    run_case("test_resnet50_equal_count_001", "equal_count_run", (((32,), (32,)), "int32", "equal_count"), "dynamic")


def main(argv):
    if argv:
        if argv[0] == "m":
            test_dynamic_manual()
        elif argv[0] == "a":
            test_dynamic_auto()
        elif argv[0] == "s":
            test_static_shape()
        elif argv[0] == "ac":
            test_all_dynamic_conv()
        elif argv[0] == "c":
            test_partial_dynamic_conv()
        elif argv[0] == "cp":
            test_partial_dynamic_conv_perf()
        elif argv[0] == "ca":
            test_partial_dynamic_conv_autotiling()
        elif argv[0] == "sc":
            test_static_conv()
        elif argv[0] == "ci":
            mini_ci()
        elif argv[0] == "cic":
            mini_ci_conv()
        elif argv[0] == "r":
            test_dynamic_resnet50()
        elif argv[0] == "mat":
            test_dynamic_matmul()
    else:
        test_dynamic_manual()
        test_dynamic_auto()
        mini_ci_conv()


if __name__ == "__main__":
    main(sys.argv[1:])

