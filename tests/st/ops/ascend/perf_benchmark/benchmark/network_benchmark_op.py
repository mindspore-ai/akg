from tests.common import boot


def test_compile_too_long():
    boot.run("conv_backprop_filter_run_019", "conv_filter_ad_run",
             ((32, 128, 56, 56), (128, 128, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1))),
    boot.run("conv_backprop_filter_run_010", "conv_backprop_filter_run",
             ((1, 3, 224, 224), (64, 3, 7, 7), (3, 3, 3, 3), (2, 2), (1, 1))),


def test_resnet_benchmark():
    boot.run("resnet50_maxpool_with_argmax_000", "maxpool_with_argmax_run",
             ((32, 4, 112, 112, 16), (3, 3), (2, 2), 'SAME', True, "float16")),
    boot.run("resnet50_bn_split_005", "bn_split_run",
             ((32, 4, 112, 112, 16), "float32", 0.1, 1e-4, "resnet50_bn_split")),
    boot.run("resnet50_conv_bn1_026", "conv_bn1_run",
             ((32, 3, 224, 224), (64, 3, 7, 7), (2, 3, 2, 3), (2, 2), (1, 1), False)),
    boot.run("resnet50_four2five_003", "four2five_run", ([32, 3, 224, 224], "float16", "NCHW", "float16")),
    boot.run("resnet50_softmax_004", "softmax_run", ((32, 1001), "float32", -1, "softmax_32")),
    boot.run("resnet50_apply_momentum_002", "apply_momentum_run", ((128, 32, 16, 16), "float32", False)),
    boot.run("resnet50_mean_000", "mean_run", ((32, 128, 7, 7, 16), "float32", (2, 3), True, "cce_mean")),


def test_bert_benchmark():
    boot.run("bert_batch_matmul_003_242", "batchmatmul_run",
             ((), 4096, 3072, 768, (3072,), "float32", False, True, "batch_matmul_output")),
    boot.run("fused_layernorm_002_1280_1024", "fused_layernorm_run", ((1280, 1024), 1, -1, 'float16')),
    boot.run("logsoftmax_grad_002", "logsoftmax_grad_run", ((160, 30522), "float32", -1, "cce_logsoftmax_fp16")),
    boot.run("unsortedsegmentsum_002", "unsortedsegmentsum_run", ([1280, 1024], [1280], 8192, "float32")),
    boot.run("transpose_002", "transpose_run", ((8, 16, 128, 64), (0, 2, 1, 3), "float32")),
    boot.run("fused_layer_norm_grad_01", "fused_layer_norm_grad_run", ((8192, 1024), -1, -1, "float16")),
    boot.run("logsoftmax_002_fp32", "logsoftmax_run", ((160, 30522), "float32", -1, "cce_logsoftmax_fp32")),
    boot.run("strided_slice_grad_002", "strided_slice_grad_run",
             ((128, 128, 768), [0, 0, 0], [128, 1, 768], [1, 1, 1], 0, 0, 0, 0, 0, (128, 1, 768), "int32"))
