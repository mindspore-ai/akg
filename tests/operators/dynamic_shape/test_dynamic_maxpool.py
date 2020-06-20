import boot

# maxpool shape
#boot.run("four2five_fp32_nhwc_001", "four2five_run", ([32, 224, 224, 4], "float32", 'NHWC', 'float16')),
boot.run("resnet50_maxpool_fp16_c", "maxpool_with_argmax_run", ((32, 4, 112, 112, 16), (3, 3), (2, 2), (1, 1, 1, 1), True, "float16"), "dynamic"),
#boot.run("four2five_020", "four2five_run", ([32, 128, 14, 14], "float16", 'NCHW', 'float16'))
