from tests.common import boot

# resnet shape
# boot.run("four2five_fp32_nhwc_001", "four2five_run", ([32, 224, 224, 4], "float32", 'NHWC', 'float16')),
# boot.run("four2five_fp32_nhwc_001", "four2five_run", ([32, 224, 224, 3], "float32", 'NHWC', 'float16')),
# boot.run("four2five_fp32_nhwc_001", "four2five_run", ([1001, 2048, 1, 1], "float32", 'NCHW', 'float16')),
# boot.run("four2five_012_fp32", "four2five_run", ([32, 1001, 1, 1], "float32", 'NCHW', 'float16')),
# boot.run("four2five_016", "four2five_run", ([1, 1024, 14, 14], "float16", 'NCHW', 'float16')),
# boot.run("four2five_017", "four2five_run", ([1, 256, 14, 14], "float16", 'NCHW', 'float16')),
# boot.run("four2five_018", "four2five_run", ([1, 512, 14, 14], "float16", 'NCHW', 'float16')),
# boot.run("four2five_019", "four2five_run", ([1, 2048, 14, 14], "float16", 'NCHW', 'float16')),
# boot.run("four2five_020", "four2five_run", ([32, 128, 14, 14], "float16", 'NCHW', 'float16'))
