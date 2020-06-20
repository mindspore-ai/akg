import boot
import pytest


@pytest.mark.softmax
@pytest.mark.level0
@pytest.mark.env_oncard
@pytest.mark.platform_x86_ascend_training
def test_softmax():
    boot.run("test_resnet50_softmax_001", "softmax_run", ((32, 10), "float16", -1, "softmax_16"), "dynamic")
    boot.run("test_resnet50_softmax_002", "softmax_run", ((32, 10), "float32", -1, "softmax_32"), "dynamic")
    boot.run("test_resnet50_softmax_003", "softmax_run", ((32, 1001), "float16", -1, "softmax_16"), "dynamic")
    boot.run("test_resnet50_softmax_004", "softmax_run", ((32, 1001), "float32", -1, "softmax_32"), "dynamic")
    