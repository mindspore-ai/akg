import pytest
from tests.common import boot


@pytest.mark.level2
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_add():
    boot.run("test_resnet50_add_000", "add_run", ([32, 128, 7, 7, 16], [32, 128, 7, 7, 16], "float32", "cce_add_fp32"),
             "dynamic")
    boot.run("test_resnet50_add_001", "add_run",
             ([32, 16, 56, 56, 16], [32, 16, 56, 56, 16], "float32", "cce_add_fp32"), "dynamic")
    boot.run("test_resnet50_add_002", "add_run",
             ([32, 32, 28, 28, 16], [32, 32, 28, 28, 16], "float32", "cce_add_fp32"), "dynamic")
    boot.run("test_resnet50_add_003", "add_run",
             ([32, 64, 14, 14, 16], [32, 64, 14, 14, 16], "float32", "cce_add_fp32"), "dynamic")
    boot.run("test_resnet50_add_004", "add_run", ([32, 128, 7, 7, 16], [32, 128, 7, 7, 16], "float16", "cce_add_fp16"),
             "dynamic")
    boot.run("test_resnet50_add_005", "add_run",
             ([32, 16, 56, 56, 16], [32, 16, 56, 56, 16], "float16", "cce_add_fp16"), "dynamic")
    boot.run("test_resnet50_add_006", "add_run",
             ([32, 32, 28, 28, 16], [32, 32, 28, 28, 16], "float16", "cce_add_fp16"), "dynamic")
    boot.run("test_resnet50_add_007", "add_run",
             ([32, 64, 14, 14, 16], [32, 64, 14, 14, 16], "float16", "cce_add_fp16"), "dynamic")
