import pytest
from tests.common import boot


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_bn():
    boot.run("test_resnet50_bn_5D_reference000", "fused_batch_norm_run",
             ((32, 128, 7, 7, 16), "float32", 0.1, 1e-4, False, "NC1HWC0", None, "resnet50_bn_5D_reference01"),
             "dynamic")
    boot.run("test_resnet50_bn_5D_reference001", "fused_batch_norm_run",
             ((32, 16, 14, 14, 16), "float32", 0.1, 1e-4, False, "NC1HWC0", None, "resnet50_bn_5D_reference01"),
             "dynamic")
    boot.run("test_resnet50_bn_5D_reference002", "fused_batch_norm_run",
             ((32, 16, 56, 56, 16), "float32", 0.1, 1e-4, False, "NC1HWC0", None, "resnet50_bn_5D_reference01"),
             "dynamic")
    boot.run("test_resnet50_bn_5D_reference003", "fused_batch_norm_run",
             ((32, 32, 28, 28, 16), "float32", 0.1, 1e-4, False, "NC1HWC0", None, "resnet50_bn_5D_reference01"),
             "dynamic")
    boot.run("test_resnet50_bn_5D_reference004", "fused_batch_norm_run",
             ((32, 32, 7, 7, 16), "float32", 0.1, 1e-4, False, "NC1HWC0", None, "resnet50_bn_5D_reference01"),
             "dynamic")
    boot.run("test_resnet50_bn_5D_reference005", "fused_batch_norm_run",
             ((32, 4, 112, 112, 16), "float32", 0.1, 1e-4, False, "NC1HWC0", None, "resnet50_bn_5D_reference01"),
             "dynamic")
    boot.run("test_resnet50_bn_5D_reference006", "fused_batch_norm_run",
             ((32, 4, 56, 56, 16), "float32", 0.1, 1e-4, False, "NC1HWC0", None, "resnet50_bn_5D_reference01"),
             "dynamic")
    boot.run("test_resnet50_bn_5D_reference007", "fused_batch_norm_run",
             ((32, 64, 14, 14, 16), "float32", 0.1, 1e-4, False, "NC1HWC0", None, "resnet50_bn_5D_reference01"),
             "dynamic")
    boot.run("test_resnet50_bn_5D_reference008", "fused_batch_norm_run",
             ((32, 8, 28, 28, 16), "float32", 0.1, 1e-4, False, "NC1HWC0", None, "resnet50_bn_5D_reference01"),
             "dynamic")
    boot.run("test_resnet50_bn_5D_reference009", "fused_batch_norm_run",
             ((32, 8, 56, 56, 16), "float32", 0.1, 1e-4, False, "NC1HWC0", None, "resnet50_bn_5D_reference010"),
             "dynamic")
    boot.run("test_resnet50_bn_5D_reference010", "fused_batch_norm_run",
             ((32, 16, 28, 28, 16), "float32", 0.1, 1e-4, False, "NC1HWC0", None, "resnet50_bn_5D_reference011"),
             "dynamic")
    boot.run("test_resnet50_bn_5D_reference011", "fused_batch_norm_run",
             ((32, 32, 14, 14, 16), "float32", 0.1, 1e-4, False, "NC1HWC0", None, "resnet50_bn_5D_reference012"),
             "dynamic")
