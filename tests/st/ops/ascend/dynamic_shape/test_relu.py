import pytest
from tests.common import boot


@pytest.mark.level2
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_relu():
    boot.run("test_resnet50_relu_000", "relu_run", ((32, 128, 7, 7, 16), "float32", 1e-5), "dynamic")
    boot.run("test_resnet50_relu_001", "relu_run", ((32, 16, 14, 14, 16), "float32", 1e-5), "dynamic")
    boot.run("test_resnet50_relu_002", "relu_run", ((32, 16, 56, 56, 16), "float32", 1e-5), "dynamic")
    boot.run("test_resnet50_relu_003", "relu_run", ((32, 32, 28, 28, 16), "float32", 1e-5), "dynamic")
    boot.run("test_resnet50_relu_004", "relu_run", ((32, 32, 7, 7, 16), "float32", 1e-5), "dynamic")
    boot.run("test_resnet50_relu_005", "relu_run", ((32, 4, 112, 112, 16), "float32", 1e-5), "dynamic")
    boot.run("test_resnet50_relu_006", "relu_run", ((32, 4, 56, 56, 16), "float32", 1e-5), "dynamic")
    boot.run("test_resnet50_relu_007", "relu_run", ((32, 64, 14, 14, 16), "float32", 1e-5), "dynamic")
    boot.run("test_resnet50_relu_008", "relu_run", ((32, 8, 28, 28, 16), "float32", 1e-5), "dynamic")
    boot.run("test_resnet50_relu_009", "relu_run", ((32, 8, 56, 56, 16), "float32", 1e-5), "dynamic")
    boot.run("test_resnet50_relu_010", "relu_run", ((32, 16, 28, 28, 16), "float32", 1e-5), "dynamic")
    boot.run("test_resnet50_relu_011", "relu_run", ((32, 32, 14, 14, 16), "float32", 1e-5), "dynamic")
    boot.run("test_resnet50_relu_012", "relu_run", ((32, 128, 7, 7, 16), "float16", 1e-5), "dynamic")
    boot.run("test_resnet50_relu_013", "relu_run", ((32, 16, 14, 14, 16), "float16", 1e-5), "dynamic")
    boot.run("test_resnet50_relu_014", "relu_run", ((32, 16, 56, 56, 16), "float16", 1e-5), "dynamic")
    boot.run("test_resnet50_relu_015", "relu_run", ((32, 32, 28, 28, 16), "float16", 1e-5), "dynamic")
    boot.run("test_resnet50_relu_016", "relu_run", ((32, 32, 7, 7, 16), "float16", 1e-5), "dynamic")
    boot.run("test_resnet50_relu_017", "relu_run", ((32, 4, 112, 112, 16), "float16", 1e-5), "dynamic")
    boot.run("test_resnet50_relu_018", "relu_run", ((32, 4, 56, 56, 16), "float16", 1e-5), "dynamic")
    boot.run("test_resnet50_relu_019", "relu_run", ((32, 64, 14, 14, 16), "float16", 1e-5), "dynamic")
    boot.run("test_resnet50_relu_020", "relu_run", ((32, 8, 28, 28, 16), "float16", 1e-5), "dynamic")
    boot.run("test_resnet50_relu_021", "relu_run", ((32, 8, 56, 56, 16), "float16", 1e-5), "dynamic")
    boot.run("test_resnet50_relu_022", "relu_run", ((32, 16, 28, 28, 16), "float16", 1e-5), "dynamic")
    boot.run("test_resnet50_relu_023", "relu_run", ((32, 32, 14, 14, 16), "float16", 1e-5), "dynamic")
