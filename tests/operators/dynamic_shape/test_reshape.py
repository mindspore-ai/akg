import boot
import pytest

@pytest.mark.reshape
@pytest.mark.level0
@pytest.mark.env_oncard
@pytest.mark.platform_x86_ascend_training
def test_reshape():
    boot.run("test_resnet50_reshape_000", "reshape_run", [(32, 2048, 1, 1), (32, 2048), "float32"], "dynamic")
    boot.run("test_resnet50_reshape_001", "reshape_run", [(32, 2048), (32, 2048, 1, 1), "float32"], "dynamic")
    boot.run("test_resnet50_reshape_002", "reshape_run", [(32, 2048, 1, 1), (32, 2048), "float16"], "dynamic")
    boot.run("test_resnet50_reshape_003", "reshape_run", [(32, 2048), (32, 2048, 1, 1), "float16"], "dynamic")
    