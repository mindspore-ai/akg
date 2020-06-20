import boot
import pytest

@pytest.mark.mean
@pytest.mark.level0
@pytest.mark.env_oncard
@pytest.mark.platform_x86_ascend_training
def test_mean():
    boot.run("test_resnet50_mean_000", "mean_run", ((32, 128, 7, 7, 16), "float32", (2, 3), True, "cce_mean"), "dynamic")
    boot.run("test_resnet50_mean_001", "mean_run", ((32, 128, 7, 7, 16), "float16", (2, 3), True, "cce_mean"), "dynamic")
    