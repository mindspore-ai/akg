import pytest
from tests.common import boot


@pytest.mark.level2
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_mean():
    boot.run("test_resnet50_mean_000", "mean_run", ((32, 128, 7, 7, 16), "float32", (2, 3), True, "cce_mean"),
             "dynamic")
    boot.run("test_resnet50_mean_001", "mean_run", ((32, 128, 7, 7, 16), "float16", (2, 3), True, "cce_mean"),
             "dynamic")
