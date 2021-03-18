import pytest
from tests.common import boot


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_equal_count():
    boot.run("test_resnet50_equal_count_001", "equal_count_run", (((32,), (32,)), "int32", "equal_count"), "dynamic")
