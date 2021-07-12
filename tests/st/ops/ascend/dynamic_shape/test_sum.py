import pytest
from tests.common import boot


@pytest.mark.level2
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_sum():
    boot.run("001_sum", "sum_run", ((1024,), (0,), False, "float32"), "dynamic")
    boot.run("001_sum", "sum_run", ((32, 1024), (1,), False, "float32"), "dynamic")
