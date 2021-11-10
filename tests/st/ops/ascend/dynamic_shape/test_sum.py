import pytest
from tests.common import boot
from tests.common.test_run import reduce_sum_run

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_sum():
    boot.run("001_sum", reduce_sum_run, ((1024,), (0,), False, "float32"), "dynamic")
    boot.run("001_sum", reduce_sum_run, ((32, 1024), (1,), False, "float32"), "dynamic")
