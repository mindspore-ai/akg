import boot
import pytest

@pytest.mark.sum
@pytest.mark.level0
@pytest.mark.env_oncard
@pytest.mark.platform_x86_ascend_training
def test_sum():
    boot.run("001_sum", "sum_run", ((1024, ), (0, ), False, "float32"),"dynamic")
    boot.run("001_sum", "sum_run", ((32, 1024 ), (1, ), False, "float32"),"dynamic")
    