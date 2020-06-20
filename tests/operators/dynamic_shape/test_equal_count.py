import boot
import pytest

@pytest.mark.equal_count
@pytest.mark.level0
@pytest.mark.env_oncard
@pytest.mark.platform_x86_ascend_training
def test_equal_count():
    boot.run("test_resnet50_equal_count_001", "equal_count_run", (((32,), (32,)), "int32", "equal_count"), "dynamic")
    