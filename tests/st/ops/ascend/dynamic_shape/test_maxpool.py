import pytest
from tests.common import boot


@pytest.mark.level2
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_maxpool():
    boot.run("resnet50_maxpool_fp16_c", "maxpool_with_argmax_run",
             ((32, 4, 112, 112, 16), (3, 3), (2, 2), (0, 1, 0, 1), True, "float16"), "dynamic")
