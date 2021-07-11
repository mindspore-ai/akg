import pytest
from tests.common import boot


@pytest.mark.level2
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_five2four():
    boot.run("test_resnet50_five2four_000", "five2four_run", ([32, 2048, 1, 1], "float16", "NCHW", "float16"),
             "dynamic")
    boot.run("test_resnet50_five2four_001", "five2four_run", ([32, 2048, 1, 1], "float32", "NCHW", "float16"),
             "dynamic")
