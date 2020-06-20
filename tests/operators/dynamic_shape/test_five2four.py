import boot
import pytest

@pytest.mark.five2four
@pytest.mark.level0
@pytest.mark.env_oncard
@pytest.mark.platform_x86_ascend_training
def test_five2four():
    boot.run("test_resnet50_five2four_000", "five2four_run", ([32, 2048, 1, 1], "float16", "NCHW", "float16"), "dynamic")
    boot.run("test_resnet50_five2four_001", "five2four_run", ([32, 2048, 1, 1], "float32", "NCHW", "float16"), "dynamic")
    