import boot
import pytest

@pytest.mark.four2five
@pytest.mark.level0
@pytest.mark.env_oncard
@pytest.mark.platform_x86_ascend_training
def test_four2five():
    boot.run("test_resnet50_four2five_000", "four2five_run", ([32, 3, 224, 224], "float32", "NCHW", "float16"), "dynamic")
    boot.run("test_resnet50_four2five_001", "four2five_run", ([32, 2048, 7, 7], "float32", "NCHW", "float16"), "dynamic")
    boot.run("test_resnet50_four2five_003", "four2five_run", ([32, 3, 224, 224], "float16", "NCHW", "float16"), "dynamic")
    boot.run("test_resnet50_four2five_004", "four2five_run", ([32, 2048, 7, 7], "float16", "NCHW", "float16"), "dynamic")
    
