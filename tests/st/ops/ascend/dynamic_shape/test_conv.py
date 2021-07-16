import pytest
from tests.common import boot


@pytest.mark.level2
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_conv1():
    boot.run_conv("conv_run001", "conv_run",
                  ((1, 1024, 14, 14), (2048, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False),
                  [14, 2048, 64, 128, 128, 14, 64], "dynamic", "bypassL1")


@pytest.mark.level2
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_conv2():
    boot.run_conv("conv_run002", "conv_run",
                  ((1, 1024, 14, 14), (256, 1024, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False),
                  [14, 256, 208, 64, 128, 14, 64], "dynamic")


@pytest.mark.level2
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_conv3():
    boot.run_conv("conv_run003", "conv_run",
                  ((1, 1024, 14, 14), (512, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False),
                  [14, 512, 64, 32, 512, 14, 64], "dynamic", "bypassL1")


@pytest.mark.level2
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_conv4():
    boot.run_conv("conv_run004", "conv_run", ((1, 128, 28, 28), (128, 128, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False),
                  [30, 128, 112, 32, 128, 30, 8], "dynamic")


@pytest.mark.level2
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_conv5():
    boot.run_conv("conv_run005", "conv_run", ((1, 128, 28, 28), (512, 128, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False),
                  [28, 512, 784, 16, 32, 28, 8], "dynamic")


@pytest.mark.level2
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_conv6():
    boot.run_conv("conv_run006", "conv_run", ((1, 2048, 7, 7), (512, 2048, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False),
                  [7, 512, 64, 32, 512, 7, 128], "dynamic", "bypassL1")


@pytest.mark.level2
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_conv7():
    boot.run_conv("conv_run007", "conv_run", ((1, 256, 14, 14), (1024, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False),
                  [7, 1024, 112, 32, 256, 14, 16], "dynamic")


@pytest.mark.level2
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_conv8():
    boot.run_conv("conv_run008", "conv_run", ((1, 256, 14, 14), (256, 256, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False),
                  [16, 256, 208, 64, 128, 16, 16], "dynamic", "bypassL1")


@pytest.mark.level2
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_conv9():
    boot.run_conv("conv_run009", "conv_run", ((1, 256, 56, 56), (128, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False),
                  [7, 128, 252, 64, 128, 56, 16], "dynamic")


@pytest.mark.level2
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_conv10():
    boot.run_conv("conv_run010", "conv_run", ((1, 256, 56, 56), (64, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False),
                  [8, 64, 224, 16, 64, 56, 16], "dynamic")


@pytest.mark.level2
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_conv11():
    boot.run_conv("conv_run011", "conv_run", ((1, 3, 224, 224), (64, 3, 7, 7), (2, 3, 2, 3), (2, 2), (1, 1), False),
                  [61, 64, 448, 16, 64, 230, 1], "dynamic", "bypassL1"),


@pytest.mark.level2
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_conv12():
    boot.run_conv("conv_run012", "conv_run", ((1, 512, 28, 28), (128, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False),
                  [14, 128, 448, 16, 64, 28, 32], "dynamic")


@pytest.mark.level2
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_conv13():
    boot.run_conv("conv_run013", "conv_run", ((1, 512, 28, 28), (256, 512, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False),
                  [13, 256, 112, 64, 256, 28, 32], "dynamic")


@pytest.mark.level2
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_conv14():
    boot.run_conv("conv_run014", "conv_run", ((1, 512, 7, 7), (2048, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False),
                  [7, 2048, 64, 16, 512, 7, 32], "dynamic", "bypassL1")


@pytest.mark.level2
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_conv15():
    boot.run_conv("conv_run015", "conv_run", ((1, 512, 7, 7), (512, 512, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False),
                  [9, 512, 49, 32, 512, 9, 32], "dynamic", "bypassL1")


@pytest.mark.level2
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_conv16():
    boot.run_conv("conv_run016", "conv_run", ((1, 64, 56, 56), (256, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False),
                  [56, 256, 784, 16, 32, 56, 4], "dynamic", "bypassL1")


@pytest.mark.level2
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_conv17():
    boot.run_conv("conv_run017", "conv_run", ((1, 64, 56, 56), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False),
                  [56, 64, 784, 16, 32, 56, 4], "dynamic")


@pytest.mark.level2
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_conv18():
    boot.run_conv("conv_run018", "conv_run", ((1, 64, 56, 56), (64, 64, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False),
                  [58, 64, 448, 16, 64, 58, 4], "dynamic")


@pytest.mark.level2
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_conv19():
    boot.run_conv("conv_run019", "conv_run", ((1, 256, 56, 56), (512, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False),
                  [7, 512, 196, 64, 256, 56, 16], "dynamic")


@pytest.mark.level2
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_conv20():
    boot.run_conv("conv_run020", "conv_run", ((1, 512, 28, 28), (1024, 512, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False),
                  [13, 1024, 112, 32, 256, 28, 32], "dynamic", "bypassL1")


@pytest.mark.level2
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_conv21():
    boot.run_conv("conv_run021", "conv_run",
                  ((2, 1024, 14, 14), (2048, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False),
                  [14, 2048, 64, 128, 128, 14, 64], "partial_dynamic", "bypassL1")


@pytest.mark.level2
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_conv22():
    boot.run_conv("conv_run022", "conv_run",
                  ((2, 1024, 14, 14), (2048, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False),
                  [14, 2048, 64, 128, 128, 14, 64], "dynamic", "bypassL1")
