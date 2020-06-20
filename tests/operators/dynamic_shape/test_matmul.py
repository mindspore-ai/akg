import boot
import pytest
from test_run.batchmatmul_run import batchmatmul_execute

@pytest.mark.matmul
@pytest.mark.level0
@pytest.mark.env_oncard
@pytest.mark.platform_x86_ascend_training
def test_matmul():
    boot.run("test_resnet50_matmul_001",batchmatmul_execute, ((), 2048, 10, 32, (), "float32", True, False, "batchmatmul_output"),    "dynamic")
    boot.run("test_resnet50_matmul_002",batchmatmul_execute, ((), 32, 2048, 10, (), "float32", False, False, "batchmatmul_output"),   "dynamic")
    boot.run("test_resnet50_matmul_003",batchmatmul_execute, ((), 2048, 1001, 32, (), "float32", True, False, "batchmatmul_output"),  "dynamic")
    boot.run("test_resnet50_matmul_004",batchmatmul_execute, ((), 32, 2048, 1001, (), "float32", False, False, "batchmatmul_output"), "dynamic")
