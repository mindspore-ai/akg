import pytest
from tests.operators.test_all import add

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_add_for_CPU():
    attrs = {"target": "llvm", "profiling": False}
    return add(True, False, attrs)
