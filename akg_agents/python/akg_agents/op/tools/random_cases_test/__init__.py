"""
Kernel泛化性随机Case测试工具

提供一个kernel和space_config，对它进行多case的泛化性测试。
"""

from .single_kernel_tester import SingleKernelTester
from .generalization_kernel_verifier import GeneralizationKernelVerifier

__all__ = [
    'SingleKernelTester',
    'GeneralizationKernelVerifier',
]

