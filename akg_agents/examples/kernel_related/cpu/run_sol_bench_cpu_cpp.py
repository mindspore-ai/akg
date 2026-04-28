import os
import asyncio
from akg_agents.op.verifier.kernel_verifier import KernelVerifier

async def main():
    # C++ implementation of ReLU via load_inline
    impl_code = """
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cpp_source = '''
#include <torch/extension.h>

torch::Tensor relu_forward(torch::Tensor x) {
    auto out = x.clone();
    auto out_data = out.data_ptr<float>();
    auto numel = out.numel();
    
    for (int i = 0; i < numel; ++i) {
        if (out_data[i] < 0.0f) {
            out_data[i] = 0.0f;
        }
    }
    return out;
}
'''

relu_module = load_inline(
    name='relu_cpp',
    cpp_sources=[cpp_source],
    functions=['relu_forward'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return relu_module.relu_forward(x.cpu().contiguous()).to(x.device)
"""

    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sol_problem_dir = os.path.join(current_dir, "mock_sol_relu")

    config = {
        "log_dir": "./logs",
        "sol_problem_dir": sol_problem_dir,
        "verify_timeout": 300
    }

    verifier = KernelVerifier(
        op_name="mock_relu",
        framework_code="", # SOL 模式下不需要 framework_code
        framework="torch",
        dsl="cpp",
        backend="cpu",
        arch="x86_64",
        config=config,
        bench_type="sol"
    )

    task_info = {
        "coder_code": impl_code
    }

    print("Starting SOL-ExecBench verification with C++...")
    passed, log = await verifier.run(task_info)
    
    print(f"Verification Passed: {passed}")
    print(f"Verification Log:\n{log}")

if __name__ == "__main__":
    asyncio.run(main())
