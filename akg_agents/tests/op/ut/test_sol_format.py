# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path

import pytest

from akg_agents.op.verifier.kernel_verifier import KernelVerifier
from akg_agents.op.verifier.sol_format import ensure_sol_problem_dir
from akg_agents.op.workflows.adaptive_search_workflow import AdaptiveSearchWorkflow
from akg_agents.op.workflows.autoresearch_workflow import AutoresearchWorkflow
from akg_agents.op.workflows.coder_only_workflow import CoderOnlyWorkflow
from akg_agents.op.workflows.connect_all_workflow import ConnectAllWorkflow
from akg_agents.op.workflows.default_workflow import DefaultWorkflow
from akg_agents.op.workflows.default_workflow_v2 import DefaultWorkflowV2
from akg_agents.op.workflows.evolve_workflow import EvolveWorkflow
from akg_agents.op.workflows.kernelgen_only_workflow import KernelGenOnlyWorkflow
from akg_agents.op.workflows.verifier_only_workflow import VerifierOnlyWorkflow


RAW_SOL_RECORD = {
    "name": "000_relu",
    "description": "ReLU raw SOL row",
    "axes": {
        "n": {"type": "var", "description": "number of elements"},
    },
    "custom_inputs_entrypoint": None,
    "inputs": {
        "x": {
            "shape": ["n"],
            "dtype": "float32",
            "description": "input tensor",
        },
    },
    "outputs": {
        "out": {
            "shape": ["n"],
            "dtype": "float32",
            "description": "output tensor",
        },
    },
    "reference": "import torch\n\ndef run(x):\n    return torch.relu(x)\n",
    "workloads": [
        {
            "uuid": "case-1",
            "axes": {"n": 128},
            "inputs": {"x": {"type": "random"}},
            "tolerance": {"max_atol": 1e-5, "max_rtol": 1e-5},
        },
        {
            "uuid": "case-2",
            "axes": {"n": 256},
            "inputs": {"x": {"type": "random"}},
            "tolerance": {"max_atol": 1e-5, "max_rtol": 1e-5},
        },
    ],
}


def test_raw_sol_record_materializes_three_required_files(tmp_path):
    case_dir = ensure_sol_problem_dir(
        config={"sol_problem_data": RAW_SOL_RECORD},
        work_dir=str(tmp_path),
        op_name="relu",
    )

    case_path = Path(case_dir)
    definition = json.loads((case_path / "definition.json").read_text())
    assert definition["name"] == "000_relu"
    assert "workloads" not in definition
    assert "def run" in (case_path / "reference.py").read_text()

    workload_lines = [
        line for line in (case_path / "workload.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(workload_lines) == 2
    assert json.loads(workload_lines[0])["uuid"] == "case-1"
    assert case_dir


def test_sol_verifier_accepts_raw_record_payload(tmp_path):
    verify_dir = tmp_path / "verify"
    verify_dir.mkdir()
    cfg = {
        "log_dir": str(tmp_path / "logs"),
        "sol_problem_data": RAW_SOL_RECORD,
    }
    verifier = KernelVerifier(
        op_name="relu",
        framework_code="",
        framework="torch",
        dsl="torch",
        backend="cpu",
        arch="x86_64",
        config=cfg,
        bench_type="sol",
    )

    impl_code = "import torch\n\nclass ModelNew(torch.nn.Module):\n    def forward(self, x):\n        return torch.relu(x)\n"
    verifier.gen_verify_project(impl_code, str(verify_dir))

    assert (verify_dir / "definition.json").is_file()
    assert (verify_dir / "workload.jsonl").is_file()
    assert (verify_dir / "reference.py").is_file()
    assert (verify_dir / "sol_runtime_fallback.py").is_file()
    assert (verify_dir / "verify_relu.py").is_file()


def test_three_file_json_payload_materializes(tmp_path):
    payload = {
        "definition.json": {
            "name": "000_relu",
            "axes": {"n": {"type": "var"}},
            "inputs": {"x": {"shape": ["n"], "dtype": "float32"}},
            "outputs": {"out": {"shape": ["n"], "dtype": "float32"}},
            "custom_inputs_entrypoint": None,
            "reference": "def run(x):\n    return x\n",
        },
        "workload.jsonl": '{"uuid":"case-1","axes":{"n":4},"inputs":{"x":{"type":"random"}}}\n',
        "reference.py": "def run(x):\n    return x\n",
    }

    case_dir = ensure_sol_problem_dir(
        config={"sol_problem_json": json.dumps(payload)},
        work_dir=str(tmp_path),
        op_name="relu",
    )

    case_path = Path(case_dir)
    assert json.loads((case_path / "definition.json").read_text())["name"] == "000_relu"
    assert (case_path / "workload.jsonl").read_text().strip().startswith('{"uuid"')
    assert case_dir


@pytest.mark.parametrize(
    "workflow_cls",
    [
        DefaultWorkflow,
        CoderOnlyWorkflow,
        KernelGenOnlyWorkflow,
        VerifierOnlyWorkflow,
        ConnectAllWorkflow,
        DefaultWorkflowV2,
        AdaptiveSearchWorkflow,
        EvolveWorkflow,
        AutoresearchWorkflow,
    ],
)
def test_op_workflow_tool_schema_exposes_sol_bench_type(workflow_cls):
    properties = workflow_cls.PARAMETERS_SCHEMA["properties"]
    bench_schema = properties["bench_type"]

    assert bench_schema["default"] == "kernelbench"
    assert set(bench_schema["enum"]) == {"kernelbench", "sol"}
    assert "sol_problem_dir" in properties
    assert "sol_problem_json" in properties
    assert "sol_task_code" in properties
