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

"""
Skill 选择与 Evolution 逻辑 UT — 不依赖 LLM，纯逻辑验证

覆盖：
1. dsl_to_dir_key 转换
2. KernelGen 实例属性、PARAMETERS_SCHEMA、run() 签名
3. _infer_case_type metadata 推断（兼容旧 case category）
4. _parse_unified_selection JSON 解析
5. _assemble_skill_contents 排序与组装（含 fix/improvement category）
6. Stage → category 注入逻辑（含 fix/improvement 识别，源码检查）
7. nodes.py / evolution_processors 接口清理验证
8. evolved_skill_loader 已删除
9. AB test build_evolve_config A/B 模式
"""

import inspect
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def kg():
    from akg_agents.op.agents.kernel_gen import KernelGen
    return KernelGen()


@pytest.fixture(scope="module")
def project_root():
    return Path(__file__).resolve().parents[3]


# ========== 1. dsl_to_dir_key ==========

class TestDslToDirKey:
    def test_underscore_to_hyphen(self):
        from akg_agents.core_v2.skill.metadata import dsl_to_dir_key
        assert dsl_to_dir_key("triton_ascend") == "triton-ascend"

    def test_upper_case(self):
        from akg_agents.core_v2.skill.metadata import dsl_to_dir_key
        assert dsl_to_dir_key("Triton_CUDA") == "triton-cuda"

    def test_package_import(self):
        from akg_agents.core_v2.skill import dsl_to_dir_key
        assert dsl_to_dir_key("triton_ascend") == "triton-ascend"


# ========== 2. KernelGen 属性 + schema + 签名 ==========

class TestKernelGenInterface:
    def test_default_attrs(self, kg):
        assert kg.exclude_skill_names == []
        assert kg.force_skill_names == []
        assert kg.extra_skills == []

    def test_attrs_settable(self):
        from akg_agents.op.agents.kernel_gen import KernelGen
        k = KernelGen()
        k.exclude_skill_names = ["a"]
        k.force_skill_names = ["b"]
        assert k.exclude_skill_names == ["a"]
        assert k.force_skill_names == ["b"]

    def test_schema_has_exclude_and_force(self, kg):
        props = kg.PARAMETERS_SCHEMA.get("properties", {})
        assert "exclude_skill_names" in props
        assert "force_skill_names" in props

    def test_run_signature(self, kg):
        params = list(inspect.signature(kg.run).parameters.keys())
        assert "exclude_skill_names" in params
        assert "force_skill_names" in params
        assert "handwrite_suggestions" in params


# ========== 3. _infer_case_type (兼容旧 category='case' 的 skill) ==========

class TestInferCaseType:
    @dataclass
    class FakeSkill:
        metadata: dict = field(default_factory=dict)

    def test_metadata_case_type_fix(self, kg):
        assert kg._infer_case_type(self.FakeSkill(metadata={"case_type": "fix"})) == "fix"

    def test_metadata_source_error_fix(self, kg):
        assert kg._infer_case_type(self.FakeSkill(metadata={"source": "error_fix"})) == "fix"

    def test_metadata_case_type_improvement(self, kg):
        assert kg._infer_case_type(self.FakeSkill(metadata={"case_type": "improvement"})) == "improvement"

    def test_default_improvement(self, kg):
        assert kg._infer_case_type(self.FakeSkill()) == "improvement"


# ========== 4. _parse_unified_selection ==========

class TestParseUnifiedSelection:
    def test_plain_json(self, kg):
        r = kg._parse_unified_selection('{"guides": ["a"], "cases": ["b"]}')
        assert r["guides"] == ["a"]

    def test_json_in_code_fence(self, kg):
        r = kg._parse_unified_selection('```json\n{"guides": ["x"]}\n```')
        assert r["guides"] == ["x"]

    def test_empty_and_malformed(self, kg):
        assert kg._parse_unified_selection("") == {}
        assert kg._parse_unified_selection("{bad}") == {}


# ========== 5. _assemble_skill_contents ==========

class TestAssembleSkillContents:
    @dataclass
    class FakeSkill:
        name: str = ""
        category: str = ""
        content: str = "content"

    def test_empty_list(self, kg):
        assert kg._assemble_skill_contents([]) == ""

    def test_section_order(self, kg):
        skills = [
            self.FakeSkill(name="c", category="case", content="c"),
            self.FakeSkill(name="f", category="fundamental", content="f"),
            self.FakeSkill(name="g", category="guide", content="g"),
            self.FakeSkill(name="e", category="example", content="e"),
        ]
        result = kg._assemble_skill_contents(skills)
        assert result.find("基础知识与规范") < result.find("算子优化指南") < \
               result.find("代码示例参考") < result.find("优化/修复案例")

    def test_fix_and_improvement_in_case_section(self, kg):
        skills = [
            self.FakeSkill(name="fx", category="fix", content="fix-content"),
            self.FakeSkill(name="imp", category="improvement", content="imp-content"),
            self.FakeSkill(name="f", category="fundamental", content="f"),
        ]
        result = kg._assemble_skill_contents(skills)
        assert "优化/修复案例" in result
        assert "fix-content" in result
        assert "imp-content" in result


# ========== 6. Stage → category 注入逻辑 ==========

class TestStageCategories:
    """验证 _select_skills_by_stage 中各 stage 的 category 注入逻辑。

    实际逻辑内嵌在方法体中（非类属性），通过源码检查确认：
    - initial: extras = []（不注入 fix/improvement）
    - debug:   extras = case_fix（fix category 全部注入）
    - optimize: extras = _sample_cases(...)（improvement 参与采样）
    - 分类时识别 fix / improvement / case 三种 category
    """

    @pytest.fixture(autouse=True)
    def _load_source(self):
        from akg_agents.op.agents.kernel_gen import KernelGen
        self.source = inspect.getsource(KernelGen._select_skills_by_stage)

    def test_initial_no_case(self):
        assert 'extras = []\n' in self.source
        assert '"none (initial)"' in self.source
        assert 'always_skills' in self.source

    def test_debug_and_optimize_have_case(self):
        assert 'extras = case_fix' in self.source or "extras = [s for s in case_fix" in self.source
        assert '_sample_cases' in self.source

    def test_recognizes_fix_and_improvement_categories(self):
        assert 'cat == "fix"' in self.source
        assert 'cat == "improvement"' in self.source


# ========== 7. 接口清理验证 ==========

class TestInterfaceCleanup:
    def test_kernel_gen_node_no_handwrite(self):
        from akg_agents.op.langgraph_op.nodes import NodeFactory
        source = inspect.getsource(NodeFactory.create_kernel_gen_node)
        assert "handwrite_suggestions" not in source

    def test_evolution_processors_no_old_refs(self):
        from akg_agents.op.utils.evolve.evolution_processors import InitializationProcessor
        source = inspect.getsource(InitializationProcessor.initialize)
        assert "evolved_suggestions" not in source
        assert "evolved_skill_loader" not in source


# ========== 8. evolved_skill_loader 已删除 ==========

class TestEvolvedLoaderDeleted:
    def test_file_not_exists(self, project_root):
        path = (project_root / "python" / "akg_agents" / "op"
                / "utils" / "evolved_skill_loader.py")
        assert not path.exists()


# ========== 9. AB test build_evolve_config ==========

class TestBuildEvolveConfig:
    @pytest.fixture(autouse=True)
    def _setup_path(self, project_root):
        import sys
        p = str(project_root / "examples" / "kernel_related" / "skill_evolution")
        if p not in sys.path:
            sys.path.insert(0, p)

    def test_a_mode_has_exclude(self, project_root):
        import yaml
        from ab_test_utils import build_evolve_config
        config_dir = project_root / "python" / "akg_agents" / "op" / "config"
        config_files = list(config_dir.glob("*.yaml"))
        if not config_files:
            pytest.skip("no config yaml found")
        with tempfile.TemporaryDirectory() as run_dir:
            path = build_evolve_config(
                group=1, ab_mode="A", run_dir=run_dir, device=0,
                evolved_skill_dir="", base_config_path=str(config_files[0]),
                max_rounds=1, project_root=project_root,
            )
            with open(Path(path).parent / "agent_config.yaml") as f:
                cfg = yaml.safe_load(f) or {}
            assert "exclude_skill_names" in cfg

    def test_b_mode_has_force(self, project_root):
        import yaml
        from ab_test_utils import build_evolve_config
        config_dir = project_root / "python" / "akg_agents" / "op" / "config"
        config_files = list(config_dir.glob("*.yaml"))
        if not config_files:
            pytest.skip("no config yaml found")
        with tempfile.TemporaryDirectory() as run_dir:
            path = build_evolve_config(
                group=1, ab_mode="B", run_dir=run_dir, device=0,
                evolved_skill_dir="", base_config_path=str(config_files[0]),
                max_rounds=1, project_root=project_root,
            )
            with open(Path(path).parent / "agent_config.yaml") as f:
                cfg = yaml.safe_load(f) or {}
            assert "force_skill_names" in cfg
