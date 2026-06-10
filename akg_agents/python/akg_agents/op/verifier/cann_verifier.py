# Copyright 2025-2026 Huawei Technologies Co., Ltd
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

"""CANN-Bench format verification project generator.

Responsibilities:
1. Generate CANN-Bench verification project (verify script + data files)
2. Generate CANN-Bench profile project (profile scripts)

Pattern follows sol_verifier.py: sys.path imports CANN-Bench source repo
for DataGenerator, ParamBuilder, etc., instead of reimplementing.
"""

import os
import shutil
import logging
import json
import yaml
from jinja2 import Template
from akg_agents import get_project_root
from akg_agents.core.worker.interface import (
    DEFAULT_WARMUP_TIMES, DEFAULT_RUN_TIMES,
)
from akg_agents.op.verifier.adapters.factory import (
    get_framework_adapter, get_dsl_adapter, get_backend_adapter,
)

logger = logging.getLogger(__name__)

PROF_CANN_BASE_TEMPLATE_PATH = os.path.join(
    get_project_root(), "op", "resources", "templates", "prof_cann_base_template.j2",
)
PROF_CANN_GENERATION_TEMPLATE_PATH = os.path.join(
    get_project_root(), "op", "resources", "templates", "prof_cann_generation_template.j2",
)

CANN_BENCH_SRC_DIR = os.path.abspath(
    os.path.join(get_project_root(), "..", "..", "thirdparty", "cann-bench", "src")
)

CANN_DATA_FILES = ["proto.yaml", "golden.py", "cases.yaml"]


def generate_cann_verify_project(verifier, impl_code: str, verify_dir: str, device_id: int = 0):
    """Generate CANN-Bench verification project files into verify_dir."""
    logger.info(
        f"[{verifier.op_name}] 开始生成 CANN-Bench 验证项目，"
        f"目录: {verify_dir}, device_id={device_id}"
    )

    cann_problem_dir = verifier.config.get("cann_problem_dir")
    if not cann_problem_dir or not os.path.exists(cann_problem_dir):
        raise ValueError(f"cann_problem_dir is missing or does not exist: {cann_problem_dir}")

    # 1. Copy CANN-Bench data files
    for file_name in CANN_DATA_FILES:
        src_file = os.path.join(cann_problem_dir, file_name)
        dst_file = os.path.join(verify_dir, file_name)
        if not os.path.exists(src_file):
            raise FileNotFoundError(f"Missing required CANN file: {src_file}")
        shutil.copy2(src_file, dst_file)

    # Copy desc.md if exists
    desc_src = os.path.join(cann_problem_dir, "desc.md")
    if os.path.exists(desc_src):
        shutil.copy2(desc_src, os.path.join(verify_dir, "desc.md"))

    # 2. Copy cann_correctness.py
    cann_correctness_src = os.path.join(
        get_project_root(), "op", "resources", "utils", "cann_correctness.py",
    )
    cann_correctness_dst = os.path.join(verify_dir, "cann_correctness.py")
    if not os.path.exists(cann_correctness_src):
        raise FileNotFoundError(f"Missing cann_correctness.py: {cann_correctness_src}")
    shutil.copy2(cann_correctness_src, cann_correctness_dst)

    # 3. Create implementation file
    if "ascendc" in verifier.dsl:
        logger.info(f"[{verifier.op_name}] 检测到AscendC DSL，生成编译项目")
        verifier.generate_ascendc_project(impl_code, verify_dir)
    else:
        file_name = f"{verifier.op_name}_{verifier.dsl}_impl.py"
        impl_file = os.path.join(verify_dir, file_name)

        try:
            dsl_adapter = get_dsl_adapter(verifier.dsl)
            import_statements = dsl_adapter.get_import_statements(verifier.framework)
        except Exception as e:
            logger.error(f"[{verifier.op_name}] DSL import语句生成失败: {e}")
            raise

        try:
            with open(impl_file, "w", encoding="utf-8") as f:
                f.write(import_statements + impl_code)
        except Exception as e:
            logger.error(f"[{verifier.op_name}] 实现文件创建失败: {impl_file}, 错误: {e}")
            raise

    # 4. Load proto.yaml for template rendering
    proto_path = os.path.join(verify_dir, "proto.yaml")
    with open(proto_path, "r", encoding="utf-8") as f:
        proto = yaml.safe_load(f)

    op_spec = proto.get("operator", {})
    precision_thresholds = op_spec.get("precision_thresholds")
    outputs = op_spec.get("outputs", [])
    ignore_output_indices = [
        i for i, out in enumerate(outputs) if out.get("compare", True) is False
    ]

    # 5. Generate verify script
    verify_file = os.path.join(verify_dir, f"verify_{verifier.op_name}.py")
    template_path = os.path.join(
        get_project_root(), "op", "resources", "templates", "verify_cann_template.j2",
    )

    try:
        with open(template_path, "r", encoding="utf-8") as f:
            template = Template(f.read())
    except Exception as e:
        logger.error(f"[{verifier.op_name}] 模板文件加载失败: {template_path}, 错误: {e}")
        raise

    try:
        framework_adapter = get_framework_adapter(verifier.framework)
        dsl_adapter = get_dsl_adapter(verifier.dsl)
        backend_adapter = get_backend_adapter(verifier.backend)
    except Exception as e:
        logger.error(f"[{verifier.op_name}] Adapters初始化失败: {e}")
        raise

    try:
        dsl_imports = dsl_adapter.get_import_statements(verifier.framework)
        dsl_impl_import = dsl_adapter.get_impl_import(verifier.op_name, verifier.impl_func_name).strip()
        if dsl_impl_import.startswith("from ") and dsl_impl_import.split(" ")[1][0].isdigit():
            module_name = dsl_impl_import.split(" ")[1]
            import_name = dsl_impl_import.split(" ")[3].strip()
            dsl_impl_import = (
                f"import importlib.util\nimport sys\n"
                f"spec = importlib.util.spec_from_file_location('{module_name}', '{module_name}.py')\n"
                f"module = importlib.util.module_from_spec(spec)\n"
                f"sys.modules['{module_name}'] = module\n"
                f"spec.loader.exec_module(module)\n"
                f"{import_name} = getattr(module, '{import_name}')"
            )
        dsl_imports += "\n" + dsl_impl_import

        backend_adapter.setup_environment(device_id, verifier.arch)
        create_impl_code = verifier._prepare_code_lines(
            dsl_adapter.create_impl_module(verifier.framework, framework_adapter)
        )
        device_setup_code = verifier._prepare_code_lines(
            framework_adapter.get_device_setup_code(verifier.backend, verifier.arch, device_id)
        )

        # Serialize precision_thresholds for template injection
        precision_thresholds_yaml = json.dumps(precision_thresholds) if precision_thresholds else "None"

        verify_script = template.render(
            op_name=verifier.op_name,
            framework=verifier.framework,
            backend=verifier.backend,
            arch=verifier.arch,
            dsl=verifier.dsl,
            device_id=device_id,
            dsl_imports=dsl_imports,
            device_setup_code=device_setup_code,
            create_impl_code=create_impl_code,
            precision_thresholds_yaml=precision_thresholds_yaml,
            ignore_output_indices=ignore_output_indices,
            schema=op_spec.get("schema", ""),
            cann_bench_src_dir=CANN_BENCH_SRC_DIR,
        )

        with open(verify_file, "w", encoding="utf-8") as f:
            f.write(verify_script)

    except Exception as e:
        logger.error(f"[{verifier.op_name}] 验证脚本生成失败: {e}")
        raise


def generate_cann_profile_project(verifier, verify_dir: str, device_id: int = 0,
                                   warmup_times: int = DEFAULT_WARMUP_TIMES,
                                   run_times: int = DEFAULT_RUN_TIMES,
                                   skip_base: bool = False):
    """Generate CANN-Bench profile project files into verify_dir.

    Produces two profile scripts:
    - profile_{op_name}_base.py: measure golden.py performance
    - profile_{op_name}_generation.py: measure generated implementation performance
    """
    logger.info(
        f"[{verifier.op_name}] 开始生成 CANN-Bench 性能测试项目，"
        f"目录: {verify_dir}, device_id={device_id}"
    )

    cann_problem_dir = verifier.config.get("cann_problem_dir")
    if not cann_problem_dir or not os.path.exists(cann_problem_dir):
        raise ValueError(f"cann_problem_dir is missing or does not exist: {cann_problem_dir}")

    # Ensure CANN data files exist (may already be copied by gen_verify_project)
    for file_name in CANN_DATA_FILES:
        dst_file = os.path.join(verify_dir, file_name)
        if not os.path.exists(dst_file):
            src_file = os.path.join(cann_problem_dir, file_name)
            if not os.path.exists(src_file):
                raise FileNotFoundError(f"Missing required CANN file: {src_file}")
            shutil.copy2(src_file, dst_file)

    desc_src = os.path.join(cann_problem_dir, "desc.md")
    if os.path.exists(desc_src):
        dst = os.path.join(verify_dir, "desc.md")
        if not os.path.exists(dst):
            shutil.copy2(desc_src, dst)

    # Ensure cann_correctness.py exists (not strictly needed for profile, but consistency)
    cann_correctness_dst = os.path.join(verify_dir, "cann_correctness.py")
    if not os.path.exists(cann_correctness_dst):
        cann_correctness_src = os.path.join(
            get_project_root(), "op", "resources", "utils", "cann_correctness.py",
        )
        shutil.copy2(cann_correctness_src, cann_correctness_dst)

    # Load proto.yaml for template
    proto_path = os.path.join(verify_dir, "proto.yaml")
    with open(proto_path, "r", encoding="utf-8") as f:
        proto = yaml.safe_load(f)
    op_spec = proto.get("operator", {})
    schema = op_spec.get("schema", "")

    # Get common template vars
    common_vars = _get_cann_common_template_vars(verifier, device_id)

    # Generate base profile script (measure golden.py)
    if not skip_base:
        try:
            with open(PROF_CANN_BASE_TEMPLATE_PATH, "r", encoding="utf-8") as f:
                base_template = Template(f.read())
            base_script = base_template.render(
                op_name=verifier.op_name,
                backend=verifier.backend,
                arch=verifier.arch,
                dsl=verifier.dsl,
                device_id=device_id,
                warmup_times=warmup_times,
                run_times=run_times,
                device_setup_code=common_vars["device_setup_code"],
                schema=schema,
                cann_bench_src_dir=common_vars["cann_bench_src_dir"],
            )
            base_path = os.path.join(verify_dir, f"profile_{verifier.op_name}_base.py")
            with open(base_path, "w", encoding="utf-8") as f:
                f.write(base_script)
            logger.info(f"[{verifier.op_name}] CANN base profile 脚本已写入: {base_path}")
        except Exception as e:
            logger.error(f"[{verifier.op_name}] CANN base profile 脚本生成失败: {e}")
            raise
    else:
        logger.info(f"[{verifier.op_name}] 跳过 CANN base profile 生成（skip_base=True）")

    # Generate generation profile script
    try:
        with open(PROF_CANN_GENERATION_TEMPLATE_PATH, "r", encoding="utf-8") as f:
            gen_template = Template(f.read())
        gen_script = gen_template.render(
            op_name=verifier.op_name,
            backend=verifier.backend,
            arch=verifier.arch,
            dsl=verifier.dsl,
            device_id=device_id,
            warmup_times=warmup_times,
            run_times=run_times,
            device_setup_code=common_vars["device_setup_code"],
            dsl_imports=common_vars["dsl_imports"],
            create_impl_code=common_vars["create_impl_code"],
            schema=schema,
            cann_bench_src_dir=common_vars["cann_bench_src_dir"],
        )
        gen_path = os.path.join(verify_dir, f"profile_{verifier.op_name}_generation.py")
        with open(gen_path, "w", encoding="utf-8") as f:
            f.write(gen_script)
        logger.info(f"[{verifier.op_name}] CANN generation profile 脚本已写入: {gen_path}")
    except Exception as e:
        logger.error(f"[{verifier.op_name}] CANN generation profile 脚本生成失败: {e}")
        raise


def _get_cann_common_template_vars(verifier, device_id: int):
    """Get common template variables for CANN profile scripts."""
    framework_adapter = get_framework_adapter(verifier.framework)
    dsl_adapter = get_dsl_adapter(verifier.dsl)
    backend_adapter = get_backend_adapter(verifier.backend)

    backend_adapter.setup_environment(device_id, verifier.arch)
    device_setup_code = verifier._prepare_code_lines(
        framework_adapter.get_device_setup_code(verifier.backend, verifier.arch, device_id)
    )

    dsl_imports = dsl_adapter.get_import_statements(verifier.framework)
    dsl_impl_import = dsl_adapter.get_impl_import(verifier.op_name, verifier.impl_func_name).strip()
    if dsl_impl_import.startswith("from ") and dsl_impl_import.split(" ")[1][0].isdigit():
        module_name = dsl_impl_import.split(" ")[1]
        import_name = dsl_impl_import.split(" ")[3].strip()
        dsl_impl_import = (
            f"import importlib.util\nimport sys\n"
            f"spec = importlib.util.spec_from_file_location('{module_name}', '{module_name}.py')\n"
            f"module = importlib.util.module_from_spec(spec)\n"
            f"sys.modules['{module_name}'] = module\n"
            f"spec.loader.exec_module(module)\n"
            f"{import_name} = getattr(module, '{import_name}')"
        )
    dsl_imports += "\n" + dsl_impl_import

    create_impl_code = verifier._prepare_code_lines(
        dsl_adapter.create_impl_module(verifier.framework, framework_adapter)
    )

    return {
        "device_setup_code": device_setup_code,
        "dsl_imports": dsl_imports,
        "create_impl_code": create_impl_code,
        "cann_bench_src_dir": CANN_BENCH_SRC_DIR,
    }