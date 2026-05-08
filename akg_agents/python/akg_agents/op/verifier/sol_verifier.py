import os
import shutil
import logging
from jinja2 import Template
from akg_agents import get_project_root
from akg_agents.op.verifier.adapters.factory import (
    get_framework_adapter, get_dsl_adapter, get_backend_adapter
)

logger = logging.getLogger(__name__)

PROF_SOL_BASE_TEMPLATE_PATH = os.path.join(
    get_project_root(), "op", "resources", "templates", "prof_sol_base_template.j2"
)
PROF_SOL_GENERATION_TEMPLATE_PATH = os.path.join(
    get_project_root(), "op", "resources", "templates", "prof_sol_generation_template.j2"
)

def generate_sol_verify_project(verifier, impl_code: str, verify_dir: str, device_id: int = 0):
    """生成 SOL-ExecBench 验证项目文件到指定目录"""
    logger.info(f"[{verifier.op_name}] 开始生成 SOL-ExecBench 验证项目，目录: {verify_dir}, device_id={device_id}")
    
    sol_problem_dir = verifier.config.get("sol_problem_dir")
    if not sol_problem_dir or not os.path.exists(sol_problem_dir):
        raise ValueError(f"sol_problem_dir is missing or does not exist: {sol_problem_dir}")
        
    # 1. 拷贝 SOL 核心文件
    for file_name in ["definition.json", "workload.jsonl", "reference.py"]:
        src_file = os.path.join(sol_problem_dir, file_name)
        dst_file = os.path.join(verify_dir, file_name)
        if not os.path.exists(src_file):
            raise FileNotFoundError(f"Missing required SOL file: {src_file}")
        shutil.copy2(src_file, dst_file)
        
    # 2. 拷贝 sol_correctness.py
    sol_correctness_src = os.path.join(get_project_root(), "op", "resources", "utils", "sol_correctness.py")
    sol_correctness_dst = os.path.join(verify_dir, "sol_correctness.py")
    if not os.path.exists(sol_correctness_src):
        raise FileNotFoundError(f"Missing sol_correctness.py: {sol_correctness_src}")
    shutil.copy2(sol_correctness_src, sol_correctness_dst)

    # 3. 创建具体实现文件
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

    # 4. 生成验证脚本
    verify_file = os.path.join(verify_dir, f"verify_{verifier.op_name}.py")
    template_path = os.path.join(get_project_root(), "op", "resources", "templates", "verify_sol_template.j2")
    
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
        # Fix for Python module names starting with numbers
        if dsl_impl_import.startswith("from ") and dsl_impl_import.split(" ")[1][0].isdigit():
            module_name = dsl_impl_import.split(" ")[1]
            import_name = dsl_impl_import.split(" ")[3].strip()
            dsl_impl_import = f"import importlib.util\nimport sys\nspec = importlib.util.spec_from_file_location('{module_name}', '{module_name}.py')\nmodule = importlib.util.module_from_spec(spec)\nsys.modules['{module_name}'] = module\nspec.loader.exec_module(module)\n{import_name} = getattr(module, '{import_name}')"
        
        dsl_imports += "\n" + dsl_impl_import
        
        backend_adapter.setup_environment(device_id, verifier.arch)
        create_impl_code = verifier._prepare_code_lines(dsl_adapter.create_impl_module(verifier.framework, framework_adapter))
        device_setup_code = verifier._prepare_code_lines(framework_adapter.get_device_setup_code(verifier.backend, verifier.arch, device_id))
        
        sol_execbench_src_dir = os.path.abspath(os.path.join(get_project_root(), "..", "..", "thirdparty", "sol-execbench", "src"))
        
        # 渲染模板
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
            sol_execbench_src_dir=sol_execbench_src_dir
        )
        
        with open(verify_file, "w", encoding="utf-8") as f:
            f.write(verify_script)
            
    except Exception as e:
        logger.error(f"[{verifier.op_name}] 验证脚本生成失败: {e}")
        raise


def _get_sol_common_template_vars(verifier, device_id: int):
    """获取 SOL profile 模板的公共变量（供 base 和 generation 模板共用）"""
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

    sol_execbench_src_dir = os.path.abspath(
        os.path.join(get_project_root(), "..", "..", "thirdparty", "sol-execbench", "src")
    )

    return {
        "device_setup_code": device_setup_code,
        "dsl_imports": dsl_imports,
        "create_impl_code": create_impl_code,
        "sol_execbench_src_dir": sol_execbench_src_dir,
    }


def generate_sol_profile_project(verifier, verify_dir: str, device_id: int = 0,
                                  warmup_times: int = 5, run_times: int = 50,
                                  skip_base: bool = False):
    """生成 SOL-ExecBench 性能测试项目文件到指定目录

    与 KernelBench 的 gen_profile_project 对应，生成两个 profile 脚本：
    - profile_{op_name}_base.py      测量 reference.run 的性能
    - profile_{op_name}_generation.py 测量生成实现的性能

    两个脚本均输出与 KernelBench 兼容的 JSON 文件
    (base_profile_result.json / generation_profile_result.json)，
    下游 profiler_utils.run_profile_scripts_and_collect_results 可直接使用。
    """
    logger.info(
        f"[{verifier.op_name}] 开始生成 SOL-ExecBench 性能测试项目，"
        f"目录: {verify_dir}, device_id={device_id}"
    )

    sol_problem_dir = verifier.config.get("sol_problem_dir")
    if not sol_problem_dir or not os.path.exists(sol_problem_dir):
        raise ValueError(f"sol_problem_dir is missing or does not exist: {sol_problem_dir}")

    # 确保 SOL 数据文件存在（gen_verify_project 通常已经拷贝过）
    for file_name in ["definition.json", "workload.jsonl", "reference.py"]:
        dst_file = os.path.join(verify_dir, file_name)
        if not os.path.exists(dst_file):
            src_file = os.path.join(sol_problem_dir, file_name)
            if not os.path.exists(src_file):
                raise FileNotFoundError(f"Missing required SOL file: {src_file}")
            shutil.copy2(src_file, dst_file)

    # 确保 sol_correctness.py 存在
    sol_correctness_dst = os.path.join(verify_dir, "sol_correctness.py")
    if not os.path.exists(sol_correctness_dst):
        sol_correctness_src = os.path.join(
            get_project_root(), "op", "resources", "utils", "sol_correctness.py"
        )
        shutil.copy2(sol_correctness_src, sol_correctness_dst)

    common_vars = _get_sol_common_template_vars(verifier, device_id)

    # 生成 base profile 脚本（测量 reference.run 的性能）
    if not skip_base:
        try:
            with open(PROF_SOL_BASE_TEMPLATE_PATH, "r", encoding="utf-8") as f:
                base_template = Template(f.read())
            base_script = base_template.render(
                op_name=verifier.op_name,
                backend=verifier.backend,
                arch=verifier.arch,
                device_id=device_id,
                warmup_times=warmup_times,
                run_times=run_times,
                device_setup_code=common_vars["device_setup_code"],
                sol_execbench_src_dir=common_vars["sol_execbench_src_dir"],
            )
            base_path = os.path.join(verify_dir, f"profile_{verifier.op_name}_base.py")
            with open(base_path, "w", encoding="utf-8") as f:
                f.write(base_script)
            logger.info(f"[{verifier.op_name}] SOL base profile 脚本已写入: {base_path}")
        except Exception as e:
            logger.error(f"[{verifier.op_name}] SOL base profile 脚本生成失败: {e}")
            raise
    else:
        logger.info(f"[{verifier.op_name}] 跳过 SOL base profile 生成（skip_base=True）")

    # 生成 generation profile 脚本（测量生成实现的性能）
    try:
        with open(PROF_SOL_GENERATION_TEMPLATE_PATH, "r", encoding="utf-8") as f:
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
            sol_execbench_src_dir=common_vars["sol_execbench_src_dir"],
        )
        gen_path = os.path.join(verify_dir, f"profile_{verifier.op_name}_generation.py")
        with open(gen_path, "w", encoding="utf-8") as f:
            f.write(gen_script)
        logger.info(f"[{verifier.op_name}] SOL generation profile 脚本已写入: {gen_path}")
    except Exception as e:
        logger.error(f"[{verifier.op_name}] SOL generation profile 脚本生成失败: {e}")
        raise
