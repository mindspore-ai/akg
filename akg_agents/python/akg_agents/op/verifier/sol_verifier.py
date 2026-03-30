import os
import shutil
import logging
from jinja2 import Template
from akg_agents import get_project_root
from akg_agents.op.verifier.adapters.factory import (
    get_framework_adapter, get_dsl_adapter, get_backend_adapter
)

logger = logging.getLogger(__name__)

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
        dsl_impl_import = dsl_adapter.get_impl_import(verifier.op_name, verifier.impl_func_name)
        dsl_imports += "\n" + dsl_impl_import
        
        backend_adapter.setup_environment(device_id, verifier.arch)
        create_impl_code = verifier._prepare_code_lines(dsl_adapter.create_impl_module(verifier.framework, framework_adapter))
        device_setup_code = verifier._prepare_code_lines(framework_adapter.get_device_setup_code(verifier.backend, verifier.arch, device_id))
        
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
            create_impl_code=create_impl_code
        )
        
        with open(verify_file, "w", encoding="utf-8") as f:
            f.write(verify_script)
            
    except Exception as e:
        logger.error(f"[{verifier.op_name}] 验证脚本生成失败: {e}")
        raise
