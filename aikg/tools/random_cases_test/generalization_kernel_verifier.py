"""
泛化性实验专用的Kernel验证器
继承并复用KernelVerifier的基础设施，只替换验证模板
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "python"))

from ai_kernel_generator.core.verifier.kernel_verifier import KernelVerifier

logger = logging.getLogger(__name__)


class GeneralizationKernelVerifier(KernelVerifier):
    """
    泛化性实验专用验证器
    
    继承KernelVerifier，只重写验证模板路径，复用所有adapter逻辑
    """
    
    def __init__(
        self,
        op_name: str,
        framework_code: str,
        task_id: str = "0",
        framework: str = "torch",
        dsl: str = "triton",
        backend: str = "cuda",
        arch: str = "a100",
        impl_func_name: str = None,
        config: Dict[str, Any] = None
    ):
        """
        初始化泛化性验证器
        
        Args:
            op_name: 算子名称
            framework_code: 框架代码（包含Model和get_inputs_dyn_list）
            task_id: 任务ID
            framework: 框架类型
            dsl: DSL类型
            backend: 后端
            arch: 架构
            impl_func_name: 实现函数名
            config: 配置字典
        """
        # 规范化framework名称（pytorch -> torch）
        if framework == "pytorch":
            framework = "torch"
        
        # 调用父类初始化
        super().__init__(
            op_name=op_name,
            framework_code=framework_code,
            task_id=task_id,
            framework=framework,
            dsl=dsl,
            backend=backend,
            arch=arch,
            impl_func_name=impl_func_name,
            config=config
        )
        
        # 替换模板路径为泛化性测试专用模板
        from ai_kernel_generator import get_project_root
        self.template_path_original = self.TEMPLATE_PATH if hasattr(self, 'TEMPLATE_PATH') else None
        self.template_path_generalization = os.path.join(
            get_project_root(),
            "resources",
            "templates",
            "generalization_verify_template.j2"
        )
        
        logger.info(f"[{self.op_name}] 使用泛化性验证模板: {self.template_path_generalization}")
    
    def gen_verify_project(self, impl_code: str, verify_dir: str, device_id: int = 0):
        """
        生成验证项目文件到指定目录
        
        重写此方法，使用泛化性测试模板
        """
        logger.info(f"[{self.op_name}] 开始生成泛化性验证项目，目录: {verify_dir}, device_id={device_id}")
        
        # 创建框架实现文件
        framework_file = os.path.join(verify_dir, f"{self.op_name}_{self.framework}.py")
        try:
            with open(framework_file, "w", encoding="utf-8") as f:
                f.write(self.framework_code)
            logger.debug(f"[{self.op_name}] 框架实现文件已创建: {framework_file}")
        except Exception as e:
            logger.error(f"[{self.op_name}] 框架实现文件创建失败: {framework_file}, 错误: {e}")
            raise

        # 创建具体实现文件（与父类相同）
        if "ascendc" in self.dsl:
            logger.info(f"[{self.op_name}] 检测到AscendC DSL，生成编译项目")
            self.generate_ascendc_project(impl_code, verify_dir)
        else:
            file_name = f"{self.op_name}_{self.dsl}.py"
            impl_file = os.path.join(verify_dir, file_name)

            # 使用adapter生成import语句
            try:
                from ai_kernel_generator.core.verifier.adapters.factory import get_dsl_adapter
                dsl_adapter = get_dsl_adapter(self.dsl)
                import_statements = dsl_adapter.get_import_statements(self.framework)
                logger.debug(f"[{self.op_name}] DSL import语句生成成功")
            except Exception as e:
                logger.error(f"[{self.op_name}] DSL import语句生成失败: {e}")
                raise

            try:
                with open(impl_file, "w", encoding="utf-8") as f:
                    f.write(import_statements + impl_code)
                logger.debug(f"[{self.op_name}] 实现文件已创建: {impl_file}")
            except Exception as e:
                logger.error(f"[{self.op_name}] 实现文件创建失败: {impl_file}, 错误: {e}")
                raise

        # 生成验证脚本（使用泛化性模板）
        verify_file = os.path.join(verify_dir, f"verify_{self.op_name}.py")

        # 从文件加载泛化性测试模板
        logger.info(f"[{self.op_name}] 使用泛化性测试模板生成验证脚本")
        try:
            from jinja2 import Template
            with open(self.template_path_generalization, "r", encoding="utf-8") as f:
                template = Template(f.read())
            logger.debug(f"[{self.op_name}] 模板文件加载成功: {self.template_path_generalization}")
        except Exception as e:
            logger.error(f"[{self.op_name}] 模板文件加载失败: {self.template_path_generalization}, 错误: {e}")
            raise

        # 检测是否为动态shape（泛化性测试总是假定为动态shape）
        is_dynamic_shape = True  # 泛化性测试总是使用多case
        logger.info(f"[{self.op_name}] 泛化性测试模式: 多case验证")

        # 获取adapters（与父类相同）
        logger.debug(f"[{self.op_name}] 初始化adapters: framework={self.framework}, dsl={self.dsl}, backend={self.backend}")
        try:
            from ai_kernel_generator.core.verifier.adapters.factory import (
                get_framework_adapter, get_dsl_adapter, get_backend_adapter
            )
            framework_adapter = get_framework_adapter(self.framework)
            dsl_adapter = get_dsl_adapter(self.dsl)
            backend_adapter = get_backend_adapter(self.backend)
            logger.debug(f"[{self.op_name}] Adapters初始化成功")
        except Exception as e:
            logger.error(f"[{self.op_name}] Adapters初始化失败: {e}")
            raise

        # 使用adapter生成代码字符串（与父类相同）
        logger.debug(f"[{self.op_name}] 开始生成代码片段...")
        try:
            framework_imports = framework_adapter.get_import_statements()
            framework_model_import = framework_adapter.get_framework_import(self.op_name, is_dynamic_shape)
            dsl_imports = dsl_adapter.get_import_statements(self.framework)
            dsl_impl_import = dsl_adapter.get_impl_import(self.op_name, self.impl_func_name)
            special_setup_code = dsl_adapter.get_special_setup_code()
            
            # 生成设备设置代码
            backend_adapter.setup_environment(device_id, self.arch)
            device_setup_code = framework_adapter.get_device_setup_code(self.backend, self.arch, device_id)
            
            # 生成输入处理代码
            process_input_code = framework_adapter.get_process_input_code(self.backend, self.dsl)
            
            # 生成创建 impl_model 的代码
            create_impl_code = dsl_adapter.create_impl_module(self.framework, framework_adapter)
            
            # 生成调用实现代码
            call_impl_code = dsl_adapter.call_impl(
                self.impl_func_name, "inputs_for_impl", device_id,
                framework_adapter, self.op_name, "data_dir", "framework_output"
            )
            
            # 生成set_seed代码
            set_seed_code = framework_adapter.get_set_seed_code(self.backend)
            
            # 生成binary I/O函数（如果需要）
            binary_io_functions = ""
            needs_binary_io = dsl_adapter.needs_binary_io()
            if needs_binary_io:
                binary_io_functions = framework_adapter.get_binary_io_functions(self.op_name)
            
            # 获取TensorType名称
            tensor_type_name = framework_adapter.get_tensor_type_name()
            
            logger.debug(f"[{self.op_name}] 所有代码片段生成成功")
        except Exception as e:
            logger.error(f"[{self.op_name}] 代码片段生成失败: {e}", exc_info=True)
            raise

        # 渲染模板
        logger.debug(f"[{self.op_name}] 开始渲染模板...")
        try:
            rendered_code = template.render(
                op_name=self.op_name,
                framework=self.framework,
                dsl=self.dsl,
                device_id=device_id,
                impl_func_name=self.impl_func_name,
                backend=self.backend,
                arch=self.arch,
                is_dynamic_shape=is_dynamic_shape,
                timeout=self.config.get('verify_timeout', 300),
                # Adapter生成的代码
                framework_imports=self._prepare_code_lines(framework_imports),
                framework_model_import=self._prepare_code_lines(framework_model_import),
                dsl_imports=self._prepare_code_lines(dsl_imports),
                dsl_impl_import=self._prepare_code_lines(dsl_impl_import),
                special_setup_code=self._prepare_code_lines(special_setup_code),
                device_setup_code=self._prepare_code_lines(device_setup_code),
                process_input_code=self._prepare_code_lines(process_input_code),
                create_impl_code=self._prepare_code_lines(create_impl_code),
                call_impl_code=self._prepare_code_lines(call_impl_code),
                set_seed_code=self._prepare_code_lines(set_seed_code),
                binary_io_functions=self._prepare_code_lines(binary_io_functions),
                needs_binary_io=needs_binary_io,
                tensor_type_name=tensor_type_name,
            )
            logger.info(f"[{self.op_name}] 模板渲染成功，渲染后代码长度: {len(rendered_code)} 字符")
        except Exception as e:
            logger.error(f"[{self.op_name}] 模板渲染失败: {e}", exc_info=True)
            raise

        # 写入文件
        try:
            with open(verify_file, "w", encoding="utf-8") as f:
                f.write(rendered_code)
            logger.info(f"[{self.op_name}] 验证脚本已写入: {verify_file}")
        except Exception as e:
            logger.error(f"[{self.op_name}] 验证脚本写入失败: {verify_file}, 错误: {e}")
            raise
    
    def run(self, task_info: Dict[str, Any], current_step: int = 0, device_id: int = 0) -> Tuple[Dict[str, Any], str]:
        """
        运行验证并返回详细结果
        
        Returns:
            Tuple[Dict[str, Any], str]: (验证结果字典, 日志)
            
            验证结果字典包含：
            {
                'total_cases': int,
                'status_counts': {'passed': N, 'assert_error': N, ...},
                'pass_rate': float,
                'detailed_results': [...]
            }
        """
        logger.info(f"GeneralizationVerifier Run - Step: {current_step}, Device: {device_id}")

        # 获取代码
        target_code = task_info.get('coder_code', '')
        if not target_code:
            logger.error("No target code found for verification")
            return {'status_counts': {}, 'pass_rate': 0.0}, "No target code found"

        # 创建验证目录
        verify_dir = self._create_verify_dir(current_step)
        
        # 生成验证项目
        self.gen_verify_project(target_code, verify_dir, device_id)
        
        # 运行验证
        verify_timeout = self.config.get('verify_timeout', 300)
        verify_res, verify_log = self.run_verify(verify_dir, timeout=verify_timeout)
        
        # 读取详细结果JSON
        results_json = os.path.join(verify_dir, "verification_results.json")
        
        if os.path.exists(results_json):
            try:
                with open(results_json, 'r', encoding='utf-8') as f:
                    verify_results = json.load(f)
                logger.info(f"[{self.op_name}] 验证结果: {verify_results['status_counts']}")
                return verify_results, verify_log
            except Exception as e:
                logger.error(f"[{self.op_name}] 读取验证结果失败: {e}")
        
        # 如果没有结果文件，返回空结果
        return {
            'total_cases': 0,
            'status_counts': {},
            'pass_rate': 0.0,
            'detailed_results': []
        }, verify_log

