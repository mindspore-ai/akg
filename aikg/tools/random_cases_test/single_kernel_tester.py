"""
单Kernel多Case验证器
提供一个kernel和space_config，对它进行多case的泛化性测试
"""

import os
import sys
import time
import json
import importlib.util
import shutil
from typing import Dict, Any
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "python"))

from ai_kernel_generator.utils.case_generator import MultiCaseGenerator
from tools.random_cases_test.generalization_kernel_verifier import GeneralizationKernelVerifier


class SingleKernelTester:
    """单Kernel泛化性测试器"""
    
    def __init__(
        self,
        kernel_path: str,
        space_config_path: str,
        num_cases: int = 100,
        output_dir: str = "results/",
        sampling_strategy: str = "mixed",
        timeout_seconds: float = 10.0,
        seed: int = 42,
        config: dict = None,
        device_id: int = 0,
        kernel_name: str = None
    ):
        """
        Args:
            kernel_path: kernel文件路径
            space_config_path: space_config.py路径
            num_cases: 采样case数量
            output_dir: 输出目录
            sampling_strategy: 采样策略 ('random', 'boundary', 'mixed')
            timeout_seconds: 单个case超时阈值
            seed: 随机种子
            config: 配置字典
            device_id: 设备ID
            kernel_name: kernel名称（用于输出，默认从文件名提取）
        """
        self.kernel_path = kernel_path
        self.space_config_path = space_config_path
        self.num_cases = num_cases
        self.output_dir = output_dir
        self.sampling_strategy = sampling_strategy
        self.timeout_seconds = timeout_seconds
        self.seed = seed
        self.config = config or {}
        self.device_id = device_id
        self.kernel_name = kernel_name or os.path.splitext(os.path.basename(kernel_path))[0]
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载space_config
        self.space_config = self._load_space_config(space_config_path)
        
        # 生成测试framework_code（包含所有cases）
        self.framework_code = self._generate_test_cases()
        
        # 解析cases数量
        self.num_actual_cases = self._count_cases_in_framework(self.framework_code)
    
    def _load_space_config(self, space_config_path: str) -> Dict:
        """加载space_config.py"""
        if not os.path.exists(space_config_path):
            raise FileNotFoundError(f"space_config文件不存在: {space_config_path}")
        
        spec = importlib.util.spec_from_file_location(
            "space_config",
            space_config_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if not hasattr(module, 'SPACE_CONFIG'):
            raise ValueError(f"space_config文件中未找到SPACE_CONFIG")
        if not hasattr(module, 'META_INFO'):
            raise ValueError(f"space_config文件中未找到META_INFO")
        if not hasattr(module, 'create_inputs'):
            raise ValueError(f"space_config文件中未找到create_inputs函数")
        
        return {
            'SPACE_CONFIG': module.SPACE_CONFIG,
            'META_INFO': module.META_INFO,
            'create_inputs': module.create_inputs,
            'Model': getattr(module, 'Model', None)
        }
    
    def _generate_test_cases(self) -> str:
        """使用MultiCaseGenerator生成测试文件，返回framework_code"""
        generator = MultiCaseGenerator(self.space_config_path, seed=self.seed)
        
        # 生成测试文件
        meta = self.space_config['META_INFO']
        multicase_file = os.path.join(
            self.output_dir, 
            f"{meta['op_name']}_multicase_generated.py"
        )
        generator.generate_multicase_file(
            output_path=multicase_file,
            num_cases=self.num_cases,
            strategy=self.sampling_strategy
        )
        
        # 读取生成的文件内容
        with open(multicase_file, 'r', encoding='utf-8') as f:
            framework_code = f.read()
        
        print(f"[OK] 生成多case测试文件: {multicase_file}")
        
        return framework_code
    
    def _count_cases_in_framework(self, framework_code: str) -> int:
        """从framework_code中统计cases数量"""
        try:
            import re
            match = re.search(r'def get_inputs_dyn_list\(\):.*?(?=\ndef |$)', 
                            framework_code, re.DOTALL)
            if match:
                func_body = match.group(0)
                append_count = len(re.findall(r'cases\.append\(', func_body))
                if append_count > 0:
                    return append_count
                
                create_count = len(re.findall(r'inputs\s*=\s*create_inputs\(', func_body))
                if create_count > 0:
                    return create_count
        except Exception as e:
            print(f"[WARNING] 解析cases数量失败: {e}")
        
        return self.num_cases
    
    def run_test(self) -> Dict[str, Any]:
        """
        运行测试
        
        Returns:
            Dict: {
                'status_counts': dict,
                'pass_rate': float,
                'detailed_results': list,
                'execution_time': float,
                'log': str,
                'num_actual_cases': int
            }
        """
        result = {
            'status_counts': {},
            'pass_rate': 0.0,
            'detailed_results': [],
            'execution_time': 0.0,
            'log': '',
            'num_actual_cases': self.num_actual_cases
        }
        
        try:
            # 读取kernel代码
            with open(self.kernel_path, 'r', encoding='utf-8') as f:
                kernel_code = f.read()
            
            # 获取元信息
            meta = self.space_config['META_INFO']
            op_name = meta['op_name']
            framework = meta['framework']
            
            # 规范化framework名称
            if framework == "pytorch":
                framework = "torch"
            
            # 从config获取dsl/backend
            dsl = self.config.get('dsl', 'triton')
            backend = self.config.get('backend', 'cuda')
            
            # 使用kernel名称作为op_name的一部分
            verifier_op_name = f"{op_name}_{self.kernel_name}"
            
            # 创建GeneralizationKernelVerifier实例
            verifier = GeneralizationKernelVerifier(
                op_name=verifier_op_name,
                framework_code=self.framework_code,
                task_id=f"test_{self.kernel_name}",
                framework=framework,
                dsl=dsl,
                backend=backend,
                arch=self.config.get('arch', 'a100'),
                config=self.config
            )
            
            # 运行验证
            print(f"运行 {self.kernel_name} kernel验证...")
            start_time = time.time()
            
            verify_results, log = verifier.run(
                task_info={
                    'coder_code': kernel_code,
                    'framework_code': self.framework_code
                },
                current_step=0,
                device_id=self.device_id
            )
            
            elapsed_time = time.time() - start_time
            
            # 填充结果
            result['status_counts'] = verify_results.get('status_counts', {})
            result['pass_rate'] = verify_results.get('pass_rate', 0.0)
            result['detailed_results'] = verify_results.get('detailed_results', [])
            result['execution_time'] = elapsed_time
            result['log'] = log
            
            # 复制结果JSON到输出目录
            expanded_log_dir = os.path.expanduser(self.config.get('log_dir', '~/aikg_logs'))
            verify_dir_actual = os.path.join(expanded_log_dir, verifier_op_name, f"Itest_{self.kernel_name}_S00_verify")
            results_json = os.path.join(verify_dir_actual, "verification_results.json")
            if os.path.exists(results_json):
                shutil.copy(
                    results_json,
                    os.path.join(self.output_dir, f"{self.kernel_name}_verification_results.json")
                )
        
        except Exception as e:
            import traceback
            result['log'] = f"验证失败: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        
        return result
    
    def print_summary(self, result: Dict[str, Any]):
        """打印测试摘要"""
        status_counts = result.get('status_counts', {})
        pass_rate = result.get('pass_rate', 0.0)
        
        print(f"\n{self.kernel_name} kernel测试完成:")
        print(f"  - 通过: {status_counts.get('passed', 0)}/{self.num_actual_cases}")
        print(f"  - 耗时: {result['execution_time']:.2f}s")
        
        print("\n" + "=" * 80)
        print(f"【{self.kernel_name}】测试结果")
        print("=" * 80)
        print(f"总cases数: {self.num_actual_cases}")
        print(f"  PASS 通过: {status_counts.get('passed', 0)}")
        print(f"  A Assert错误: {status_counts.get('assert_error', 0)}")
        print(f"  P 精度错误: {status_counts.get('precision_error', 0)}")
        print(f"  C 崩溃: {status_counts.get('crash', 0)}")
        print(f"  T 超时: {status_counts.get('timeout', 0)}")
        print(f"\n通过率: {pass_rate * 100:.2f}%")
    
    def save_report(self, result: Dict[str, Any], report_name: str = "test_report.json"):
        """保存测试报告"""
        status_counts = result.get('status_counts', {})
        pass_rate = result.get('pass_rate', 0.0)
        
        report = {
            'kernel': self.kernel_name,
            'kernel_path': self.kernel_path,
            'num_cases': self.num_actual_cases,
            'status_counts': status_counts,
            'pass_rate': pass_rate,
            'execution_time': result['execution_time'],
            'config': {
                'num_cases_requested': self.num_cases,
                'sampling_strategy': self.sampling_strategy,
                'timeout_seconds': self.timeout_seconds,
                'random_seed': self.seed,
            }
        }
        
        report_path = os.path.join(self.output_dir, report_name)
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 保存详细结果
        detail_path = os.path.join(self.output_dir, f"{self.kernel_name}_detailed_results.json")
        detailed = {
            'kernel': self.kernel_name,
            'kernel_path': self.kernel_path,
            'num_cases': self.num_actual_cases,
            'status_counts': status_counts,
            'pass_rate': pass_rate,
            'execution_time': result['execution_time'],
            'detailed_results': result.get('detailed_results', []),
            'log': result['log']
        }
        with open(detail_path, 'w', encoding='utf-8') as f:
            json.dump(detailed, f, indent=2, ensure_ascii=False)
        
        print(f"\n报告已保存到: {report_path}")
        print(f"详细结果已保存到: {detail_path}")
        
        return report_path