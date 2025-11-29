"""
多case测试文件生成器
读取space_config.py，采样生成multicase_task_desc.py
"""

import os
import re
import importlib.util
from typing import Dict, List, Any
from .space_sampler import SpaceSampler


class MultiCaseGenerator:
    """多case测试文件生成器"""
    
    def __init__(self, space_config_path: str, seed: int = 42):
        """
        Args:
            space_config_path: space_config.py的路径
            seed: 随机种子
        """
        self.space_config_path = space_config_path
        self.seed = seed
        
        # 加载space_config
        self.config = self._load_space_config()
        self.space = self.config['SPACE_CONFIG']
        self.meta = self.config['META_INFO']
    
    def _load_space_config(self) -> Dict:
        """动态加载space_config.py"""
        spec = importlib.util.spec_from_file_location(
            "space_config", 
            self.space_config_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        return {
            'SPACE_CONFIG': module.SPACE_CONFIG,
            'META_INFO': module.META_INFO,
            'source': self._read_source()  # 保存源码用于提取函数
        }
    
    def _read_source(self) -> str:
        """读取space_config.py源码"""
        with open(self.space_config_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def generate_multicase_file(
        self, 
        output_path: str,
        num_cases: int = 10,
        strategy: str = 'mixed'
    ):
        """
        生成multicase_task_desc.py文件
        
        Args:
            output_path: 输出文件路径
            num_cases: 采样case数量
            strategy: 采样策略 ('random', 'boundary', 'mixed')
        """
        # 1. 采样
        sampler = SpaceSampler(
            self.space, 
            self.meta['param_names'],
            seed=self.seed
        )
        cases = sampler.sample(num_cases, strategy)
        
        # 2. 生成文件内容
        content = self._generate_file_content(cases)
        
        # 3. 写入文件
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"生成多case文件: {output_path}")
        print(f"采样策略: {strategy}, case数量: {len(cases)}")
    
    def _generate_file_content(self, cases: List[Dict]) -> str:
        """生成完整文件内容"""
        
        # 文件头
        header = f'''"""
自动生成的多case测试文件
算子: {self.meta['op_name']}
框架: {self.meta['framework']}
测试case数量: {len(cases)}
"""

'''
        
        # 提取各部分（从space_config源码）
        imports = self._extract_imports_from_source()
        model_class = self._extract_model_class_from_source()
        helper_funcs = self._extract_section('# ===+ HELPER', '# ===')
        referenced_funcs = self._extract_referenced_functions(model_class)
        create_inputs = self._extract_create_inputs_function()
        
        # 生成get_inputs_dyn_list
        dyn_list = self._generate_dyn_list_function(cases)
        
        # 生成get_init_inputs
        init_func = self._generate_init_inputs()
        
        # 组合
        parts = [header]
        if imports:
            parts.append(imports + '\n')
        if referenced_funcs:  # 先放引用的函数定义
            parts.append(referenced_funcs + '\n')
        if model_class:
            parts.append(model_class + '\n')
        if helper_funcs:
            parts.append(helper_funcs + '\n')
        parts.append(create_inputs + '\n')
        parts.append(dyn_list)
        parts.append(init_func)
        
        return '\n'.join(parts)
    
    def _extract_section(self, start_marker: str, end_marker: str) -> str:
        """从源码中提取标记的section"""
        source = self.config['source']
        pattern = f'{start_marker}(.*?){end_marker}'
        match = re.search(pattern, source, re.DOTALL)
        return match.group(1).strip() if match else ''
    
    def _extract_imports_from_source(self) -> str:
        """提取import语句"""
        lines = self.config['source'].split('\n')
        imports = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(('import ', 'from ')) and not stripped.startswith('#'):
                imports.append(line)
        return '\n'.join(imports)
    
    def _extract_model_class_from_source(self) -> str:
        """提取Model类定义"""
        # 查找class Model开始到下一个class或def或文件末尾
        pattern = r'class Model\([^)]+\):.*?(?=\nclass |\ndef create_|\n# ===|$)'
        match = re.search(pattern, self.config['source'], re.DOTALL)
        return match.group(0).strip() if match else ''
    
    def _extract_referenced_functions(self, model_class: str) -> str:
        """提取Model类的forward方法中引用的外部函数定义"""
        if not model_class:
            return ''
        
        # 只从forward方法中提取函数调用
        forward_pattern = r'def forward\(.*?\):.*?(?=\n    def |\nclass |\n# ===|$)'
        forward_match = re.search(forward_pattern, model_class, re.DOTALL)
        if not forward_match:
            return ''
        
        forward_method = forward_match.group(0)
        
        # 匹配函数调用模式：identifier(
        func_calls = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', forward_method)
        
        # 过滤掉内置函数、方法调用、torch/mindspore等常见模块的函数
        # 以及Model类的方法名（forward, __init__等）
        builtins = {'print', 'len', 'range', 'enumerate', 'zip', 'super', 'isinstance', 
                   'hasattr', 'getattr', 'setattr', 'torch', 'ms', 'return', 'self',
                   'forward', '__init__', '__call__', '__repr__', '__str__'}
        referenced_funcs = set(func_calls) - builtins
        
        # 在源码中查找这些函数的定义
        source = self.config['source']
        extracted_funcs = []
        
        for func_name in referenced_funcs:
            # 匹配函数定义：def func_name(...): ... 直到下一个def/class或特殊标记
            pattern = rf'(def {func_name}\([^)]*\):.*?)(?=\ndef |\nclass |\n# ===|$)'
            match = re.search(pattern, source, re.DOTALL)
            if match:
                func_def = match.group(1).strip()
                # 确保不是create_inputs等已经单独提取的函数
                if not func_name.startswith('create_'):
                    extracted_funcs.append(func_def)
        
        return '\n\n'.join(extracted_funcs) if extracted_funcs else ''
    
    def _extract_create_inputs_function(self) -> str:
        """提取create_inputs函数"""
        pattern = r'def create_inputs\([^)]*\):.*?(?=\ndef |\nclass |\n# ===|$)'
        match = re.search(pattern, self.config['source'], re.DOTALL)
        if not match:
            raise ValueError("space_config中未找到create_inputs函数")
        return match.group(0).strip()
    
    def _generate_dyn_list_function(self, cases: List[Dict]) -> str:
        """生成get_inputs_dyn_list函数"""
        code = 'def get_inputs_dyn_list():\n'
        code += '    """多case测试输入"""\n'
        code += '    cases = []\n\n'
        
        for i, case in enumerate(cases, 1):
            # 注释：显示参数
            case_desc = ', '.join([f'{k}={v}' for k, v in case.items()])
            code += f'    # Case {i}: {case_desc}\n'
            
            # 调用create_inputs
            params = ', '.join([
                self._format_value(case[p]) 
                for p in self.meta['param_names']
            ])
            code += f'    inputs = create_inputs({params})\n'
            code += f'    cases.append(inputs)\n\n'
        
        code += '    return cases\n\n'
        return code
    
    def _format_value(self, value: Any) -> str:
        """格式化参数值为代码字符串"""
        if isinstance(value, str):
            return f'"{value}"'
        elif isinstance(value, bool):
            return 'True' if value else 'False'
        else:
            return str(value)
    
    def _generate_init_inputs(self) -> str:
        """生成get_init_inputs函数"""
        source = self.config['source']
        
        # 1. 首先尝试提取原始的 get_init_inputs 函数
        pattern = r'def get_init_inputs\([^)]*\):.*?(?=\ndef |\nclass |\n# ===|$)'
        match = re.search(pattern, source, re.DOTALL)
        if match:
            # 如果找到了原始的 get_init_inputs，直接使用
            return match.group(0).strip()
        
        # 2. 如果没有 get_init_inputs，检查是否有 create_init_inputs
        create_pattern = r'def create_init_inputs\([^)]*\):'
        if re.search(create_pattern, source):
            # 如果有 create_init_inputs，生成调用它的函数
            return '''def get_init_inputs():
    # 注意：如果init_inputs依赖动态参数，需要手动调整
    return []
'''
        
        # 3. 都没有，返回空函数
        return '''def get_init_inputs():
    return []
'''

