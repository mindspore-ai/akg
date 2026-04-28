"""
参数空间采样器
从参数空间配置中采样测试case
"""

import random
from typing import Dict, List, Any


class SpaceSampler:
    """参数空间采样器"""
    
    def __init__(self, space: Dict[str, Any], param_names: List[str], seed: int = 42):
        """
        Args:
            space: 参数空间配置字典
            param_names: 参数名称列表（保持顺序）
            seed: 随机种子
        """
        self.space = space
        self.param_names = param_names
        random.seed(seed)
    
    def sample(self, num_cases: int, strategy: str = 'mixed') -> List[Dict]:
        """
        采样测试case
        
        Args:
            num_cases: 采样数量
            strategy: 采样策略
                - 'random': 完全随机
                - 'boundary': 边界值（最小、最大、中间）
                - 'mixed': 边界 + 随机（推荐）
        
        Returns:
            List[Dict]: 每个Dict是一组参数值
        """
        if strategy == 'boundary':
            return self._boundary_sample()
        elif strategy == 'mixed':
            boundary = self._boundary_sample()
            remaining = max(0, num_cases - len(boundary))
            if remaining > 0:
                # 传入已有的boundary cases，确保random采样不重复
                random_cases = self._random_sample(remaining, existing_cases=boundary)
                return boundary + random_cases
            return boundary[:num_cases]
        else:  # random
            return self._random_sample(num_cases)
    
    def _random_sample(self, num_cases: int, existing_cases: List[Dict] = None) -> List[Dict]:
        """
        随机采样
        
        Args:
            num_cases: 需要采样的数量
            existing_cases: 已有的cases，新采样的case不会与这些重复
        """
        if existing_cases is None:
            existing_cases = []
        
        cases = existing_cases.copy()  # 复制一份用于去重检查
        new_cases = []  # 只存储新生成的cases
        max_retries = 100  # 最大重试次数，避免死循环
        
        for _ in range(num_cases):
            retries = 0
            while retries < max_retries:
                case = {}
                for param_name in self.param_names:
                    config = self.space[param_name]
                    case[param_name] = self._sample_param(config)
                
                # 检查是否与已有case（包括existing和new）重复
                if not self._is_duplicate(case, cases):
                    cases.append(case)
                    new_cases.append(case)
                    break
                
                retries += 1
            
            # 如果超过最大重试次数，说明参数空间可能太小，直接添加
            if retries >= max_retries and not self._is_duplicate(case, cases):
                cases.append(case)
                new_cases.append(case)
        
        return new_cases
    
    def _is_duplicate(self, case: Dict, cases: List[Dict]) -> bool:
        """检查case是否与已有cases重复"""
        for existing_case in cases:
            if all(case.get(param) == existing_case.get(param) for param in self.param_names):
                return True
        return False
    
    def _boundary_sample(self) -> List[Dict]:
        """边界采样：最小、最大、中间值"""
        min_case = {}
        max_case = {}
        mid_case = {}
        
        for param_name in self.param_names:
            config = self.space[param_name]
            min_case[param_name] = self._get_min(config)
            max_case[param_name] = self._get_max(config)
            mid_case[param_name] = self._get_mid(config)
        
        # 去重：只添加不重复的case
        boundary_cases = []
        for case in [min_case, max_case, mid_case]:
            if not self._is_duplicate(case, boundary_cases):
                boundary_cases.append(case)
        
        return boundary_cases
    
    def _sample_param(self, config: Dict) -> Any:
        """从单个参数配置中采样一个值"""
        if config['type'] == 'choice':
            return random.choice(config['values'])
        
        elif config['type'] == 'range':
            values = list(range(
                config['min'],
                config['max'] + 1,
                config.get('step', 1)
            ))
            return random.choice(values)
        
        elif config['type'] == 'fixed':
            return config['value']
        
        elif config['type'] == 'power_of_2':
            pow_val = random.randint(config['min_pow'], config['max_pow'])
            return 2 ** pow_val
        
        return None
    
    def _get_min(self, config: Dict) -> Any:
        """获取最小值"""
        if config['type'] == 'choice':
            return min(config['values'])
        elif config['type'] == 'range':
            return config['min']
        elif config['type'] == 'fixed':
            return config['value']
        elif config['type'] == 'power_of_2':
            return 2 ** config['min_pow']
        return None
    
    def _get_max(self, config: Dict) -> Any:
        """获取最大值"""
        if config['type'] == 'choice':
            return max(config['values'])
        elif config['type'] == 'range':
            return config['max']
        elif config['type'] == 'fixed':
            return config['value']
        elif config['type'] == 'power_of_2':
            return 2 ** config['max_pow']
        return None
    
    def _get_mid(self, config: Dict) -> Any:
        """获取中间值"""
        if config['type'] == 'choice':
            return config['values'][len(config['values']) // 2]
        elif config['type'] == 'range':
            return (config['min'] + config['max']) // 2
        elif config['type'] == 'fixed':
            return config['value']
        elif config['type'] == 'power_of_2':
            mid_pow = (config['min_pow'] + config['max_pow']) // 2
            return 2 ** mid_pow
        return None

