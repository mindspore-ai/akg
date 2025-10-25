# Copyright 2025 Huawei Technologies Co., Ltd
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

import os

# 全局变量存储配置信息
_collected_config_timings = {}

# 需要过滤的底层实现参数
_FILTERED_CONFIG_PARAMS = {
    'num_warps',
    'num_ctas', 
    'num_stages',
    'num_buffers_warp_spec',
    'num_consumer_groups',
    'reg_dec_producer',
    'reg_inc_consumer',
    'maxnreg'
}


def _filter_config_string(config_str: str) -> str:
    """过滤配置字符串，移除底层实现参数
    
    处理终端打印格式：
    "BLOCK_B: 32, BLOCK_C: 32, num_warps: 4, num_ctas: 1, ..."
    
    Args:
        config_str: 原始配置字符串
        
    Returns:
        过滤后的配置字符串，如 "BLOCK_B: 32, BLOCK_C: 32"
    """
    # 分割参数（按逗号分隔）
    params = []
    for param in config_str.split(','):
        param = param.strip()
        if not param:
            continue
        
        # 提取参数名（支持冒号和等号）
        if ':' in param:
            param_name = param.split(':', 1)[0].strip()
        elif '=' in param:
            param_name = param.split('=', 1)[0].strip()
        else:
            # 没有分隔符的参数保留
            params.append(param)
            continue
        
        # 只保留非过滤参数
        if param_name not in _FILTERED_CONFIG_PARAMS:
            params.append(param)
    
    # 重新组装
    return ', '.join(params)


def patch_triton_autotuner():
    """动态补丁triton autotuner，添加配置信息收集功能"""
    try:
        import triton.runtime.autotuner as autotuner_module
    except ImportError:
        return True

    try:
        import triton.runtime.autotiling_tuner as autotiling_module
    except ImportError:
        autotiling_module = None

    if not hasattr(autotuner_module, 'Autotuner'):
        return True

    original_autotuner_run = getattr(autotuner_module.Autotuner, 'run', None)
    if original_autotuner_run is None:
        return True

    original_autotiling_run = None
    if autotiling_module and hasattr(autotiling_module, 'AutoTilingTuner'):
        original_autotiling_run = getattr(autotiling_module.AutoTilingTuner, 'run', None)

    def _process_config_timings(self):
        """处理配置时间信息"""
        if not (hasattr(self, 'best_config') and
                hasattr(self, 'configs_timings') and
                self.configs_timings and
                isinstance(self.configs_timings, dict)):
            return

        func_name = "unknown_function"
        try:
            if hasattr(self, 'base_fn') and hasattr(self.base_fn, '__name__'):
                func_name = self.base_fn.__name__
            elif hasattr(self, 'fn') and hasattr(self.fn, '__name__'):
                func_name = self.fn.__name__
        except (AttributeError, TypeError):
            pass

        try:
            sorted_timings = sorted(self.configs_timings.items(), key=lambda x: x[1])

            config_data = []
            for i, (config, timing) in enumerate(sorted_timings):
                try:
                    is_best = config == self.best_config
                    timing_value = timing[0] if isinstance(timing, list) else timing
                    # profiler_npu返回的已经是微秒，无需转换
                    timing_us = timing_value
                    
                    # 过滤配置字符串
                    config_str = _filter_config_string(str(config))

                    config_data.append({
                        "config": config_str,
                        "timing_us": float(timing_us),
                        "is_best": is_best,
                        "rank": i + 1
                    })
                except (TypeError, ValueError, AttributeError):
                    continue

            if config_data:
                global _collected_config_timings
                if func_name not in _collected_config_timings:
                    _collected_config_timings[func_name] = config_data

                    if os.getenv("TRITON_PRINT_AUTOTUNING", None) == "1":
                        print(f"All config timings for {func_name}:")
                        for i, (config, timing) in enumerate(sorted_timings):
                            try:
                                status = " (BEST)" if config == self.best_config else ""
                                timing_value = timing[0] if isinstance(timing, list) else timing
                                timing_us = timing_value
                                # 过滤配置字符串
                                config_str = _filter_config_string(str(config))
                                print(f"  Config {i+1}: {config_str} -> {timing_us:.4f}us{status}")
                            except (TypeError, ValueError, AttributeError):
                                continue

        except (TypeError, ValueError, AttributeError):
            pass

    def patched_autotuner_run(self, *args, **kwargs):
        """补丁后的Autotuner.run方法"""
        result = original_autotuner_run(self, *args, **kwargs)
        try:
            _process_config_timings(self)
        except Exception:
            pass
        return result

    def patched_autotiling_run(self, *args, **kwargs):
        """补丁后的AutoTilingTuner.run方法"""
        result = original_autotiling_run(self, *args, **kwargs)
        try:
            _process_config_timings(self)
        except Exception:
            pass
        return result

    try:
        autotuner_module.Autotuner.run = patched_autotuner_run
    except (AttributeError, TypeError):
        pass

    if original_autotiling_run is not None:
        try:
            autotiling_module.AutoTilingTuner.run = patched_autotiling_run
        except (AttributeError, TypeError):
            pass

    return True


def get_collected_config_timings():
    """获取收集的配置时间信息"""
    global _collected_config_timings
    return _collected_config_timings.copy()


def clear_collected_config_timings():
    """清除收集的配置时间信息"""
    global _collected_config_timings
    _collected_config_timings = {}


def patch_driver_benchmarker():
    """补丁driver.active.get_benchmarker()，让autotune使用我们的profiler方法"""
    try:
        from triton.runtime import driver

        # 检查是否已经被补丁过了
        if hasattr(driver.active.get_benchmarker, '_aikg_patched'):
            return True

        original_get_benchmarker = driver.active.get_benchmarker

        def patched_get_benchmarker():
            def custom_benchmarker(kernel_call, quantiles=(0.5, 0.2, 0.8)):
                try:
                    from ai_kernel_generator.core.verifier.profiler import profiler_npu

                    time_us = profiler_npu(
                        kernel_call,
                        warmup=5,
                        active=30,
                        suppress_warnings=True
                    )
                    return [time_us] * 3

                except ImportError:
                    original_benchmarker = original_get_benchmarker()
                    return original_benchmarker(kernel_call, quantiles)

            return custom_benchmarker

        driver.active.get_benchmarker = patched_get_benchmarker
        # 标记已经被补丁过了
        driver.active.get_benchmarker._aikg_patched = True
        return True

    except ImportError:
        return False
    except Exception as e:
        print(f"Warning: Failed to patch driver benchmarker: {e}")
        return False


def apply_triton_patches():
    """应用所有triton补丁"""
    success1 = patch_triton_autotuner()
    success2 = patch_driver_benchmarker()
    return success1 or success2


if __name__ != "__main__":
    apply_triton_patches()

if __name__ == "__main__":
    print("Testing Triton patches...")
    os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

    success1 = patch_triton_autotuner()
    success2 = patch_driver_benchmarker()

    if success1:
        print("Autotuner patch applied successfully!")
    if success2:
        print("Driver benchmarker patch applied successfully!")

    if not any([success1, success2]):
        print("Failed to apply patches")
