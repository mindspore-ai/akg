"""
测试构造与预运行 - 为构建好的任务生成测试并预运行验证

功能:
  1. 验证任务代码语法
  2. 预运行: 实例化 Model, 调用 get_inputs/get_init_inputs, 执行 forward
  3. 检查输出是否包含 nan/inf
  4. 一致性检查: 两次运行结果一致
  5. 输出统计: mean/std/min/max
"""
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any

from ..config import CODE_EXEC_TIMEOUT, PYTHON_EXECUTABLE


# 预运行验证代码模板 (用 $TASK_CODE 占位符，避免与 f-string 的 {} 冲突)
PRERUN_TEMPLATE = '''\
import sys
import torch
import numpy as np

# ====== 加载任务代码 ======
$TASK_CODE

# ====== 预运行验证 ======
def validate():
    errors = []
    
    # 1. 实例化 Model
    try:
        init_inputs = get_init_inputs()
        model = Model(*init_inputs)
        print(f"[OK] Model 实例化成功, init_inputs={init_inputs}")
    except Exception as e:
        errors.append(f"Model 实例化失败: {e}")
        print(f"[FAIL] Model 实例化失败: {e}")
        return errors

    # 2. 获取输入
    try:
        inputs = get_inputs()
        print(f"[OK] get_inputs() 成功, 共 {len(inputs)} 个输入")
        for i, inp in enumerate(inputs):
            if isinstance(inp, torch.Tensor):
                print(f"  input[{i}]: shape={inp.shape}, dtype={inp.dtype}")
            else:
                print(f"  input[{i}]: type={type(inp).__name__}, value={inp}")
    except Exception as e:
        errors.append(f"get_inputs() 失败: {e}")
        print(f"[FAIL] get_inputs() 失败: {e}")
        return errors

    # 3. 执行 forward
    try:
        model.eval()
        with torch.no_grad():
            output = model(*inputs)
        print(f"[OK] forward 执行成功")
        if isinstance(output, torch.Tensor):
            print(f"  output: shape={output.shape}, dtype={output.dtype}")
        elif isinstance(output, (tuple, list)):
            for i, o in enumerate(output):
                if isinstance(o, torch.Tensor):
                    print(f"  output[{i}]: shape={o.shape}, dtype={o.dtype}")
    except Exception as e:
        errors.append(f"forward 执行失败: {e}")
        print(f"[FAIL] forward 执行失败: {e}")
        return errors

    # 4. 检查 nan/inf
    def check_tensor(t, name):
        if isinstance(t, torch.Tensor):
            if torch.isnan(t).any():
                errors.append(f"{name} 包含 NaN")
                print(f"[FAIL] {name} 包含 NaN")
            if torch.isinf(t).any():
                errors.append(f"{name} 包含 Inf")
                print(f"[FAIL] {name} 包含 Inf")
    
    if isinstance(output, torch.Tensor):
        check_tensor(output, "output")
    elif isinstance(output, (tuple, list)):
        for i, o in enumerate(output):
            check_tensor(o, f"output[{i}]")

    if not errors:
        print("[OK] NaN/Inf 检查通过")

    # 5. 一致性检查：用相同输入跑两次，结果应一致
    try:
        model.eval()
        with torch.no_grad():
            output2 = model(*inputs)
        
        def compare_outputs(o1, o2, name="output"):
            if isinstance(o1, torch.Tensor) and isinstance(o2, torch.Tensor):
                if o1.shape != o2.shape:
                    errors.append(f"{name} 形状不一致: {o1.shape} vs {o2.shape}")
                    print(f"[FAIL] {name} 形状不一致")
                elif not torch.allclose(o1.float(), o2.float(), atol=1e-6, rtol=1e-5):
                    max_diff = (o1.float() - o2.float()).abs().max().item()
                    errors.append(f"{name} 两次运行结果不一致, max_diff={max_diff:.2e}")
                    print(f"[FAIL] {name} 两次运行不一致, max_diff={max_diff:.2e}")
        
        if isinstance(output, torch.Tensor):
            compare_outputs(output, output2)
        elif isinstance(output, (tuple, list)):
            for i, (o1, o2) in enumerate(zip(output, output2)):
                compare_outputs(o1, o2, f"output[{i}]")
        
        if not any("不一致" in e for e in errors):
            print("[OK] 一致性检查通过（两次运行结果相同）")
    except Exception as e:
        print(f"[WARN] 一致性检查跳过: {e}")
    
    # 6. 输出统计（辅助人工判断）
    try:
        def tensor_stats(t, name):
            if isinstance(t, torch.Tensor) and t.numel() > 0:
                t_f = t.float()
                print(f"  {name}: mean={t_f.mean().item():.4f}, std={t_f.std().item():.4f}, "
                      f"min={t_f.min().item():.4f}, max={t_f.max().item():.4f}")
        
        print("\\n--- 输出统计 ---")
        if isinstance(output, torch.Tensor):
            tensor_stats(output, "output")
        elif isinstance(output, (tuple, list)):
            for i, o in enumerate(output):
                tensor_stats(o, f"output[{i}]")
    except Exception:
        pass

    return errors

errors = validate()
if errors:
    print(f"\\n===== 验证失败 ({len(errors)} 个错误) =====")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("\\n===== 验证通过 =====")
    sys.exit(0)
'''


# ====== Reference 对比测试模板 ======
REFERENCE_TEST_TEMPLATE = '''\
import sys
import torch
import traceback

# ====== 加载任务代码 ======
$TASK_CODE

# ====== Reference 对比代码 ======
$REFERENCE_CODE

# ====== 多组输入生成 ======
$MULTI_INPUTS_CODE

# ====== 对比测试 ======
def compare_tensors(our, ref, name, atol=1e-5, rtol=1e-5):
    """对比两个输出（支持单 tensor 和 tuple）"""
    errors = []
    if isinstance(our, torch.Tensor) and isinstance(ref, torch.Tensor):
        if our.shape != ref.shape:
            errors.append(f"{name}: shape mismatch: {our.shape} vs {ref.shape}")
        elif not torch.allclose(our.float(), ref.float(), atol=atol, rtol=rtol):
            diff = (our.float() - ref.float()).abs()
            errors.append(f"{name}: max_diff={diff.max().item():.2e}, mean_diff={diff.mean().item():.2e}")
    elif isinstance(our, (tuple, list)) and isinstance(ref, (tuple, list)):
        if len(our) != len(ref):
            errors.append(f"{name}: output count mismatch: {len(our)} vs {len(ref)}")
        else:
            for i, (o, r) in enumerate(zip(our, ref)):
                errors.extend(compare_tensors(o, r, f"{name}[{i}]", atol, rtol))
    else:
        errors.append(f"{name}: type mismatch: {type(our).__name__} vs {type(ref).__name__}")
    return errors

def run_tests():
    all_errors = []
    
    # 默认 init_inputs
    default_init_inputs = get_init_inputs()
    
    # 获取多组输入
    try:
        test_cases = get_multi_test_inputs()
        print(f"[INFO] 共 {len(test_cases)} 组测试用例")
    except NameError:
        # 如果没定义 get_multi_test_inputs，用默认的 get_inputs
        test_cases = [{"name": "default", "inputs": get_inputs()}]
        print(f"[INFO] 使用默认输入（1组）")
    
    passed = 0
    failed = 0
    
    for tc in test_cases:
        tc_name = tc.get("name", "unnamed")
        inputs = tc["inputs"]
        # 支持 per-case init_inputs 覆盖（解决不同 test case 需要不同参数的问题）
        tc_init_inputs = tc.get("init_inputs", default_init_inputs)
        
        try:
            # 每次用 tc_init_inputs 重新实例化 Model
            model = Model(*tc_init_inputs)
            model.eval()
        except Exception as e:
            failed += 1
            all_errors.append(f"{tc_name}: Model init failed: {e}")
            print(f"[FAIL] {tc_name}: Model init failed: {e}")
            continue
        
        try:
            # 运行我们的实现
            with torch.no_grad():
                our_output = model(*inputs)
            
            # 运行 reference（传入 tc_init_inputs）
            with torch.no_grad():
                ref_output = reference_forward(inputs, tc_init_inputs)
            
            # 对比
            errors = compare_tensors(our_output, ref_output, tc_name)
            
            if errors:
                failed += 1
                for e in errors:
                    print(f"[FAIL] {e}")
                    all_errors.append(e)
            else:
                passed += 1
                # 打印简要信息
                if isinstance(our_output, torch.Tensor):
                    print(f"[PASS] {tc_name}: shape={our_output.shape}")
                elif isinstance(our_output, (tuple, list)):
                    shapes = [o.shape if isinstance(o, torch.Tensor) else type(o).__name__ for o in our_output]
                    print(f"[PASS] {tc_name}: {len(shapes)} outputs")
                else:
                    print(f"[PASS] {tc_name}")
        
        except Exception as e:
            failed += 1
            all_errors.append(f"{tc_name}: {e}")
            print(f"[FAIL] {tc_name}: {e}")
            traceback.print_exc()
    
    print(f"\\n===== Reference 对比结果: {passed} passed, {failed} failed =====")
    if all_errors:
        for e in all_errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        sys.exit(0)

run_tests()
'''


class TestConstructor:
    """测试构造与预运行验证"""

    @staticmethod
    def build_prerun_script(task_code: str) -> str:
        """
        生成预运行验证脚本

        Args:
            task_code: KernelBench 格式的任务代码

        Returns:
            完整的验证脚本代码
        """
        return PRERUN_TEMPLATE.replace("$TASK_CODE", task_code)

    @staticmethod
    def run_validation(task_code: str, timeout: int = None) -> Dict[str, Any]:
        """
        预运行验证任务代码

        Args:
            task_code: KernelBench 格式的任务代码
            timeout: 超时秒数

        Returns:
            {"status": "success"|"error", "output": str, "error": str}
        """
        timeout = timeout or CODE_EXEC_TIMEOUT
        script = TestConstructor.build_prerun_script(task_code)

        try:
            tmp = tempfile.NamedTemporaryFile(
                suffix=".py", mode="w", delete=False, encoding="utf-8"
            )
            tmp.write(script)
            tmp.close()

            result = subprocess.run(
                [PYTHON_EXECUTABLE, tmp.name],
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            combined = ""
            if stdout:
                combined += stdout + "\n"
            if stderr:
                combined += f"[stderr]\n{stderr}\n"

            if result.returncode == 0:
                return {"status": "success", "output": combined, "error": ""}
            else:
                return {"status": "error", "output": combined,
                        "error": f"验证失败 (exit {result.returncode})"}

        except subprocess.TimeoutExpired:
            return {"status": "error", "output": "", "error": f"验证超时 ({timeout}s)"}
        except Exception as e:
            return {"status": "error", "output": "", "error": str(e)}
        finally:
            try:
                Path(tmp.name).unlink(missing_ok=True)
            except Exception:
                pass

    @staticmethod
    def run_reference_test(
        task_code: str,
        reference_code: str,
        multi_inputs_code: str = "",
        timeout: int = None,
    ) -> Dict[str, Any]:
        """
        对比测试：将 Model 的输出与 reference 函数对比。

        Args:
            task_code: KernelBench 格式的任务代码
            reference_code: reference 代码，需定义 reference_forward(inputs, init_inputs) 函数
            multi_inputs_code: 可选，定义 get_multi_test_inputs() 函数返回多组输入
            timeout: 超时秒数

        Returns:
            {"status": "success"|"error", "output": str, "error": str}
        """
        timeout = timeout or CODE_EXEC_TIMEOUT
        script = REFERENCE_TEST_TEMPLATE
        script = script.replace("$TASK_CODE", task_code)
        script = script.replace("$REFERENCE_CODE", reference_code)
        script = script.replace("$MULTI_INPUTS_CODE", multi_inputs_code or "# (no multi_inputs_code)")

        try:
            tmp = tempfile.NamedTemporaryFile(
                suffix=".py", mode="w", delete=False, encoding="utf-8"
            )
            tmp.write(script)
            tmp.close()

            result = subprocess.run(
                [PYTHON_EXECUTABLE, tmp.name],
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            combined = ""
            if stdout:
                combined += stdout + "\n"
            if stderr:
                combined += f"[stderr]\n{stderr}\n"

            if result.returncode == 0:
                return {"status": "success", "output": combined, "error": ""}
            else:
                return {"status": "error", "output": combined,
                        "error": f"Reference 对比失败 (exit {result.returncode})"}

        except subprocess.TimeoutExpired:
            return {"status": "error", "output": "", "error": f"对比测试超时 ({timeout}s)"}
        except Exception as e:
            return {"status": "error", "output": "", "error": str(e)}
        finally:
            try:
                Path(tmp.name).unlink(missing_ok=True)
            except Exception:
                pass
