"""
算子鲁棒性测试通用运行器。

提供精度验证、性能测试、DevicePool 并行等通用功能。
算子特定的部分（CONFIG, TEST_CASES, make_inputs）由调用脚本定义。

性能测试方法参考 akg_agents 仓库的实际实现：
  - Ascend (NPU): profiler_npu（硬件 profiler + L2 cache 清除）
  - CUDA  (GPU): triton.testing.do_bench
  - CPU:         time.perf_counter 循环计时

使用方法：
    from robustness_test_runner import RobustnessTestRunner
    runner = RobustnessTestRunner(CONFIG, TEST_CASES, make_inputs)
    runner.run()
"""

import sys
import os
import time
import json
import importlib
import asyncio
import torch

from akg_agents.core.async_pool.device_pool import DevicePool


# ============================================================
# 精度比对
# ============================================================

def get_limit(dtype):
    """根据 dtype 返回误差容忍度（与 akg_agents verify 模板一致）"""
    return {
        torch.float16: 0.004,
        torch.bfloat16: 0.03,
        torch.int8: 0.01,
    }.get(dtype, 0.02)


def compare_outputs(fw_out, impl_out, limit):
    """
    精度比对，逻辑与 akg_agents FrameworkAdapterTorch.get_compare_code() 一致：
    - 检查 shape 一致性
    - 检查 NaN/Inf 位置一致性
    - 对有限值计算相对误差，超限元素数 > 总元素数 * limit 则 FAIL
    """
    fw_flat = fw_out.flatten().detach().cpu().float()
    impl_flat = impl_out.flatten().detach().cpu()
    if isinstance(impl_flat, torch.Tensor):
        impl_flat = impl_flat.float()
    else:
        impl_flat = torch.tensor(impl_flat, dtype=torch.float32)

    if fw_flat.shape != impl_flat.shape:
        raise AssertionError(
            f"输出 shape 不一致: framework={fw_out.shape}, impl={impl_out.shape}"
        )

    size = fw_flat.numel()

    # 检查 NaN 位置一致性
    fw_nan = torch.isnan(fw_flat)
    impl_nan = torch.isnan(impl_flat)
    if not torch.equal(fw_nan, impl_nan):
        raise AssertionError(
            f"NaN 位置不匹配: framework={fw_nan.sum().item()}/{size}, "
            f"impl={impl_nan.sum().item()}/{size}"
        )

    # 检查 Inf 位置一致性
    fw_inf = torch.isinf(fw_flat)
    impl_inf = torch.isinf(impl_flat)
    if not torch.equal(fw_inf, impl_inf):
        raise AssertionError(
            f"Inf 位置不匹配: framework={fw_inf.sum().item()}/{size}, "
            f"impl={impl_inf.sum().item()}/{size}"
        )
    if fw_inf.any() and not torch.equal(
        torch.sign(fw_flat[fw_inf]), torch.sign(impl_flat[impl_inf])
    ):
        raise AssertionError("Inf 符号不匹配")

    # 对有限值做精度比较
    finite_mask = torch.isfinite(fw_flat) & torch.isfinite(impl_flat)
    finite_count = finite_mask.sum().item()
    if finite_count == 0:
        return

    fw_fin = fw_flat[finite_mask]
    impl_fin = impl_flat[finite_mask]

    if fw_fin.dtype == torch.bool:
        if not torch.equal(fw_fin, impl_fin):
            raise AssertionError("布尔值不匹配")
        return

    abs_diff = torch.abs(fw_fin - impl_fin)
    abs_ref = torch.abs(fw_fin).clamp(min=1e-8)
    rel_err = abs_diff / abs_ref

    err_cnt = (rel_err > limit).sum().item()
    limit_cnt = int(finite_count * limit)

    if err_cnt > limit_cnt:
        max_err = rel_err.max().item()
        mean_err = rel_err.mean().item()
        raise AssertionError(
            f"精度不达标: 超限元素 {err_cnt}/{limit_cnt}, "
            f"max_rel_err={max_err:.6e}, mean_rel_err={mean_err:.6e}"
        )


# ============================================================
# 性能测试
# ============================================================

def run_benchmark(model, inputs, backend="cpu", dsl="cpp",
                  warmup=5, run_times=50):
    """
    性能基准测试，按后端选择正确的计时方法（与 akg_agents 仓库一致）:

      - Ascend(NPU): 使用 profiler_npu（硬件 profiler + L2 cache 清除）
                     参考: akg_agents.op.verifier.profiler.profiler_npu
      - CUDA(GPU) + Triton: 使用 triton.testing.do_bench
                     参考: DSLAdapterTritonCuda.benchmark_impl
      - CUDA(GPU) 非 Triton: time.perf_counter + torch.cuda.synchronize
      - CPU:         使用 time.perf_counter 循环计时
                     参考: DSLAdapterCpp.benchmark_impl

    Args:
        model: 模型实例（FrameworkModel 或 ModelNew）
        inputs: 输入 tensor 列表
        backend: "ascend" / "cuda" / "cpu"
        dsl: DSL 类型，影响计时方式选择
        warmup: 预热次数
        run_times: 有效测量次数

    Returns:
        float: 平均执行时间 (微秒 us)
    """
    def benchmark_fn():
        return model(*inputs)

    if backend == "ascend":
        # NPU: 使用 akg_agents 的 profiler_npu
        # 支持 L2 cache 清除以获得一致的测量结果
        from akg_agents.op.verifier.profiler import profiler_npu
        dsl_type = "triton_ascend" if "triton_ascend" in dsl else "other"
        exec_time_us = profiler_npu(
            benchmark_fn,
            warmup=warmup,
            active=run_times,
            prof_dir_name="prof_robustness",
            keep_res=False,
            suppress_warnings=True,
            clear_l2_cache=True,
            dsl=dsl_type,
        )
        return exec_time_us

    elif backend == "cuda" and "triton" in dsl:
        # CUDA + Triton: 使用 triton.testing.do_bench（内部处理 synchronize）
        import triton.testing
        exec_time_ms = triton.testing.do_bench(
            benchmark_fn,
            warmup=warmup,
            rep=run_times,
            return_mode="median",
        )
        return exec_time_ms * 1000  # ms → us

    elif backend == "cuda":
        # CUDA 非 Triton: 手动 synchronize + 循环计时
        for _ in range(warmup):
            benchmark_fn()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(run_times):
            benchmark_fn()
        torch.cuda.synchronize()
        end = time.perf_counter()
        return (end - start) * 1e6 / run_times

    else:
        # CPU: 循环计时
        for _ in range(warmup):
            benchmark_fn()
        start = time.perf_counter()
        for _ in range(run_times):
            benchmark_fn()
        end = time.perf_counter()
        return (end - start) * 1e6 / run_times


# ============================================================
# 工具函数
# ============================================================

def setup_device(backend, device_id=0):
    """初始化设备上下文并返回 torch.device。
    """
    if backend == "ascend":
        import torch_npu  # noqa: F401
        os.environ["DEVICE_ID"] = str(device_id)
        torch.npu.set_device(device_id)
        return torch.device(f"npu:{device_id}")
    elif backend == "cuda":
        torch.cuda.set_device(device_id)
        return torch.device(f"cuda:{device_id}")
    return torch.device("cpu")


def format_shape(params):
    """将 params dict 格式化为可读的 shape 字符串"""
    return ", ".join(f"{k}={v}" for k, v in params.items())


# ============================================================
# RobustnessTestRunner
# ============================================================

class RobustnessTestRunner:
    """
    算子鲁棒性测试通用运行器。

    将精度验证、性能测试、并行调度等通用逻辑封装在此，
    算子特定的部分（CONFIG, TEST_CASES, make_inputs）由调用脚本提供。

    精度验证和性能测试均以子进程方式执行：
      1. 精度验证：算子首次前向传播会触发编译（C++/Triton/AscendC），编译过程
         会启动新的子进程。信号量（signal.alarm）无法杀死编译子进程，因此使用
         asyncio.create_subprocess_exec + asyncio.wait_for + process.kill()。
      2. 性能测试：torch_npu.profiler.profile() 会修改进程级全局状态，退出
         profiler 上下文后 NPU 设备状态被污染，同一进程内后续的 profiler_npu()
         调用会触发 "vector core exception"。akg_agents 仓库的做法也是每个
         profile 脚本作为独立进程运行（通过 run_command(["python", script])）。

    使用方法：
        runner = RobustnessTestRunner(CONFIG, TEST_CASES, make_inputs)
        runner.run()
    """

    def __init__(self, config, test_cases, make_inputs_fn, get_init_params_fn=None):
        """
        Args:
            config: dict, 测试配置，必需字段:
                verify_dir, task_module, kernel_module,
                op_name, framework, dsl, backend, arch,
                device_ids, seed, warmup_times, run_times, verify_timeout
            test_cases: list of (tag, params_dict, dtype, description)
            make_inputs_fn: callable(params_dict, dtype, device) -> list[Tensor]
            get_init_params_fn: callable(params_dict) -> list or None (可选,
                用于参数绑定维度需要重建模型的场景；返回 None 表示使用默认 get_init_inputs())
        """
        self.config = config
        self.test_cases = test_cases
        self.make_inputs_fn = make_inputs_fn
        self.get_init_params_fn = get_init_params_fn

        # 导入模型
        verify_dir = os.path.expanduser(config["verify_dir"])
        if verify_dir not in sys.path:
            sys.path.insert(0, verify_dir)

        task_mod = importlib.import_module(config["task_module"])
        self.FrameworkModel = task_mod.Model
        self.default_get_init_inputs = task_mod.get_init_inputs

        try:
            kernel_mod = importlib.import_module(config["kernel_module"])
            self.ModelNew = kernel_mod.ModelNew
        except Exception as e:
            print(f"ERROR: 无法导入 kernel_code: {e}", file=sys.stderr)
            print("请检查编译环境和依赖（如 C++ 编译器、CUDA toolkit 等）", file=sys.stderr)
            sys.exit(1)

    # ---- 入口 ----

    def run(self):
        """根据命令行参数分派主模式或子进程模式"""
        if len(sys.argv) > 1 and sys.argv[1] == "--verify-case":
            self._run_verify_subprocess()
        elif len(sys.argv) > 1 and sys.argv[1] == "--benchmark-case":
            self._run_benchmark_subprocess()
        else:
            self._run_main()

    # ---- 子进程: 精度验证 ----

    def _run_verify_subprocess(self):
        case_idx = int(sys.argv[2])
        device_id = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        tag, params, dtype, desc = self.test_cases[case_idx]
        try:
            self._verify_single(params, dtype, device_id)
        except Exception as e:
            print(str(e), file=sys.stderr)
            sys.exit(1)
        sys.exit(0)

    def _run_benchmark_subprocess(self):
        """子进程模式：执行单个 case 的性能测试，结果写入 JSON 文件。

        参考 profiler_utils.py 的做法：子进程写文件，主进程读文件。
        不依赖 stdout 传递数据（profiler_npu 的 suppress_output() 会
        替换 sys.stdout，导致 stdout 输出不可靠）。

        用法: script --benchmark-case <case_idx> <device_id> <result_file>
        """
        case_idx = int(sys.argv[2])
        device_id = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        result_file = sys.argv[4]
        tag, params, dtype, desc = self.test_cases[case_idx]
        cfg = self.config
        device = setup_device(cfg["backend"], device_id)
        init_params = self._get_init_params(params)

        torch.manual_seed(cfg["seed"])
        fw_model = self.FrameworkModel(*init_params)
        impl_model = self.ModelNew(*init_params)

        if device.type != "cpu":
            fw_model = fw_model.to(device)
            impl_model = impl_model.to(device)

        torch.manual_seed(cfg["seed"])
        inputs_perf = self.make_inputs_fn(params, dtype, device)

        base_us = run_benchmark(
            fw_model, inputs_perf,
            backend=cfg["backend"], dsl=cfg["dsl"],
            warmup=cfg["warmup_times"], run_times=cfg["run_times"],
        )
        gen_us = run_benchmark(
            impl_model, inputs_perf,
            backend=cfg["backend"], dsl=cfg["dsl"],
            warmup=cfg["warmup_times"], run_times=cfg["run_times"],
        )

        # 结果写入文件（参考 profiler_utils.py 的 profile 脚本写 JSON 文件模式）
        # profiler_npu 失败时返回 float('inf')，用 None 表示失败
        result = {
            "base_time_us": base_us if base_us != float("inf") else None,
            "gen_time_us": gen_us if gen_us != float("inf") else None,
        }
        with open(result_file, "w") as f:
            json.dump(result, f)
        sys.exit(0)

    def _get_init_params(self, params):
        """获取模型初始化参数（支持按 case 覆盖）"""
        if self.get_init_params_fn:
            override = self.get_init_params_fn(params)
            if override is not None:
                return override
        return self.default_get_init_inputs()

    def _verify_single(self, params, dtype, device_id):
        """独立精度验证（在子进程中运行）"""
        cfg = self.config
        device = setup_device(cfg["backend"], device_id)
        init_params = self._get_init_params(params)

        torch.manual_seed(cfg["seed"])
        fw_model = self.FrameworkModel(*init_params)
        impl_model = self.ModelNew(*init_params)

        if device.type != "cpu":
            fw_model = fw_model.to(device)
            impl_model = impl_model.to(device)

        torch.manual_seed(cfg["seed"])
        inputs_fw = self.make_inputs_fn(params, dtype, device)
        torch.manual_seed(cfg["seed"])
        inputs_impl = self.make_inputs_fn(params, dtype, device)

        fw_out = fw_model(*inputs_fw)
        impl_out = impl_model(*inputs_impl)

        # 统一为 list
        fw_outs = fw_out if isinstance(fw_out, (list, tuple)) else [fw_out]
        impl_outs = impl_out if isinstance(impl_out, (list, tuple)) else [impl_out]

        if len(fw_outs) != len(impl_outs):
            raise AssertionError(
                f"输出个数不一致: framework={len(fw_outs)}, impl={len(impl_outs)}"
            )

        limit = get_limit(dtype)
        for fw_o, impl_o in zip(fw_outs, impl_outs):
            compare_outputs(fw_o, impl_o, limit)

    # ---- 单 case 测试（精度验证 + 性能测试） ----

    async def _test_single_case(self, case_idx, tag, params, dtype, desc, device_id):
        """对单个 shape/dtype 执行精度验证和性能测试（均通过子进程执行）"""
        cfg = self.config

        result = {
            "tag": tag,
            "shape": format_shape(params),
            "dtype": str(dtype),
            "description": desc,
            "device_id": device_id,
            "accuracy": None,
            "base_time_us": None,
            "gen_time_us": None,
            "speedup": None,
            "perf_status": None,
            "error": None,
        }

        # --- 精度验证（子进程 + 超时保护）---
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        # 脚本路径是调用脚本（非 runner），确保子进程能重建 runner
        script = os.path.abspath(sys.argv[0])
        proc = await asyncio.create_subprocess_exec(
            sys.executable, script,
            "--verify-case", str(case_idx), str(device_id),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=cfg["verify_timeout"]
            )
            if proc.returncode != 0:
                result["accuracy"] = "FAIL"
                result["error"] = stderr.decode(errors="replace").strip()
                result["perf_status"] = "SKIP"
                return result
            result["accuracy"] = "PASS"

        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            result["accuracy"] = "TIMEOUT"
            result["error"] = f"精度验证超时 ({cfg['verify_timeout']}s)，已终止子进程"
            result["perf_status"] = "SKIP"
            return result

        # --- 性能测试（子进程写文件，主进程读文件）---
        # 参考 profiler_utils.py：子进程将结果写入 JSON 文件，
        # 主进程读文件获取结果。不依赖 stdout（会被 profiler 污染）。
        import tempfile
        result_file = tempfile.mktemp(
            suffix=".json", prefix=f"bench_{case_idx}_"
        )
        try:
            bench_proc = await asyncio.create_subprocess_exec(
                sys.executable, script,
                "--benchmark-case", str(case_idx), str(device_id), result_file,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            _, bench_stderr = await asyncio.wait_for(
                bench_proc.communicate(), timeout=cfg["verify_timeout"]
            )

            if bench_proc.returncode != 0:
                result["perf_status"] = "ERROR"
                result["error"] = bench_stderr.decode(errors="replace").strip()[-1000:]
                return result

            # 读取结果文件
            with open(result_file, "r") as f:
                bench_data = json.load(f)

            base_us = bench_data["base_time_us"]
            gen_us = bench_data["gen_time_us"]

            if base_us is None or gen_us is None:
                result["perf_status"] = "ERROR"
                result["error"] = f"profiler 数据收集失败: base={base_us}, gen={gen_us}"
                return result

            speedup = base_us / gen_us if gen_us > 0 else float("inf")
            result["base_time_us"] = round(base_us, 2)
            result["gen_time_us"] = round(gen_us, 2)
            result["speedup"] = round(speedup, 3)
            result["perf_status"] = (
                "PASS" if speedup >= 0.95
                else "WARN" if speedup >= 0.8
                else "FAIL"
            )

        except asyncio.TimeoutError:
            try:
                bench_proc.kill()
            except ProcessLookupError:
                pass
            result["perf_status"] = "ERROR"
            result["error"] = f"性能测试超时 ({cfg['verify_timeout']}s)"

        except Exception as e:
            result["perf_status"] = "ERROR"
            result["error"] = str(e)

        finally:
            # 清理临时文件
            if os.path.exists(result_file):
                os.unlink(result_file)

        return result

    # ---- DevicePool 并行 ----

    async def _run_all_tests(self):
        """使用 DevicePool 并行分配设备，执行所有测试 case"""
        cfg = self.config
        pool = DevicePool(cfg["device_ids"])

        async def run_on_device(idx, case):
            device_id = await pool.acquire_device()
            try:
                tag, params, dtype, desc = case
                return await self._test_single_case(
                    idx, tag, params, dtype, desc, device_id
                )
            finally:
                await pool.release_device(device_id)

        tasks = [run_on_device(i, c) for i, c in enumerate(self.test_cases)]
        return list(await asyncio.gather(*tasks))

    # ---- 结果输出 ----

    def _print_and_save_results(self, results):
        """打印结果表格并保存 JSON 汇总"""
        cfg = self.config

        header = (
            f"{'Tag':15s} {'Shape':30s} {'dtype':12s} "
            f"{'Acc':6s} {'base(us)':>10s} {'gen(us)':>10s} {'speedup':>8s} {'Perf':6s}"
        )
        print(f"\n{header}")
        print("-" * 100)

        for r in results:
            base = f"{r['base_time_us']:>10.2f}" if r.get("base_time_us") is not None else "       N/A"
            gen = f"{r['gen_time_us']:>10.2f}" if r.get("gen_time_us") is not None else "       N/A"
            spd = f"{r['speedup']:>7.3f}x" if r.get("speedup") is not None else "     N/A"
            perf = r.get("perf_status", "N/A")

            print(
                f"{r['tag']:15s} {r['shape']:30s} {r['dtype']:12s} "
                f"{r['accuracy']:6s} {base} {gen} {spd} {perf:6s}"
            )
            if r.get("error"):
                print(f"  -> {r['error']}")

        total = len(results)
        acc_pass = sum(1 for r in results if r["accuracy"] == "PASS")
        acc_fail = sum(1 for r in results if r["accuracy"] == "FAIL")
        acc_timeout = sum(1 for r in results if r["accuracy"] == "TIMEOUT")
        print("-" * 100)

        parts = [f"PASS {acc_pass}"]
        if acc_fail:
            parts.append(f"FAIL {acc_fail}")
        if acc_timeout:
            parts.append(f"TIMEOUT {acc_timeout}")
        print(f"精度: {' / '.join(parts)} (共 {total})")

        perf_pass = sum(1 for r in results if r.get("perf_status") == "PASS")
        perf_warn = sum(1 for r in results if r.get("perf_status") == "WARN")
        perf_fail = sum(1 for r in results if r.get("perf_status") == "FAIL")
        perf_tested = perf_pass + perf_warn + perf_fail
        if perf_tested:
            print(f"性能通过率 (speedup >= 0.95): {perf_pass}/{perf_tested}")

        # JSON 汇总
        original = self.test_cases[0]
        summary = {
            "metadata": {
                "op_name": cfg["op_name"],
                "test_date": time.strftime("%Y-%m-%d"),
                "framework": cfg["framework"],
                "dsl": cfg["dsl"],
                "backend": cfg["backend"],
                "arch": cfg["arch"],
                "original_shape": format_shape(original[1]),
                "original_dtype": str(original[2]),
            },
            "summary": {
                "total_cases": total,
                "accuracy_pass": acc_pass,
                "accuracy_fail": acc_fail,
                "accuracy_timeout": acc_timeout,
                "perf_pass": perf_pass,
                "perf_warn": perf_warn,
                "perf_fail": perf_fail,
                "perf_skip": sum(1 for r in results if r.get("perf_status") == "SKIP"),
            },
            "cases": results,
        }

        out_path = f"{cfg['op_name']}_robustness_summary.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"详细结果: {out_path}")

    # ---- 主流程 ----

    def _run_main(self):
        """主模式入口：打印配置 → 并行执行 → 输出结果"""
        cfg = self.config
        original = self.test_cases[0]

        print("=" * 100)
        print(f"算子鲁棒性测试: {cfg['op_name']}")
        print(f"配置: framework={cfg['framework']}, dsl={cfg['dsl']}, "
              f"backend={cfg['backend']}, arch={cfg['arch']}")
        print(f"原始通过: shape=({format_shape(original[1])}), dtype={original[2]}")
        print(f"设备: {cfg['device_ids']} (共 {len(cfg['device_ids'])} 个)")
        print(f"精度验证时间限制: {cfg['verify_timeout']}s")
        print("=" * 100)

        results = asyncio.run(self._run_all_tests())
        self._print_and_save_results(results)
