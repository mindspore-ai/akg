#!/usr/bin/env python3
# Copyright 2026 Huawei Technologies Co., Ltd
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

"""算子验证脚本 — 对比框架实现 (Model) 与生成实现 (ModelNew) 的输出一致性。

支持多框架（torch / mindspore）、多后端（cuda / ascend / cpu）。

用法:
    python verify.py --op_name <算子名> --dsl <dsl> --backend <backend> \
        --framework <framework> [--device_id <id>] \
        [--verify_dir <验证目录>] [--timeout <超时秒数>]

前置条件（验证目录下需存在以下文件）:
    {op_name}_{framework}.py       — 包含 Model, get_inputs, get_init_inputs
    {op_name}_{dsl}_impl.py  — 包含 ModelNew
"""
import argparse
import os
import sys
import subprocess


def get_limit(data_type):
    """根据数据类型获取精度阈值"""
    import torch
    if data_type == torch.float16:
        return 0.004
    elif data_type == torch.bfloat16:
        return 0.03
    elif data_type == torch.int8:
        return 0.01
    else:
        return 0.02


def compare(fw_out, impl_out, limit, data_type):
    """对比框架输出和实现输出"""
    import torch
    fw_flat = fw_out.flatten().detach().cpu()
    impl_flat = impl_out.flatten()
    if isinstance(impl_flat, torch.Tensor):
        impl_flat = impl_flat.detach().cpu()
    else:
        impl_flat = torch.tensor(impl_flat, dtype=fw_flat.dtype)

    size = fw_flat.numel()

    if fw_flat.shape != impl_flat.shape:
        raise AssertionError(
            f"验证失败，输出形状不一致: framework={fw_flat.shape}, impl={impl_flat.shape}"
        )

    fw_nan_mask = torch.isnan(fw_flat)
    impl_nan_mask = torch.isnan(impl_flat)
    if not torch.equal(fw_nan_mask, impl_nan_mask):
        raise AssertionError(
            f"验证失败，NaN 位置不匹配: Framework={fw_nan_mask.sum().item()}/{size}, "
            f"Implementation={impl_nan_mask.sum().item()}/{size}"
        )

    fw_inf_mask = torch.isinf(fw_flat)
    impl_inf_mask = torch.isinf(impl_flat)
    if not torch.equal(fw_inf_mask, impl_inf_mask):
        raise AssertionError(
            f"验证失败，Inf 位置不匹配: Framework={fw_inf_mask.sum().item()}/{size}, "
            f"Implementation={impl_inf_mask.sum().item()}/{size}"
        )
    if fw_inf_mask.any():
        if not torch.equal(
            torch.sign(fw_flat[fw_inf_mask]),
            torch.sign(impl_flat[impl_inf_mask]),
        ):
            raise AssertionError("验证失败，Inf 符号不匹配")

    finite_mask = torch.isfinite(fw_flat) & torch.isfinite(impl_flat)
    finite_count = finite_mask.sum().item()
    if finite_count == 0:
        print("警告: 所有值都是非有限值，跳过精度检查")
        return

    fw_finite = fw_flat[finite_mask]
    impl_finite = impl_flat[finite_mask]

    if fw_finite.dtype == torch.bool:
        if not torch.equal(fw_finite, impl_finite):
            raise AssertionError(f"验证失败，布尔值不匹配: dtype={data_type}")
        return

    if impl_finite.dtype != fw_finite.dtype:
        impl_finite = impl_finite.to(fw_finite.dtype)

    abs_diff = torch.abs(fw_finite.float() - impl_finite.float())
    abs_ref = torch.abs(fw_finite.float())
    eps = 1e-8
    relative_error = torch.where(abs_ref > eps, abs_diff / abs_ref, abs_diff)

    err_cnt = (relative_error > limit).sum().item()
    limit_cnt = int(finite_count * limit)

    if err_cnt > limit_cnt:
        max_error = relative_error.max().item()
        mean_error = relative_error.mean().item()
        mismatch_mask = relative_error > limit
        mismatch_indices = torch.where(mismatch_mask)[0]
        num_to_show = min(10, len(mismatch_indices))

        error_msg = (
            f"验证失败，输出不一致(误差数/最大容忍误差数): "
            f"err_cnt={err_cnt} / {limit_cnt}, dtype={data_type}, limit={limit}\n"
        )
        error_msg += f"最大相对误差: {max_error:.6e}, 平均相对误差: {mean_error:.6e}\n"
        error_msg += f"前 {num_to_show} 个不一致的值:\n"
        for i in range(num_to_show):
            idx = mismatch_indices[i].item()
            error_msg += (
                f"  位置[{idx}]: framework={fw_finite[idx]:.6e}, "
                f"impl={impl_finite[idx]:.6e}, "
                f"相对误差={relative_error[idx]:.6e}\n"
            )
        raise AssertionError(error_msg)


def setup_device(framework, backend, device_id):
    """根据 framework/backend/device_id 配置运行设备并返回 device 对象"""
    if framework == "torch":
        import torch
        if backend == "ascend":
            import torch_npu  # noqa: F401
            if device_id >= 0:
                torch.npu.set_device(device_id)
            return torch.device(f"npu:{device_id}" if device_id >= 0 else "npu")
        elif backend == "cuda":
            if device_id >= 0:
                torch.cuda.set_device(device_id)
            return torch.device(f"cuda:{device_id}" if device_id >= 0 else "cuda")
        elif backend == "cpu":
            return torch.device("cpu")
        else:
            raise ValueError(f"不支持的 backend: {backend}")
    elif framework == "mindspore":
        import mindspore as ms
        if backend == "ascend":
            ms.set_context(device_target="Ascend", device_id=max(device_id, 0))
        elif backend == "cpu":
            ms.set_context(device_target="CPU")
        else:
            raise ValueError(f"mindspore 不支持 backend: {backend}")
        return None
    else:
        raise ValueError(f"不支持的 framework: {framework}")


def set_seed(framework, backend, seed=0):
    """设置随机种子"""
    if framework == "torch":
        import torch
        torch.manual_seed(seed)
        if backend == "ascend":
            torch.npu.manual_seed(seed)
        elif backend == "cuda":
            torch.cuda.manual_seed(seed)
    elif framework == "mindspore":
        import mindspore as ms
        ms.set_seed(seed)


def verify_implementations(op_name, dsl, backend, framework, device_id, verify_dir):
    """验证框架实现和生成实现的结果一致性"""
    device = setup_device(framework, backend, device_id)

    sys.path.insert(0, verify_dir)

    torch_module = __import__(f"{op_name}_{framework}")
    FrameworkModel = torch_module.Model
    get_inputs = torch_module.get_inputs
    get_init_inputs = torch_module.get_init_inputs

    impl_module = __import__(f"{op_name}_{dsl}_impl")
    ModelNew = impl_module.ModelNew

    if framework == "torch":
        import torch

        set_seed(framework, backend)
        init_params = get_init_inputs()
        framework_model = FrameworkModel(*init_params)
        impl_model = ModelNew(*init_params)
        if device is not None:
            framework_model = framework_model.to(device)
            impl_model = impl_model.to(device)

        set_seed(framework, backend)
        inputs_for_impl = [
            x.to(device) if isinstance(x, torch.Tensor) and device is not None else x
            for x in get_inputs()
        ]

        set_seed(framework, backend)
        inputs_for_framework = [
            x.to(device) if isinstance(x, torch.Tensor) and device is not None else x
            for x in get_inputs()
        ]

        with torch.no_grad():
            impl_output = impl_model(*inputs_for_impl)
        with torch.no_grad():
            framework_output = framework_model(*inputs_for_framework)

        if not isinstance(framework_output, (list, tuple)):
            framework_output = [framework_output]
        if not isinstance(impl_output, (list, tuple)):
            impl_output = [impl_output]

        if len(framework_output) != len(impl_output):
            raise AssertionError(
                f"验证失败，输出数量不一致: framework={len(framework_output)}, "
                f"impl={len(impl_output)}"
            )

        for i, (fw_out, impl_out) in enumerate(zip(framework_output, impl_output)):
            if fw_out is None or impl_out is None:
                raise AssertionError(
                    f"输出 {i} 为 None: framework={fw_out is None}, impl={impl_out is None}"
                )
            if isinstance(fw_out, torch.Tensor) and isinstance(impl_out, torch.Tensor):
                data_type = fw_out.dtype
                limit = get_limit(data_type)
                compare(fw_out, impl_out, limit, data_type)

    else:
        raise NotImplementedError(f"framework={framework} 的验证逻辑尚未实现")

    print("验证成功")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="算子验证脚本")
    parser.add_argument("--op_name", required=True, help="算子名称")
    parser.add_argument("--dsl", required=True, help="DSL（如 triton_cuda, triton_ascend, cpp）")
    parser.add_argument("--backend", required=True, help="后端（cuda / ascend / cpu）")
    parser.add_argument("--framework", default="torch", help="框架（torch / mindspore，默认 torch）")
    parser.add_argument("--device_id", type=int, default=0, help="设备 ID（默认 0，-1 表示自动）")
    parser.add_argument("--verify_dir", default=".", help="验证目录（默认当前目录）")
    parser.add_argument("--timeout", type=int, default=300, help="超时秒数（默认 300）")
    parser.add_argument("--_run", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    verify_dir = os.path.abspath(args.verify_dir)
    if not os.path.isdir(verify_dir):
        print(f"错误: 验证目录不存在: {verify_dir}", file=sys.stderr)
        sys.exit(1)

    if args._run:
        try:
            verify_implementations(
                args.op_name, args.dsl, args.backend,
                args.framework, args.device_id, verify_dir,
            )
        except Exception as e:
            print(f"{e}", file=sys.stderr)
            sys.exit(1)
    else:
        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--op_name", args.op_name,
            "--dsl", args.dsl,
            "--backend", args.backend,
            "--framework", args.framework,
            "--device_id", str(args.device_id),
            "--verify_dir", verify_dir,
            "--_run",
        ]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = proc.communicate(timeout=args.timeout)
            sys.stdout.buffer.write(stdout)
            sys.stdout.buffer.flush()
            if proc.returncode != 0:
                sys.stderr.buffer.write(stderr)
                sys.stderr.buffer.flush()
                sys.exit(proc.returncode)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            print(f"验证超时（{args.timeout}秒），已终止子进程", file=sys.stderr)
            sys.exit(1)
