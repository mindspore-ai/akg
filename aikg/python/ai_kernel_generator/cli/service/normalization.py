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

from __future__ import annotations

from typing import List

from ai_kernel_generator.cli.constants import Backend, Framework
from ai_kernel_generator.core.utils import VALID_CONFIGS, check_backend_arch, check_dsl


def normalize_backend(v: str) -> str:
    return (v or "").strip().lower()


def normalize_framework(v: str) -> str:
    return (v or "").strip().lower()


def normalize_dsl(v: str) -> str:
    return (v or "").strip().lower()


def validate_basic(framework: str, backend: str, arch: str, dsl: str) -> List[str]:
    errors: List[str] = []
    if framework not in [Framework.TORCH, Framework.MINDSPORE]:
        errors.append(
            f"framework 非法: {framework}（期望: {Framework.TORCH}/{Framework.MINDSPORE}）"
        )
    if backend not in [Backend.CUDA, Backend.ASCEND]:
        errors.append(
            f"backend 非法: {backend}（期望: {Backend.CUDA}/{Backend.ASCEND}）"
        )
    if not arch:
        errors.append("arch 不能为空")
    if not dsl:
        errors.append("dsl 不能为空")
    return errors


def _sorted_keys(d: dict) -> List[str]:
    try:
        return sorted(list(d.keys()))
    except Exception:
        return list(d.keys())


def validate_target_config(
    framework: str, backend: str, arch: str, dsl: str
) -> List[str]:
    """
    校验 framework/backend/arch/dsl 的合法性与组合兼容性。
    - 对缺失参数：给出该参数在当前上下文下的可选范围
    - 对非法/不兼容参数：说明原因，并给出可选范围/推荐修正
    """
    errors: List[str] = []

    fw = normalize_framework(framework)
    be = normalize_backend(backend)
    ar = (arch or "").strip().lower()
    ds = normalize_dsl(dsl)

    if not fw:
        errors.append(
            f"framework 不能为空（可选: {'/'.join(_sorted_keys(VALID_CONFIGS))}）"
        )
        return errors
    if fw not in VALID_CONFIGS:
        errors.append(
            f"framework 非法: {fw}（可选: {'/'.join(_sorted_keys(VALID_CONFIGS))}）"
        )
        return errors

    fw_cfg = VALID_CONFIGS.get(fw, {})
    if not be:
        errors.append(
            f"backend 不能为空（{fw} 可选: {'/'.join(_sorted_keys(fw_cfg))}）"
        )
        return errors
    if be not in fw_cfg:
        errors.append(
            f"backend 不支持: {be}（在 framework={fw} 下可选: {'/'.join(_sorted_keys(fw_cfg))}）"
        )
        return errors

    be_cfg = fw_cfg.get(be, {})
    if not ar:
        errors.append(
            f"arch 不能为空（framework={fw}, backend={be} 可选: {'/'.join(_sorted_keys(be_cfg))}）"
        )
        return errors
    if ar not in be_cfg:
        errors.append(
            f"arch 不支持: {ar}（在 framework={fw}, backend={be} 下可选: {'/'.join(_sorted_keys(be_cfg))}）"
        )
        return errors

    allowed_dsls = be_cfg.get(ar, [])
    if not ds:
        errors.append(
            f"dsl 不能为空（framework={fw}, backend={be}, arch={ar} 可选: {'/'.join(allowed_dsls)}）"
        )
        return errors
    if ds not in allowed_dsls:
        errors.append(
            f"dsl 不兼容: {ds}（framework={fw}, backend={be}, arch={ar} 仅支持: {'/'.join(allowed_dsls)}）"
        )

    # 补充：沿用 core 侧的更细粒度合法性校验，错误信息更明确
    try:
        check_backend_arch(be, ar)
    except Exception as e:
        errors.append(str(e))
    try:
        check_dsl(ds)
    except Exception as e:
        errors.append(str(e))

    return errors
