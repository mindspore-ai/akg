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

import os
from typing import Any, Mapping


def _normalize_lang(value: str | None) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return "zh"
    if raw in {"zh", "cn", "zh-cn", "zh_cn", "zh-hans", "zh_hans", "chinese", "中文"}:
        return "zh"
    if raw in {"en", "en-us", "en_us", "english"}:
        return "en"
    if raw.startswith("zh"):
        return "zh"
    if raw.startswith("en"):
        return "en"
    return "zh"


_LANG: str = _normalize_lang(os.environ.get("AKG_AGENTS_LANG"))


# 说明：
# - key 使用稳定的“语义 id”，避免直接用中文做 key
# - 翻译缺失时：先回退到中文，再回退到 key 本身（便于排查）
_STRINGS: Mapping[str, Mapping[str, str]] = {
    "zh": {
        # ===== 通用 =====
        "lang.zh": "中文",
        "lang.en": "English",
        # ===== Presenter =====
        "presenter.user_request": "👤 ",
        "presenter.goodbye": "再见！",
        "presenter.error": "错误",
        "presenter.node": "节点",
        "presenter.done": "完成",
        "presenter.duration": "耗时",
        "presenter.call_llm": "调用 LLM",
        "presenter.response_done": "响应完成",
        "presenter.response": "响应",
        "presenter.code_generated": "代码已生成",
        "presenter.verify_pass": "验证通过",
        "presenter.verify_fail": "验证失败",
        "presenter.verifier_error_log": "验证错误日志",
        "presenter.perf.baseline": "基准",
        "presenter.perf.optimized": "优化",
        "presenter.perf.speedup": "加速比",
        "presenter.perf.performance": "性能",
        "presenter.sketch_generated": "Sketch 生成完成",
        "presenter.error_analysis_done": "错误分析完成",
        # ===== Ops prompts / messages (Textual) =====
        "ops.error.config_invalid": "配置校验失败：",
        "ops.msg.intent_empty_exit": "需求为空，已退出。",
    },
    "en": {
        # ===== Common =====
        "lang.zh": "Chinese",
        "lang.en": "English",
        # ===== Presenter =====
        "presenter.user_request": "User request",
        "presenter.goodbye": "Bye!",
        "presenter.error": "Error",
        "presenter.node": "Node",
        "presenter.done": "done",
        "presenter.duration": "time",
        "presenter.call_llm": "Call LLM",
        "presenter.response_done": "Response done",
        "presenter.response": "Response",
        "presenter.code_generated": "Code generated",
        "presenter.verify_pass": "Verify PASS",
        "presenter.verify_fail": "Verify FAIL",
        "presenter.verifier_error_log": "Verifier error log",
        "presenter.perf.baseline": "Baseline",
        "presenter.perf.optimized": "Optimized",
        "presenter.perf.speedup": "Speedup",
        "presenter.perf.performance": "Performance",
        "presenter.sketch_generated": "Sketch generated",
        "presenter.error_analysis_done": "Error analysis done",
        # ===== Ops prompts / messages (Textual) =====
        "ops.error.config_invalid": "Config validation failed:",
        "ops.msg.intent_empty_exit": "Empty request, exited.",
    },
}


def get_lang() -> str:
    return _LANG


def set_lang(lang: str) -> str:
    global _LANG
    _LANG = _normalize_lang(lang)
    return _LANG


def toggle_lang() -> str:
    global _LANG
    _LANG = "en" if _LANG == "zh" else "zh"
    os.environ["AKG_AGENTS_LANG"] = _LANG
    return _LANG


def t(key: str, **kwargs: Any) -> str:
    lang = _LANG
    text = _STRINGS.get(lang, {}).get(key)
    if text is None:
        text = _STRINGS.get("zh", {}).get(key) or key
    if not kwargs:
        return text
    try:
        return text.format(**kwargs)
    except (KeyError, IndexError, ValueError):
        return text


def lang_display(lang: str | None = None) -> str:
    l = _normalize_lang(lang) if lang is not None else _LANG
    return t("lang.zh") if l == "zh" else t("lang.en")
