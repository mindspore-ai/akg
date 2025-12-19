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


_LANG: str = _normalize_lang(os.environ.get("AIKG_LANG"))


# 说明：
# - key 使用稳定的“语义 id”，避免直接用中文做 key
# - 翻译缺失时：先回退到中文，再回退到 key 本身（便于排查）
_STRINGS: Mapping[str, Mapping[str, str]] = {
    "zh": {
        # ===== 通用 =====
        "lang.zh": "中文",
        "lang.en": "English",
        "lang.switched": "语言已切换：{lang}",
        "theme.dark": "深色",
        "theme.light": "浅色",
        # ===== TUI: bindings =====
        "tui.binding.quit": "退出",
        "tui.binding.clear_chat": "清空 Chat",
        "tui.binding.clear_task": "清空右侧",
        "tui.binding.scroll_end": "跳到底部",
        "tui.binding.focus_chat": "回看/滚动",
        "tui.binding.focus_input": "输入聚焦",
        "tui.binding.focus_trace": "Trace 聚焦",
        "tui.binding.toggle_tail": "追底开关",
        "tui.binding.save_tui": "保存屏幕文本",
        "tui.binding.copy_task_desc": "复制 task_desc",
        "tui.binding.copy_kernel_code": "复制 kernel_code",
        "tui.binding.copy_job_id": "复制 job_id",
        "tui.binding.watch_prev": "上一个并发任务",
        "tui.binding.watch_next": "下一个并发任务",
        "tui.binding.toggle_language": "切换语言",
        "tui.binding.toggle_theme": "切换主题",
        # ===== TUI: titles / placeholders =====
        "tui.title.tasks_bar": "任务（点击/左右键切换）",
        "tui.title.chat": "对话",
        "tui.title.task_info": "任务信息",
        "tui.title.workflow": "工作流",
        "tui.title.trace": "Trace（点击跳转至开头）",
        "tui.hint.trace_click": "（点击跳转至开头）",
        "tui.placeholder.input_initial": "输入下一条需求（支持多行；Ctrl+J 换行；任务运行中会禁用输入）...",
        "tui.placeholder.input_enabled_hint": "输入（支持多行；Ctrl+J 换行）...",
        "tui.placeholder.input_disabled": "任务执行中（输入已禁用）...",
        "tui.placeholder.resume_readonly": "resume 回放模式（只读，按 Ctrl+C 退出）...",
        "tui.placeholder.input_done": "任务已完成，可输入新的需求（支持多行；Ctrl+J 换行；尚未接入二次生成逻辑）...",
        "tui.tail.on": "追底:开",
        "tui.tail.off": "追底:关",
        # ===== TUI: messages =====
        "tui.msg.wait_task_info": "等待任务信息...",
        "tui.msg.wait_workflow": "等待工作流...",
        "tui.msg.workflow_cancelled": "⚠️ 工作流被取消（Ctrl+C）",
        "tui.msg.workflow_error": "工作流错误: {error}",
        "tui.msg.workflow_done": "✅ 工作流执行完成！",
        "tui.msg.press_ctrl_c_exit": "按 Ctrl+C 退出",
        "tui.msg.input_disabled_warn": "当前输入被禁用（任务执行中）",
        "tui.msg.no_copy_content": "没有可复制内容: {key}",
        "tui.msg.copy_ok": "已复制 {key} 到剪贴板（{n} chars）",
        "tui.msg.copy_failed": "复制失败: {error}",
        "tui.msg.theme_switched": "主题已切换：{theme}",
        "tui.msg.theme_switch_failed": "主题切换失败: {error}",
        "tui.msg.save_tui_ok": "已保存当前屏幕文本：{path}",
        "tui.msg.save_tui_failed": "保存失败: {error}",
        "tui.msg.save_tui_empty": "当前屏幕无可保存内容",
        "tui.msg.confirm_save_session": "检测到 Ctrl+C 退出：是否保存本次会话以便后续 resume？",
        "tui.btn.save": "保存",
        "tui.btn.discard": "不保存",
        # ===== Presenter =====
        "presenter.label.workflow": "工作流",
        "presenter.label.framework": "框架",
        "presenter.label.backend": "后端",
        "presenter.label.arch": "架构",
        "presenter.label.dsl": "DSL",
        "presenter.label.current": "当前",
        "presenter.label.graph": "图",
        "presenter.label.progress": "进度",
        "presenter.label.watch_task": "观察任务",
        "presenter.status.running": "运行中",
        "presenter.status.done": "完成",
        "presenter.wait_node_events": "（等待节点事件…）",
        "presenter.hint.watch_switch_short": "（[ / ] 或 F8/F9 切换）",
        "presenter.hint.switch_watch": (
            "切换观察并发任务：按F8。\n"
            "提示：只有进入 evolve 且出现多个 task_id 后才可切换。\n"
        ),
        "presenter.trace.title": "Trace | {tid}",
        "presenter.trace.title_no_task": "Trace | (无 task_id)",
        "presenter.evolve.watch_hint": "👁 观察: {tid}  （按 [ / ] 或 F8/F9 切换；Ctrl+E 跳到底）",
        "presenter.evolve.replay_failed": "回放失败:",
        "presenter.evolve.switch_unavailable": "（提示）并发切换不可用：尚未进入 evolve 并发或任务数不足。",
        "presenter.evolve.replay_summary": "回放: task={tid} | events={events} | node={node} | llm_end={llm_end}",
        "presenter.evolve.progress_header": "Evolve 进度 | 轮次 {round}/{max_rounds} | 完成 {done}/{total} | 通过 {ok} | 失败 {fail}",
        "presenter.evolve.status.done": "完成",
        "presenter.evolve.status.fail": "失败",
        "presenter.evolve.status.run": "运行",
        "presenter.evolve.status.queue": "排队",
        "presenter.user_request": "用户需求",
        "presenter.workflow_done": "工作流执行完成！",
        "presenter.goodbye": "再见！",
        "presenter.error": "错误",
        "presenter.replay": "回放",
        "presenter.node": "节点",
        "presenter.done": "完成",
        "presenter.duration": "耗时",
        "presenter.call_llm": "调用 LLM",
        "presenter.response_done": "响应完成",
        "presenter.response": "响应",
        "presenter.code_generated": "代码已生成",
        "presenter.generated_triton_kernel_code": "生成的 Triton Kernel 代码:",
        "presenter.verify_pass": "验证通过",
        "presenter.verify_fail": "验证失败",
        "presenter.verifier_error_log": "验证错误日志",
        "presenter.perf.baseline": "基准",
        "presenter.perf.optimized": "优化",
        "presenter.perf.speedup": "加速比",
        "presenter.perf.performance": "性能",
        "presenter.sketch_generated": "Sketch 生成完成",
        "presenter.generated_sketch": "生成的 Sketch:",
        "presenter.error_analysis_done": "错误分析完成",
        # ===== Ops prompts / messages (Textual) =====
        "ops.prompt.select_framework": "请选择框架（torch/mindspore）：",
        "ops.prompt.select_backend": "请选择后端（cuda/ascend）：",
        "ops.prompt.input_arch": "请输入架构（如 a100/ascend910b4）：",
        "ops.prompt.input_dsl": "请输入 DSL（如 triton_cuda/triton_ascend）：",
        "ops.error.config_invalid": "配置校验失败：",
        "ops.error.missing_worker": "缺少 worker",
        "ops.error.missing_worker_hint": "；请通过 --worker_url 注册匹配 worker（或先启动 worker 服务）。",
        "ops.prompt.workers_missing_server": (
            "server 未检测到匹配 worker，请输入 worker_url（逗号分隔，如 localhost:9001,1.2.3.4:9002）："
        ),
        "ops.prompt.workers_missing": (
            "未检测到可用 worker，请输入 worker_url（逗号分隔，如 localhost:9001,1.2.3.4:9002）："
        ),
        "ops.error.workers_parse_failed": "worker_url 解析失败: {error}",
        "ops.error.worker_still_missing": "注册后仍未找到匹配 worker backend={backend}, arch={arch}",
        "ops.msg.will_run_files": "将执行 {n} 个 KernelBench 文件",
        "ops.msg.done_file": "完成",
        "ops.msg.all_files_done": "全部文件执行完成",
        "ops.prompt.op_intent": "请输入算子需求：",
        "ops.msg.intent_empty_exit": "需求为空，已退出。",
        "ops.msg.extra_empty_exit": "未提供补充信息，已退出。",
        "ops.msg.user_modify": "用户补充/修改：{extra}",
        "ops.msg.user_feedback": "用户反馈（生成后迭代）：{feedback}",
        "ops.msg.supplement_round": "补充信息（第{round}轮）：{answer}",
        "ops.msg.start_generate_with_desc": "开始生成（基于已确认的 KernelBench task_desc）",
        "ops.msg.generation_done": "生成完成 op={op} | verify={verify} | time={time}",
        "ops.msg.taskinit_cannot_continue_default": "TaskInit 未能继续",
        "ops.error.taskinit_ready_but_missing_desc": (
            "错误: TaskInit 返回 ready 但未生成 task_desc（无法进入生成阶段）"
        ),
        "ops.msg.taskinit_ready_code": "TaskInit 已就绪，生成的 KernelBench 代码如下：",
        "ops.msg.taskinit_only_done": "已按 --task-init-only 完成 TaskInit，未进入生成阶段。",
        "ops.prompt.can_start_generate": "前置信息已满足，是否可以开始生成？",
        "ops.prompt.modify_intent": "请补充/修改你的需求（回到 TaskInit）：",
        "ops.prompt.satisfied": "对结果是否满意？",
        "ops.prompt.continue_new_op": "是否继续生成新算子？",
        "ops.prompt.new_op_intent": "请输入新算子需求（不要输入 y/n；直接回车退出）：",
        "ops.prompt.feedback": "请描述你不满意的点/新约束（将进入下一轮生成）：",
        "ops.msg.need_more_info_default": "请补充更多信息（例如：shape/dtype/layout/后端/性能目标）",
        "ops.msg.need_more_info": "需要补全信息：{question}",
        "ops.prompt.your_supplement": "你的补充：",
        "ops.msg.cannot_continue": "无法继续：status={status} {msg}",
        # ===== Session notifications =====
        "session.notify.done_ok": "✅ AIKG 任务完成",
        "session.notify.done_fail": "❌ AIKG 任务完成",
        "session.notify.failed": "❌ AIKG 任务失败",
        "session.notify.op": "算子",
        "session.notify.verify": "验证",
        "session.notify.verify_pass": "通过",
        "session.notify.verify_fail": "失败",
        "session.notify.time": "耗时",
        "session.notify.error": "错误",
        # ===== Summary =====
        "summary.title": "执行摘要",
        "summary.col.item": "项目",
        "summary.col.value": "内容",
        "summary.row.op_name": "算子名称",
        "summary.row.verify": "验证结果",
        "summary.row.error": "错误",
        "summary.row.perf_rounds": "性能测试轮次",
        "summary.row.round_perf": "Round {round} 性能",
        "summary.row.baseline_time": "基准耗时",
        "summary.row.optimized_time": "优化耗时",
        "summary.row.speedup": "加速比",
        "summary.row.total_time": "总耗时",
        "summary.row.log_dir": "日志目录",
        "summary.row.task_desc_path": "task_desc 路径",
        "summary.row.kernel_code_path": "kernel 代码路径",
        "summary.evolve.round_summary": "Evolve 轮次汇总",
        "summary.evolve.task_detail": "Evolve 并发任务明细",
        "summary.evolve.error_brief": "error(摘要)",
        "summary.tokens.title": "LLM Tokens & 耗时",
        "summary.tokens.col.agent": "Agent",
        "summary.tokens.col.input": "输入",
        "summary.tokens.col.output": "输出",
        "summary.tokens.col.total": "总计",
        "summary.tokens.col.time": "耗时(s)",
        "summary.tokens.total_row": "合计",
        "summary.node_timings.title": "节点耗时",
        "summary.node_timings.col.node": "节点",
        "summary.node_timings.col.duration": "耗时",
        "summary.node_timings.col.percentage": "占比",
        "summary.unit.seconds": "秒",
    },
    "en": {
        # ===== Common =====
        "lang.zh": "Chinese",
        "lang.en": "English",
        "lang.switched": "Language switched: {lang}",
        "theme.dark": "Dark",
        "theme.light": "Light",
        # ===== TUI: bindings =====
        "tui.binding.quit": "Quit",
        "tui.binding.clear_chat": "Clear chat",
        "tui.binding.clear_task": "Clear right",
        "tui.binding.scroll_end": "Scroll to end",
        "tui.binding.focus_chat": "Scrollback",
        "tui.binding.focus_input": "Focus input",
        "tui.binding.focus_trace": "Focus trace",
        "tui.binding.toggle_tail": "Toggle tail",
        "tui.binding.save_tui": "Save screen text",
        "tui.binding.copy_task_desc": "Copy task_desc",
        "tui.binding.copy_kernel_code": "Copy kernel_code",
        "tui.binding.copy_job_id": "Copy job_id",
        "tui.binding.watch_prev": "Previous task",
        "tui.binding.watch_next": "Next task",
        "tui.binding.toggle_language": "Toggle language",
        "tui.binding.toggle_theme": "Toggle theme",
        # ===== TUI: titles / placeholders =====
        "tui.title.tasks_bar": "Tasks (click / ←→ to switch)",
        "tui.title.chat": "Chat",
        "tui.title.task_info": "Task Info",
        "tui.title.workflow": "Workflow",
        "tui.title.trace": "Trace (click to jump to start)",
        "tui.hint.trace_click": "(click to jump to start)",
        "tui.placeholder.input_initial": "Type next request (multiline; Ctrl+J Newline; input disabled while running)…",
        "tui.placeholder.input_enabled_hint": "Input (multiline; Ctrl+J Newline)…",
        "tui.placeholder.input_disabled": "Running (input disabled)…",
        "tui.placeholder.resume_readonly": "Resume replay (read-only, press Ctrl+C to exit)…",
        "tui.placeholder.input_done": "Done. You can type a new request (multiline; Ctrl+J Newline; regeneration not wired yet)…",
        "tui.tail.on": "TAIL:ON",
        "tui.tail.off": "TAIL:OFF",
        # ===== TUI: messages =====
        "tui.msg.wait_task_info": "Waiting for task info...",
        "tui.msg.wait_workflow": "Waiting for workflow...",
        "tui.msg.workflow_cancelled": "⚠️ Workflow cancelled (Ctrl+C)",
        "tui.msg.workflow_error": "Workflow error: {error}",
        "tui.msg.workflow_done": "✅ Workflow finished!",
        "tui.msg.press_ctrl_c_exit": "Press Ctrl+C to exit",
        "tui.msg.input_disabled_warn": "Input is disabled (task running)",
        "tui.msg.no_copy_content": "Nothing to copy: {key}",
        "tui.msg.copy_ok": "Copied {key} to clipboard ({n} chars)",
        "tui.msg.copy_failed": "Copy failed: {error}",
        "tui.msg.theme_switched": "Theme switched: {theme}",
        "tui.msg.theme_switch_failed": "Theme switch failed: {error}",
        "tui.msg.save_tui_ok": "Saved screen text: {path}",
        "tui.msg.save_tui_failed": "Save failed: {error}",
        "tui.msg.save_tui_empty": "Nothing to save on screen",
        "tui.msg.confirm_save_session": "Ctrl+C detected: save this session for later resume?",
        "tui.btn.save": "Save",
        "tui.btn.discard": "Discard",
        # ===== Presenter =====
        "presenter.label.workflow": "Workflow",
        "presenter.label.framework": "Framework",
        "presenter.label.backend": "Backend",
        "presenter.label.arch": "Arch",
        "presenter.label.dsl": "DSL",
        "presenter.label.current": "Current",
        "presenter.label.graph": "Graph",
        "presenter.label.progress": "Progress",
        "presenter.label.watch_task": "Watch",
        "presenter.status.running": "RUNNING",
        "presenter.status.done": "DONE",
        "presenter.wait_node_events": "(waiting for node events...)",
        "presenter.hint.switch_watch": (
            "Switch watched task: press F8.\n"
            "Tip: only available in evolve mode with multiple task_id.\n"
        ),
        "presenter.trace.title": "Trace | {tid}",
        "presenter.trace.title_no_task": "Trace | (no task_id)",
        "presenter.evolve.replay_failed": "Replay failed:",
        "presenter.evolve.switch_unavailable": "(tip) Switch unavailable: not in evolve mode or not enough tasks.",
        "presenter.evolve.replay_summary": "Replay: task={tid} | events={events} | node={node} | llm_end={llm_end}",
        "presenter.evolve.progress_header": "Evolve Progress | round {round}/{max_rounds} | done {done}/{total} | ok {ok} | fail {fail}",
        "presenter.evolve.status.done": "DONE",
        "presenter.evolve.status.fail": "FAIL",
        "presenter.evolve.status.run": "RUN",
        "presenter.evolve.status.queue": "QUEUE",
        "presenter.user_request": "User request",
        "presenter.workflow_done": "Workflow finished!",
        "presenter.goodbye": "Bye!",
        "presenter.error": "Error",
        "presenter.replay": "Replay",
        "presenter.node": "Node",
        "presenter.done": "done",
        "presenter.duration": "time",
        "presenter.call_llm": "Call LLM",
        "presenter.response_done": "Response done",
        "presenter.response": "Response",
        "presenter.code_generated": "Code generated",
        "presenter.generated_triton_kernel_code": "Generated Triton kernel code:",
        "presenter.verify_pass": "Verify PASS",
        "presenter.verify_fail": "Verify FAIL",
        "presenter.verifier_error_log": "Verifier error log",
        "presenter.perf.baseline": "Baseline",
        "presenter.perf.optimized": "Optimized",
        "presenter.perf.speedup": "Speedup",
        "presenter.perf.performance": "Performance",
        "presenter.sketch_generated": "Sketch generated",
        "presenter.generated_sketch": "Generated Sketch:",
        "presenter.error_analysis_done": "Error analysis done",
        # ===== Ops prompts / messages (Textual) =====
        "ops.prompt.select_framework": "Select framework (torch/mindspore):",
        "ops.prompt.select_backend": "Select backend (cuda/ascend):",
        "ops.prompt.input_arch": "Enter arch (e.g. a100/ascend910b4):",
        "ops.prompt.input_dsl": "Enter DSL (e.g. triton_cuda/triton_ascend):",
        "ops.error.config_invalid": "Config validation failed:",
        "ops.error.missing_worker": "Missing worker",
        "ops.error.missing_worker_hint": "; please register a matching worker via --worker_url (or start worker service).",
        "ops.prompt.workers_missing_server": (
            "No matching worker detected on server. Enter worker_url (comma-separated, e.g. localhost:9001,1.2.3.4:9002):"
        ),
        "ops.prompt.workers_missing": (
            "No available worker detected. Enter worker_url (comma-separated, e.g. localhost:9001,1.2.3.4:9002):"
        ),
        "ops.error.workers_parse_failed": "Failed to parse worker_url: {error}",
        "ops.error.worker_still_missing": "Still no matching worker after register backend={backend}, arch={arch}",
        "ops.msg.will_run_files": "Will run {n} KernelBench files",
        "ops.msg.done_file": "Done",
        "ops.msg.all_files_done": "All files completed",
        "ops.prompt.op_intent": "Enter op request:",
        "ops.msg.intent_empty_exit": "Empty request, exited.",
        "ops.msg.extra_empty_exit": "No extra info, exited.",
        "ops.msg.user_modify": "User addition/modification: {extra}",
        "ops.msg.user_feedback": "User feedback (iterate): {feedback}",
        "ops.msg.supplement_round": "Supplement (round {round}): {answer}",
        "ops.msg.start_generate_with_desc": "Start generation (using confirmed KernelBench task_desc)",
        "ops.msg.generation_done": "Generated op={op} | verify={verify} | time={time}",
        "ops.msg.taskinit_cannot_continue_default": "TaskInit cannot continue",
        "ops.error.taskinit_ready_but_missing_desc": (
            "Error: TaskInit returned ready but no task_desc generated (cannot proceed)."
        ),
        "ops.msg.taskinit_ready_code": "TaskInit ready. Generated KernelBench code:",
        "ops.msg.taskinit_only_done": "Finished TaskInit with --task-init-only; generation skipped.",
        "ops.prompt.can_start_generate": "Prerequisites satisfied. Start generation?",
        "ops.prompt.modify_intent": "Add/modify your request (back to TaskInit):",
        "ops.prompt.satisfied": "Are you satisfied with the result?",
        "ops.prompt.continue_new_op": "Generate a new op?",
        "ops.prompt.new_op_intent": "Enter new op request (don't type y/n; empty to exit):",
        "ops.prompt.feedback": "Describe what's wrong / new constraints (next round):",
        "ops.msg.need_more_info_default": (
            "Please provide more info (e.g. shape/dtype/layout/backend/perf target)."
        ),
        "ops.msg.need_more_info": "Need more info: {question}",
        "ops.prompt.your_supplement": "Your input:",
        "ops.msg.cannot_continue": "Cannot continue: status={status} {msg}",
        # ===== Session notifications =====
        "session.notify.done_ok": "✅ AIKG job finished",
        "session.notify.done_fail": "❌ AIKG job finished",
        "session.notify.failed": "❌ AIKG job failed",
        "session.notify.op": "Op",
        "session.notify.verify": "Verify",
        "session.notify.verify_pass": "PASS",
        "session.notify.verify_fail": "FAIL",
        "session.notify.time": "Time",
        "session.notify.error": "Error",
        # ===== Summary =====
        "summary.title": "Summary",
        "summary.col.item": "Item",
        "summary.col.value": "Value",
        "summary.row.op_name": "Op",
        "summary.row.verify": "Verify",
        "summary.row.error": "Error",
        "summary.row.perf_rounds": "Perf rounds",
        "summary.row.round_perf": "Round {round} performance",
        "summary.row.baseline_time": "Baseline time",
        "summary.row.optimized_time": "Optimized time",
        "summary.row.speedup": "Speedup",
        "summary.row.total_time": "Total time",
        "summary.row.log_dir": "Log dir",
        "summary.row.task_desc_path": "task_desc path",
        "summary.row.kernel_code_path": "kernel code path",
        "summary.evolve.round_summary": "Evolve round summary",
        "summary.evolve.task_detail": "Evolve tasks",
        "summary.evolve.error_brief": "error (brief)",
        "summary.tokens.title": "LLM Tokens & Time",
        "summary.tokens.col.agent": "Agent",
        "summary.tokens.col.input": "Input",
        "summary.tokens.col.output": "Output",
        "summary.tokens.col.total": "Total",
        "summary.tokens.col.time": "Time (s)",
        "summary.tokens.total_row": "Total",
        "summary.node_timings.title": "Node timings",
        "summary.node_timings.col.node": "Node",
        "summary.node_timings.col.duration": "Time",
        "summary.node_timings.col.percentage": "Share",
        "summary.unit.seconds": "s",
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
    os.environ["AIKG_LANG"] = _LANG
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
