"""compile_gate：sanity launch 失败的 config 必须被剔除并记原因。"""

from __future__ import annotations

from akg_agents.op.dynamic_tune.config import Config
from akg_agents.op.dynamic_tune.measure.compile_gate import (
    CompileGate,
    CompileGateOutcome,
    CompileGateRejection,
)


def _config(name: str, value: int):
    return Config({"BLOCK_M": value}, config_id=name)


def test_compile_gate_keeps_all_when_no_failure():
    configs = [_config("a", 64), _config("b", 128)]

    def launch_factory(shape, config):
        return lambda: None

    outcome = CompileGate().filter(
        configs=configs, sanity_shape=(16,), launch_factory=launch_factory
    )
    assert isinstance(outcome, CompileGateOutcome)
    assert outcome.kept_config_ids() == ("a", "b")
    assert outcome.rejections == ()


def test_compile_gate_rejects_failing_config_and_records_reason():
    bad = _config("bad", 9999)
    good_one = _config("good_one", 64)
    good_two = _config("good_two", 128)
    configs = [good_one, bad, good_two]

    def launch_factory(shape, config):
        if config.config_id == "bad":
            def _launch():
                raise RuntimeError("local memory exceeds limit")

            return _launch
        return lambda: None

    rejections: list[CompileGateRejection] = []

    outcome = CompileGate(on_reject=rejections.append).filter(
        configs=configs, sanity_shape=(16,), launch_factory=launch_factory
    )
    assert outcome.kept_config_ids() == ("good_one", "good_two")
    assert len(outcome.rejections) == 1
    assert outcome.rejections[0].config.config_id == "bad"
    assert "local memory" in outcome.rejections[0].reason
    assert rejections == list(outcome.rejections)


def test_compile_gate_handles_construction_failure():
    """launch_factory 自身抛异常也应被捕获到对应 config 上。"""

    bad = _config("bad", 0)
    good = _config("good", 64)

    def launch_factory(shape, config):
        if config.config_id == "bad":
            raise ValueError("cannot allocate buffer")
        return lambda: None

    outcome = CompileGate().filter(
        configs=[bad, good], sanity_shape=(16,), launch_factory=launch_factory
    )
    assert outcome.kept_config_ids() == ("good",)
    assert outcome.rejections[0].config.config_id == "bad"
    assert "cannot allocate buffer" in outcome.rejections[0].reason
