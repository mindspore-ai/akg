from __future__ import annotations

from akg_agents.op.dynamic_tune.cases.contract import _ModelNewContractValidator


def test_validate_contract_accepts_config_param_flow() -> None:
    code = """
from akg_agents.op.dynamic_tune import load_deployed_selector

class ModelNew:
    def _select_config(self, shape_key):
        selector = load_deployed_selector()
        return selector.select_config(shape_key)

    def forward(self, X, config=None):
        if config is None:
            config = self._select_config((int(X.shape[0]),))
        return config.param("BLOCK_M")
"""
    assert _ModelNewContractValidator.validate_code(code) == ()


def test_validate_contract_rejects_missing_required_methods() -> None:
    code = """
class ModelNew:
    def forward(self, X):
        return X
"""
    errors = _ModelNewContractValidator.validate_code(code)
    assert "缺少 `ModelNew._select_config`" in errors
    assert "`ModelNew.forward` 必须声明 `config=None`" in errors


def test_validate_contract_rejects_wrong_selector_import() -> None:
    code = """
from akg_agents.op.dynamic_tune.cases.deploy_utils import load_deployed_selector

class ModelNew:
    def _select_config(self, shape_key):
        selector = load_deployed_selector()
        return selector.select_config(shape_key)

    def forward(self, X, config=None):
        if config is None:
            config = self._select_config((int(X.shape[0]),))
        return config.param("BLOCK_M")
"""
    errors = _ModelNewContractValidator.validate_code(code)
    assert "`load_deployed_selector` 必须从 `akg_agents.op.dynamic_tune` 导入" in errors
