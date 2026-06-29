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

from types import SimpleNamespace

from akg_agents.core.llm import thinking_chat_model
from akg_agents.core_v2.config import AKGSettings, EmbeddingConfig, ModelConfig
from akg_agents.core_v2.config.settings import _load_env_config
from akg_agents.core_v2.llm import factory
from akg_agents.core_v2.llm.providers import anthropic_provider, openai_provider
from akg_agents.core_v2.llm.providers.embedding_provider import OpenAICompatibleEmbeddings


class _DummyOpenAIClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.base_url = kwargs.get("base_url")


class _DummyAnthropicClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _DummyHttpClient:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def test_openai_provider_enables_ssl_verification_by_default(monkeypatch):
    monkeypatch.setattr(openai_provider, "AsyncOpenAI", _DummyOpenAIClient)
    monkeypatch.setattr(openai_provider.httpx, "AsyncClient", _DummyHttpClient)

    provider = openai_provider.LLMProvider(model_name="gpt-4", api_key="sk-test")

    assert provider.verify_ssl is True
    assert provider.client.kwargs["http_client"].kwargs["verify"] is True


def test_openai_provider_allows_explicit_ssl_verification_disable(monkeypatch):
    monkeypatch.setattr(openai_provider, "AsyncOpenAI", _DummyOpenAIClient)
    monkeypatch.setattr(openai_provider.httpx, "AsyncClient", _DummyHttpClient)

    provider = openai_provider.LLMProvider(
        model_name="gpt-4",
        api_key="sk-test",
        verify_ssl=False,
    )

    assert provider.verify_ssl is False
    assert provider.client.kwargs["http_client"].kwargs["verify"] is False


def test_anthropic_provider_enables_ssl_verification_by_default(monkeypatch):
    monkeypatch.setattr(anthropic_provider, "AsyncAnthropic", _DummyAnthropicClient)
    monkeypatch.setattr(anthropic_provider.httpx, "AsyncClient", _DummyHttpClient)

    provider = anthropic_provider.AnthropicProvider(model_name="claude", api_key="sk-test")

    assert provider.verify_ssl is True
    assert provider.client.kwargs["http_client"].kwargs["verify"] is True


def test_embedding_provider_enables_ssl_verification_by_default():
    embeddings = OpenAICompatibleEmbeddings(
        api_url="https://api.example.com/v1/embeddings",
        model_name="embedding-model",
    )

    assert embeddings.verify_ssl is True


def test_embedding_factory_enables_ssl_verification_by_default(monkeypatch):
    captured = {}

    class DummyEmbeddings:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    settings = SimpleNamespace(
        embedding=SimpleNamespace(
            base_url="https://api.example.com/v1",
            api_key="sk-test",
            model_name="embedding-model",
            timeout=60,
            verify_ssl=True,
        )
    )

    monkeypatch.setattr(factory, "get_settings", lambda: settings)
    monkeypatch.setattr(factory, "OpenAICompatibleEmbeddings", DummyEmbeddings)

    factory.create_embedding_model()

    assert captured["verify_ssl"] is True


def test_global_verify_ssl_env_disables_all_model_levels(monkeypatch):
    monkeypatch.setenv("AKG_AGENTS_VERIFY_SSL", "false")

    settings = AKGSettings()
    settings.models["complex"] = ModelConfig(
        base_url="https://api.deepseek.com/v1",
        api_key="sk-test",
        model_name="deepseek-reasoner",
    )
    settings.models["standard"] = ModelConfig(
        base_url="https://api.deepseek.com/v1",
        api_key="sk-test",
        model_name="deepseek-chat",
    )
    settings.models["fast"] = ModelConfig(
        base_url="https://api.deepseek.com/v1",
        api_key="sk-test",
        model_name="deepseek-chat",
    )

    settings = _load_env_config(settings)

    assert settings.models["complex"].verify_ssl is False
    assert settings.models["standard"].verify_ssl is False
    assert settings.models["fast"].verify_ssl is False
    assert settings.models["standard"].base_url == "https://api.deepseek.com/v1"
    assert settings.models["standard"].model_name == "deepseek-chat"


def test_model_level_verify_ssl_env_overrides_single_model_env(monkeypatch):
    monkeypatch.setenv("AKG_AGENTS_VERIFY_SSL", "false")
    monkeypatch.setenv("AKG_AGENTS_STANDARD_VERIFY_SSL", "true")

    settings = AKGSettings()
    settings.models["complex"] = ModelConfig(
        base_url="https://api.deepseek.com/v1",
        api_key="sk-test",
        model_name="deepseek-reasoner",
    )
    settings.models["standard"] = ModelConfig(
        base_url="https://api.deepseek.com/v1",
        api_key="sk-test",
        model_name="deepseek-chat",
    )
    settings.models["fast"] = ModelConfig(
        base_url="https://api.deepseek.com/v1",
        api_key="sk-test",
        model_name="deepseek-chat",
    )

    settings = _load_env_config(settings)

    assert settings.models["complex"].verify_ssl is False
    assert settings.models["standard"].verify_ssl is True
    assert settings.models["fast"].verify_ssl is False
    assert settings.models["standard"].base_url == "https://api.deepseek.com/v1"
    assert settings.models["standard"].model_name == "deepseek-chat"


def test_model_level_env_inherits_global_verify_ssl_env(monkeypatch):
    monkeypatch.setenv("AKG_AGENTS_VERIFY_SSL", "false")
    monkeypatch.setenv("AKG_AGENTS_STANDARD_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("AKG_AGENTS_STANDARD_API_KEY", "sk-test")
    monkeypatch.setenv("AKG_AGENTS_STANDARD_MODEL_NAME", "deepseek-chat")

    settings = _load_env_config(AKGSettings())

    assert settings.models["standard"].base_url == "https://api.deepseek.com/v1"
    assert settings.models["standard"].model_name == "deepseek-chat"
    assert settings.models["standard"].verify_ssl is False


def test_embedding_verify_ssl_env_works_without_other_embedding_env(monkeypatch):
    monkeypatch.setenv("AKG_AGENTS_EMBEDDING_VERIFY_SSL", "false")

    settings = AKGSettings(
        embedding=EmbeddingConfig(
            base_url="https://api.example.com/v1",
            api_key="sk-test",
            model_name="embedding-model",
        )
    )
    settings = _load_env_config(settings)

    assert settings.embedding.verify_ssl is False
    assert settings.embedding.base_url == "https://api.example.com/v1"
    assert settings.embedding.model_name == "embedding-model"


def test_model_config_merge_preserves_lower_verify_ssl_when_higher_omits_it():
    lower = AKGSettings(
        models={
            "standard": ModelConfig(
                base_url="https://api.example.com/v1",
                api_key="sk-test",
                model_name="lower-model",
                verify_ssl=False,
            )
        }
    )
    higher = AKGSettings.from_dict(
        {"models": {"standard": {"model_name": "higher-model"}}},
        use_defaults=False,
    )

    merged = lower.merge_with(higher)

    assert merged.models["standard"].model_name == "higher-model"
    assert merged.models["standard"].verify_ssl is False


def test_embedding_config_merge_preserves_lower_verify_ssl_when_higher_omits_it():
    lower = AKGSettings(
        embedding=EmbeddingConfig(
            base_url="https://api.example.com/v1",
            api_key="sk-test",
            model_name="lower-embedding",
            verify_ssl=False,
        )
    )
    higher = AKGSettings.from_dict(
        {"embedding": {"model_name": "higher-embedding"}},
        use_defaults=False,
    )

    merged = lower.merge_with(higher)

    assert merged.embedding.model_name == "higher-embedding"
    assert merged.embedding.verify_ssl is False

    higher_without_embedding = AKGSettings.from_dict(
        {"default_model": "standard"},
        use_defaults=False,
    )
    merged_without_embedding = lower.merge_with(higher_without_embedding)

    assert merged_without_embedding.embedding.model_name == "lower-embedding"
    assert merged_without_embedding.embedding.verify_ssl is False


def test_thinking_chat_model_enables_ssl_verification_by_default(monkeypatch):
    sync_clients = []
    async_clients = []

    def fake_client(*args, **kwargs):
        client = _DummyHttpClient(*args, **kwargs)
        sync_clients.append(client)
        return client

    def fake_async_client(*args, **kwargs):
        client = _DummyHttpClient(*args, **kwargs)
        async_clients.append(client)
        return client

    def fake_chat_deepseek_init(self, **kwargs):
        object.__setattr__(self, "init_kwargs", kwargs)

    monkeypatch.setattr(thinking_chat_model.httpx, "Client", fake_client)
    monkeypatch.setattr(thinking_chat_model.httpx, "AsyncClient", fake_async_client)
    monkeypatch.setattr(thinking_chat_model.ChatDeepSeek, "__init__", fake_chat_deepseek_init)

    thinking_chat_model.ThinkingAwareChatDeepSeek(model="deepseek-chat", api_key="sk-test")

    assert sync_clients[0].kwargs["verify"] is True
    assert async_clients[0].kwargs["verify"] is True
