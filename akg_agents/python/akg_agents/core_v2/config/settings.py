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

"""
配置管理系统 - 支持多层级配置

优先级（从高到低）：
1. 环境变量（AKG_AGENTS_* 或 AIKG_* 前缀，兼容旧版）
2. Local: .akg/settings.local.json（仅本人此项目，gitignored）
3. Project: .akg/settings.json（此项目所有协作者，提交到 git）
4. User: ~/.akg/settings.json（跨所有项目）
5. 默认值

参考：https://code.claude.com/docs/zh-CN/settings
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ========================= 环境变量工具函数 =========================

def get_akg_env_var(var_name: str, default: Optional[str] = None) -> Optional[str]:
    """
    获取 AKG 环境变量，同时支持 AIKG_* 和 AKG_AGENTS_* 两种前缀
    
    优先级：
    1. AKG_AGENTS_{var_name} (新版)
    2. AIKG_{var_name} (旧版，兼容性)
    
    Args:
        var_name: 环境变量名（不含前缀），如 "BASE_URL", "API_KEY"
        default: 默认值
        
    Returns:
        环境变量值，如果都不存在则返回 default
        
    Examples:
        get_akg_env_var("BASE_URL") -> 查找 AKG_AGENTS_BASE_URL 或 AIKG_BASE_URL
        get_akg_env_var("MODEL_NAME", "gpt-4") -> 查找变量，不存在返回 "gpt-4"
    """
    # 优先使用新版 AKG_AGENTS_*
    new_key = f"AKG_AGENTS_{var_name}"
    value = os.getenv(new_key)
    if value is not None:
        return value
    
    # 降级到旧版 AIKG_*
    old_key = f"AIKG_{var_name}"
    value = os.getenv(old_key)
    if value is not None:
        logger.debug(f"Using legacy env var {old_key} (consider migrating to {new_key})")
        return value
    
    return default


def _detect_env_prefix(*var_names: str) -> str:
    """
    检测环境变量实际使用的前缀（AKG_AGENTS 优先于 AIKG）
    
    Args:
        *var_names: 要检测的变量名（不含前缀），如 "BASE_URL", "API_KEY"
    
    Returns:
        "AKG_AGENTS" 或 "AIKG"
    """
    for var_name in var_names:
        if os.getenv(f"AKG_AGENTS_{var_name}") is not None:
            return "AKG_AGENTS"
    return "AIKG"


# ========================= 数据类 =========================

@dataclass
class EmbeddingConfig:
    """Embedding 模型配置"""
    base_url: str = ""
    api_key: str = ""
    model_name: str = ""
    timeout: int = 60
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmbeddingConfig":
        """从字典创建"""
        return cls(
            base_url=data.get("base_url", ""),
            api_key=data.get("api_key", ""),
            model_name=data.get("model_name", ""),
            timeout=data.get("timeout", 60),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "base_url": self.base_url,
            "api_key": self.api_key,
            "model_name": self.model_name,
            "timeout": self.timeout,
        }
    
    def is_configured(self) -> bool:
        """检查是否已配置"""
        return bool(self.base_url and self.api_key and self.model_name)
    
    def merge_with(self, other: "EmbeddingConfig") -> "EmbeddingConfig":
        """合并配置（other 优先级更高）"""
        return EmbeddingConfig(
            base_url=other.base_url if other.base_url else self.base_url,
            api_key=other.api_key if other.api_key else self.api_key,
            model_name=other.model_name if other.model_name else self.model_name,
            timeout=other.timeout if other.timeout != 60 else self.timeout,
        )


@dataclass
class ModelConfig:
    """单个模型配置"""
    base_url: str
    api_key: str
    model_name: str
    temperature: float = 0.2
    max_tokens: int = 8192
    top_p: float = 0.9
    frequency_penalty: Optional[float] = None  # 可选，来自配置文件
    presence_penalty: Optional[float] = None   # 可选，来自配置文件
    timeout: int = 300
    extra_body: Dict[str, Any] = field(default_factory=dict)  # 透传到 API 请求的额外参数（如 thinking/reasoning）
    extra: Dict[str, Any] = field(default_factory=dict)       # 其他扩展字段（不透传到 API）
    provider_type: str = "openai"  # Provider 类型: "openai" (OpenAI 兼容协议) 或 "anthropic" (Anthropic 协议)
    
    @property
    def thinking_enabled(self) -> bool:
        """向后兼容属性：检查 extra_body 中是否包含 thinking 相关配置"""
        return bool(self.extra_body)
    
    @classmethod
    def _parse_extra_body(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        解析 extra_body 配置，支持新旧格式（向后兼容）
        
        新格式（推荐）: "extra_body": {"thinking": {"type": "enabled"}}
        旧格式（兼容）: "thinking_enabled": true  → 转为 {"thinking": {"type": "enabled"}}
        """
        # 新格式优先：直接使用 extra_body
        if "extra_body" in data:
            return data["extra_body"]
        # 旧格式向后兼容：thinking_enabled: true
        if data.get("thinking_enabled"):
            return {"thinking": {"type": "enabled"}}
        return {}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], use_defaults: bool = True) -> "ModelConfig":
        """
        从字典创建
        
        Args:
            data: 配置字典
            use_defaults: 是否使用默认值，False 时缺失字段为空/None（用于合并）
        """
        extra_body = cls._parse_extra_body(data)
        
        if use_defaults:
            return cls(
                base_url=data.get("base_url", "https://api.openai.com/v1"),
                api_key=data.get("api_key", ""),
                model_name=data.get("model_name", "gpt-4"),
                temperature=data.get("temperature", 0.2),
                max_tokens=data.get("max_tokens", 8192),
                top_p=data.get("top_p", 0.9),
                frequency_penalty=data.get("frequency_penalty"),
                presence_penalty=data.get("presence_penalty"),
                timeout=data.get("timeout", 300),
                extra_body=extra_body,
                extra=data.get("extra", {}),
                provider_type=data.get("provider_type", "openai"),
            )
        else:
            return cls(
                base_url=data.get("base_url", ""),
                api_key=data.get("api_key", ""),
                model_name=data.get("model_name", ""),
                temperature=data.get("temperature", 0.2),
                max_tokens=data.get("max_tokens", 8192),
                top_p=data.get("top_p", 0.9),
                frequency_penalty=data.get("frequency_penalty"),
                presence_penalty=data.get("presence_penalty"),
                timeout=data.get("timeout", 300),
                extra_body=extra_body,
                extra=data.get("extra", {}),
                provider_type=data.get("provider_type", "openai"),
            )
    
    @classmethod
    def from_env(cls, prefix: str = "", thinking_enabled: bool = False) -> "ModelConfig":
        """
        从环境变量构建 ModelConfig
        
        Args:
            prefix: 环境变量前缀，如 "" (单模型) 或 "COMPLEX_" (多模型)
            thinking_enabled: 是否启用 thinking 模式（环境变量兼容，生成默认 extra_body）
        """
        env_temp = get_akg_env_var(f"{prefix}TEMPERATURE")
        env_max_tokens = get_akg_env_var(f"{prefix}MAX_TOKENS")
        env_timeout = get_akg_env_var(f"{prefix}TIMEOUT")
        
        # 环境变量模式下，thinking_enabled 生成默认的 extra_body
        extra_body: Dict[str, Any] = {}
        if thinking_enabled:
            extra_body = {"thinking": {"type": "enabled"}}
        
        return cls(
            base_url=get_akg_env_var(f"{prefix}BASE_URL", "https://api.openai.com/v1"),
            api_key=get_akg_env_var(f"{prefix}API_KEY", ""),
            model_name=get_akg_env_var(f"{prefix}MODEL_NAME", "gpt-4"),
            temperature=float(env_temp) if env_temp else 0.2,
            max_tokens=int(env_max_tokens) if env_max_tokens else 8192,
            timeout=int(env_timeout) if env_timeout else 300,
            extra_body=extra_body,
            provider_type=get_akg_env_var(f"{prefix}PROVIDER_TYPE", "openai"),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            "base_url": self.base_url,
            "api_key": self.api_key,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "timeout": self.timeout,
            "extra": self.extra,
            "provider_type": self.provider_type,
        }
        if self.extra_body:
            result["extra_body"] = self.extra_body
        if self.frequency_penalty is not None:
            result["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            result["presence_penalty"] = self.presence_penalty
        return result
    
    def merge_with(self, other: "ModelConfig") -> "ModelConfig":
        """合并配置（other 优先级更高，只覆盖显式设置的值）"""
        return ModelConfig(
            base_url=other.base_url if other.base_url else self.base_url,
            api_key=other.api_key if other.api_key else self.api_key,
            model_name=other.model_name if other.model_name else self.model_name,
            temperature=other.temperature if other.temperature != 0.2 else self.temperature,
            max_tokens=other.max_tokens if other.max_tokens != 8192 else self.max_tokens,
            top_p=other.top_p if other.top_p != 0.9 else self.top_p,
            frequency_penalty=other.frequency_penalty if other.frequency_penalty is not None else self.frequency_penalty,
            presence_penalty=other.presence_penalty if other.presence_penalty is not None else self.presence_penalty,
            timeout=other.timeout if other.timeout != 300 else self.timeout,
            extra_body={**self.extra_body, **other.extra_body},
            extra={**self.extra, **other.extra},
            provider_type=other.provider_type if other.provider_type != "openai" else self.provider_type,
        )


@dataclass
class AKGSettings:
    """AIKG 配置类 - 支持多模型配置（complex / standard / fast）"""
    
    # 多模型配置
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    
    # Embedding 模型配置
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    
    # 默认使用的模型级别
    default_model: str = "standard"
    
    # 模型上下文窗口大小（用于智能历史压缩判断）
    # 默认 128k tokens；仅当 prompt 总 token 数超过 80% 时才触发历史压缩
    context_window: int = 128000
    
    # 流式输出
    stream_output: bool = False
    
    # 扩展字段
    extra: Dict[str, Any] = field(default_factory=dict)
    
    # 来源追踪（不参与序列化和反序列化）
    _model_sources: Dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _embedding_source: str = field(default="default", init=False, repr=False)
    
    def get_model_source(self, level: str) -> str:
        """获取指定模型级别的配置来源"""
        return self._model_sources.get(level, "default")
    
    def get_embedding_source(self) -> str:
        """获取 embedding 配置的来源"""
        return self._embedding_source
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于保存，不含来源追踪字段）"""
        result = {
            "models": {k: v.to_dict() for k, v in self.models.items()},
            "default_model": self.default_model,
            "context_window": self.context_window,
            "stream_output": self.stream_output,
            "extra": self.extra,
        }
        if self.embedding.is_configured():
            result["embedding"] = self.embedding.to_dict()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], use_defaults: bool = True) -> "AKGSettings":
        """
        从字典创建
        
        Args:
            data: 配置字典
            use_defaults: 是否使用默认值，False 时用于合并场景
        """
        models = {}
        if "models" in data:
            for level, model_data in data["models"].items():
                models[level] = ModelConfig.from_dict(model_data, use_defaults=use_defaults)
        
        # 解析 embedding 配置
        embedding = EmbeddingConfig()
        if "embedding" in data:
            embedding = EmbeddingConfig.from_dict(data["embedding"])
        
        return cls(
            models=models,
            embedding=embedding,
            default_model=data.get("default_model", "standard" if use_defaults else ""),
            context_window=data.get("context_window", 128000 if use_defaults else 128000),
            stream_output=data.get("stream_output", False),
            extra=data.get("extra", {}),
        )
    
    def merge_with(self, other: "AKGSettings") -> "AKGSettings":
        """合并配置（other 优先级更高，只覆盖显式设置的值）"""
        merged_models = dict(self.models)
        
        for level, model_config in other.models.items():
            if level in merged_models:
                merged_models[level] = merged_models[level].merge_with(model_config)
            else:
                merged_models[level] = model_config
        
        # 合并 embedding 配置
        merged_embedding = self.embedding.merge_with(other.embedding)
        
        result = AKGSettings(
            models=merged_models,
            embedding=merged_embedding,
            # 只有显式设置了才覆盖（空字符串表示未设置）
            default_model=other.default_model if other.default_model else self.default_model,
            context_window=other.context_window if other.context_window != 128000 else self.context_window,
            stream_output=other.stream_output or self.stream_output,
            extra={**self.extra, **other.extra},
        )
        
        # 继承并更新来源追踪
        result._model_sources = dict(self._model_sources)
        result._model_sources.update(other._model_sources)
        result._embedding_source = (
            other._embedding_source if other._embedding_source != "default" else self._embedding_source
        )
        
        return result


# ========================= 路径解析 =========================

def get_settings_path() -> Path:
    """获取用户级配置路径：~/.akg/settings.json"""
    home = Path.home()
    akg_dir = home / ".akg"
    akg_dir.mkdir(exist_ok=True)
    return akg_dir / "settings.json"


def find_project_root() -> Optional[Path]:
    """
    查找项目根目录（包含 .akg/ 或 .git/ 的目录）
    
    从当前目录向上查找，直到找到包含 .akg/ 或 .git/ 的目录
    """
    current = Path.cwd()
    
    for parent in [current] + list(current.parents):
        # 优先检查 .akg 目录
        if (parent / ".akg").is_dir():
            return parent
        # 其次检查 .git 目录（项目根目录标志）
        if (parent / ".git").is_dir():
            return parent
    
    return None


def get_project_settings_path() -> Optional[Path]:
    """获取项目级配置路径：.akg/settings.json"""
    project_root = find_project_root()
    if project_root:
        return project_root / ".akg" / "settings.json"
    return None


def get_local_settings_path() -> Optional[Path]:
    """获取本地配置路径：.akg/settings.local.json"""
    project_root = find_project_root()
    if project_root:
        return project_root / ".akg" / "settings.local.json"
    return None


# ========================= 配置加载 =========================

def load_json_file(path: Path) -> Optional[Dict[str, Any]]:
    """从 JSON 文件加载配置"""
    if not path or not path.exists():
        return None
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.debug(f"Loaded settings from {path}")
        return data
    except Exception as e:
        logger.warning(f"Failed to load settings from {path}: {e}")
        return None


def _get_config_layers() -> List[Tuple[str, Optional[Path], bool]]:
    """
    返回配置文件层列表，按优先级从低到高排列
    
    Returns:
        [(source_label, path, use_defaults), ...]
    """
    user_path = get_settings_path()
    project_path = get_project_settings_path()
    local_path = get_local_settings_path()
    
    return [
        (f"user: {user_path}", user_path, True),
        (f"project: {project_path}", project_path, False),
        (f"local: {local_path}", local_path, False),
    ]


def _parse_thinking_enabled(env_value: Optional[str]) -> bool:
    """解析 thinking 环境变量"""
    if not env_value:
        return False
    return env_value.lower() in ("enabled", "true", "1", "yes", "on")


def _load_env_config(settings: AKGSettings) -> AKGSettings:
    """
    从环境变量加载配置（最高优先级）
    
    支持两种方式：
    1. 单模型配置（兼容旧版）：AKG_AGENTS_BASE_URL/AIKG_BASE_URL, AKG_AGENTS_API_KEY/AIKG_API_KEY 等
       -> 自动覆盖所有级别（complex/standard/fast）
    
    2. 多模型配置：AKG_AGENTS_COMPLEX_BASE_URL/AIKG_COMPLEX_BASE_URL, AKG_AGENTS_STANDARD_API_KEY/AIKG_STANDARD_API_KEY 等
       -> 分别设置 complex/standard/fast
    """
    thinking_enabled = _parse_thinking_enabled(get_akg_env_var("MODEL_ENABLE_THINK"))
    
    # 方式 1：单模型配置（兼容旧版，自动覆盖所有级别：complex/standard/fast）
    if get_akg_env_var("BASE_URL") or get_akg_env_var("API_KEY") or get_akg_env_var("MODEL_NAME"):
        single_config = ModelConfig.from_env("", thinking_enabled)
        prefix = _detect_env_prefix("BASE_URL", "API_KEY", "MODEL_NAME")
        source = f"env: {prefix}_*"
        for level in ["complex", "standard", "fast"]:
            settings.models[level] = single_config
            settings._model_sources[level] = source
        logger.debug(f"Loaded single model config from env ({source}), applied to all levels")
    
    # 方式 2：多模型配置（优先级高于单模型，会覆盖方式 1 的设置）
    for level in ["complex", "standard", "fast"]:
        level_upper = level.upper()
        if get_akg_env_var(f"{level_upper}_BASE_URL") or get_akg_env_var(f"{level_upper}_API_KEY"):
            # 多模型模式也继承全局 thinking_enabled 设置
            settings.models[level] = ModelConfig.from_env(f"{level_upper}_", thinking_enabled)
            prefix = _detect_env_prefix(f"{level_upper}_BASE_URL", f"{level_upper}_API_KEY")
            settings._model_sources[level] = f"env: {prefix}_{level_upper}_*"
            logger.debug(f"Loaded '{level}' model config from env")
    
    # AKG_AGENTS_DEFAULT_MODEL / AIKG_DEFAULT_MODEL
    if default_model := get_akg_env_var("DEFAULT_MODEL"):
        settings.default_model = default_model
        logger.debug(f"Loaded default_model from env: {default_model}")
    
    # AKG_AGENTS_STREAM_OUTPUT / AIKG_STREAM_OUTPUT
    if stream := get_akg_env_var("STREAM_OUTPUT"):
        settings.stream_output = stream.lower() in ("true", "1", "yes", "on")
        logger.debug(f"Loaded stream_output from env: {settings.stream_output}")
    
    # Embedding 配置（AKG_AGENTS_EMBEDDING_* / AIKG_EMBEDDING_*）
    env_emb_base_url = get_akg_env_var("EMBEDDING_BASE_URL")
    env_emb_model_name = get_akg_env_var("EMBEDDING_MODEL_NAME")
    env_emb_api_key = get_akg_env_var("EMBEDDING_API_KEY")
    
    if env_emb_base_url or env_emb_model_name or env_emb_api_key:
        settings.embedding = EmbeddingConfig(
            base_url=env_emb_base_url or "",
            api_key=env_emb_api_key or "",
            model_name=env_emb_model_name or "",
            timeout=int(get_akg_env_var("EMBEDDING_TIMEOUT", "60")),
        )
        prefix = _detect_env_prefix("EMBEDDING_BASE_URL", "EMBEDDING_API_KEY")
        settings._embedding_source = f"env: {prefix}_EMBEDDING_*"
        logger.debug(f"Loaded embedding config from env: model={env_emb_model_name}")
    
    return settings


def get_settings() -> AKGSettings:
    """
    获取配置（按优先级加载并合并）
    
    优先级（从低到高）：
    1. 默认值
    2. User: ~/.akg/settings.json
    3. Project: .akg/settings.json
    4. Local: .akg/settings.local.json
    5. 环境变量（AKG_AGENTS_* 或 AIKG_*，兼容旧版）
    
    高优先级配置会覆盖低优先级配置（只覆盖显式设置的字段）
    
    Returns:
        AKGSettings 配置对象
    """
    settings = AKGSettings()
    
    # 遍历配置文件层（从低到高优先级）
    for source_label, path, use_defaults in _get_config_layers():
        data = load_json_file(path) if path else None
        if data is None:
            continue
        layer = AKGSettings.from_dict(data, use_defaults=use_defaults)
        # 标记此层中各配置项的来源
        for level in layer.models:
            layer._model_sources[level] = source_label
        if "embedding" in data:
            layer._embedding_source = source_label
        settings = settings.merge_with(layer)
        logger.debug(f"Merged settings from {source_label}")
    
    # 环境变量（最高优先级）
    settings = _load_env_config(settings)
    
    return settings


# ========================= 工具函数 =========================

def save_settings_file(settings: AKGSettings, path: Optional[Path] = None) -> None:
    """
    保存配置到文件
    
    Args:
        settings: 要保存的配置
        path: 保存路径，默认为 ~/.akg/settings.json
    """
    if path is None:
        path = get_settings_path()
    
    # 确保目录存在
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(settings.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Saved settings to {path}")
    except Exception as e:
        logger.error(f"Failed to save settings to {path}: {e}")
        raise


def get_all_settings_paths() -> Dict[str, Optional[Path]]:
    """获取所有配置文件路径"""
    return {
        "user": get_settings_path(),
        "project": get_project_settings_path(),
        "local": get_local_settings_path(),
    }


def _mask_key(value: str) -> str:
    """遮蔽 API Key"""
    if len(value) > 12:
        return value[:8] + "***" + value[-4:]
    return "***"


def _print_model_info(level: str, model: "ModelConfig", source: str, indent: str = "   "):
    """打印单个模型级别的配置信息（含来源）"""
    print(f"{indent}[{level}] (from {source})")
    print(f"{indent}  model_name: {model.model_name}")
    print(f"{indent}  base_url: {model.base_url}")
    print(f"{indent}  api_key: {_mask_key(model.api_key)}")
    print(f"{indent}  provider_type: {model.provider_type}")
    print(f"{indent}  temperature: {model.temperature}")
    if model.extra_body:
        print(f"{indent}  extra_body: {model.extra_body}")
    else:
        print(f"{indent}  extra_body: (none)")


def print_settings_info(model_level: Optional[str] = None) -> None:
    """
    打印配置信息（用于调试），包含配置来源
    
    Args:
        model_level: 可选，只显示指定级别的模型配置。
                     如 "complex"/"standard"/"fast"，不指定则显示全部。
    """
    settings = get_settings()
    
    print("=" * 60)
    
    if model_level:
        # 只显示指定级别
        if model_level in settings.models:
            _print_model_info(
                model_level, settings.models[model_level],
                settings.get_model_source(model_level),
            )
        else:
            print(f"⚠️  模型级别 '{model_level}' 未配置")
            print(f"   可用级别: {list(settings.models.keys())}")
    else:
        # 显示全部配置
        print(f"⚙️  生效配置 (models: {list(settings.models.keys())})")
        
        for level, model in settings.models.items():
            print()
            _print_model_info(level, model, settings.get_model_source(level))
        
        if settings.embedding.is_configured():
            print(f"\n   [embedding] (from {settings.get_embedding_source()})")
            print(f"     model_name: {settings.embedding.model_name}")
            print(f"     base_url: {settings.embedding.base_url}")
    
    print("\n💡 优先级: 环境变量 > Local > Project > User > 默认值")
    print("💡 环境变量支持: AKG_AGENTS_* (推荐) 和 AIKG_* (兼容)")
    print("=" * 60)


def check_model_config(model_level: Optional[str] = None) -> bool:
    """
    检查模型配置是否完整（base_url, api_key, model_name 必须非空）
    
    Args:
        model_level: 要检查的模型级别，如 "complex"/"standard"/"fast"。
                     不指定则检查所有已配置的级别。
    
    Returns:
        bool: 配置是否完整
    
    Examples:
        check_model_config("complex")   # 只检查 complex
        check_model_config()            # 检查所有已配置的级别
    """
    settings = get_settings()
    
    levels_to_check = [model_level] if model_level else list(settings.models.keys())
    
    if not levels_to_check:
        print("❌ 未找到任何模型配置")
        print("   请通过环境变量或配置文件设置模型信息")
        return False
    
    all_ok = True
    for level in levels_to_check:
        if level not in settings.models:
            print(f"❌ 模型级别 '{level}' 未配置")
            print(f"   可用级别: {list(settings.models.keys())}")
            all_ok = False
            continue
        
        model = settings.models[level]
        missing = []
        if not model.base_url:
            missing.append("base_url")
        if not model.api_key:
            missing.append("api_key")
        if not model.model_name:
            missing.append("model_name")
        
        if missing:
            print(f"❌ [{level}] 缺少必要配置: {', '.join(missing)}")
            all_ok = False
        else:
            print(f"✅ [{level}] {model.model_name} @ {model.base_url}")
    
    if not all_ok:
        print("\n💡 配置方式:")
        print("   环境变量: export AKG_AGENTS_BASE_URL=... AKG_AGENTS_API_KEY=... AKG_AGENTS_MODEL_NAME=...")
        print("   配置文件: ~/.akg/settings.json 或 .akg/settings.json")
    
    return all_ok


def create_default_settings_file() -> None:
    """创建默认的 settings.json 文件（如果不存在）"""
    settings_path = get_settings_path()
    
    if settings_path.exists():
        logger.info(f"Settings file already exists: {settings_path}")
        return
    
    # 创建示例配置
    default_settings = AKGSettings()
    
    # 添加示例模型配置
    api_key_value = get_akg_env_var("API_KEY", "your-api-key-here")
    default_settings.models = {
        "complex": ModelConfig(
            base_url="https://api.openai.com/v1",
            api_key=api_key_value,
            model_name="gpt-4",
            temperature=0.0
        ),
        "standard": ModelConfig(
            base_url="https://api.openai.com/v1",
            api_key=api_key_value,
            model_name="gpt-3.5-turbo",
            temperature=0.0
        ),
        "fast": ModelConfig(
            base_url="https://api.openai.com/v1",
            api_key=api_key_value,
            model_name="gpt-3.5-turbo",
            temperature=0.3
        )
    }
    
    save_settings_file(default_settings)
    
    print(f"✨ Created default settings file at: {settings_path}")
    print("\n📝 Configuration priority (high to low):")
    print("   1. Environment variables (AKG_AGENTS_* or AIKG_*)")
    print("   2. Local:   .akg/settings.local.json  (gitignored)")
    print("   3. Project: .akg/settings.json        (shared with team)")
    print("   4. User:    ~/.akg/settings.json      (personal)")


def load_settings_file() -> Optional[Dict[str, Any]]:
    """加载配置"""
    return load_json_file(get_settings_path())
