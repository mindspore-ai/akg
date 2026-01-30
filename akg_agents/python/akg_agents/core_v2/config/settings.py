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
1. 环境变量（AKG_AGENTS_* 前缀）
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
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


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
    thinking_enabled: Optional[bool] = None  # 是否启用 thinking 模式（None 表示未设置）
    extra: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], use_defaults: bool = True) -> "ModelConfig":
        """
        从字典创建
        
        Args:
            data: 配置字典
            use_defaults: 是否使用默认值，False 时缺失字段为空/None（用于合并）
        """
        if use_defaults:
            return cls(
                base_url=data.get("base_url", "https://api.openai.com/v1"),
                api_key=data.get("api_key", ""),
                model_name=data.get("model_name", "gpt-4"),
                temperature=data.get("temperature", 0.2),
                max_tokens=data.get("max_tokens", 8192),
                top_p=data.get("top_p", 0.9),
                frequency_penalty=data.get("frequency_penalty"),  # 可选，无默认值
                presence_penalty=data.get("presence_penalty"),    # 可选，无默认值
                timeout=data.get("timeout", 300),
                thinking_enabled=data.get("thinking_enabled", False),
                extra=data.get("extra", {})
            )
        else:
            # 用于合并时，只保留显式设置的值（None 表示未设置）
            return cls(
                base_url=data.get("base_url", ""),
                api_key=data.get("api_key", ""),
                model_name=data.get("model_name", ""),
                temperature=data.get("temperature", 0.2),
                max_tokens=data.get("max_tokens", 8192),
                top_p=data.get("top_p", 0.9),
                frequency_penalty=data.get("frequency_penalty"),  # 可选
                presence_penalty=data.get("presence_penalty"),    # 可选
                timeout=data.get("timeout", 300),
                thinking_enabled=data.get("thinking_enabled"),  # None 如果未设置
                extra=data.get("extra", {})
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
            "thinking_enabled": self.thinking_enabled if self.thinking_enabled is not None else False,
            "extra": self.extra
        }
        # 可选参数仅在设置时包含
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
            # 可选参数：other 设置了则用 other，否则保留 self
            frequency_penalty=other.frequency_penalty if other.frequency_penalty is not None else self.frequency_penalty,
            presence_penalty=other.presence_penalty if other.presence_penalty is not None else self.presence_penalty,
            timeout=other.timeout if other.timeout != 300 else self.timeout,
            # thinking_enabled: None 表示未设置，保留原值；显式设置则覆盖
            thinking_enabled=other.thinking_enabled if other.thinking_enabled is not None else self.thinking_enabled,
            extra={**self.extra, **other.extra}
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
    
    # 流式输出
    stream_output: bool = False
    
    # 数据收集
    data_collect: bool = False
    
    # 扩展字段
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于保存）"""
        result = {
            "models": {k: v.to_dict() for k, v in self.models.items()},
            "default_model": self.default_model,
            "stream_output": self.stream_output,
            "data_collect": self.data_collect,
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
            stream_output=data.get("stream_output", False),
            data_collect=data.get("data_collect", False),
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
        
        return AKGSettings(
            models=merged_models,
            embedding=merged_embedding,
            # 只有显式设置了才覆盖（空字符串表示未设置）
            default_model=other.default_model if other.default_model else self.default_model,
            stream_output=other.stream_output or self.stream_output,
            data_collect=other.data_collect or self.data_collect,
            extra={**self.extra, **other.extra},
        )


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
        logger.info(f"Loaded settings from {path}")
        return data
    except Exception as e:
        logger.warning(f"Failed to load settings from {path}: {e}")
        return None


def load_from_env(settings: AKGSettings) -> AKGSettings:
    """
    从环境变量加载配置（最高优先级）
    
    支持两种方式：
    1. 单模型配置（兼容旧版）：AKG_AGENTS_BASE_URL, AKG_AGENTS_API_KEY 等
       -> 自动设置为 'standard' 级别
    
    2. 多模型配置：AKG_AGENTS_COMPLEX_BASE_URL, AKG_AGENTS_STANDARD_API_KEY 等
       -> 分别设置 complex/standard/fast
    """
    
    def _parse_thinking_enabled(env_value: Optional[str]) -> bool:
        """解析 thinking 环境变量"""
        if not env_value:
            return False
        return env_value.lower() in ("enabled", "true", "1", "yes", "on")
    
    thinking_enabled = _parse_thinking_enabled(os.getenv("AKG_AGENTS_MODEL_ENABLE_THINK"))
    
    # 方式 1：单模型配置（兼容旧版，自动设置为 standard）
    if os.getenv("AKG_AGENTS_BASE_URL") or os.getenv("AKG_AGENTS_API_KEY") or os.getenv("AKG_AGENTS_MODEL_NAME"):
        standard_config = ModelConfig(
            base_url=os.getenv("AKG_AGENTS_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.getenv("AKG_AGENTS_API_KEY", ""),
            model_name=os.getenv("AKG_AGENTS_MODEL_NAME", "gpt-4"),
            temperature=float(os.getenv("AKG_AGENTS_TEMPERATURE", "0.0")),
            max_tokens=int(os.getenv("AKG_AGENTS_MAX_TOKENS")) if os.getenv("AKG_AGENTS_MAX_TOKENS") else None,
            timeout=int(os.getenv("AKG_AGENTS_TIMEOUT", "300")),
            thinking_enabled=thinking_enabled,
        )
        settings.models["standard"] = standard_config
        logger.debug(f"Loaded single model config from env as 'standard' (thinking={thinking_enabled})")
    
    # 方式 2：多模型配置
    for level in ["complex", "standard", "fast"]:
        prefix = f"AKG_AGENTS_{level.upper()}_"
        
        if os.getenv(f"{prefix}BASE_URL") or os.getenv(f"{prefix}API_KEY"):
            model_config = ModelConfig(
                base_url=os.getenv(f"{prefix}BASE_URL", "https://api.openai.com/v1"),
                api_key=os.getenv(f"{prefix}API_KEY", ""),
                model_name=os.getenv(f"{prefix}MODEL_NAME", "gpt-4"),
                temperature=float(os.getenv(f"{prefix}TEMPERATURE", "0.0")),
                max_tokens=int(os.getenv(f"{prefix}MAX_TOKENS")) if os.getenv(f"{prefix}MAX_TOKENS") else None,
                timeout=int(os.getenv(f"{prefix}TIMEOUT", "300"))
            )
            settings.models[level] = model_config
            logger.debug(f"Loaded '{level}' model config from env")
    
    # AKG_AGENTS_DEFAULT_MODEL
    if default_model := os.getenv("AKG_AGENTS_DEFAULT_MODEL"):
        settings.default_model = default_model
        logger.debug(f"Loaded default_model from env: {default_model}")
    
    # AKG_AGENTS_STREAM_OUTPUT
    if stream := os.getenv("AKG_AGENTS_STREAM_OUTPUT"):
        settings.stream_output = stream.lower() in ("true", "1", "yes", "on")
        logger.debug(f"Loaded stream_output from env: {settings.stream_output}")
    
    # AKG_AGENTS_DATA_COLLECT
    if collect := os.getenv("AKG_AGENTS_DATA_COLLECT"):
        settings.data_collect = collect.lower() in ("true", "1", "yes", "on")
        logger.debug(f"Loaded data_collect from env: {settings.data_collect}")
    
    # Embedding 配置（AKG_AGENTS_EMBEDDING_*）
    env_emb_base_url = os.getenv("AKG_AGENTS_EMBEDDING_BASE_URL")
    env_emb_model_name = os.getenv("AKG_AGENTS_EMBEDDING_MODEL_NAME")
    env_emb_api_key = os.getenv("AKG_AGENTS_EMBEDDING_API_KEY")
    
    if env_emb_base_url or env_emb_model_name or env_emb_api_key:
        settings.embedding = EmbeddingConfig(
            base_url=env_emb_base_url or "",
            api_key=env_emb_api_key or "",
            model_name=env_emb_model_name or "",
            timeout=int(os.getenv("AKG_AGENTS_EMBEDDING_TIMEOUT", "60")),
        )
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
    5. 环境变量（AKG_AGENTS_*）
    
    高优先级配置会覆盖低优先级配置（只覆盖显式设置的字段）
    
    Returns:
        AKGSettings 配置对象
    """
    # 1. 从默认值开始
    settings = AKGSettings()
    
    # 2. User: ~/.akg/settings.json（完整配置，使用默认值）
    user_path = get_settings_path()
    if user_data := load_json_file(user_path):
        user_settings = AKGSettings.from_dict(user_data, use_defaults=True)
        settings = settings.merge_with(user_settings)
        logger.debug(f"Merged user settings from {user_path}")
    
    # 3. Project: .akg/settings.json（部分配置，不使用默认值）
    project_path = get_project_settings_path()
    if project_path and (project_data := load_json_file(project_path)):
        project_settings = AKGSettings.from_dict(project_data, use_defaults=False)
        settings = settings.merge_with(project_settings)
        logger.debug(f"Merged project settings from {project_path}")
    
    # 4. Local: .akg/settings.local.json（部分配置，不使用默认值）
    local_path = get_local_settings_path()
    if local_path and (local_data := load_json_file(local_path)):
        local_settings = AKGSettings.from_dict(local_data, use_defaults=False)
        settings = settings.merge_with(local_settings)
        logger.debug(f"Merged local settings from {local_path}")
    
    # 5. 环境变量（最高优先级）
    settings = load_from_env(settings)
    
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


def print_settings_info() -> None:
    """打印配置信息（用于调试）"""
    import os
    # 检查环境变量
    env_vars = {
        "AKG_AGENTS_BASE_URL": os.getenv("AKG_AGENTS_BASE_URL"),
        "AKG_AGENTS_API_KEY": os.getenv("AKG_AGENTS_API_KEY"),
        "AKG_AGENTS_MODEL_NAME": os.getenv("AKG_AGENTS_MODEL_NAME"),
        "AKG_AGENTS_MODEL_ENABLE_THINK": os.getenv("AKG_AGENTS_MODEL_ENABLE_THINK"),
    }
    has_env = any(v for v in env_vars.values())
    
    if has_env:
        print("=" * 60)
        print("🌍 环境变量 (最高优先级):")
        for key, value in env_vars.items():
            if value:
                masked = value[:8] + "***" + value[-4:] if "KEY" in key and len(value) > 12 else value
                print(f"   ✓ {key}={masked}")
    else:
        paths = get_all_settings_paths()
        settings = get_settings()
        
        print("=" * 60)
        print("📁 配置文件位置:")
        for scope, path in paths.items():
            if path and path.exists():
                print(f"   ✓ {scope}: {path}")
            elif path:
                print(f"   ○ {scope}: {path} (不存在)")
            else:
                print(f"   ○ {scope}: (未找到项目根目录)")

        print(f"\n⚙️  当前配置:")
        print(f"   default_model: {settings.default_model}")
        print(f"   stream_output: {settings.stream_output}")
        print(f"   models: {list(settings.models.keys())}")
        
        for level, model in settings.models.items():
            masked_key = model.api_key[:8] + "*" * (len(model.api_key) - 12) + \
                model.api_key[-4:] if len(model.api_key) > 12 else "***"
            
            # 判断配置来源
            source = "env" if has_env and level == "standard" else "file"
            
            print(f"\n   [{level}] (from {source})")
            print(f"     model_name: {model.model_name}")
            print(f"     base_url: {model.base_url}")
            print(f"     api_key: {masked_key}")
            print(f"     thinking_enabled: {model.thinking_enabled}")
    
    print("\n💡 优先级: 环境变量 > Local > Project > User > 默认值")
    print("=" * 60)


def create_default_settings_file() -> None:
    """创建默认的 settings.json 文件（如果不存在）"""
    settings_path = get_settings_path()
    
    if settings_path.exists():
        logger.info(f"Settings file already exists: {settings_path}")
        return
    
    # 创建示例配置
    default_settings = AKGSettings()
    
    # 添加示例模型配置
    default_settings.models = {
        "complex": ModelConfig(
            base_url="https://api.openai.com/v1",
            api_key=os.getenv("AKG_AGENTS_API_KEY", "your-api-key-here"),
            model_name="gpt-4",
            temperature=0.0
        ),
        "standard": ModelConfig(
            base_url="https://api.openai.com/v1",
            api_key=os.getenv("AKG_AGENTS_API_KEY", "your-api-key-here"),
            model_name="gpt-3.5-turbo",
            temperature=0.0
        ),
        "fast": ModelConfig(
            base_url="https://api.openai.com/v1",
            api_key=os.getenv("AKG_AGENTS_API_KEY", "your-api-key-here"),
            model_name="gpt-3.5-turbo",
            temperature=0.3
        )
    }
    
    save_settings_file(default_settings)
    
    print(f"✨ Created default settings file at: {settings_path}")
    print("\n📝 Configuration priority (high to low):")
    print("   1. Environment variables (AKG_AGENTS_*)")
    print("   2. Local:   .akg/settings.local.json  (gitignored)")
    print("   3. Project: .akg/settings.json        (shared with team)")
    print("   4. User:    ~/.akg/settings.json      (personal)")


def load_settings_file() -> Optional[Dict[str, Any]]:
    """加载配置"""
    return load_json_file(get_settings_path())
