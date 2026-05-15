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

"""Plugin registry for panel plugins"""

from typing import Optional

from akg_agents.cli.ui.panel_plugin import PanelPlugin


class PanelPluginRegistry:
    """Panel plugin registry"""

    def __init__(self):
        self._plugins: dict[str, PanelPlugin] = {}

    def register(self, plugin: PanelPlugin) -> None:
        """注册插件"""
        self._plugins[plugin.get_name()] = plugin

    def get_plugin(self, name: str) -> Optional[PanelPlugin]:
        """获取插件"""
        return self._plugins.get(name)

    def get_default_plugin(self) -> Optional[PanelPlugin]:
        """获取默认插件（kernel_impl_list）"""
        return self.get_plugin("kernel_impl_list")


# 全局注册表实例
_registry = PanelPluginRegistry()


def register_plugin(plugin: PanelPlugin) -> None:
    """注册插件"""
    _registry.register(plugin)


def get_plugin(name: str) -> Optional[PanelPlugin]:
    """获取插件"""
    return _registry.get_plugin(name)


def get_default_plugin() -> Optional[PanelPlugin]:
    """获取默认插件"""
    return _registry.get_default_plugin()
