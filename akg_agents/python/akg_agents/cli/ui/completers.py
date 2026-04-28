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

"""自动补全器 - 斜杠命令自动补全"""

from __future__ import annotations

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from typing import Iterable


class SlashCommandCompleter(Completer):
    """斜杠命令自动补全器"""
    
    def __init__(self, registry):
        self.registry = registry
    
    def get_completions(self, document: Document, complete_event) -> Iterable[Completion]:
        text = document.text_before_cursor
        
        # 只在输入斜杠开头时提示命令
        if text.startswith('/'):
            word = text[1:]  # 去掉前导 /
            
            for cmd in self.registry.list_all():
                # 匹配命令名或别名
                if cmd.name.startswith(word):
                    yield self._make_completion(cmd, word, False)
                else:
                    for alias in cmd.aliases:
                        if alias.startswith(word):
                            yield self._make_completion(cmd, word, True, alias)
                            break
    
    def _make_completion(self, cmd, word: str, is_alias: bool, alias: str = None) -> Completion:
        """构造补全项"""
        display_name = alias if is_alias else cmd.name
        display_text = f'/{display_name}'
        
        # 构建描述信息
        meta_parts = [cmd.description]
        if is_alias:
            meta_parts.append(f'(→ {cmd.name})')
        if cmd.category:
            meta_parts.append(f'[{cmd.category.value}]')
        
        return Completion(
            text=display_name,
            start_position=-len(word),
            display=display_text,
            display_meta=' '.join(meta_parts),
            style='fg:ansigreen bold' if not is_alias else 'fg:ansiyellow'
        )


class EnhancedCompleter(Completer):
    """增强的自动补全器（组合多种补全源）"""
    
    def __init__(self, command_registry):
        self.slash_completer = SlashCommandCompleter(command_registry)
    
    def get_completions(self, document: Document, complete_event) -> Iterable[Completion]:
        text = document.text_before_cursor
        
        # 斜杠命令补全
        if text.startswith('/'):
            yield from self.slash_completer.get_completions(document, complete_event)
