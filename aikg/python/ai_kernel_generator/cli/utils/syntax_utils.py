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

from typing import Union

from rich.style import Style
from rich.syntax import PygmentsSyntaxTheme, SyntaxTheme

try:
    from pygments.token import Error, Token
except Exception:  # pragma: no cover - defensive fallback
    Error = None
    Token = None


class ErrorNeutralSyntaxTheme(SyntaxTheme):
    """Wrap a base theme and neutralize Token.Error styling."""

    def __init__(self, base: SyntaxTheme) -> None:
        self._base = base
        self._text_style = (
            base.get_style_for_token(Token.Text) if Token is not None else Style.null()
        )

    def get_style_for_token(self, token_type):
        if Token is not None and token_type in (Token.Error, Error):
            return self._text_style
        return self._base.get_style_for_token(token_type)

    def get_background_style(self) -> Style:
        return self._base.get_background_style()


def build_syntax_theme(
    theme: Union[str, SyntaxTheme], *, suppress_error_tokens: bool = True
) -> Union[str, SyntaxTheme]:
    if not suppress_error_tokens:
        return theme
    try:
        base = theme if isinstance(theme, SyntaxTheme) else PygmentsSyntaxTheme(theme)
    except Exception:
        return theme
    return ErrorNeutralSyntaxTheme(base)
