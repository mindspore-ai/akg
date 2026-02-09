"""
输入解析器 - 识别用户输入类型并加载代码内容

修复:
  - 用户输入 "路径 + 描述" 时，正确拆分路径和描述
  - 目录输入直接进入该目录，不退回上级
  - 文件/目录附带行数等元信息
"""
import re
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass, field


@dataclass
class ParsedInput:
    """解析后的用户输入"""
    input_type: str  # "code" | "file" | "directory"
    code: str = ""                  # 代码内容（或文件读取后的内容）
    file_path: Optional[str] = None  # 文件路径
    dir_path: Optional[str] = None   # 目录路径
    description: str = ""            # 用户对代码的描述
    files: Dict[str, str] = field(default_factory=dict)  # 目录: {相对路径: 行数}
    total_lines: int = 0


class InputParser:
    """解析用户的原始输入，判断类型并加载内容"""

    @classmethod
    def parse(cls, raw_input: str, description: str = "") -> ParsedInput:
        """
        解析用户输入。

        支持格式:
          - 纯代码文本
          - 文件绝对/相对路径
          - 目录绝对/相对路径
          - "路径 描述文本" （自动拆分）
        """
        text = raw_input.strip()

        # 尝试从 "路径 + 描述" 中拆分
        resolved_path, extra_desc = cls._split_path_and_desc(text)

        # 合并描述
        if extra_desc and not description:
            description = extra_desc
        elif extra_desc and description:
            description = f"{description}; {extra_desc}"

        # 如果拆分出了有效路径
        if resolved_path and resolved_path.exists():
            if resolved_path.is_file():
                return cls._parse_file(resolved_path, description)
            elif resolved_path.is_dir():
                return cls._parse_directory(resolved_path, description)

        # 整体当作路径检查
        path = Path(text).expanduser()
        if path.exists():
            if path.is_file():
                return cls._parse_file(path, description)
            elif path.is_dir():
                return cls._parse_directory(path, description)

        # 检查是否"看起来像"路径（但不存在）
        if cls._looks_like_path(text):
            return ParsedInput(
                input_type="file",
                file_path=text,
                description=description,
            )

        # 当作代码文本
        return ParsedInput(
            input_type="code",
            code=text,
            description=description,
        )

    @classmethod
    def _split_path_and_desc(cls, text: str):
        """
        尝试从 "C:\\path\\to\\dir 帮我做xxx" 中拆分路径和描述。
        策略: 从左到右尝试，找到最长的有效路径。
        """
        if "\n" in text:
            return None, ""

        # 按空格分割，逐步尝试更长的前缀
        parts = text.split()
        best_path = None
        best_idx = 0

        for i in range(1, len(parts) + 1):
            candidate = " ".join(parts[:i])
            p = Path(candidate).expanduser()
            if p.exists():
                best_path = p
                best_idx = i

        if best_path:
            desc = " ".join(parts[best_idx:]).strip()
            return best_path, desc

        return None, ""

    @classmethod
    def _parse_file(cls, path: Path, description: str) -> ParsedInput:
        try:
            content = path.read_text(encoding="utf-8")
            total_lines = content.count("\n") + 1
        except Exception:
            content = ""
            total_lines = 0
        return ParsedInput(
            input_type="file",
            code=content,
            file_path=str(path.resolve()),
            description=description,
            total_lines=total_lines,
        )

    @classmethod
    def _parse_directory(cls, path: Path, description: str) -> ParsedInput:
        """扫描目录，不读取完整内容，只记录文件名和行数"""
        files = {}
        total_lines = 0
        py_files = sorted(path.rglob("*.py"))
        for f in py_files[:200]:
            try:
                rel = str(f.relative_to(path))
                if any(part.startswith('.') for part in f.relative_to(path).parts):
                    continue
                if any(skip in rel for skip in ["__pycache__", "egg-info", ".git"]):
                    continue
                line_count = f.read_text(encoding="utf-8").count("\n") + 1
                files[rel] = str(line_count)  # 只存行数
                total_lines += line_count
            except (UnicodeDecodeError, PermissionError):
                continue
        return ParsedInput(
            input_type="directory",
            dir_path=str(path.resolve()),
            description=description,
            files=files,
            total_lines=total_lines,
        )

    @staticmethod
    def _looks_like_path(text: str) -> bool:
        if "\n" in text:
            return False
        if len(text) > 500:
            return False
        return bool(re.match(r'^[a-zA-Z]:\\|^[~/.]|^/', text))
