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
Skill安装管理模块

提供统一的skill安装位置和资源访问机制
"""

import json
import shutil
import hashlib
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import logging

from .metadata import SkillMetadata
from .loader import SkillLoader

logger = logging.getLogger(__name__)


class SkillInstaller:
    """
    Skill安装管理器
    
    核心功能：
    1. 将skills从开发目录复制到标准安装位置（~/.akg/skills/）
    2. 基于文件哈希的增量更新（未copy的文件复制，哈希不一致的覆盖，一致的跳过）
    3. 提供统一的资源路径解析API
    
    安装目录结构：
        ~/.akg/skills/
        ├── skill-name-1/
        │   ├── SKILL.md
        │   ├── templates/
        │   ├── scripts/
        │   └── .install.json       # 文件清单+哈希
        ├── skill-name-2/
        │   └── ...
        └── .registry.json          # 全局安装清单
    """
    
    def __init__(self, install_root: Optional[Path] = None):
        """
        初始化安装管理器
        
        Args:
            install_root: 安装根目录，默认 ~/.akg/skills
        """
        self.install_root = install_root or (Path.home() / ".akg" / "skills")
        self.registry_file = self.install_root / ".registry.json"
        self._ensure_directories()
        self._loader = SkillLoader()
    
    def _ensure_directories(self):
        """确保安装目录存在"""
        self.install_root.mkdir(parents=True, exist_ok=True)
        
        # 初始化全局清单
        if not self.registry_file.exists():
            self._save_registry({
                "version": "1.0",
                "skills": {},
                "last_updated": datetime.now().isoformat()
            })
    
    # ===== 哈希计算 =====
    
    @staticmethod
    def _calculate_file_hash(file_path: Path) -> str:
        """
        计算文件的SHA256哈希值
        
        Args:
            file_path: 文件路径
        
        Returns:
            十六进制哈希字符串
        """
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _calculate_directory_hashes(self, directory: Path) -> Dict[str, str]:
        """
        计算目录下所有文件的哈希值
        
        Args:
            directory: 目录路径
        
        Returns:
            {相对路径: 哈希值}
        """
        hashes = {}
        
        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.name != ".install.json":
                relative_path = str(file_path.relative_to(directory))
                hashes[relative_path] = self._calculate_file_hash(file_path)
        
        return hashes
    
    # ===== 安装信息管理 =====
    
    def _load_install_info(self, skill_dir: Path) -> Dict[str, Any]:
        """加载skill的安装信息"""
        info_file = skill_dir / ".install.json"
        if info_file.exists():
            with open(info_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_install_info(self, skill_dir: Path, info: Dict[str, Any]):
        """保存skill的安装信息"""
        info_file = skill_dir / ".install.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
    
    def _load_registry(self) -> Dict[str, Any]:
        """加载全局安装清单"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"version": "1.0", "skills": {}, "last_updated": ""}
    
    def _save_registry(self, registry: Dict[str, Any]):
        """保存全局安装清单"""
        registry["last_updated"] = datetime.now().isoformat()
        with open(self.registry_file, 'w', encoding='utf-8') as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)
    
    def _update_registry(self, skill_name: str, info: Dict[str, Any]):
        """更新全局清单中的skill信息"""
        registry = self._load_registry()
        registry["skills"][skill_name] = {
            "version": info.get("version", "1.0.0"),  # 记录version
            "installed_at": info.get("installed_at", datetime.now().isoformat()),
            "updated_at": datetime.now().isoformat(),
            "source": info.get("source", ""),
            "file_count": len(info.get("files", {}))
        }
        self._save_registry(registry)
    
    # ===== 文件同步 =====
    
    def _sync_files(
        self, 
        source_dir: Path, 
        target_dir: Path,
        existing_hashes: Dict[str, str]
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        同步文件（基于哈希的增量更新）
        
        Args:
            source_dir: 源目录
            target_dir: 目标目录
            existing_hashes: 已存在文件的哈希字典
        
        Returns:
            (操作统计, 新的哈希字典)
            操作统计: {"copied": 复制数, "updated": 更新数, "skipped": 跳过数}
        """
        target_dir.mkdir(parents=True, exist_ok=True)
        
        stats = {"copied": 0, "updated": 0, "skipped": 0}
        new_hashes = {}
        
        # 遍历源目录所有文件
        for source_file in source_dir.rglob("*"):
            if not source_file.is_file():
                continue
            
            # 计算相对路径
            relative_path = str(source_file.relative_to(source_dir))
            target_file = target_dir / relative_path
            
            # 计算源文件哈希
            source_hash = self._calculate_file_hash(source_file)
            new_hashes[relative_path] = source_hash
            
            # 判断是否需要复制
            if relative_path not in existing_hashes:
                # 文件不存在 → 复制
                target_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_file, target_file)
                stats["copied"] += 1
                logger.info(f"复制新文件: {relative_path}")
            
            elif existing_hashes[relative_path] != source_hash:
                # 哈希不一致 → 覆盖
                shutil.copy2(source_file, target_file)
                stats["updated"] += 1
                logger.info(f"更新文件: {relative_path}")
            
            else:
                # 哈希一致 → 跳过
                stats["skipped"] += 1
                logger.debug(f"跳过文件: {relative_path}")
        
        return stats, new_hashes
    
    # ===== URL安装相关 =====
    
    def install_from_url(
        self,
        url: str,
        skill_path: Optional[str] = None,
        branch: str = "main"
    ) -> Dict[str, Any]:
        """
        从Git URL安装skill
        
        支持：
        - GitHub仓库
        - GitLab仓库
        - 任何Git仓库
        
        Args:
            url: Git仓库URL
            skill_path: 仓库内skill的相对路径（如果仓库包含多个skills）
            branch: 分支名（默认main）
        
        Returns:
            安装结果字典
        
        示例：
            # 安装整个仓库作为一个skill
            installer.install_from_url(
                "https://github.com/user/my-skill.git"
            )
            
            # 安装仓库中的特定skill
            installer.install_from_url(
                "https://github.com/vercel-labs/agent-skills.git",
                skill_path="vercel-react-best-practices"
            )
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            
            try:
                # 克隆仓库
                logger.info(f"克隆仓库: {url}")
                subprocess.run(
                    ["git", "clone", "--depth", "1", "--branch", branch, url, str(tmp_path)],
                    check=True,
                    capture_output=True,
                    text=True
                )
                
                # 确定skill目录
                if skill_path:
                    source_dir = tmp_path / skill_path
                    if not source_dir.exists():
                        raise ValueError(f"仓库中不存在路径: {skill_path}")
                else:
                    source_dir = tmp_path
                
                # 安装skill
                result = self.install(source_dir)
                result["source_url"] = url
                result["source_path"] = skill_path
                
                return result
                
            except subprocess.CalledProcessError as e:
                error_msg = f"Git克隆失败: {e.stderr if e.stderr else str(e)}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            except Exception as e:
                logger.error(f"从URL安装失败: {e}")
                raise
    
    def install_from_github(
        self,
        repo: str,
        skill_path: Optional[str] = None,
        branch: str = "main"
    ) -> Dict[str, Any]:
        """
        从GitHub仓库安装skill（便捷方法）
        
        Args:
            repo: GitHub仓库（格式：owner/repo）
            skill_path: 仓库内skill的相对路径
            branch: 分支名
        
        Returns:
            安装结果字典
        
        示例：
            installer.install_from_github(
                "vercel-labs/agent-skills",
                skill_path="vercel-react-best-practices"
            )
        """
        url = f"https://github.com/{repo}.git"
        return self.install_from_url(url, skill_path, branch)
    
    # ===== 本地安装相关 =====
    
    def install(
        self, 
        source_path: Path,
        skill_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        从源目录安装skill
        
        流程：
        1. 检测SKILL.md，解析skill名称
        2. 计算所有文件的哈希
        3. 对比已安装版本，执行增量更新
        4. 保存安装信息
        
        Args:
            source_path: skill源目录（包含SKILL.md）
            skill_name: skill名称（如不指定则从SKILL.md读取）
        
        Returns:
            安装结果字典，包含操作统计
        """
        # 1. 验证源目录
        if not source_path.exists():
            raise ValueError(f"源目录不存在: {source_path}")
        
        skill_md = source_path / "SKILL.md"
        if not skill_md.exists():
            raise ValueError(f"源目录缺少SKILL.md: {source_path}")
        
        # 2. 解析skill元数据
        skill = self._loader.load_single(skill_md)
        if not skill:
            raise ValueError(f"无法从SKILL.md解析元数据: {skill_md}")
        
        skill_name = skill_name or skill.name
        
        # 3. 确定目标目录
        target_dir = self.install_root / skill_name
        
        # 4. 加载已有安装信息
        existing_info = self._load_install_info(target_dir)
        existing_hashes = existing_info.get("files", {})
        
        # 5. 同步文件
        stats, new_hashes = self._sync_files(
            source_path,
            target_dir,
            existing_hashes
        )
        
        # 6. 保存安装信息（包含version用于校验）
        install_info = {
            "skill_name": skill_name,
            "version": skill.version,  # 记录version作为"文件是否被修改"的校验
            "installed_at": existing_info.get("installed_at", datetime.now().isoformat()),
            "updated_at": datetime.now().isoformat(),
            "source": str(source_path.absolute()),
            "files": new_hashes
        }
        self._save_install_info(target_dir, install_info)
        
        # 7. 更新全局清单
        self._update_registry(skill_name, install_info)
        
        # 8. 返回结果
        result = {
            "skill_name": skill_name,
            "target_dir": str(target_dir),
            "stats": stats,
            "total_files": len(new_hashes)
        }
        
        logger.info(
            f"安装完成: {skill_name} "
            f"(复制{stats['copied']}个，更新{stats['updated']}个，跳过{stats['skipped']}个)"
        )
        
        return result
    
    def install_from_directory(
        self, 
        skills_dir: Path,
        pattern: str = "**/SKILL.md"
    ) -> List[Dict[str, Any]]:
        """
        批量安装目录下所有skills
        
        Args:
            skills_dir: 包含多个skill的目录
            pattern: 查找SKILL.md的glob模式
        
        Returns:
            安装结果列表
        """
        if not skills_dir.exists():
            raise ValueError(f"目录不存在: {skills_dir}")
        
        results = []
        skill_files = list(skills_dir.glob(pattern))
        
        logger.info(f"发现 {len(skill_files)} 个skill")
        
        for skill_file in skill_files:
            skill_dir = skill_file.parent
            try:
                result = self.install(skill_dir)
                results.append(result)
            except Exception as e:
                logger.error(f"安装失败: {skill_dir} - {e}")
                results.append({
                    "skill_name": skill_dir.name,
                    "error": str(e),
                    "success": False
                })
        
        return results
    
    # ===== 卸载相关 =====
    
    def uninstall(self, skill_name: str) -> bool:
        """
        卸载skill
        
        Args:
            skill_name: skill名称
        
        Returns:
            是否成功
        """
        target_dir = self.install_root / skill_name
        
        if not target_dir.exists():
            logger.warning(f"Skill未安装: {skill_name}")
            return False
        
        # 删除目录
        shutil.rmtree(target_dir)
        
        # 更新全局清单
        registry = self._load_registry()
        if skill_name in registry["skills"]:
            del registry["skills"][skill_name]
            self._save_registry(registry)
        
        logger.info(f"卸载成功: {skill_name}")
        return True
    
    # ===== 查询相关 =====
    
    def list_installed(self) -> Dict[str, Dict[str, Any]]:
        """
        列出所有已安装的skills
        
        Returns:
            {skill_name: {安装信息}}
        """
        registry = self._load_registry()
        return registry.get("skills", {})
    
    def is_installed(self, skill_name: str) -> bool:
        """
        检查skill是否已安装
        
        Args:
            skill_name: skill名称
        
        Returns:
            是否已安装
        """
        target_dir = self.install_root / skill_name
        return target_dir.exists() and (target_dir / "SKILL.md").exists()
    
    def get_skill_path(self, skill_name: str) -> Optional[Path]:
        """
        获取已安装skill的路径
        
        Args:
            skill_name: skill名称
        
        Returns:
            skill目录路径，未安装返回None
        """
        target_dir = self.install_root / skill_name
        
        if self.is_installed(skill_name):
            return target_dir
        
        return None
    
    def resolve_resource(
        self,
        skill_name: str,
        resource_path: str
    ) -> Optional[Path]:
        """
        解析skill内的资源路径（核心功能，供Agent/工具使用）
        
        Example:
            >>> installer.resolve_resource(
            ...     "triton-syntax", 
            ...     "templates/kernel.triton"
            ... )
            Path("/home/user/.akg/skills/triton-syntax/templates/kernel.triton")
        
        Args:
            skill_name: skill名称
            resource_path: 资源相对路径
        
        Returns:
            资源绝对路径，不存在返回None
        """
        skill_path = self.get_skill_path(skill_name)
        
        if not skill_path:
            logger.warning(f"Skill未安装: {skill_name}")
            return None
        
        resource = skill_path / resource_path
        
        if not resource.exists():
            logger.warning(f"资源不存在: {skill_name}/{resource_path}")
            return None
        
        return resource
    
    # ===== 验证相关 =====
    
    def verify(self, skill_name: str) -> Tuple[bool, Optional[str]]:
        """
        验证已安装skill的完整性
        
        检查：
        1. 目录存在
        2. SKILL.md存在
        3. .install.json中记录的文件是否都存在
        
        Args:
            skill_name: skill名称
        
        Returns:
            (是否有效, 错误信息)
        """
        skill_path = self.get_skill_path(skill_name)
        
        if not skill_path:
            return False, f"Skill未安装: {skill_name}"
        
        # 检查SKILL.md
        if not (skill_path / "SKILL.md").exists():
            return False, "SKILL.md文件丢失"
        
        # 检查安装信息
        install_info = self._load_install_info(skill_path)
        if not install_info:
            return False, ".install.json文件丢失或损坏"
        
        # 检查记录的文件
        recorded_files = install_info.get("files", {})
        for relative_path in recorded_files.keys():
            file_path = skill_path / relative_path
            if not file_path.exists():
                return False, f"文件丢失: {relative_path}"
        
        return True, None
    
    def get_install_info(self, skill_name: str) -> Optional[Dict[str, Any]]:
        """
        获取skill的安装信息
        
        Args:
            skill_name: skill名称
        
        Returns:
            安装信息字典，未安装返回None
        """
        skill_path = self.get_skill_path(skill_name)
        
        if not skill_path:
            return None
        
        return self._load_install_info(skill_path)
