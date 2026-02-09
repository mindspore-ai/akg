# Snapshot Filesystem 设计文档

## 概述

Snapshot Filesystem 是一个轻量级的代码版本管理系统，专为 AKG Agent 的树形搜索场景设计。它使用**文件复制 + 硬链接**实现节点间代码状态的隔离与高效继承，无需外部 Git 依赖。

## 核心优势

| 特性           | 描述                                       |
| -------------- | ------------------------------------------ |
| **零外部依赖** | 仅使用 Python 标准库 (`shutil`, `os.link`) |
| **安全性**     | 无命令注入风险，无 Git hooks 执行          |
| **空间效率**   | 使用硬链接实现写时复制 (Copy-on-Write)     |
| **简单性**     | 所有状态以普通文件形式存储，易于调试和备份 |

## 目录结构

```
conversations/{task_id}/
├── current_node.txt         # 当前活动节点 ID
├── workspace/               # 活动工作区 (可变)
│   └── *.py                 # 当前节点的代码文件
├── nodes/
│   ├── root/
│   │   ├── state.json       # 节点元数据
│   │   ├── code/            # 代码快照 (不可变)
│   │   │   └── *.py
│   │   └── actions/         # 动作历史
│   └── node_001/
│       ├── state.json
│       ├── code/            # 硬链接自父节点 (写时复制)
│       └── ...
└── logs/                    # 执行日志
```

## 核心概念

### 1. 工作区 (Workspace)
- 位置: `workspace/`
- 用途: 当前活动节点的代码镜像
- 特性: 随 `set_current_node()` 自动切换

### 2. 代码快照 (Code Snapshot)
- 位置: `nodes/{node_id}/code/`
- 用途: 节点代码的不可变副本
- 特性: 
  - 创建后内容不变
  - 新节点通过**硬链接**继承父节点快照 (O(1) 空间)
  - 修改时先 `unlink()` 再写入，保证隔离性

### 3. 写时复制 (Copy-on-Write)
当修改文件时，系统会先打破硬链接再写入：

```python
# save_code_file 内部逻辑
if snapshot_file.exists():
    snapshot_file.unlink()  # 打破硬链接
shutil.copy2(workspace_file, snapshot_file)  # 写入新副本
```

## 公共 API

### 代码文件操作

| 方法                                         | 描述                            |
| -------------------------------------------- | ------------------------------- |
| `save_code_file(node_id, filename, content)` | 保存代码到工作区和快照          |
| `load_code_file(node_id, filename) -> str`   | 从快照加载代码                  |
| `list_code_files(node_id) -> List[str]`      | 列出节点的所有代码文件          |
| `export_node_code(node_id, target_dir)`      | 导出节点代码到指定目录          |
| `diff_file(node_a, node_b, filename) -> str` | 比较单一文件在两个节点间的差异  |
| `diff_nodes(node_a, node_b) -> Path`         | 生成节点间所有文件的 Patch 补丁 |

### 节点管理

| 方法                              | 描述                              |
| --------------------------------- | --------------------------------- |
| `copy_node_state(from_id, to_id)` | 复制节点状态 (代码通过硬链接继承) |
| `set_current_node(node_id)`       | 切换工作区到指定节点              |
| `get_current_node() -> str`       | 获取当前活动节点 ID               |
| `find_lca(id1, id2) -> str`       | (TraceSystem) 寻找最近公共祖先    |
| `merge_nodes(target, source)`     | (TraceSystem) 执行三路合并        |

### 状态持久化

| 方法                                    | 描述               |
| --------------------------------------- | ------------------ |
| `save_node_state(node_id, state)`       | 保存节点元数据     |
| `load_node_state(node_id) -> NodeState` | 加载节点元数据     |
| `update_node_state(node_id, **kwargs)`  | 增量更新节点元数据 |

## 上层感知

### 对调用者透明的变化
- ✅ 所有公共 API 保持不变
- ✅ `NodeState.file_state` 格式保持 `{"code/{filename}": {"size": N, ...}}`
- ✅ 代码内容一致性有保证

### 行为差异
| 场景     | Git 版本                 | Snapshot 版本         |
| -------- | ------------------------ | --------------------- |
| 初始化   | 创建 `.git` 目录         | 无 `.git` 目录        |
| 保存代码 | `git add` + `git commit` | 文件复制到快照目录    |
| 切换节点 | `git checkout {hash}`    | 清空工作区 + 复制快照 |
| 节点复制 | 复制 `commit_hash`       | 硬链接快照目录        |

### 已移除的字段
- `NodeState.commit_hash` - 不再需要

## 使用示例

```python
from akg_agents.core_v2.filesystem import FileSystemState

# 初始化
fs = FileSystemState("task_001", base_dir="/path/to/data")
fs.initialize_task()

# 保存代码
fs.save_code_file("root", "kernel.py", "def main(): pass")

# 创建子节点 (代码通过硬链接继承)
fs.copy_node_state("root", "node_a")
fs.set_current_node("node_a")

# 修改代码 (自动打破硬链接)
fs.save_code_file("node_a", "kernel.py", "def main(): print('v2')")

# 切换回 root (工作区自动恢复)
fs.set_current_node("root")
content = (fs.workspace_dir / "kernel.py").read_text()
assert content == "def main(): pass"  # root 的代码未受影响
```

## 注意事项

### 1. 硬链接限制
硬链接要求源和目标在同一文件系统。跨文件系统时自动回退到复制：
```python
def _hardlink_or_copy(self, src, dst):
    try:
        os.link(src, dst)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)  # 回退到复制
```

### 2. 写时复制隔离
修改共享文件前**必须**先 `unlink()`，否则会污染其他节点的快照。`save_code_file()` 已内置此逻辑。

### 3. 工作区是临时的
工作区内容在 `set_current_node()` 时会被完全覆盖。如需保存更改，请先调用 `save_code_file()`。

## Diff & Merge 设计

### Diff 存储 (Workspace-Centric)
为了保持内存和 LLM 上下文轻量，Diff 结果不会持久化在节点中，而是作为临时补丁存储在 `workspace/.akg/diffs/` 下。
- **路径**: `workspace/.akg/diffs/{node_a}_to_{node_b}.patch`
- **生命周期**: 随工作区存在，属于“工作记忆”，由 Agent 按需读取。

### 三路合并 (3-way Merge)
合并逻辑实现在 `TraceSystem` 中，流程如下：
1. **寻找 LCA**: 通过 Trace 树寻找目标节点和源节点的最近公共祖先。
2. **三路比较**:
   - 如果只有一边修改，自动采纳修改。
   - 如果两边修改不一致，产生冲突。
3. **冲突处理**: 在文件中插入标准冲突标记 (`<<<<<<< YOURS` 等)，并将节点状态标记为 `conflict`。

## 测试与验证

### 1. 自动化测试套件
运行以下命令执行全量验证：
```bash
$env:PYTHONPATH="akg_agents/python"; python -m pytest -v akg_agents/python/akg_agents/core_v2/tests/
```

### 2. 测试覆盖范围
系统经过 93 个测试用例的严苛验证，覆盖以下核心领域：

| 维度         | 测试项         | 验证内容                                                         |
| :----------- | :------------- | :--------------------------------------------------------------- |
| **基础功能** | Snapshot/DFS   | 初始化、增量保存、快照加载、节点副本创建、工作区自动恢复。       |
| **空间效率** | Hard Link 校验 | 验证未修改文件共享物理磁盘块；修改文件时自动打破硬链接 (CoW)。   |
| **版本管理** | Diff & Merge   | 三路合并 (3-way Merge)、LCA 寻踪、自动生成统一 Patch 补丁。      |
| **鲁棒性**   | 边缘情况       | 嵌套深层目录、空节点、覆盖写入、非 ASCII 字符处理。              |
| **兼容性**   | 跨平台路径     | 自动处理并归一化 Windows (`\`) 和 Unix (`/`) 风格的路径分隔符。  |
| **性能**     | 深树搜索       | 模拟 50+ 层深度的 Trace 树，验证历史重建与状态读取的性能稳定性。 |
| **异常处理** | 冲突检测       | 验证合并冲突时插入标记、节点状态转为 `conflict` 且不破坏工作区。 |

### 3. 集成测试
`test_system_integration.py` 模拟了真实的 Agent 并行搜索与合并工作流，确保 `FileSystemState` 与 `TraceSystem` 在复杂业务逻辑下保持强一致性。
