# AIKG Trace 系统设计（Tree 版本）

> 单一树结构，多分叉探索，节点切换

---

## 一、设计理念

### 核心原则

1. **单一树结构**：整个 Trace 就是一个树（Tree），不需要 "branch" 的概念
2. **节点分叉**：一个节点可以有多个子节点（children），形成分叉
3. **节点切换**：用户通过 `/trace switch <node_id>` 切换到任意节点
4. **自动分叉**：在已有子节点的节点上执行新操作时，自动创建新的子节点（分叉）

---

## 二、文件系统结构

```
~/.akg/conversations/{task_id}/
├─ trace.json                      # Trace 树元数据 ⭐
├─ current_node.txt                # 当前节点 ID
│
├─ nodes/                          # 每个节点的状态 ⭐
│   ├─ root/
│   │   └─ state.json              # 初始状态
│   │
│   ├─ node_001/
│   │   ├─ state.json
│   │   ├─ thinking.json
│   │   ├─ actions/
│   │   │   ├─ action_history.json          # 压缩版（渲染用）
│   │   │   ├─ action_history_fact.json     # 增量版（只保存本节点的 action）⭐
│   │   │   └─ pending_tools.json           # 待执行工具
│   │   └─ code/
│   │
│   ├─ node_002/
│   │   ├─ state.json
│   │   ├─ thinking.json
│   │   ├─ actions/
│   │   │   └─ action_history_fact.json     # 只保存 node_002 的 action
│   │   └─ code/
│   │
│   ├─ node_003/                   # node_002 的子节点（路径1）
│   │   ├─ state.json
│   │   ├─ actions/
│   │   │   └─ action_history_fact.json     # 只保存 node_003 的 action
│   │   └─ code/
│   │
│   └─ node_004/                   # node_002 的子节点（路径2，分叉！）
│       ├─ state.json
│       ├─ actions/
│       │   └─ action_history_fact.json     # 只保存 node_004 的 action
│       └─ code/
│
└─ logs/
```

---

## 三、trace.json 设计（核心）

### 完整示例

```json
{
  "task_id": "task_20260121_001",
  "created_at": "2026-01-21T10:00:00Z",
  "current_node": "node_003",
  
  "tree": {
    "root": {
      "node_id": "root",
      "parent_id": null,
      "state_snapshot": {
        "turn": 0,
        "status": "init"
      },
      "action": null,
      "result": null,
      "timestamp": "2026-01-21T10:00:00Z",
      "children": ["node_001"],
      "metrics": {}
    },
    
    "node_001": {
      "node_id": "node_001",
      "parent_id": "root",
      "state_snapshot": {
        "turn": 1,
        "status": "running"
      },
      "action": {
        "type": "call_designer",
        "params": {}
      },
      "result": {
        "success": true,
        "output": "设计方案..."
      },
      "timestamp": "2026-01-21T10:05:00Z",
      "children": ["node_002"],
      "metrics": {
        "token_used": 1200
      }
    },
    
    "node_002": {
      "node_id": "node_002",
      "parent_id": "node_001",
      "state_snapshot": {
        "turn": 2,
        "status": "running"
      },
      "action": {
        "type": "call_coder",
        "params": {}
      },
      "result": {
        "success": true,
        "output": "生成代码..."
      },
      "timestamp": "2026-01-21T10:10:00Z",
      "children": ["node_003", "node_004"],
      "metrics": {
        "token_used": 2000
      }
    },
    
    "node_003": {
      "node_id": "node_003",
      "parent_id": "node_002",
      "state_snapshot": {
        "turn": 3,
        "status": "running"
      },
      "action": {
        "type": "verify_tool",
        "params": {}
      },
      "result": {
        "success": true,
        "performance": 65
      },
      "timestamp": "2026-01-21T10:15:00Z",
      "children": ["node_005"],
      "metrics": {
        "performance": 65,
        "token_used": 500
      }
    },
    
    "node_004": {
      "node_id": "node_004",
      "parent_id": "node_002",
      "state_snapshot": {
        "turn": 3,
        "status": "running"
      },
      "action": {
        "type": "call_coder",
        "params": {
          "strategy": "shared_memory"
        }
      },
      "result": {
        "success": true,
        "output": "使用 shared memory 优化..."
      },
      "timestamp": "2026-01-21T10:13:00Z",
      "children": ["node_006"],
      "metrics": {
        "token_used": 2500
      }
    },
    
    "node_005": {
      "node_id": "node_005",
      "parent_id": "node_003",
      "action": {
        "type": "call_coder",
        "params": {}
      },
      "result": {
        "success": true,
        "output": "优化代码..."
      },
      "children": [],
      "metrics": {
        "token_used": 1800
      }
    },
    
    "node_006": {
      "node_id": "node_006",
      "parent_id": "node_004",
      "action": {
        "type": "verify_tool",
        "params": {}
      },
      "result": {
        "success": true,
        "performance": 85
      },
      "children": [],
      "metrics": {
        "performance": 85,
        "token_used": 500
      }
    }
  }
}
```

### 树结构可视化

```
root
  ↓
node_001 (call_designer)
  ↓
node_002 (call_coder: 120 行)
  ├─ node_003 (verify: 65%) → node_005 (优化中...)
  │
  └─ node_004 (call_coder: SM 优化, 150 行) → node_006 (verify: 85%)
```

---

## 四、用户交互场景

### 场景1: 启动任务

```bash
# 通过 akg_cli op 启动任务
$ akg_cli op --task "实现 softmax"

🌳 创建任务: task_20260121_001
📍 当前节点: root
📝 开始生成...
```

### 场景2: 查看 Trace 树

在 AIKG 对话界面中：

```
👤 User: /trace show

🤖 AIKG:
🌳 Trace Tree (当前: node_003):

root
  ↓
[1] call_designer
    → 策略: Block Reduce + Warp Shuffle
  ↓
[2] call_coder
    → 代码: 120 行
  ↓
  ├─ [3] verify_tool ⭐ 当前
  │   → 性能: 65%
  │   ↓
  │   [5] call_coder (优化中...)
  │
  └─ [4] call_coder (SM 优化)
      → 代码: 150 行
      ↓
      [6] verify_tool
      → 性能: 85% ✅

💡 使用 /trace switch <node> 切换到指定节点
```

### 场景3: 切换节点并创建分叉

```
👤 User: /trace switch 2

🤖 AIKG: 
✅ 已切换到节点 [2] call_coder
📝 当前代码: 120 行

---

👤 User: 使用 shared memory 优化代码

🤖 AIKG: 
🌿 检测到节点 [2] 已有子节点 [3]
✅ 创建新分叉: node_004
🔄 正在使用 shared memory 优化...

✅ call_coder 完成
📝 生成代码: 150 行
📍 当前节点: node_004

---

👤 User: 验证性能

🤖 AIKG:
✅ verify_tool 完成
⚡ 性能: 85% ✅
📍 当前节点: node_006
```

### 场景4: 对比不同路径

```
👤 User: /trace compare 3 6

🤖 AIKG:
📊 路径对比: node_003 vs node_006

路径 A (root → 1 → 2 → 3):
  • 操作: 默认策略
  • 性能: 65%
  • Token: 3700

路径 B (root → 1 → 2 → 4 → 6):
  • 操作: Shared Memory 优化
  • 性能: 85% ✅
  • Token: 5000

💡 使用 /trace switch 6 切换到更优节点
```

### 场景5: 并行探索多个方案

```
👤 User: /parallel 4 call_design

🤖 AIKG:
🔄 从当前节点 [root] 创建 4 个分叉并行设计...

✅ 并行执行完成: call_design × 4

📊 结果摘要:

1. node_007 ✅
   策略: Block Reduce + Shared Memory
   Token: 1200

2. node_008 ✅
   策略: Warp Shuffle + Tree Reduction
   Token: 1500

3. node_009 ✅
   策略: Parallel Scan + Coalesced Access
   Token: 1300

4. node_010 ✅
   策略: Hybrid Approach (Block + Warp)
   Token: 1400

💡 使用 /trace switch <node> 切换到指定节点继续
```

### 场景6: 继续任务

```bash
# 继续任务（恢复上次状态）
$ akg_cli op continue task_20260121_001

🔄 恢复任务: task_20260121_001
📍 当前节点: node_006
⚡ 性能: 85%

# 进入对话界面，用户可以继续操作
👤 User: 继续优化...
```

---

## 五、对话式命令设计

### 5.1 /trace show [node_id]

显示 Trace 树或指定节点详情

```python
# 命令格式
/trace show                          # 显示整个树
/trace show <node_id>                # 显示指定节点详情
/trace show --full                   # 显示完整树（不折叠）

# 示例
/trace show                          # 显示整个树
/trace show 3                        # 显示节点 3 详情
```

**树视图输出（折叠）**：

```
🌳 Trace Tree (当前: node_003):

root
  ↓
[1] call_designer (折叠)
  ↓
[2] call_coder
    → 代码: 120 行
  ↓
  ├─ [3] verify_tool ⭐ 当前
  │   → 性能: 65%
  │   ↓
  │   [5] call_coder (优化中...)
  │
  └─ [4] call_coder (SM 优化)
      → 代码: 150 行
      ↓
      [6] verify_tool
      → 性能: 85% ✅

💡 使用 /trace show <node> 查看节点详情
💡 使用 /trace switch <node> 切换节点
```

**节点详情输出**：

```
📋 节点详情: node_002

📍 节点信息:
  • ID: node_002
  • 父节点: node_001
  • 子节点: [node_003, node_004] (2 个分叉)
  • 时间: 2026-01-21 10:10:00

📝 执行动作:
  • 类型: call_coder
  • 参数: {}

📊 执行结果:
  • 状态: success
  • 输出: 生成代码...
  • 代码: 120 行

📈 指标:
  • Token 消耗: 2000

💡 /trace switch 2 切换到此节点
```

### 5.2 /trace switch <node_id>

切换到指定节点

```python
# 命令格式
/trace switch <node_id>

# 示例
/trace switch 2                     # 切换到节点 2
/trace switch node_002              # 切换到 node_002
```

**输出**：

```
✅ 已切换到: node_002
📝 当前代码: 120 行
📊 子节点: 2 个 (node_003, node_004)

💡 如果执行新操作，将创建新的分叉 (node_XXX)
```

### 5.3 /trace compare <node_1> <node_2>

对比两个节点（及其路径）

```python
# 命令格式
/trace compare <node_1> <node_2>

# 示例
/trace compare 3 6                  # 对比节点 3 和节点 6
```

**输出**：

```
📊 路径对比: node_003 vs node_006

路径 A (root → 1 → 2 → 3):
  • 长度: 3 步
  • 总 Token: 3700
  • 性能: 65%
  • 描述: 默认策略

路径 B (root → 1 → 2 → 4 → 6):
  • 长度: 4 步
  • 总 Token: 5000
  • 性能: 85% ✅
  • 描述: Shared Memory 优化

差异分析:
  • 分叉点: node_002
  • node_003: verify_tool (性能: 65%)
  • node_004: call_coder (SM 优化) + verify_tool (性能: 85%)

💡 使用 /trace switch 6 切换到更优节点
```

### 5.4 /trace path <node_id>

显示从 root 到指定节点的路径

```python
# 命令格式
/trace path <node_id>

# 示例
/trace path 6                       # 显示到节点 6 的路径
```

**输出**：

```
📍 路径: root → node_006

root (初始状态)
  ↓
[1] call_designer
    → 策略: Block Reduce + Warp Shuffle
    Token: 1200
  ↓
[2] call_coder
    → 代码: 120 行
    Token: 2000
  ↓
[4] call_coder (SM 优化) 🌿 分叉点
    → 代码: 150 行
    Token: 2500
  ↓
[6] verify_tool
    → 性能: 85% ✅
    Token: 500

总计:
  • 步数: 4
  • Token: 6200
  • 性能: 85%
```

### 5.5 /parallel N <action> [params]

从当前节点创建 N 个分叉，并行执行指定 action

```python
# 命令格式
/parallel <N> <action> [params]

# 支持的 action
/parallel 4 call_design              # 并行设计 4 个方案
/parallel 3 call_codegen             # 并行生成 3 个代码实现
/parallel 5 call_evolve              # 并行演进 5 个优化版本
/parallel 3 call_adaptive_search     # 并行搜索 3 个参数配置
```

**输出示例**：

```
👤 User: /parallel 3 call_evolve

🤖 AIKG:
🔄 从当前节点 [node_006] 创建 3 个分叉并行演进...

✅ 并行执行完成: call_evolve × 3

📊 结果摘要:

1. node_011 ✅
   优化: 减少 bank conflict
   性能: 82% (+17%)

2. node_012 ✅
   优化: 优化 warp 利用率
   性能: 88% (+23%) 🏆

3. node_013 ❌
   优化: 激进优化（编译失败）
   性能: -

🏆 推荐节点: node_012 (性能: 88%)

💡 使用 /trace switch 12 切换到最优节点
```

---

## 六、命令总结

| 命令 | 功能 | 示例 |
|------|------|------|
| `/trace show` | 显示整个树 | `/trace show` |
| `/trace show <node>` | 显示指定节点详情 | `/trace show 3` |
| `/trace switch <node>` | 切换到指定节点 | `/trace switch 2` |
| `/trace compare <n1> <n2>` | 对比两个节点路径 | `/trace compare 3 6` |
| `/trace path <node>` | 显示到节点的路径 | `/trace path 6` |
| `/parallel N <action>` | 并行执行创建分叉 | `/parallel 4 call_design` |

### Shell 命令

| 命令 | 功能 | 示例 |
|------|------|------|
| `akg_cli op --task "..."` | 启动新任务 | `akg_cli op --task "实现 softmax"` |
| `akg_cli op continue <task_id>` | 继续任务 | `akg_cli op continue task_001` |

---

## 七、核心实现类

### 7.1 TraceSystem 类

```python
class TraceSystem:
    """Trace 树管理器"""
    
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.base_dir = Path.home() / ".aikg" / "conversations" / task_id
        self.trace_file = self.base_dir / "trace.json"
        self.current_node_file = self.base_dir / "current_node.txt"
        self.nodes_dir = self.base_dir / "nodes"
        
    def add_node(self, action: Dict, result: Dict) -> str:
        """在当前节点添加子节点"""
        current_node_id = self.get_current_node()
        
        # 1. 生成新节点 ID
        new_node_id = self._generate_node_id()
        
        # 2. 创建节点目录
        node_dir = self.nodes_dir / new_node_id
        node_dir.mkdir(parents=True, exist_ok=True)
        
        # 3. 从父节点复制状态
        parent_state = self._get_node_state(current_node_id)
        self._save_node_state(new_node_id, parent_state)
        
        # 4. 保存本节点的 action（增量保存）⭐
        actions_dir = node_dir / "actions"
        actions_dir.mkdir(exist_ok=True)
        
        action_history_fact = {
            "node_id": new_node_id,
            "parent_node_id": current_node_id,
            "actions": [
                {
                    "action_id": self._generate_action_id(),
                    "tool_name": action["type"],
                    "arguments": action.get("params", {}),
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                }
            ]
        }
        
        with open(actions_dir / "action_history_fact.json", "w") as f:
            json.dump(action_history_fact, f, indent=2)
        
        # 5. 创建新节点
        new_node = {
            "node_id": new_node_id,
            "parent_id": current_node_id,
            "state_snapshot": {...},
            "action": action,
            "result": result,
            "timestamp": datetime.now().isoformat(),
            "children": [],
            "metrics": {...}
        }
        
        # 6. 更新 trace.json
        trace = self._load_trace()
        trace["tree"][new_node_id] = new_node
        trace["tree"][current_node_id]["children"].append(new_node_id)
        trace["current_node"] = new_node_id
        self._save_trace(trace)
        
        # 7. 更新 current_node.txt
        self.current_node_file.write_text(new_node_id)
        
        return new_node_id
    
    def switch_node(self, node_id: str):
        """切换到指定节点"""
        # 1. 验证节点存在
        if not self._node_exists(node_id):
            raise ValueError(f"节点 {node_id} 不存在")
        
        # 2. 更新 current_node.txt
        self.current_node_file.write_text(node_id)
        
        # 3. 更新 trace.json 中的 current_node
        trace = self._load_trace()
        trace["current_node"] = node_id
        self._save_trace(trace)
    
    def get_current_node(self) -> str:
        """获取当前节点 ID"""
        return self.current_node_file.read_text().strip()
    
    def get_node_state(self, node_id: str) -> Dict:
        """获取指定节点的状态"""
        state_file = self.nodes_dir / node_id / "state.json"
        return json.loads(state_file.read_text())
    
    def get_path_to_node(self, node_id: str) -> List[str]:
        """获取从 root 到指定节点的路径"""
        path = []
        current = node_id
        trace = self._load_trace()
        
        while current != "root":
            path.append(current)
            current = trace["tree"][current]["parent_id"]
        
        path.append("root")
        path.reverse()
        
        return path
    
    def get_full_action_history(self, node_id: str) -> List[Dict]:
        """
        获取从 root 到指定节点的完整 action history
        
        每个 node 只保存自己的 action（增量），此方法沿 parent 链回溯重建完整历史
        """
        # 1. 获取路径
        path = self.get_path_to_node(node_id)
        
        # 2. 沿路径收集所有 action
        full_history = []
        for node in path:
            if node == "root":
                continue
            
            # 读取该 node 的 action_history_fact
            actions_file = self.nodes_dir / node / "actions" / "action_history_fact.json"
            if actions_file.exists():
                with open(actions_file, "r") as f:
                    actions_data = json.load(f)
                    full_history.extend(actions_data["actions"])
        
        return full_history
    
    def compare_nodes(self, node_1: str, node_2: str) -> Dict:
        """对比两个节点的路径"""
        trace = self._load_trace()
        
        # 1. 获取两个路径
        path_1 = self.get_path_to_node(node_1)
        path_2 = self.get_path_to_node(node_2)
        
        # 2. 找到分叉点
        fork_point = None
        for i in range(min(len(path_1), len(path_2))):
            if path_1[i] != path_2[i]:
                fork_point = path_1[i-1] if i > 0 else "root"
                break
        
        # 3. 计算每条路径的指标
        metrics_1 = self._calculate_path_metrics(path_1, trace)
        metrics_2 = self._calculate_path_metrics(path_2, trace)
        
        return {
            "path_1": path_1,
            "path_2": path_2,
            "fork_point": fork_point,
            "metrics_1": metrics_1,
            "metrics_2": metrics_2
        }
    
    def visualize_tree(self) -> str:
        """可视化 Trace 树"""
        trace = self._load_trace()
        current_node = trace["current_node"]
        
        # 从 root 开始递归构建树
        tree_str = self._build_tree_str("root", trace, current_node, indent=0)
        
        return f"🌳 Trace Tree (当前: {current_node}):\n\n{tree_str}"
    
    def _build_tree_str(self, node_id: str, trace: Dict, current_node: str, indent: int) -> str:
        """递归构建树的字符串表示"""
        node = trace["tree"][node_id]
        
        # 构建当前节点的字符串
        prefix = "  " * indent
        is_current = "⭐" if node_id == current_node else ""
        
        if node_id == "root":
            node_str = f"{prefix}root\n"
        else:
            action_type = node["action"]["type"]
            result_summary = self._summarize_result(node["result"])
            node_str = f"{prefix}[{node_id.split('_')[1]}] {action_type} {is_current}\n"
            node_str += f"{prefix}    → {result_summary}\n"
        
        # 递归处理子节点
        children = node.get("children", [])
        if len(children) == 0:
            return node_str
        elif len(children) == 1:
            node_str += f"{prefix}  ↓\n"
            node_str += self._build_tree_str(children[0], trace, current_node, indent)
        else:
            # 多个子节点，显示分叉
            for i, child in enumerate(children):
                if i == 0:
                    node_str += f"{prefix}  ↓\n"
                    node_str += f"{prefix}  ├─ "
                else:
                    node_str += f"{prefix}  │\n"
                    node_str += f"{prefix}  └─ "
                
                node_str += self._build_tree_str(child, trace, current_node, indent + 2)
        
        return node_str
    
    def create_parallel_forks(self, n: int, action: str, params: Dict) -> List[str]:
        """从当前节点创建 N 个分叉"""
        current_node = self.get_current_node()
        fork_nodes = []
        
        for i in range(n):
            # 不切换节点，直接创建子节点
            new_node_id = self._generate_node_id()
            
            # 创建节点（但不设置为 current_node）
            # ...
            
            fork_nodes.append(new_node_id)
        
        return fork_nodes
```

---

## 八、典型用户流程（完整示例）

```bash
# 1. 启动任务
$ akg_cli op --task "实现 softmax"

🌳 创建任务: task_001
📍 当前节点: root
📝 分析需求中...
```

进入 AIKG 对话界面：

```
👤 User: 设计方案

🤖 AIKG: ✅ call_designer 执行完成
         📝 设计方案: Block Reduce + Warp Shuffle
         📍 当前节点: node_001

---

👤 User: 实现代码

🤖 AIKG: ✅ call_coder 执行完成
         📝 生成代码: 120 行
         📍 当前节点: node_002

---

👤 User: 验证性能

🤖 AIKG: ✅ verify_tool 执行完成
         ⚡ 性能: 65%
         📍 当前节点: node_003

---

👤 User: /trace show

🤖 AIKG:
🌳 Trace Tree (当前: node_003):

root
  ↓
[1] call_designer
    → 策略: Block Reduce + Warp Shuffle
  ↓
[2] call_coder
    → 代码: 120 行
  ↓
[3] verify_tool ⭐ 当前
    → 性能: 65%

---

👤 User: /trace switch 2

🤖 AIKG: ✅ 已切换到节点 [2] call_coder
         📝 当前代码: 120 行
         📊 子节点: 1 个 (node_003)

---

👤 User: 使用 shared memory 优化

🤖 AIKG: 🌿 创建新分叉: node_004 (node_002 的第 2 个子节点)
         ✅ call_coder 完成
         📝 生成代码: 150 行
         📍 当前节点: node_004

---

👤 User: 验证性能

🤖 AIKG: ✅ verify_tool 完成
         ⚡ 性能: 85% ✅
         📍 当前节点: node_005

---

👤 User: /trace show

🤖 AIKG:
🌳 Trace Tree (当前: node_005):

root
  ↓
[1] call_designer
    → 策略: Block Reduce + Warp Shuffle
  ↓
[2] call_coder
    → 代码: 120 行
  ↓
  ├─ [3] verify_tool
  │   → 性能: 65%
  │
  └─ [4] call_coder (SM 优化) ✨ 新分叉
      → 代码: 150 行
      ↓
      [5] verify_tool ⭐ 当前
      → 性能: 85% ✅

---

👤 User: /trace compare 3 5

🤖 AIKG:
📊 路径对比: node_003 vs node_005

路径 A (root → 1 → 2 → 3):
  • 步数: 3
  • Token: 3700
  • 性能: 65%

路径 B (root → 1 → 2 → 4 → 5):
  • 步数: 4
  • Token: 5000
  • 性能: 85% ✅

差异:
  • 分叉点: node_002
  • 路径 A: 直接验证
  • 路径 B: SM 优化后验证

---

👤 User: 保存最终代码

🤖 AIKG: ✅ 代码已保存到 ~/.akg/conversations/task_001/nodes/node_005/code/
```

---

## 九、实现优先级

### P0（必须，第一阶段）
- [x] `trace.json` 树结构设计
- [x] `nodes/` 目录组织
- [x] `TraceSystem.add_node()` - 添加子节点
- [x] `TraceSystem.get_current_node()` - 获取当前节点
- [x] `/trace show` - 查看树结构
- [x] `TraceSystem.switch_node()` - 切换节点

### P1（重要，第二阶段）
- [ ] `TraceSystem.get_path_to_node()` - 获取路径
- [ ] `TraceSystem.compare_nodes()` - 对比节点
- [ ] `/trace switch` - CLI 切换节点
- [ ] `/trace path` - 显示路径
- [ ] `akg_cli op continue` - 继续任务

### P2（增强，第三阶段）
- [ ] `/trace compare` - 对比节点
- [ ] `/parallel` - 并行探索

### P3（未来）
- [ ] Web UI 可视化树
- [ ] 自动评分（自动选择最优节点）

---

**总结**:
- **单一树结构**：只有一个 Trace Tree，没有 "branch" 的概念
- **节点分叉**：一个节点可以有多个子节点，形成分叉
- **节点切换**：用户通过 `/trace switch <node>` 切换
- **自动分叉**：在有子节点的节点上执行新操作，自动创建新分叉
- **路径对比**：对比两个节点从 root 到该节点的路径
- **文件组织**：`nodes/` 目录，每个节点独立存储状态

**关键优势**:
1. **概念简洁**：只有 node 和 tree，没有 branch
2. **自然分叉**：树的分叉就是"多分支"
3. **灵活探索**：随时切换到任意节点继续
4. **路径对比**：清晰对比不同路径的效果

**下一步**: 实现 P0 优先级功能
