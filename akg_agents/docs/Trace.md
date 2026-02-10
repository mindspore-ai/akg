# TraceSystem Design Document

## Overview
TraceSystem is a tree-based inference tracing system in AKG Agents, responsible for completely recording all operation traces during the AI Kernel generation process. It supports multi-fork exploration, node switching, incremental action history, and checkpoint resume, with persistence implemented via `FileSystemState`.

## Architecture Overview

```
core_v2/filesystem/
├── trace_system.py         # TraceSystem main class
├── state.py                # FileSystemState file system state management
├── models.py               # Data model definitions
├── compressor.py           # ActionCompressor action compression
└── exceptions.py           # Custom exceptions

Storage directory structure:
~/.akg_agents/conversations/{task_id}/
├── trace.json              # Trace tree structure
├── current_node.txt        # Current active node
├── nodes/
│   ├── root/
│   │   └── state.json
│   ├── node_001/
│   │   ├── state.json                  # Node state snapshot
│   │   ├── thinking.json               # Thinking process
│   │   ├── actions/
│   │   │   ├── action_history_fact.json      # Incremental action history
│   │   │   ├── action_history_compressed.json # Compressed action history
│   │   │   └── pending_tools.json            # Pending tools
│   │   ├── code/                       # Code files
│   │   └── system_prompts/             # System prompts
│   └── ...
└── logs/
```

## Core Concepts

### Trace Tree (TraceTree)
The entire trace record is a single tree — there is no concept of "branches". Each node can have multiple children, forming forks. Stored in `trace.json`.

### Node (TraceNode)
Each node in the tree records:
- `node_id`: Unique identifier (e.g., `root`, `node_001`)
- `parent_id`: Parent node
- `action`: Executed action
- `result`: Execution result
- `metrics`: Metric data (token usage, duration, performance, etc.)
- `state_snapshot`: State snapshot (turn, status)
- `children`: List of child nodes

### Action Record (ActionRecord)
Record of a single tool call:
- `action_id`: Unique action identifier
- `tool_name`: Tool name
- `arguments`: Call arguments
- `result`: Execution result
- `duration_ms`: Execution duration

### Incremental Action History (ActionHistoryFact)
Each node only stores its **own new** actions (incremental), avoiding duplicate storage. Complete history is reconstructed by tracing back along the parent chain.

### Compressed Action History (ActionHistoryCompressed)
LLM-compressed history summary for reducing token consumption. Includes a caching mechanism with `source_path` validation for cache validity.

## Core API

### Initialization
```python
from akg_agents.core_v2.filesystem.trace_system import TraceSystem

# Create TraceSystem
trace = TraceSystem(task_id="my_task", base_dir="~/.akg_agents")

# Initialize (creates trace.json and root node)
trace.initialize(task_input="User request content")

# Checkpoint resume (automatically loads existing trace)
trace.initialize()  # Loads if trace already exists
```

### Node Operations
```python
# Get current node
current = trace.get_current_node()  # → "root" or "node_001"

# Add child node (automatically created under current node)
node_id = trace.add_node(
    action={"type": "call_kernel_gen", "arguments": {...}},
    result={"status": "success", "code": "..."},
    metrics={"token_used": 1500, "duration_ms": 3200}
)

# Update node result
trace.update_node_result(
    node_id="node_001",
    result={"status": "success", "performance": 0.95},
    metrics={"token_used": 500}
)

# Mark node completed/failed
trace.mark_node_completed("node_001", metrics={"performance": 0.95})
trace.mark_node_failed("node_001", error="Verification failed: accuracy below threshold")

# Switch to a specific node
trace.switch_node("node_001")
```

### Path and History
```python
# Get path from root to node
path = trace.get_path_to_node("node_003")  # → ["root", "node_001", "node_002", "node_003"]

# Get full action history (reconstructed by tracing parent chain)
history = trace.get_full_action_history("node_003")

# Get compressed history for LLM (with caching)
compressed = await trace.get_compressed_history_for_llm(
    llm_client=client,
    node_id="node_003",
    max_tokens=2000
)
```

### Parallel Forks
```python
# Create N parallel fork nodes
fork_nodes = trace.create_parallel_forks(
    n=3,
    action_template={"type": "parallel_gen"}
)
# → ["node_004", "node_005", "node_006"]

# Complete a fork node
trace.complete_fork(
    node_id="node_004",
    result={"status": "success", "code": "..."},
    metrics={"performance": 0.92}
)
```

### Node Comparison
```python
# Compare paths of two nodes
comparison = trace.compare_nodes("node_003", "node_005")
# → {
#     "path_1": [...], "path_2": [...],
#     "fork_point": "node_001",
#     "metrics_1": {"steps": 3, "total_token": 4500, ...},
#     "metrics_2": {"steps": 2, "total_token": 3000, ...}
# }

# Get best leaf node
best = trace.get_best_leaf_node(metric="performance")
```

### Visualization
```python
# Print tree string representation
print(trace.visualize_tree())

# Get node detail
print(trace.get_node_detail("node_001"))

# Get path detail
print(trace.get_path_detail("node_003"))
```

### Task Resume
```python
# Get resume information (for checkpoint resume)
resume_info = trace.get_resume_info()
# → {
#     "task_id": "my_task",
#     "current_node": "node_003",
#     "state": NodeState(...),
#     "action_history": [...],
#     "pending_tools": PendingToolsState(...),
#     "thinking": ThinkingState(...),
#     "path": ["root", "node_001", "node_003"]
# }
```

## Data Models

### NodeState (Node State Snapshot)
Corresponds to `nodes/{node_id}/state.json`:

| Field | Type | Description |
|-------|------|-------------|
| node_id | str | Node ID |
| turn | int | Turn number |
| status | str | Status: `init` / `running` / `completed` / `failed` |
| agent_info | dict | Agent information |
| task_info | dict | Task information |
| execution_info | dict | Execution information |
| file_state | dict | File state (path → FileInfo) |
| metrics | dict | Metric data |

### ThinkingState (Thinking Process)
Corresponds to `nodes/{node_id}/thinking.json`:

| Field | Type | Description |
|-------|------|-------------|
| node_id | str | Node ID |
| turn | int | Turn number |
| current_plan | dict | Current plan |
| latest_thinking | str | Latest thinking |
| decision_history | list | Decision history |

### PendingToolsState (Pending Tools)
Corresponds to `nodes/{node_id}/actions/pending_tools.json`, used for checkpoint resume:

| Field | Type | Description |
|-------|------|-------------|
| node_id | str | Node ID |
| turn | int | Turn number |
| pending_tools | list | List of pending tools |

## FileSystemState

`FileSystemState` is the underlying file system state manager. All persistence operations go through it. Core principles:
1. **File System as Source of Truth**: All state is persisted to the file system
2. **Human Readable**: Uses JSON/Text formats for easy debugging and auditing
3. **Incremental Saving**: Avoids duplication, saves space
4. **Framework Agnostic**: No dependency on specific frameworks, easy to migrate
5. **Checkpoint Resume Support**: Saves sufficient information for execution recovery

## Comparison with Old Trace

| Feature | Old Trace | New TraceSystem |
|---------|-----------|-----------------|
| Data Structure | Linear file list | Tree structure (`TraceTree`) |
| Storage | Flat files | Structured directories + JSON |
| Fork Support | Not supported | Multi-fork exploration |
| Checkpoint Resume | Not supported | Fully supported |
| History Management | Full copy | Incremental + compressed |
| Node Switching | Not supported | Switch to any node |
| Naming Convention | `I{task_id}_S{step}_{action}_*.txt` | `nodes/{node_id}/` directory structure |
| Parallel Exploration | Not supported | Parallel forking supported |
| State Management | None | `FileSystemState` full management |

## Related Documentation
- [Workflow System](./Workflow.md)
- [Skill System Documentation](./SkillSystem.md)
