[中文版](./CN/Trace.md)

# Trace System

## 1. Overview

The Trace System provides tree-based inference tracing for AKG Agents. It records the entire reasoning and execution process of agents, supporting:

- **Tree structure**: Single tree with multi-fork exploration
- **Node switching**: Navigate to any node in the trace tree
- **Incremental history**: Action records accumulated per node
- **Checkpoint resume**: Persist state and resume from any point
- **Visualization**: Text and Rich terminal visualization
- **Fork & merge**: Fork at ask_user nodes, merge branches, parallel exploration
- **Blame**: Track which node introduced each line of generated code

## 2. Core Concepts

### Trace Tree

The entire trace is a single tree — no separate "branch" concept. A node can have multiple children (forks). When a new action is performed on a node that already has children, a new child node is automatically created.

### Data Models

| Model | Description |
|-------|-------------|
| `TraceTree` | Root data structure containing all nodes and metadata |
| `TraceNode` | A single node in the tree (agent info, task info, state, children) |
| `NodeState` | Per-node state snapshot |
| `ActionRecord` | A single action record (tool call + result) |
| `ActionHistoryFact` | Factual action history accumulated per node |
| `ActionHistoryCompressed` | Compressed action history for LLM context |
| `AgentInfo` | Agent metadata (name, id) |
| `TaskInfo` | Task metadata (task_id, input, domain, custom metadata) |
| `ExecutionInfo` | Execution counters (tool calls, turns) |
| `FileInfo` | File metadata for code snapshots |
| `Metrics` | Performance and quality metrics for a node |
| `ThinkingState` | Plan and decision state (thinking.json) |
| `PlanState` | Structured plan state within ThinkingState |
| `DecisionRecord` | A single thinking/decision record |
| `PendingTool` | A pending tool call awaiting completion |
| `PendingToolsState` | Collection of pending tools (pending_tools.json) |

## 3. TraceSystem

`TraceSystem` is the main entry point for trace management.

### Constructor

```python
trace = TraceSystem(task_id="my_task", base_dir="~/.akg")
```

### Initialization

```python
# Initialize (creates trace.json and root node)
trace.initialize(task_input="Generate a relu kernel")

# Force re-initialization (overwrites existing trace)
trace.initialize(task_input="Generate a relu kernel", force=True)

# Resume existing trace (auto-loads if trace.json exists)
trace.initialize(force=False)
```

### Node Operations

```python
# Add a new node (returns node_id)
node_id = trace.add_node(
    action={"type": "verify_kernel", "arguments": {"op_name": "relu"}},
    result={"status": "success", "output": "..."},
    metrics={"performance": 1.5},          # optional
    state_snapshot={"iteration": 3},       # optional
)

# Get current node ID
current = trace.get_current_node()

# Get a node object
node = trace.get_node(node_id)

# Switch to a different node
trace.switch_node(node_id)

# Get path from root to a node
path = trace.get_path_to_node(node_id)

# Get node depth in the tree
depth = trace.get_node_depth(node_id)
```

### Action History

```python
# Get full action history for a node (includes all ancestors)
history = trace.get_full_action_history(node_id)

# Get compressed history for LLM context (async, requires LLM client)
compressed = await trace.get_compressed_history_for_llm(
    llm_client, node_id, max_tokens=2000
)
```

### Fork & Merge

```python
# Fork at an ask_user node (creates new child with same question, empty answer)
new_node_id = trace.fork_ask_user(node_id)

# Create parallel exploration forks
fork_ids = trace.create_parallel_forks(n=3, action_template={"type": "explore"})

# Complete a fork with results
trace.complete_fork(fork_id, result={"status": "success"}, metrics={...})

# Merge two branches (three-way merge of code files)
merged_node_id = trace.merge_nodes(target_node_id, source_node_id)
```

### Node Status

```python
# Mark a node as completed
trace.mark_node_completed(node_id, metrics={"performance": 1.5})

# Mark a node as failed
trace.mark_node_failed(node_id, error="Verification timeout")

# Update a node's result
trace.update_node_result(node_id, result={...}, metrics={...})
```

### Tree Queries

```python
# Get all leaf nodes
leaves = trace.get_all_leaf_nodes()

# Get the best leaf node by a metric
best = trace.get_best_leaf_node(metric="performance")

# Compare two nodes
diff = trace.compare_nodes(node_1, node_2)

# Find lowest common ancestor
lca = trace.find_lca(node_id1, node_id2)
```

### Code Blame

```python
# Blame: track which node introduced each line
blame_info = trace.blame_file(node_id, "kernel.py")

# Blame all files at a node
all_blame = trace.blame_all_files(node_id)
```

### Resume

```python
# Get resume info for checkpoint recovery
resume_info = trace.get_resume_info()
```

## 4. FileSystemState

`FileSystemState` handles persistence of trace data to the filesystem.

```
~/.akg/tasks/{task_id}/
├── trace.json              # Trace tree structure
├── .traceconfig            # Trace configuration
└── nodes/
    ├── root/
    │   ├── state.json      # Node state
    │   ├── result.json     # Action results
    │   ├── action_history_fact.json
    │   ├── thinking.json   # Plan/decision state
    │   ├── pending_tools.json
    │   ├── code/           # Code snapshots (CoW)
    │   ├── logs/           # Verification logs, etc.
    │   └── system_prompts/ # Per-turn system prompts
    ├── node_001/
    │   ├── state.json
    │   ├── result.json
    │   └── ...
    └── ...
```

### Key Methods

| Method | Description |
|--------|-------------|
| `save_node_state(node_id, state)` | Persist node state to `state.json` |
| `load_node_state(node_id)` | Load node state from `state.json` |
| `append_action(node_id, action)` | Append an action record to the node's history |
| `save_code_file(node_id, filename, content)` | Save a code file snapshot (CoW) |
| `load_code_file(node_id, filename)` | Load a code file from a node's snapshot |
| `diff_nodes(node_a, node_b)` | Generate a diff between two nodes' code |
| `copy_node_state(from_node, to_node)` | Copy state between nodes |
| `save_system_prompt(node_id, turn, prompt)` | Save per-turn system prompt |

## 5. ActionCompressor

`ActionCompressor` compresses action history to fit within LLM context windows. It summarizes older actions while preserving recent ones in full detail.

```python
from akg_agents.core_v2.filesystem import ActionCompressor

compressor = ActionCompressor(llm_client)
compressed = await compressor.compress_history(action_history, max_tokens=4000)
```

> Note: `ActionCompressor` requires an `LLMClient` instance for summarization.

## 6. Visualization

### Text Visualization

```python
from akg_agents.core_v2.filesystem.trace_visualizer import visualize_text

text = visualize_text(trace, focus_node="node_005", depth=4)
print(text)
```

### Rich Terminal Visualization

```python
from akg_agents.core_v2.filesystem.trace_visualizer import visualize_rich

rich_text = visualize_rich(trace, focus_node="node_005", depth=4)
```

### Node Detail

```python
from akg_agents.core_v2.filesystem.trace_visualizer import format_node_detail_rich

detail = format_node_detail_rich(trace, "node_003")
```

## 7. CLI: /trace Command

In `akg_cli` interactive mode, the `/trace` slash command (alias: `/t`) provides trace tree inspection and forking capabilities.

### Usage

```
/trace [<id>|root|show <id>|node <id>|history|fork <id>] [-n depth]
```

### Subcommands

| Subcommand | Example | Description |
|------------|---------|-------------|
| (none) | `/trace` | Show path view centered on current node (default depth: 4) |
| `root` | `/trace root` | Show tree from root node downward |
| `<id>` | `/trace 005` | Show path view centered on the specified node (`005` is auto-expanded to `node_005`) |
| `-n <N>` | `/trace -n 8` | Set display depth to N |
| `show <id>` | `/trace show 003` | Show detailed info for a node (action, arguments, result) |
| `node <id>` | `/trace node 003` | Same as `show` |
| `history` | `/trace history` | Show full action history of the current node |
| `fork <id>` | `/trace fork 005` | Fork at an `ask_user` node to provide a different answer |

### Fork Behavior

`/trace fork <id>` only works on `ask_user` type nodes. When executed:

1. Copies the original ask_user node's action (the question), clears the user response
2. Creates a new child node under the same parent
3. Updates the agent's current node and action history
4. Re-displays the original question and waits for a new answer

### Examples

```bash
# View the trace tree
/trace

# View from root with depth 8
/trace root -n 8

# Inspect node 003
/trace show 003

# Fork at node 005 to try a different answer
/trace fork 005
```

## 8. Exceptions

| Exception | Description |
|-----------|-------------|
| `FileSystemStateError` | Base exception for filesystem state errors |
| `NodeNotFoundError` | Node does not exist in the trace |
| `TraceSystemError` | General trace system error |
| `InvalidNodeStateError` | Node state is invalid |
| `TraceNotInitializedError` | Trace has not been initialized |
| `TraceAlreadyExistsError` | Trace already exists (when force=False) |
