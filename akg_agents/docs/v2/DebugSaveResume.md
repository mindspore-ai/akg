[中文版](./CN/DebugSaveResume.md)

# Debug Save/Resume Design

## 1. Background

AKG Agents orchestrates operator generation with LangGraph. A single task may include design, code generation, static checks, verification, profiling, and several repair rounds. Final logs alone are not enough for debugging; developers need to:

- continue from completed nodes after an interruption
- persist a failing state for later reproduction
- keep the implementation aligned with LangGraph's native save/resume contract

The Debug module keeps two responsibilities separate:

- LangGraph checkpointing saves and resumes workflow state
- AKG Trace / WorkflowLogger / ReplayGuard records readable logs, node outputs, and LLM replay data

## 2. Goals

- Disabled by default, with no behavior change for normal tasks
- Supports `LangGraphTask`, KernelAgent workflow tool execution, and `OpTaskBuilderWorkflow`
- Uses LangGraph checkpointers and `configurable.thread_id`
- Provides a local file-backed saver so developer environments do not need database packages
- Fails fast when `resume` is requested but no checkpoint exists

## 3. Configuration

Recommended configuration:

```yaml
debug:
  enabled: true
  resume: false
  session_id: "relu-debug-001"
  checkpoint_dir: "~/.akg/langgraph_checkpoints"
```

Resume from a checkpoint:

```yaml
debug:
  enabled: true
  resume: true
  session_id: "relu-debug-001"
  checkpoint_dir: "~/.akg/langgraph_checkpoints"
```

Flat keys are also supported for CLI and tests:

```yaml
debug_enabled: true
debug_resume: false
debug_session_id: "relu-debug-001"
debug_checkpoint_dir: "~/.akg/langgraph_checkpoints"
```

| Field | Description |
|---|---|
| `debug.enabled` | Enable checkpoint saving |
| `debug.resume` | Continue from an existing checkpoint |
| `debug.session_id` / `debug.thread_id` | LangGraph `thread_id`; use the same value for save and resume |
| `debug.checkpoint_dir` | Checkpoint directory |
| `debug.checkpoint_file` | Explicit checkpoint file |

Default checkpoint path:

```text
{checkpoint_dir}/{workflow_name}_{thread_id}.pkl
```

## 4. Save Flow

1. `BaseWorkflow.compile()` checks `debug.enabled/debug.resume`
2. It creates a `FileCheckpointSaver` when debug checkpointing is enabled
3. `StateGraph.compile(checkpointer=..., debug=True)` uses LangGraph's native checkpoint mechanism
4. The invoke config injects the required `thread_id`:

```python
{
    "recursion_limit": ...,
    "configurable": {"thread_id": "..."}
}
```

5. Every checkpoint write flushes the in-memory LangGraph saver state to a local pickle file with atomic replace

## 5. Resume Flow

1. Set `debug.resume: true`
2. Reuse the same `thread_id` and checkpoint path
3. Before execution, call `app.aget_state(invoke_config)` to verify saved state exists
4. If no state exists, fail immediately instead of silently rerunning from the beginning
5. If state exists, call `app.ainvoke(None, config=invoke_config)` so LangGraph resumes from the checkpoint

The implementation covers:

- `BaseLangGraphTask.run()`
- `LangGraphTask.run()`
- `ToolExecutor._execute_workflow()`
- `OpTaskBuilderWorkflow.run()`

## 6. Trace And Replay

Checkpointing saves LangGraph state and answers "where should execution continue?"

Trace / WorkflowLogger saves readable logs and node artifacts, answering "what happened?"

ReplayGuard / LLM cache helps reproduce LLM outputs. When checkpoint debug is enabled, `LangGraphTask` does not put `_replay_guard` into state, avoiding non-serializable objects in checkpoints.

These mechanisms are complementary rather than interchangeable.

## 7. Failure Behavior

- Corrupted checkpoint file: raises `RuntimeError` with the file path
- `debug.resume=true` without saved state: fails fast
- Debug disabled: keeps the original `graph.compile()` and `ainvoke()` behavior
- The file saver uses pickle and is only intended for trusted local debug artifacts

## 8. Acceptance

Run the focused test:

```bash
PYTHONPATH=/workspace/akg/akg_agents/python \
pytest -q akg_agents/tests/ut/test_langgraph_debug_checkpoint.py
```

Expected behavior:

- the first run creates a checkpoint file
- a new graph instance can load the saved state
- `ainvoke(None, config=...)` resumes from that checkpoint
