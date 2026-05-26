# AutoResearch

AutoResearch is the kernel-optimization agent: given a reference implementation
and a seed kernel, it iteratively edits the kernel and measures each edit
against an eval budget, steered by a structured plan and a skill catalogue.
All state (plan, skills, counters, op_summary.md, plan_analysis.md) is
persisted so a run is resumable.

This document is the design contract for the module at
`python/akg_agents/op/autoresearch/`. It describes the runtime architecture,
the data that flows between components, and the invariants that keep the
loop terminating and the context bounded. Implementation details that a
grep-able code base answers better (exact function bodies, config defaults)
are intentionally out of scope.

---

## 1. Contract

| | |
|---|---|
| Tool name | `call_autoresearch_workflow` |
| Scenario | Iterative optimization of an existing kernel with a known reference |
| Inputs | `op_name`, `task_desc` (reference impl as `Model/get_inputs/get_init_inputs`), `dsl`, `framework`, `backend`, `arch`, optional `previous_code`, optional `max_rounds` (default 20) |
| Output | Optimized kernel + profile result (latency, speedup vs reference) |
| Halt | Eval budget exhausted, terminal LLM error, compact-recovery failure, or explicit `finish` tool call |

Accepted parameter values are listed in the top-level `AGENTS.md` (backend /
dsl / arch enums). Preflight fails closed if validation fails; see §3.

### 1.1 When to use it

AutoResearch is the deep-iterative-optimization lane. Pick a different
workflow when:

| Feature | KernelGenOnly | Evolve / AdaptiveSearch | **AutoResearch** |
|---|---|---|---|
| Strategy | single-shot | population-based | agent-driven ReAct |
| Iterations | 1 | many (parallel) | many (sequential) |
| Autonomy | none | template-driven | full: plans, reads docs, self-diagnoses |
| Best for | first draft | wide exploration | deep tuning |
| Failure recovery | none | population absorbs | diagnose subagent + agent self-acknowledgement |

Rule of thumb: use KernelGen for the first draft, Evolve/AdaptiveSearch
to fan out in parallel, and AutoResearch once you have a correct kernel
and want an agent to push on it.

---

## 2. Architecture

```text
                 call_autoresearch_workflow
                             │
                             ▼
     ┌──────────────────── AgentLoop ────────────────────┐
     │                                                    │
     │   [Preflight] ─► [Baseline eval] ─► [Turn loop]   │
     │                                                    │
     │       ┌───────────── Turn loop ──────────────┐    │
     │       │                                       │    │
     │   LLM call ─► TurnExecutor ─► post-eval hook │    │
     │                    │                          │    │
     │                    ├── tools ─► edit / plan / │    │
     │                    │           skill search / │    │
     │                    │           compact / fin  │    │
     │                    │                          │    │
     │                    └── eval (quick_check →    │    │
     │                           run eval script →   │    │
     │                           git commit if KEEP) │    │
     │       │                                       │    │
     │       └── on failure / no_edit / threshold ─► │    │
     │                  Diagnose subagent             │    │
     └────────────────────────────────────────────────────┘

State owners (single writer per artefact):
  • plan.md             ← FeedbackBuilder
  • session.json        ← SessionStore (snapshot of counters + plan + skill)
  • messages_*.jsonl    ← ConversationBuffer
  • op_summary.md       ← compress.auto_compact (LLM #1; force_rebuild writes fallback)
  • plan_analysis.md    ← compress.auto_compact (LLM #2; force_rebuild writes fallback)
  • ranking.md / log.jsonl / perf_log.md / report.md ← RoundLogger / Runner
```

The key structural rule: **one component owns each piece of state**. The LLM
never writes plan.md directly; it submits `update_plan(...)` and the
`FeedbackBuilder` is the single validator and writer. The LLM never assigns
a `backing_skill`; it emits `keywords` and the `SkillPool` + `TurnExecutor`
do the matching. This keeps trust boundaries explicit.

### 2.1 Component map

| Module | Role |
|---|---|
| `agent/loop.py` | AgentLoop — preflight, baseline, turn loop, post-eval hooks, session save on every turn |
| `agent/turn.py` | TurnExecutor — one turn: LLM → tool dispatch → eval → feedback |
| `agent/feedback.py` | Plan state machine + plan.md writer + feedback assembly |
| `agent/skill_pool.py` | Ranked candidate skill list; refill / match / reference-match |
| `agent/skill_builder.py` | Per-skill registry (selected / active / applied / previously unbound; no terminal state) |
| `agent/skill_adapter.py` | Catalogue filter, keyword generation, keyword-based ranking |
| `agent/skill_rendering.py` | Skill → markdown (index mode vs. full mode) |
| `agent/subagents.py` | Post-eval diagnose subagent |
| `agent/conversation.py` | ConversationBuffer: message list, skill injection, compact hooks |
| `agent/compress.py` | microcompact / auto_compact / force_rebuild, STATE_ATTACHMENT rebuild |
| `agent/session.py` | session.json, heartbeat, resume load with HEAD / dirty guards |
| `agent/counters.py` | RunCounters: eval budget, consecutive-failure bookkeeping |
| `agent/prompt_builder.py` | System prompt and initial user message assembly |
| `framework/runner.py` | ExperimentRunner: quick_check, eval subprocess, rollback, git commit |
| `framework/logger.py` | RoundLogger: log.jsonl, perf_log.md, ranking.md |
| `framework/git_repo.py` | Git snapshot / commit / rollback per round |

---

## 3. Preflight

Fail-closed validation runs before the main loop so a broken setup cannot
consume the eval budget:

1. **Worker acquisition** (mandatory). No worker available → abort.
2. **Reference validation** — static AST check (`Model`, `get_inputs`,
   `get_init_inputs` present with expected signatures) and a runtime
   execution of the reference on the worker. Either fails → abort. The
   reference is the ground truth; a broken reference would invalidate
   every subsequent correctness check.
3. **Seed resolution** — `previous_code` is runtime-verified via
   `KernelVerifier.run()`. If that fails, the path falls through to
   KernelGen (up to `gen_retries` attempts, each gated by CodeChecker
   static analysis and KernelVerifier runtime), errors feeding back to
   KernelGen for the next retry.
4. **Worker release** is in `finally`, so any abort path still returns
   the worker to the pool.

Preflight output feeds the main loop a verified baseline code + baseline
metric. The loop is allowed to start from here.

CodeChecker is used by every workflow that generates kernel code, not
only AutoResearch. Pipeline and YAML policy
(`op/config/code_checker.yaml`) are documented in
[CodeChecker.md](./CodeChecker.md).

---

## 4. Runtime

### 4.1 Startup

`AgentLoop` construction wires the stable deps (LLM client, runner, git
repo, session store) and the per-run state (feedback, skill builder, skill
pool, conversation buffer, counters). After baseline eval commits to git,
the **startup refill** seeds the skill pool concurrently with the baseline
measurement:

```text
SkillPool.refill(mode="replace", include_categories=["guide"])
```

Only guides enter the pool at startup — they are the category the
initial prompt indexes and the default binding surface. Fundamentals
are not pool entries; they reach the agent verbatim in the system
prompt's `## DSL Fundamentals` block, sourced from
`task_dir/skills/<name>/SKILL.md` by `build_system_prompt`. Examples
and cases are intentionally deferred — the agent pulls them on demand
via `search_skills(hint=...)`, or the system auto-widens the pool
when a keyword-requested item's keywords don't match any pool entry
(§6.3).

If the session is a resume, `SkillBuilder.skill_state_from_dict`
restores the registry (v5 format with `applied_versions` /
`unbound_at_versions` badges; v4 terminal-abandoned payloads are
migrated on load), and `_rehydrate_pool_from_plan` pulls in any
`backing_skill` named by the stored plan items that is not already in
the pool after the refill.

### 4.2 Turn loop

A **turn** is one LLM call plus its tool-dispatch fallout:

1. **Context guard** — before the LLM call, `ConversationBuffer`
   auto-compacts if an estimated token count crosses
   `compression_threshold × context_limit`. If a prompt-too-long error
   still fires, `compact_failures` is recorded and the loop escalates
   through auto_compact → force_rebuild → abort at `compact_max_failures`.
2. **Skill injection** — if the active plan item carries a `backing_skill`,
   `ConversationBuffer.inject_backing_skill` appends the matched SKILL.md
   as a plain user message with a marker prefix
   (`[skill auto-injected for <item> v<N> (<name>)]`), dedup-protected by
   `(plan_version, item_id, skill_name)`. On settle the content is
   elided through the same path as voluntary `read_file` of a
   `skills/...` path (§8.4).
3. **LLM call** → tool-call list.
4. **Tool dispatch** (§4.3). After tools run, if any edit occurred, the
   turn enters the **eval path**: quick_check (import + smoke) → full
   eval → KEEP/FAIL/DISCARD decision → git commit on KEEP, rollback on
   DISCARD/FAIL → `FeedbackBuilder.settle_active` → feedback message
   assembled for the next turn.
5. **Post-eval hook** — only `consecutive_failures` crossing a
   multiple of `diagnose_suggest_threshold` triggers the diagnose
   subagent (which in turn forces a replan via `require_replan`).
   `consecutive_no_edit_turns` is handled separately inside
   `TurnExecutor._nudge_if_no_edits` — it appends a warning user
   message but does NOT trigger diagnose.
6. **Session save** — `SessionStore.save` writes counters + plan_state +
   skill_state + last_diagnosis after every turn. Heartbeat is refreshed.

### 4.3 Tools

The LLM sees a stable schema; the trust boundary is `TurnExecutor`.

| Tool | Args | Effect |
|---|---|---|
| `update_plan` | `items: [{text, rationale, keywords?}]` | Submit a new plan. Rationale is validated (length + banned-generic check). Keywords are matched by the pool to assign `backing_skill`. Agent-supplied `backing_skill` is stripped. |
| `read_file` | `path` | Sandboxed read. Returns truncated content with a stale-file hash so subsequent `patch_file` calls fail fast if the file changed. |
| `patch_file` | `path, old_str, new_str` | In-place edit via unique-match replace. Forbidden patterns are checked. **Blocked on bound items until `acknowledge_skill` is called for the active item.** |
| `write_file` | `path, content` | Full file rewrite. Same acknowledge-gate as `patch_file`. |
| `acknowledge_skill` | `plan_item_id, valuable_aspects, kernel_application, applicability` | Required before the first edit on any item whose `backing_skill != None`. Schema-strict: `valuable_aspects` (100–500 chars, skill-level value) + `kernel_application` (100–500 chars, concrete change on THIS kernel — prefer STRUCTURAL edits over parameter tuning); applicability ∈ {apply, unbind}. `unbind` releases the binding for THIS item (item stays active as free exploration) and appends to the skill's `unbound_at_versions`, **but the skill remains bindable** — a later KEEP on the same skill promotes it back to top priority. |
| `search_skills` | `hint: str` | `SkillPool.refill(mode="append")`: keyword pipeline re-runs with the hint, new candidates are appended. No plan mutation. |
| `compact` | — | Agent-initiated auto_compact. |
| `finish` | `summary?` | Explicit termination. |

Illegal tool combinations (e.g. `finish` with pending edits) return a
rejection reply and do not mutate state. Rejection messages are
self-describing so the agent can correct its own call.

### 4.4 Termination

The loop ends on any of:

- Eval budget exhausted (`eval_calls_made >= max_rounds`).
- API call budget exhausted (`total_api_calls >= max_rounds × max_turns_multiplier`).
- Compact failures reach `compact_max_failures`.
- LLM connection error after `llm_max_retries`.
- Explicit `finish` call.

The `best_result` (highest primary_metric under `lower_is_better` /
`higher_is_better`) is written back to the task directory as the final
artefact; the corresponding git commit is the reproducible checkpoint.

---

## 5. Plan state

### 5.1 Shape

A `plan_item` is a dict with fields:

```text
id               "p1", "p2", ...
text             short action description
rationale        validated one-sentence reasoning (required)
status           pending | active | done_ok | done_fail
keywords         list[str] (system-cleaned)
backing_skill    name of bound skill, or None (cleared on `unbind` ack)
skill_ack        recorded acknowledgement once the agent has read the
                 injected SKILL.md: {valuable_aspects, kernel_application,
                 applicability, skill}; absent for unbound items and
                 items not yet acknowledged
sketch           optional code sketch
```

The first pending item is promoted to `active` on plan submission; the
active item is what the agent edits.

### 5.2 Lifecycle

```text
update_plan(items) ──► all items pending; first → active
                         │
                         ▼ (if item.backing_skill != None)
           loop auto-injects SKILL.md via inject_backing_skill
                         │
                         ▼
           agent MUST call acknowledge_skill(item_id,
                          valuable_aspects, kernel_application,
                          applicability)
             │
             ├─ applicability = apply
             │     → skill_ack stored; edits unblocked
             │
             └─ applicability = unbind
                   → SkillBuilder.mark_unbound(backing_skill);
                     item.backing_skill = None; item continues as
                     unbound free exploration. Skill is NOT terminal —
                     it stays bindable (just tier-2 until next KEEP)
                         │
                         ▼ (patch_file / write_file + eval)
                ┌─ KEEP    → done_ok; SkillBuilder.record_applied(bs, v);
                │            _advance() to next pending item
                ├─ FAIL    → done_fail; _advance() to next pending item
                │            (consecutive_failures++ may later trigger
                │            diagnose, but the item itself is already
                │            settled and no longer active)
                └─ DISCARD → done_fail; _advance() to next pending item

diagnose → require_replan ──► failing item tagged
                              "abandoned (diagnose)" (new history
                              entry for pre-eval fails, retroactive
                              retag for post-eval fails + rewind of
                              any promoted-but-unrun pending item);
                              must_replan = true; next update_plan
                              replaces the abandoned item. The
                              backing_skill registry is NOT touched —
                              diagnose only rewinds the plan.
```

`plan_version` increments on every accepted `update_plan`. The
SkillBuilder registry (§6.4) stores `registered_at_version`,
`applied_versions[]`, and `unbound_at_versions[]` for lineage in plan.md
and for the binding-priority tier computation.

### 5.3 plan.md layout

`FeedbackBuilder._persist_plan` writes plan.md on every state change, in
three sections:

```text
# Plan vN
- [status] p1: text  (keywords: ...) (skill: backing_skill)
  ...

## Optimization History
### Plan v(N-k)
- [O]/[X] ... settlement entries with reason + metric
...

## Skill State
### Active (adopting now)
  - [>>>] name (category)  reason / backing items / excerpt
### Applied (pattern adopted successfully)
### Selected (candidates, not yet bound)
### Previously unbound (last-resort tier; still selectable, ranked last)
```

Every skill in the registry lands in exactly one bucket per render:
Active wins when the skill is currently bound to the active item;
otherwise Applied if `applied_versions` is non-empty; otherwise
Previously unbound if `unbound_at_versions` is non-empty; otherwise
Selected. None of these are terminal — Previously unbound is a
demotion signal, not a block. A later KEEP promotes the skill back
to Applied (tier 0) and its unbound history stays as a badge.

The Optimization History grows unboundedly across plan versions. Skill
State is the recovery point: every skill's full history
(`applied_versions`, `unbound_at_versions`) is preserved so a
post-compact prompt can still see "X was unbound at v12, applied at
v17" without replaying the chain.

---

## 6. Skills

### 6.1 Three layers

| Layer | Where | What |
|---|---|---|
| 0 — system prompt | Static preamble | `## DSL Fundamentals` block: full content of every `fundamental`-category SKILL.md under `task_dir/skills/` (greedy-packed under `system_fundamentals_max_chars`, default 20 000). Plus task metadata, context_files (hardware_info etc.), constraints, tool protocol. |
| 1 — `task_dir/skills/<name>/SKILL.md` | On demand | Every guide / example / case SKILL.md the agent may want is mirrored into the task directory at scaffold time. Agent pulls via `read_file('skills/<name>/SKILL.md')`. Content lives in the buffer only while the owning item is active — evicted at settle by `unload_item_reads`. |
| 2 — skill pool index | Initial user message | Compact index (`name [category] @ skills/<name>/SKILL.md: desc`) listing bindable candidates (guide). No content. Used by the agent to pick `keywords`, and by the framework as the candidate list for binding. |

### 6.2 Binding model

A single slot on the plan item:

- **`backing_skill`** — the matched skill name (or None when the item is
  unbound). Set by the framework at `update_plan` time from `keywords`;
  agent-submitted values are stripped at the trust boundary. Cleared on
  agent's `acknowledge_skill(applicability="unbind")` — the item
  degrades to unbound free exploration. The skill itself is NOT
  excluded; it stays in the registry at a lower priority tier.

Binding at `update_plan`:

```text
agent submits item with keywords
  ↓
pool.match_by_keywords(keywords, exclude_names=already_bound)
  → top-1 result, sorted by (tier, -score) → item.backing_skill
```

`match_by_keywords` walks `residual_bindable` (all trackable-category
skills in the pool — there is no terminal exclusion). A keyword-hit
minimum requirement means category prior alone is never enough — at
least one keyword must appear in the skill's name / description /
content. Within the eligible set the result is sorted by
``SkillBuilder.tier()`` first, then by keyword score:

- **Tier 0 (applied)** — skill has non-empty ``applied_versions``.
  Proven effective on this run; tried first.
- **Tier 1 (fresh)** — registered, never applied, never unbound.
- **Tier 2 (previously unbound)** — non-empty ``unbound_at_versions``
  and empty ``applied_versions``. Last resort, but still selectable —
  a later KEEP promotes it back to tier 0.

### 6.3 Pool depletion auto-widen

The startup pool is narrow (guide only) so the initial prompt stays
lean. Over a long run the agent may request patterns that only
`example` / `case` skills cover. `_match_keywords_to_skills` runs a
two-pass match: any keyword-requested item whose keywords don't hit
the current pool triggers a one-shot
`SkillPool.refill(mode="append", include_categories=["example", "case", "method", "implementation"])`,
then the misses are retried. The trigger is **"a specific keyword
request couldn't be served"**, not **"pool has zero bindable"** — with
guides still selected but semantically unrelated to the request, the
pool is non-empty yet every match comes up empty.

To preserve "a rejected `update_plan` has no side effects",
**rationale validation runs before** the auto-widen call; an invalid
plan returns a rejection without touching the pool.

### 6.4 SkillBuilder registry

SkillBuilder is a flat registry with no terminal states. Every
registered skill stays bindable; two monotonic badge lists drive the
binding priority tier:

```text
    registered  ◄──► record_applied  → applied_versions = [v, ...]
        │                                        │
        │                                        └─ tier 0 (top priority)
        │
        └──────► mark_unbound   →  unbound_at_versions = [v, ...]
                                              │
                                              └─ tier 2 (last resort,
                                                  unless also applied)
```

- **register** — creates or updates the record. Idempotent; later
  registrations just refresh `registered_reason` and
  `registered_at_version`.
- **mark_unbound** — called by the agent's
  `acknowledge_skill(applicability="unbind")` ack (via
  `FeedbackBuilder.record_skill_acknowledgement`). Appends
  `plan_version` to the skill's `unbound_at_versions` list and clears
  the current item's `backing_skill`. Diagnose / `require_replan` do
  NOT touch the registry — they only rewind the plan. **NOT terminal**:
  the skill remains bindable; `tier()` returns 2 unless it has also
  been applied.
- **record_applied** — called when a backing-skill-bound item settles
  KEEP. Appends `plan_version` to `applied_versions`. `tier()` returns
  0 whenever `applied_versions` is non-empty, so a single KEEP after
  a prior unbind promotes the skill back to the top priority.

"Which skill is currently active?" is computed on the fly from
`plan_items` (the item with `status == "active"` AND
`backing_skill != None`). The registry does not store that state.

Session persistence format: v5 (native `unbound_at_versions` list).
v4 payloads (legacy terminal `abandoned=True` + scalar
`abandoned_at_version`) are silently migrated by mapping to
`unbound_at_versions=[abandoned_at_version]` — those skills load as
tier 2, not as permanently excluded.

### 6.5 Keyword pipeline

`skill_adapter.generate_query_keywords` builds a `QueryKeywords` triplet
from `op_name + task_desc + stage + dsl/backend/arch/framework` via an
LLM call (with a deterministic fallback). `rank_skills_by_keywords`
fuses three signals: keyword-hit score over skill text, category prior
for the current stage, and operator-metadata bonuses from the catalogue.
The result is a single ranked list.

Canonical categories collapse `method → guide` and
`implementation → example` before ranking so the two taxonomies on disk
don't split the prior.

---

## 7. Skill acknowledgement and diagnose

### 7.1 Agent self-acknowledgement (pre-edit gate)

There is no external supervisor. Enforcement of "the agent actually
read and understood the injected SKILL.md" is done by the agent
itself via a mandatory structured tool call.

When a plan item has `backing_skill != None`:

1. At turn prologue the loop calls
   `ConversationBuffer.inject_backing_skill(item_id, skill_name,
   content, plan_version)` — the SKILL.md lands in the buffer as a
   user message with a marker prefix (dedup-protected by
   `(plan_version, item_id, skill_name)`).
2. `TurnExecutor._dispatch_tools` **blocks** `patch_file` /
   `write_file` on that item until the agent calls
   `acknowledge_skill(plan_item_id, valuable_aspects,
   kernel_application, applicability)`.
3. The handler validates the schema (both fields 100–500 chars,
   applicability ∈ {apply, unbind}) and hands the ack to
   `FeedbackBuilder.record_skill_acknowledgement`. The tool
   description explicitly nudges `kernel_application` toward
   STRUCTURAL edits (algorithm / access pattern / memory hierarchy /
   kernel fusion-split) FIRST, with parameter tuning only when
   backed by a specific structural hypothesis.
4. On `unbind` the feedback builder:
  - calls `SkillBuilder.mark_unbound(backing_skill, reason)`
     (NOT terminal — the skill stays in the registry, tier 2);
  - clears `item.backing_skill` (the item becomes unbound);
  - opens the edit gate — edits continue as free exploration.
5. On `apply` the ack is stored on the item under `skill_ack`
   (rendered in plan.md, persisted in session.json); the gate opens.

Items without `backing_skill` skip this flow entirely — free
exploration has no acknowledgement requirement. The structured
payload is also the audit trail: `valuable_aspects` and
`kernel_application` show up in the plan history so future reviewers
can see both what the agent thought the skill was worth and what
concrete change it planned to make before editing.

### 7.2 Post-eval diagnose

Triggered when `consecutive_failures` crosses a multiple of
`diagnose_suggest_threshold` (`consecutive_no_edit_turns` does NOT
feed this — that path is a warning nudge inside `TurnExecutor`,
not a subagent run). The diagnose subagent receives the current
plan, recent history, and the editable code; it returns a
diagnostic report, and the handler calls `require_replan` with the
report — the report surfaces as `last_diagnosis` in the next
prompt and the agent is forced into replan mode.

`require_replan` tags the failing item (the last-settled history
entry on post-eval paths, or the still-active item on
`edit_fail` / `quick_check_fail` paths) as
`abandoned (diagnose)`, sets `must_replan = true`, and records the
diagnosis. It does **not** touch the `SkillBuilder` registry. The
next `update_plan` enters **replace mode**: agent submits one or
more replacement items for the abandoned position; pending items
from the outgoing plan carry over unchanged.

`last_diagnosis` is persisted into session.json so a resume after a
replan-required state keeps the reasoning visible.

---

## 8. Context management

### 8.1 ConversationBuffer

Owner of `_msgs` and the skill-read tracking maps. It is the only
component that mutates the message list. Public operations:

- `append` / `extend` — ordinary message append.
- `inject_backing_skill(item_id, skill_name, content, plan_version,
  max_chars)` — append the injected SKILL.md as a plain user message
  with a marker prefix
  (`[skill auto-injected for <item> v<N> (<name>)]`). Idempotent per
  `(plan_version, item_id, skill_name)`. The marker is registered in
  `_item_inject_markers[item_id]` so `unload_item_reads` can find it
  later. Note: this is NOT a raw `tool_result` block — doing so would
  break Anthropic's tool_use/tool_result pairing.
- `track_item_skill_read(tool_use_id, item_id)` — called by
  TurnExecutor when the agent `read_file('skills/...')` while `item_id`
  is active. Records the tool_use_id in `_item_read_ids[item_id]`.
- `unload_item_reads(item_id)` — at settle. Walks two tracks:
  synthetic markers (match on `msg.content` prefix for top-level user
  messages) and real tool_use_ids (descend into user-message content
  arrays, flip the matching `tool_result` block). Replaces the body
  with `[skill read elided — plan item settled]`; preserves the
  `tool_use_id` so the API pairing stays valid.
- `on_buffer_rebuilt` — post auto_compact / force_rebuild hook. The
  new buffer carries recent rounds forward verbatim, so we do NOT
  clear the tracking maps blindly. Instead:
    - auto-inject track: rescan surviving user messages for the
      `[skill auto-injected for pN vM (name)]` marker and repopulate
      `_skill_inject_keys` + `_item_inject_markers` from what
      actually survived. Dedup continues to block re-inject if the
      marker is still in recent rounds; releases if compaction
      dropped it.
    - voluntary-read track: intersect each item's tool_use_id set
      with the ids still present as `tool_result` blocks in the new
      buffer. Ids the rebuild threw away are purged; ids that
      survived stay elide-able at settle time.
- `save_full_increment` / `load_latest` / `save_latest` — JSONL
  persistence (`messages_full.jsonl` grows, `messages_latest.jsonl`
  overwrites).

### 8.2 Compaction tiers

| Tier | When | What |
|---|---|---|
| `microcompact` | Per-turn | Trim stale `tool_result` bodies (keep the latest N). Zero LLM cost. |
| `auto_compact` | `estimate > threshold × context_limit` or the agent's `compact` tool | Multi-step pipeline: **two independent LLM calls** run concurrently (operator summary + structured plan.md analysis). The new buffer has five marker messages (boundary + bootstrap + 3 state attachments) plus the preserved recent rounds. See §8.3. |
| `force_rebuild` | Second PTL, or `auto_compact` no-op / exception | Same buffer layout as `auto_compact`, but **no LLM calls**: the operator summary degrades to a plain keyword dump, the plan analysis degrades to a truncated raw `plan.md` with an "analysis unavailable" label. |

`compact_failures` counts consecutive PTL events between successful LLM
calls; at `compact_max_failures` the loop aborts rather than burn the
budget on unusable prompts.

### 8.3 auto_compact pipeline

After `auto_compact` the buffer layout is:

```text
[COMPACT_BOUNDARY]                  — marker only
[BOOTSTRAP]                         — current plan vN + [OPERATOR_SUMMARY]
[STATE_ATTACHMENT:KERNEL]           — editable files, full content (80k sanity cap)
[STATE_ATTACHMENT:PLAN]             — 5-section structured plan analysis
[STATE_ATTACHMENT:RANKING]          — ranking.md, full content
<recent rounds from input>
```

**LLM call #1 — operator summary** (`_summarize_operator_from_keywords`):

- Input: `config.name`, `reference.py` head excerpt, historical keyword
  frequencies aggregated from `_plan_items + _settled_history` across
  the whole run.
- Output: ≤500 tokens markdown with sections `## Operator Shape`,
  `## Computation Components`, `## Exploration Signals`.
- Persisted: `agent_session/op_summary.md` (overwritten each compact).
- Fallback: plain `Keywords seen so far: a×3, b, c (counts: …)` dump.

**LLM call #2 — plan analysis** (`_analyze_plan_md`):

- Input: the full `task_dir/plan.md` — no truncation.
- Output: ≤1500 tokens markdown with EXACTLY five sections:
  `## Current Status`, `## What's Working`, `## High-ROI Operations`,
  `## Repeated Failures`, `## Dead Directions`.
- Persisted: `agent_session/plan_analysis.md` (overwritten each compact).
- Fallback: raw `plan.md` truncated to
  `compact_plan_raw_fallback_chars` with an "analysis unavailable"
  banner.

Both calls run via `asyncio.gather(..., return_exceptions=True)`; a
failure in one does not block the other. `kernel.py` and `ranking.md`
are attached in full (`compact_kernel_sanity_cap` is a defensive 80k
upper bound, not a default truncation).

### 8.4 Skill content lifecycle

Skill content enters the buffer **on turn prologue** (not at
submission) via `inject_backing_skill`, once per `(plan_version,
item_id, skill_name)`. It is a plain user message carrying a marker
prefix — not a raw `tool_result` block, because a synthetic
`tool_use_id` with no prior `tool_use` in an assistant message would
be rejected by the Anthropic API.

Two eviction tracks at settle time, both handled by
`unload_item_reads(item_id)`:

- **Synthetic inject** — match on the stored marker prefix; replace
  the user message's content wholesale with the elision marker.
- **Voluntary read_file** — agent read `skills/<name>/SKILL.md` while
  this item was active; the tool_result block's `content` is replaced
  while keeping the `tool_use_id` intact so the assistant /
  tool_result pairing stays valid.

On rebuild (auto_compact / force_rebuild) both tracking maps and the
inject dedup set are cleared; the active item's skill re-injects into
the fresh buffer on the next turn.

---

## 9. Persistence and resume

### 9.1 session.json

```json
{
  "version": 3,
  "task_name": "...",
  "model": "...",
  "counters": { ... RunCounters.to_dict() ... },
  "baseline_commit": "<sha>",
  "head_commit":     "<sha at save time>",
  "saved_at":        "YYYY-MM-DD HH:MM:SS",
  "plan_state":      { ... FeedbackBuilder.plan_state_to_dict() ... },
  "skill_state":     { ... SkillBuilder.skill_state_to_dict() ... },
  "last_diagnosis":  "..."
}
```

Written atomically (temp + os.replace) by `SessionStore.save()` on every
turn. Legacy v2 sessions with counters at top level are auto-migrated
by `RunCounters.from_dict`.

### 9.2 Resume guards

`SessionStore.load()` refuses to restore when:

- `task_name` mismatch (belongs to a different task).
- `head_commit` differs from the repo's current HEAD (someone advanced
  the tree; the session's eval history no longer matches).
- Any of the semantic files (`editable_files`, `eval_script`, etc.) are
  dirty in git.

On refusal, the log emits a warning and the run starts fresh. This is
deliberate: resume is an optimization, not a correctness requirement.

### 9.3 Pool rehydration on resume

`SkillBuilder.skill_state_from_dict` restores every SkillRecord
(including previously-unbound ones — they stay bindable, just tier 2).
The pool itself is not persisted; the
startup refill reconstructs it. Because the refill is category-limited,
any `example` / `case` that the previous session added via
`search_skills` is not automatically reloaded. `_rehydrate_pool_from_plan`
walks the restored plan items, collects every `backing_skill` name, and
calls `pool.append_new(...)` with the catalogue entries so content
lookups during skill injection work after a resume.

---

## 10. Budget and counters

`RunCounters` (`agent/counters.py`) holds every semantic threshold the
loop consults:

| Field | Purpose |
|---|---|
| `eval_calls_made` | Primary budget; gated against `max_rounds`. |
| `total_api_calls` | Secondary budget; gated against `max_rounds × max_turns_multiplier`. |
| `consecutive_failures` | Incremented on FAIL **and** on pre-eval failures (`edit_fail` / `quick_check_fail`); reset on KEEP **and** on DISCARD (correct-but-no-improvement is not a failure). Feeds diagnose threshold. |
| `consecutive_no_edit_turns` | Incremented on edit-less turns; reset on any patch. At `max_no_edit_turns` the turn handler appends a "you must edit now" warning user message (NOT a diagnose / replan). |
| `consecutive_no_tool_turns` | Incremented when the LLM replies with no tool call at all. Hard stop protection. |
| `compact_failures` | PTL escalation counter. |

Counters are the single source of truth. Every decision that looks like
"should I replan / diagnose / abort" reads from here — the loop logic
itself doesn't carry duplicate flags.

---

## 11. Configuration

AutoResearch reads from the standard AKG config system (`load_config` +
`build_langgraph_task_config`). Per-task overrides go in the
`agent.config` block of `task.yaml`. Full field list lives in
`framework/config.py`; below is a curated view.

### 11.1 Task-level keys

| Key | Default | Purpose |
|---|---|---|
| `max_step` | 20 | Max eval rounds (the `max_rounds` argument overrides). |
| `agent_model_config.coder` | `"standard"` | LLM model level. |
| `eval_timeout` | 120 | Per-round eval timeout (seconds). |
| `workflow_timeout` | dynamic | Total workflow timeout. Auto-computed as `max(1800, max_rounds × (eval_timeout + 60) + 300)`. |
| `profile_settings.run_times` | 50 | Profiling repetitions. |
| `profile_settings.warmup_times` | 5 | Profiling warmups. |

### 11.2 AgentConfig highlights

| Key | Default | Purpose |
|---|---|---|
| `max_consecutive_failures` | 10 | Abort after N consecutive failure rounds. Counts FAIL evals and pre-eval failures (edit/quick_check); DISCARD resets the counter. |
| `max_turns_multiplier` | 8 | Hard cap: total API calls ≤ `max_rounds × this`. |
| `max_no_edit_turns` | 3 | After N turns with no edits, `TurnExecutor._nudge_if_no_edits` appends a warning user message. Does NOT force replan or trigger diagnose. |
| `context_limit` | 150000 | Model context window (tokens). auto_compact fires at `× compression_threshold`. |
| `compression_threshold` | 0.75 | Fraction of `context_limit` that triggers auto_compact. |
| `compact_max_failures` | — | PTL escalation cap before loop aborts. |
| `skill_block_max_chars` | 8000 | Initial-prompt skill index total budget. |
| `skill_block_top_k` | 5 | Top-K skill index entries rendered. |
| `skill_inject_max_chars` | 6000 | Clamp for skill content injected on item activation. |
| `skill_narrow_timeout` | 30.0 | Hard timeout for the keyword-generation LLM call. |
| `diagnose_suggest_threshold` | 3 | Trigger diagnose subagent every N consecutive failures. |
| `subagent_max_iterations` | 15 | Max iterations for the diagnose subagent. |
| `system_fundamentals_max_chars` | 20000 | Budget for the `## DSL Fundamentals` block in the system prompt. |
| `skill_inject_max_chars` | 6000 | Clamp applied to the injected SKILL.md body. |
| `plan_item_rationale_min_chars` | 30 | Min length for plan item rationale. |
| `plan_item_rationale_max_chars` | 400 | Truncation cap for rationale. |
| `skill_keyword_max_per_item` | 5 | Max `keywords` accepted per plan item. |

### 11.3 Grouping (for quick navigation)

- **Context** — `context_limit`, `compression_threshold`,
  `compact_keep_recent_rounds`, `compact_op_summary_max_tokens`,
  `compact_plan_analysis_max_tokens`, `compact_kernel_sanity_cap`,
  `compact_plan_raw_fallback_chars`, `compact_post_check_ratio`,
  `compact_max_failures`.
- **Truncation** — `editable_file_truncate`,
  `system_context_file_truncate`, `system_context_total_truncate`,
  `plan_max_chars`, `skill_block_max_chars`, `skill_inject_max_chars`,
  `eval_feedback_tail`, `raw_output_tail`.
- **Loop control** — `max_consecutive_failures`, `max_no_edit_turns`,
  `max_turns_multiplier`, `llm_max_tokens`, `thinking_budget`.
- **Skill** — `skill_*`, `system_fundamentals_max_chars`,
  `plan_item_rationale_*`.
- **Diagnose / feedback** — `diagnose_suggest_threshold`,
  `compact_diagnosis_truncate`, `ranking_description_truncate`,
  `finish_hint_threshold`.
- **LLM** — `llm_connection_check_timeout`, `call_timeout`,
  `llm_max_retries`, `chars_per_token`.
- **Paths** — `session_dir` (default `"agent_session"`),
  `heartbeat_file`.

---

## 12. Task directory layout

Everything the run produces is under `task_dir`:

| Path | Writer | Purpose |
|---|---|---|
| `session.json` | SessionStore | Resume state |
| `plan.md` | FeedbackBuilder | Current plan + history + skill state |
| `{session_dir}/messages_full.jsonl` | ConversationBuffer | Complete message archive (append-only) |
| `{session_dir}/messages_latest.jsonl` | ConversationBuffer | Current buffer snapshot |
| `{session_dir}/op_summary.md` | compress.auto_compact (LLM #1) | Operator-level summary rebuilt each compact |
| `{session_dir}/plan_analysis.md` | compress.auto_compact (LLM #2) | Structured 5-section plan.md analysis rebuilt each compact |
| `log.jsonl` | RoundLogger | One JSON per evaluated round |
| `perf_log.md` | RoundLogger | Human-readable results table |
| `ranking.md` | RoundLogger | Top-K correct + failed attempts |
| `report.md` | Runner | Optimization report (markdown with inline SVG curve, no external deps) |
| `agent.log` | FileLogger | Stdout tee with timestamps |
| `RUNNING` | SessionStore | PID + status heartbeat |

The `git_repo.py` policy is that these run artefacts are **never
committed** — the repository is reserved for eval checkpoints (one
commit per KEEP), and the task directory is the uncommitted workspace.

---

## 13. Invariants

The system relies on a handful of invariants that every change should
preserve:

1. **Single writer per artefact.** plan.md → FeedbackBuilder only;
   session.json → SessionStore only; messages_*.jsonl → ConversationBuffer
   only. A caller wanting to change any of these state types goes
   through the owner.
2. **Rejected tool calls have no side effects.** If `update_plan` rejects
   on rationale validation, the SkillPool, SkillBuilder, and plan.md are
   untouched. If `patch_file` rejects on stale-hash, the filesystem is
   untouched.
3. **SkillBuilder has no terminal state.** Every registered skill
   stays bindable. `unbound_at_versions` only demotes a skill to
   tier 2 in the binding-priority sort; a later KEEP promotes it
   back to tier 0.
4. **Budget accounting is counter-only.** The loop's "continue /
   terminate / escalate" decisions read from `RunCounters` exclusively —
   never from ad-hoc flags.
5. **plan.md is the recovery point.** After any compact cycle,
   plan.md + session.json are enough to reconstruct the agent's
   world-view. Section-aware truncation exists specifically to
   protect this.
6. **Tool schemas are the trust boundary.** Anything beyond the declared
   tool args (including agent-emitted `backing_skill`) is stripped by
   `TurnExecutor` before it can mutate state.

---

## 14. Usage

The single production entry point is the CLI script
`scripts/run_autoresearch.py`. Four input combinations:

```bash
# Natural language (LLM writes reference, KernelGen writes seed)
python scripts/run_autoresearch.py \
  --desc "fused ReLU + LayerNorm, input (32, 1024), fp16" --backend cuda

# Natural language + initial kernel (skip KernelGen)
python scripts/run_autoresearch.py \
  --desc "fused ReLU + LayerNorm, input (32, 1024), fp16" \
  --kernel path/to/kernel.py --backend cuda

# Reference file (KernelGen writes seed)
python scripts/run_autoresearch.py --ref path/to/reference.py --backend cuda

# Reference + initial kernel (skip all generation)
python scripts/run_autoresearch.py \
  --ref path/to/reference.py --kernel path/to/kernel.py --backend cuda
```

Key flags:

| Flag | Purpose |
|------|---------|
| `--desc` / `--ref` / `--kernel` | Input source; see the four combinations above. |
| `--op-name` | Operator name (drives task directory and log category). |
| `--backend` / `--arch` | Target backend and arch; must match the worker that runs eval. |
| `--max-rounds` | Eval-call budget (overrides `max_step` from `task.yaml`). |
| `--device-id` | Local worker device id (default 0). Mutually exclusive with `--worker-url`. |
| `--worker-url` | Remote Worker Service HTTP endpoint (`host:port`, comma-separated for multiple). Use this when the local machine has no NPU / CUDA and eval must be forwarded to a remote worker. Worker startup (`akg_cli worker --start`) is covered in [AKG_CLI.md](./AKG_CLI.md) §4; the Server / Worker / WorkerManager architecture is documented in [ServerArchitecture.md](../v1/ServerArchitecture.md). |

> KernelAgent reaches the same workflow internally via
> `ToolExecutor → prepare_config() → build_initial_state() → workflow
> execution` when a user requests deep iterative optimization; end
> users do not call it directly.
>
> **Quick remote-worker setup**: on a machine with NPU / CUDA, run
> `akg_cli worker --start --backend <backend> --arch <arch>
> --devices <ids> --port <port>`; if the local host cannot reach it
> directly, forward the port via an SSH tunnel, then pass
> `--worker-url host:port`. See [AKG_CLI.md](./AKG_CLI.md) §4.

### 14.1 Debug entry points

Non-production paths, for debugging only. They skip the preprocessing
done by the CLI script (arg validation, worker registration, scaffold);
callers are responsible for supplying valid arguments — invalid input
surfaces as an error at the component boundary.

**(a) Driving `LangGraphTask` directly** (skip `run_autoresearch.py`,
assemble config by hand):

```python
from akg_agents.op.langgraph_op.task import LangGraphTask
from akg_agents.op.config.config_validator import load_config
from akg_agents.core.worker.manager import register_local_worker
from akg_agents.utils.task_label import resolve_task_label

await register_local_worker([0], backend="cuda", arch="a100")

config = load_config(dsl="triton_cuda", backend="cuda")
config["task_label"] = resolve_task_label(op_name="my_op", parallel_index=1)
config["max_step"] = 20

task = LangGraphTask(
    op_name="my_op",
    task_desc=open("reference.py").read(),
    task_id="my_op_001",
    backend="cuda", arch="a100",
    dsl="triton_cuda", framework="torch",
    config=config,
    workflow="autoresearch",
)
op_name, success, final_state = await task.run()
```

**(b) Dropping into `AgentLoop` / `manual_eval`** (bypass the workflow
layer; must use the fully-qualified module path — relative imports
make bare `python manual_eval.py` not work):

```bash
# Direct AgentLoop (skip preflight / seed / workflow wrapper)
python -m akg_agents.op.autoresearch.agent \
  --task <task_dir> --max-rounds 20 --device-id 0

# Manual eval helper (no LLM; single-round eval / status / report)
python -m akg_agents.op.autoresearch.manual_eval \
  --task <task_dir> --eval-only
python -m akg_agents.op.autoresearch.manual_eval \
  --task <task_dir> --status
python -m akg_agents.op.autoresearch.manual_eval \
  --task <task_dir> --report
```

---

## 15. Testing

Tests auto-detect NPU / GPU and skip if neither is available. The
smoke-test operator is `relu(x)` with shape `(11, 37, 8191)` — prime
dimensions force mask handling; `8191 = 2^13 − 1` stresses UB sizing.

| Suite | Path | Coverage |
|---|---|---|
| End-to-end | `tests/op/st/test_autoresearch.py` | preflight → seed verify → AgentLoop → eval_fn → KernelVerifier; performance gate (speedup > 1.0×). |

Component unit tests live in `tests/op/ut/`:

- `test_skill_builder.py` — SkillBuilder transitions (settle / supersede / replan / serialize).
- `test_skill_pipeline.py` — keyword pipeline primitives + `SkillPool` read side.
- `test_acknowledge_skill.py` — tool schema validation, ack→feedback wiring, `inject_backing_skill` / `unload_item_reads` round-trip through the buffer.
- `test_fundamentals_layout.py` — task_dir/skills layout, Layer 0 fundamentals scan, budget cap.
- `test_runtime_skill_binding.py` — `mark_selected` idempotence, `SkillPool.refill` (replace + append), `_handle_update_plan` augmentation, `_handle_search_skills` wrapper.
- `test_session_persistence.py` — resume flow + skill state round-trip.
- `test_skill_prompt_injection.py` — initial-message wording + compress bootstrap skill index rendering.
- `test_conversation_buffer.py` — buffer skill injection lifecycle.
- `test_scaffold_roundtrip.py` — `scaffold_task_dir` → `load_yaml_config` field preservation.

---

## Appendix A: Public constants

```text
compress.COMPACT_BOUNDARY    "[COMPACT_BOUNDARY]"
compress.STATE_ATTACHMENT    "[STATE_ATTACHMENT]"
compress.BOOTSTRAP_MARKER    "[BOOTSTRAP]"

skill_adapter.TRACKABLE_PATTERN_CATEGORIES
    frozenset({"guide", "example", "case", "method", "implementation"})

skill_pool.SkillPool._REFERENCE_CATEGORIES
    frozenset({"fundamental", "reference"})
```

## Appendix B: Parameter reference

Allowed values for task parameters (`framework`, `backend`, `dsl`,
`arch`) are defined in the repo-root `AGENTS.md`; they are enforced at
`TaskConfig` construction.
