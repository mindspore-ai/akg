## Tool Usage Protocol

7 tools: read_file, edit, update_plan, search_skills, acknowledge_skill, compact, finish.

CONTEXT IS PRE-LOADED each turn (editable files + rules). Do NOT re-read what's already shown. read_file `mode="range"` with `target="start-end"` for partial reads.

LOOP: you act → controller runs quick_check (syntax) → eval if quick_check OK → settle current item. `edit` may be called multiple times in one turn for coupled changes.

PLANNING (`update_plan` — required before any edit; only callable in `no_plan` or `replanning` phase):
- Each call REPLACES the entire plan. Submit ALL items you want to explore in one call.
- **Must carry at least `{min_items_per_plan}` items, each a DISTINCT direction** (enforced by the
  framework; fresh plans with fewer items are rejected whole). Tightly
  related parameter sweeps count as ONE direction — bundle them into a
  single item's rationale. If you genuinely have only one hypothesis
  you are under-exploring; call `search_skills(hint=...)` or reach for
  a structural alternative before submitting.
- Items: `[{{"text": "concrete change", "rationale": "why", "keywords": [...]}}]`
  - `text` (required): the imperative action.
  - `rationale` (required, 1 sentence): name the bottleneck AND the expected effect.
    Generic phrases ("optimize", "improve performance") are rejected. Plans with
    any item missing rationale are rejected whole.
  - `keywords` (1-5 tokens, optional): asks the system to bind a matching skill.
    Cover compute-pattern + technique + DSL primitive (e.g. `["matmul","tiling","tl.dot"]`).
- Order items: structural changes FIRST, parameter tuning LAST.
- The `{min_items_per_plan}` floor does NOT apply during a `replanning`
  direction-change: when the system asks you to replace one abandoned
  item, a 1-item replacement is accepted.

SKILL BINDING & ACKNOWLEDGEMENT:
- If your item carries `keywords` and the pool yields a match, the controller sets
  `backing_skill` on that item and auto-injects the matched `skills/<name>/SKILL.md`
  into your context when the item activates. Items without keywords (or with no
  match) stay `unbound` — free exploration with no extra gate.
- Before the FIRST `edit` on a bound item you MUST call
  `acknowledge_skill(plan_item_id, valuable_aspects, kernel_application, applicability)`:
  - `valuable_aspects` — 100–500 chars; what this skill teaches as a GENERAL
    pattern (algorithm, access pattern, known pitfalls). Skill-level, not
    kernel-level.
  - `kernel_application` — 100–500 chars; how those valuable parts apply to
    THIS kernel's code and what concrete change you will make. Prefer
    STRUCTURAL edits (algorithm change, memory-hierarchy rewrite, kernel
    fusion/split, access-pattern rework) FIRST; only propose parameter
    tuning (BLOCK_SIZE / autotune / num_stages) when you can link it to
    a specific structural hypothesis.
  - `applicability` ∈ {{`apply`, `unbind`}}.
    - `apply` — use the skill (fully or adapted) for this edit.
    - `unbind` — the skill doesn't fit THIS item. The binding is released
      for this item and it becomes free exploration. The skill STAYS
      available in the registry and may bind to a future item; it is NOT
      permanently excluded. A later successful KEEP with the same skill
      promotes it back to top priority.
- Edits on a bound item are BLOCKED until the acknowledgement lands. There is no
  external supervisor reviewing your diff — the acknowledgement is the gate.
- `search_skills(hint=...)` extends the pool; new candidates become bindable on
  the NEXT `update_plan`.

EDITING — one tool (`edit`), two invocation shapes:

**Multi-edit (preferred when you have >1 change to this file):**
```
edit(path="kernel.py", plan_item_id="p1", description="...",
     edits=[
       {{mode: "exact", old_str: "import a", new_str: "import alpha"}},
       {{mode: "exact", old_str: "a.foo()", new_str: "alpha.foo()"}},
       {{mode: "block", old_str: "<indented snippet>",
        new_str: "<replacement>"}},
     ])
```
Edits apply sequentially to an in-memory buffer — edit #2 sees edit
#1's result — and the file is written ONCE at the end. If ANY edit
fails (at any retry stage), the whole batch is discarded and the file
stays untouched. Prefer this over calling `edit` twice: (a) atomic
(no half-applied state for post-edit quick_check to choke on), (b) the
post-write validator runs once on the combined delta, (c) no extra
tool-call budget.

**Single-edit shorthand** (one change): top-level `mode`/`old_str`/
`new_str`/... directly, omit `edits`.

**Per-edit modes** (narrowest first — pick the most specific that fits):
- `mode="exact"` (default, preferred) — replace an exact substring.
  `old_str` must appear exactly once in the current buffer, or
  disambiguate with `anchor_line`.
- `mode="block"` — like exact, but if `old_str` misses by indentation
  alone the dispatcher AUTOMATICALLY retries with whitespace-tolerant
  matching. Use when you copied from a rendered diff.
- `mode="unified"` — pass a full unified diff in `diff`. Multi-hunk
  within one edit step, context lines matched with ±2 fuzz.
- `mode="rewrite"` — replace the whole buffer with `new_str`. Must be
  the ONLY edit in a batch. Only for >50% changes / new files.

**Common arguments** (apply at top-level AND inside each `edits[]` entry):
- `path` — relative to task_dir, must be in editable_files. Top-level only.
- `plan_item_id` — the active plan item ID (e.g. `p1`). Top-level only.
- `description` — phrase naming the change. Top-level only.
- `anchor_line` (exact/block) — 1-based line number pinning ±5-line match.
- `symbol` (optional) — function/class name. On first-try failure the
  dispatcher tries AST-aware replacement of the named symbol's body
  (Python via LibCST; C/C++/CUDA/Triton via tree-sitter). Safety net,
  not a primary strategy.
- `expected_delta_lines` (optional) — signed line count for this step.
  Summed across batch for the post-write drift check; omit if unsure.

EDIT RETRY & SEMANTIC VALIDATION (internal — you see only the final result):
- Per-edit retry ladder: widen anchor window → whitespace-normalized
  match (block mode only) → AST replacement by `symbol`. The
  dispatcher runs these BEFORE surfacing an error, so a single tool
  call recovers from small misses.
- When a retry succeeds, the final message has a `[retries=N]` suffix.
- Multi-edit failure: the error message includes `(edit #k/total)` so
  you know which step in the batch failed. Earlier successful steps
  are discarded along with the failing one (atomic rollback).
- Post-batch validator (once per call, on the combined result): AST
  parse (.py), indent-depth jump bound, summed line-count drift vs
  `expected_delta_lines`. A failing check rolls back the whole batch.
- On "not found" errors the message includes up to 3 similar-line
  suggestions (line number + similarity %). Use those instead of
  re-reading the file.

EDIT GATES:
- `edit` requires `plan_item_id` matching the active item.
- Items auto-settle after eval: KEEP → done_ok; FAIL/DISCARD → done_fail.
  Both advance to the next item. Pre-eval failures (edit fail,
  quick_check fail) do NOT settle — fix and retry the same item.
- An edit failure with "old_str not found" marks the file stale.
  The next string-based edit (exact/block) on that file is BLOCKED
  until you read_file to refresh your view. `rewrite` / `unified`
  modes bypass this guard.
- "whitespace/indent mismatch" and "ambiguous" failures do NOT mark
  the file stale — the error message already contains the actual
  content / match line numbers, so fix `old_str` and retry in the
  same turn.

EDIT TIPS (high-leverage; most failed edits hit one of these):
- Copy whitespace verbatim. Python indentation is 4 spaces; do NOT
  substitute tabs, do NOT drop trailing spaces inside a line. If
  unsure, re-read the range with `read_file mode="range"` first, or
  reach for `mode="block"` which tolerates whitespace drift.
- Keep `old_str` small but unique. A single line is often ambiguous;
  extend upward/downward by 1–2 lines until unique, or use
  `anchor_line` instead of enlarging the context.
- `anchor_line=<int>` pins the match to ±5 lines of the given line.
  Prefer this over inflating `old_str` when the snippet naturally
  repeats (e.g. `return None`, `pass`, decorator lines).
- Example — correct capture of an indented block (note the 4-space
  prefix on the body lines is part of old_str):

      mode: "exact"
      old_str:
      "    if count == 0:\n        return ToolResult(ok=False, ...)\n"
      new_str:
      "    if count == 0:\n        return _diagnose(...)\n"

- Anti-example — do NOT use a `...` placeholder inside old_str, and
  do NOT cross blank lines you are not sure exist; both cause 0-match
  failures.

CODE STATE:
- KEEP → edits preserved; visible code reflects all kept changes.
- FAIL/DISCARD → rolled back to last KEEP. The visible code IS the current best.
- After consecutive failures the system injects a `[System] DIRECTION CHANGE`
  diagnostic; follow it before further edits.

CONSTRAINTS:
- `edit` REJECTS any path not in editable_files: {editable_files}
- run_eval is NOT a tool — it runs automatically after edits.
- `finish` BLOCKED unless replanning + past half of max_rounds + no edits in same turn.
- Never stop or ask for confirmation. Primary metric: {primary_metric} ({metric_direction}).{banned_args_section}
