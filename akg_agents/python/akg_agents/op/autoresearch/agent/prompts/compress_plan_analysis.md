You are analyzing the progression of an optimization run. You will be
given the full contents of plan.md, which contains:

- The current plan items (latest version, with keywords + backing skills)
- An "Optimization History" section (past plan versions, each item's
  KEEP / FAIL / DISCARD outcome and the actual edit description)
- A "Skill State" section that lists every skill as one of Active,
  Selected, Applied, or Previously unbound. Note: NO skill is
  permanently terminal. "Previously unbound" means the agent
  acknowledged the skill as not fitting a particular item; the skill
  stays registered and can bind to a future item. "Applied" beats
  "Previously unbound" — a KEEP on an unbound skill promotes it back
  to top priority.

Produce a structured markdown summary with EXACTLY these FIVE sections
in order:

## Current Status
Best metric value and recent trajectory, rough eval-budget consumed,
whether the run is still making progress or stuck.

## What's Working
KEEP items ordered by contribution magnitude. For each, say ONE line
whether it was a structural change (altered the actual computation
flow) or a configuration / parameter adjustment. Do not re-list
anything already labeled Dead below.

## High-ROI Operations
Top 2-3 single-step wins. For each: the edit description and a
one-sentence hypothesis for WHY it worked (cite the plan item id).

## Repeated Failures
Cluster failure modes by the FAILURE PATTERN, not by domain. For each
cluster: a short generic name (e.g. "parameter sweep without
structural change", "invariant violation", "resource-limit exceeded",
"correctness regression on DISCARD") plus the rough repeat count.
Derive the name from the plan.md you were given — do not invent
categories that aren't supported by its content.

## Dead Directions
Directions with repeated, structurally-similar FAIL / DISCARD outcomes
AND no successful variant — the agent should avoid re-running those
same edits. Do NOT list a skill as dead just because it appears under
"Previously unbound": that only means the skill did not fit one
specific item; it may still apply to a future item with different
keywords, and a successful KEEP anywhere will promote it back to
top priority. Only a direction that has been tried and kept failing
qualifies here.

Rules:
- Respond with TEXT only. Do NOT call any tools.
- Use the exact 5 section names above (H2 level).
- Be concrete: cite plan item ids (p1, p2, ...) and round numbers (R15,
  R46, ...) when helpful.
- Stay under 1200 words total.
- Do NOT introduce domain-specific vocabulary (hardware names,
  framework specifics, library calls) beyond what appears verbatim in
  plan.md.
