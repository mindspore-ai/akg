## Custom SubAgent Routing

You have the following custom subagents available via `task(subagent_type=...)`:

| SubAgent | Trigger | Call Pattern | Note |
|----------|---------|--------------|------|
| `akg-installer` | User asks to install / configure / setup akg_agents | `task(subagent_type="akg-installer", load_skills=[], run_in_background=false, description="Install akg_agents", prompt="<user request>")` | After `akg-installer` returns, its output is already a user-facing report — use it as your response directly. Do not rewrite it into your own words. |

**General rule**: If any subagent listed above can fulfill a user request, delegate to it immediately. Do NOT attempt to implement the same functionality yourself.
