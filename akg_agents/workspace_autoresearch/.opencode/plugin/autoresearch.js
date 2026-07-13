// Copyright 2026 Huawei Technologies Co., Ltd
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// AutoResearch guardrail plugin for opencode — the opencode ADAPTER.
//
// It does NO phase logic. Every decision comes from the SAME Python brain
// the Claude Code build uses (scripts/decide.py), reached through the door
// (.opencode/door.py). This file only translates: opencode hook event →
// neutral AgentEvent (base64 argv to the door), neutral Decision → opencode
// mechanism (throw to block; append guidance to the tool result).
//
// Mirror of .claude/hooks/cc_hook.py. The door test
// (tests/opencode_door/run_door_test.py) proves both adapters agree.
//
// ── Verified against opencode docs ──
//   plugin export shape; tool.execute.before/after(input, output);
//   block-by-throw in tool.execute.before; plugin lives in .opencode/plugin/.
// ── NOT covered by the official docs — VERIFY on your opencode version ──
//   1. `stop` hook existence + throw-to-block (community-documented, not in
//      the official plugin page). If absent, early-stop can't be forced;
//      the /autoresearch command still tells the agent to keep going.
//   2. tool.execute.after `output` object shape. We append guidance to
//      `output.output` (the model-visible result). If the field name or
//      mutability differs, adjust _injectGuidance below.
//   3. callID field name on the hook `input` (used to carry the bash
//      command from before→after). We try input.callID then input.callId.

import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { appendFileSync } from "node:fs";

// Opt-in trace (set AR_PLUGIN_TRACE=<file> to enable). Off by default, but
// kept in-tree as the canonical way to smoke-test hook firing on a live
// opencode where the host's own logs don't surface plugin activity.
function trace(msg) {
  const f = process.env.AR_PLUGIN_TRACE;
  if (!f) return;
  try { appendFileSync(f, `[ar-plugin] ${msg}\n`); } catch (_) {}
}
trace("module loaded");

const _DIR = (() => {
  try {
    if (import.meta.dir) return import.meta.dir;
  } catch (_) { /* fall through */ }
  return dirname(fileURLToPath(import.meta.url));
})();
const DOOR = join(_DIR, "..", "door.py");
const PYTHON = process.env.AR_PYTHON || "python";

// opencode names its tools lowercase; decide() branches on the canonical
// tokens (Bash/Edit/Write/MultiEdit/NotebookEdit/Task). Map here — this is
// the one piece of opencode-specific vocabulary the adapter owns.
// opencode's native tool names -> decide()'s neutral tool taxonomy
// ("shell" | "edit" | "subagent"). This map is the opencode adapter's own
// business; decide() never sees a native name and privileges no agent's
// vocabulary. (Claude's adapter has the symmetric map for Bash/Edit/Task.)
const TOOL_KIND = {
  bash: "shell",
  edit: "edit",
  write: "edit",
  patch: "edit",
  task: "subagent",
};
function toolKind(name) {
  return TOOL_KIND[name] || "";
}

function argString(args, ...keys) {
  if (!args) return "";
  for (const k of keys) {
    const v = args[k];
    if (typeof v === "string" && v) return v;
  }
  return "";
}

function shellSingleQuote(value) {
  return `'${String(value).replaceAll("'", `'"'"'`)}'`;
}

export const AutoResearch = async ({ $, client }) => {
  // before→after command relay, keyed by callID (after's `output` may not
  // carry the original tool args; decide(post_tool) needs the bash command).
  const argsByCall = new Map();
  // Per-session auto-continue counter (runaway backstop for the
  // session.idle re-prompt that stands in for the absent `stop` hook).
  const continuesBySession = new Map();
  const MAX_CONTINUES = Number(process.env.AR_MAX_CONTINUES || "200");

  async function callDoor(event) {
    try {
      const b64 = Buffer.from(JSON.stringify(event), "utf8").toString("base64");
      const res = await $`${PYTHON} ${DOOR} event ${b64}`.quiet().nothrow();
      const out = (res && res.stdout != null)
        ? res.stdout.toString()
        : String(res ?? "");
      return JSON.parse(out.trim());
    } catch (e) {
      // The door never throws by design; if spawning/parsing failed, fail
      // OPEN (allow) so a broken adapter can't wedge the agent.
      return { block: false, status: [], context: null };
    }
  }

  function buildEvent(kind, input, args) {
    const nativeTool = (input && input.tool) || "";
    return {
      kind,
      tool_kind: toolKind(nativeTool),   // neutral: shell | edit | subagent
      tool: nativeTool,                  // raw native name (logging only)
      command: argString(args, "command"),
      file_path: argString(args, "filePath", "file_path", "path"),
      subagent_type: argString(args, "subagentType", "subagent_type", "agent"),
      output: "",
      stop_reason: "stop",
      session_id: (input && (input.sessionID || input.sessionId)) || "",
    };
  }

  function callId(input) {
    return (input && (input.callID || input.callId)) || "";
  }

  function injectGuidance(output, decision) {
    const lines = [...(decision.status || []), decision.context].filter(Boolean);
    // Best-effort plan mirror. The neutral `todos` channel is dropped above,
    // but opencode DOES have a todo tool — surface the live plan so the agent
    // can mirror it into todowrite. Not forced; plan.md is the source of truth.
    if (decision.todos_header && Array.isArray(decision.todos)) {
      const items = decision.todos.length
        ? decision.todos.map((t) => `  - [${t.status}] ${t.content}`).join("\n")
        : "  (no live items)";
      lines.push(
        `${decision.todos_header}\nIf you have a todo/task-list tool ` +
        `(e.g. todowrite), mirror the current plan into it, replacing existing ` +
        `entries:\n${items}`,
      );
    }
    if (!lines.length) return;
    const note = lines.join("\n");
    // VERIFY: `output.output` is assumed to be the model-visible result
    // string. Append our guidance so the model reads it like a hook would.
    if (output && typeof output.output === "string") {
      output.output = output.output ? `${output.output}\n\n${note}` : note;
    } else if (output) {
      output.output = note;
    }
  }

  return {
    // Inject the neutral session id into every shell call. Environment-file
    // activation is handled after the original command passes the phase gate
    // below; shell.env is not reliable for login-shell PATH restoration in
    // all supported OpenCode versions.
    "shell.env": async (input, output) => {
      if (!output) return;
      output.env = output.env || {};
      const sid = input && input.sessionID;
      if (sid) output.env.AR_SESSION_ID = sid;
    },

    "tool.execute.before": async (input, output) => {
      const args = (output && output.args) || {};
      const envFile = process.env.AR_OPENCODE_ENV_FILE;
      const envPrefix = envFile ? `. ${shellSingleQuote(envFile)}\n` : "";
      const policyArgs = {...args};
      // Some OpenCode versions re-enter before-hooks after an earlier hook
      // mutates args. Strip only our exact prefix before policy evaluation so
      // repeated delivery remains idempotent and the shared brain always sees
      // the model's original command.
      if (toolKind(input && input.tool) === "shell" && envPrefix &&
          policyArgs.command && policyArgs.command.startsWith(envPrefix)) {
        policyArgs.command = policyArgs.command.slice(envPrefix.length);
      }
      const ev = buildEvent("pre_tool", input, policyArgs);
      const id = callId(input);
      // An empty/unrecognised call-id cannot safely key concurrent tools;
      // prefer after-hook output.args instead of making them share one slot.
      if (id) argsByCall.set(id, policyArgs);
      const decision = await callDoor(ev);
      trace(`before tool=${ev.tool} kind=${ev.tool_kind} block=${decision.block}` +
            (decision.block ? ` reason=${(decision.block_reason || "").slice(0, 60)}` : ""));
      if (decision.block) {
        if (id) argsByCall.delete(id);
        throw new Error(decision.block_reason || "[AR] blocked");
      }
      // OpenCode may run Bash tools through a login shell that replaces the
      // activated PATH. Source one explicit, user-selected environment file
      // only AFTER the original command has passed the shared phase policy.
      // The stored after-hook args remain the original command, so this
      // adapter detail never enters state-machine semantics.
      if (ev.tool_kind === "shell" && envPrefix && args.command &&
          !args.command.startsWith(envPrefix)) {
        args.command = `${envPrefix}${args.command}`;
      }
    },

    "tool.execute.after": async (input, output) => {
      const id = callId(input);
      const args = (id && argsByCall.get(id)) || (output && output.args) || {};
      if (id) argsByCall.delete(id);
      const ev = buildEvent("post_tool", input, args);
      const decision = await callDoor(ev);
      trace(`after tool=${ev.tool} status=${(decision.status || []).length} ` +
            `ctx=${decision.context ? 1 : 0} todos=${decision.todos_header != null ? 1 : 0}`);
      injectGuidance(output, decision);
    },

    // opencode 1.17.7 has NO `stop` hook (verified against
    // @opencode-ai/plugin's hook list), so early-stop can't be hard-blocked
    // by throwing. Instead we watch `session.idle` (fires when the agent
    // ends its turn) and, while the phase machine says "keep going" (not
    // FINISH), re-inject the phase guidance as a fresh prompt — a
    // push-continue that stands in for Claude Code's Stop block. Capped per
    // session as a runaway backstop; fail-safe (any SDK error → no-op, and
    // the /autoresearch command instruction remains the soft guard).
    event: async ({ event }) => {
      if (!event) return;
      const props = event.properties || {};
      const sid = props.sessionID || props.sessionId || "";
      if (event.type === "session.deleted") {
        if (sid) continuesBySession.delete(sid);
        return;
      }
      if (event.type !== "session.idle") return;
      if (!sid) return;
      // run_loop.py already owns headless continuation. Keeping this idle
      // push active there creates two independent loop drivers.
      if (process.env.AR_EXTERNAL_LOOP === "1") return;
      const decision = await callDoor({
        kind: "stop", tool: "", command: "", file_path: "",
        subagent_type: "", output: "", stop_reason: "idle", session_id: sid,
      });
      trace(`idle session=${sid.slice(0, 8)} block=${decision.block}`);
      if (!decision.block) continuesBySession.delete(sid);
      if (!decision.block) return;       // FINISH or no active task → allow stop
      const n = continuesBySession.get(sid) || 0;
      if (n >= MAX_CONTINUES) {
        trace(`idle cap reached (${n}) — not continuing`);
        return;
      }
      continuesBySession.set(sid, n + 1);
      try {
        await client.session.prompt({
          path: { id: sid },
          body: {
            parts: [{
              type: "text",
              text: decision.block_reason ||
                "[AR] Continue the autoresearch loop; do not stop before FINISH.",
            }],
          },
        });
        trace(`idle re-prompted (#${n + 1})`);
      } catch (e) {
        trace(`idle re-prompt failed: ${String(e).slice(0, 80)}`);
      }
    },
  };
};
