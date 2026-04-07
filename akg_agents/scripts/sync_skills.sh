#!/usr/bin/env bash
# Copyright 2025-2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Sync .opencode/skills/ to .claude/skills/ and .cursor/skills/
# via individual symlinks so Claude Code and Cursor can discover them.
#
# Usage:
#   ./scripts/sync_skills.sh              # sync akg_agents/
#   ./scripts/sync_skills.sh workspace    # sync workspace/
#   ./scripts/sync_skills.sh all          # sync both

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

sync_dir() {
    local dir="$1"
    local src="$dir/.opencode/skills"

    if [ ! -d "$src" ]; then
        echo "跳过 $dir: .opencode/skills/ 不存在"
        return
    fi

    for tool_dir in ".claude/skills" ".cursor/skills"; do
        local target="$dir/$tool_dir"
        rm -rf "$target"
        mkdir -p "$target"

        for skill in "$src"/*/; do
            [ -d "$skill" ] || continue
            local name
            name=$(basename "$skill")
            ln -sf "../../.opencode/skills/$name" "$target/$name"
        done

        local count
        count=$(ls -1 "$target" 2>/dev/null | wc -l | tr -d ' ')
        echo "  $tool_dir: $count skills"
    done
}

case "${1:-all}" in
    workspace)
        echo "=== workspace/ ==="
        sync_dir "$REPO_ROOT/workspace"
        ;;
    all)
        echo "=== akg_agents/ ==="
        sync_dir "$REPO_ROOT"
        echo ""
        echo "=== workspace/ ==="
        sync_dir "$REPO_ROOT/workspace"
        ;;
    *)
        echo "=== $1/ ==="
        sync_dir "$REPO_ROOT/$1"
        ;;
esac

echo ""
echo "Done. 新增 skill 后重新运行此脚本即可同步。"
