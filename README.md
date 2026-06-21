# Highway MCP

The MCP front door for **Highway Agents**. It lets any MCP host (Claude
Desktop/Code, opencode, Cursor, Claude mobile) drive durable, always-on agents
that run server-side on the Highway Workflow Engine — the user runs no
infrastructure, just an MCP client and a Highway API key.

It exposes three tools, each a thin wrapper over the Highway REST API:

| Tool | What it does | REST call |
|------|--------------|-----------|
| `run_goal(goal)` | Start a durable agent run; returns `workflow_run_id`. The agent keeps running even if the client disconnects. | `POST /api/v1/workflows` |
| `get_status(workflow_run_id)` | Status, progress, result, and `pending_approvals`. | `GET /api/v1/workflows/<id>` + `GET /api/v1/approvals` |
| `approve(approval_key, approved, comment)` | Resolve a human-in-the-loop approval to resume the run (works across devices/sessions). | `POST /api/v1/approvals/<key>/approve\|reject` |

The agent loop itself is the engine tool `tools.agent.run_goal` driven by the
`agent_run_goal` workflow (one durable model turn per iteration, HITL approval
before outbound actions). See the engine repo (`enterprise/tools/agent.py`,
`api/dsl_templates/agent_run_goal.py`) and issue #749.

## Prerequisites

- The Highway stack running and reachable (default `http://localhost:7822`).
- A Highway API key (`hw_k1_...`) with permissions `submit_workflows`,
  `view_workflows`, `view_approvals`, `approve_workflows`. Mint one via
  `POST /api/v1/admin/api-keys` (see engine docs).

## Run

```bash
pip install -e .

# stdio (for a local host like opencode/Claude Desktop)
HIGHWAY_BASE_URL=http://localhost:7822 HIGHWAY_API_KEY=hw_k1_xxx highway-mcp

# remote / streamable-http (reachable by laptop + mobile clients)
TRANSPORT=http PORT=8848 HIGHWAY_BASE_URL=https://highway.rodmena.app \
  HIGHWAY_API_KEY=hw_k1_xxx highway-mcp
```

## Connect from opencode (dogfood)

Add to the host's MCP config (stdio example):

```json
{
  "mcpServers": {
    "highway-agents": {
      "command": "highway-mcp",
      "env": {
        "HIGHWAY_BASE_URL": "http://localhost:7822",
        "HIGHWAY_API_KEY": "hw_k1_xxx"
      }
    }
  }
}
```

Then: `run_goal("summarise what files are in /tmp")` → the agent asks to run a
shell command → `get_status(<id>)` shows a pending approval → `approve(<key>)` →
the run resumes and completes; `get_status` returns the result.

## Configuration

| Env var | Default | Purpose |
|---------|---------|---------|
| `HIGHWAY_BASE_URL` | `http://localhost:7822` | Engine base URL |
| `HIGHWAY_API_KEY` | (required) | `hw_k1_...` key, forwarded as Bearer |
| `TRANSPORT` | `stdio` | `stdio` or `http`/`streamable-http` |
| `PORT` | `8848` | HTTP port when `TRANSPORT=http` |

## Status / follow-ups (Phase 0 spike)

- **Auth is single-key from env.** Production needs per-connection auth so each
  customer's own key (from the MCP session/OAuth) is forwarded — not one shared
  server key.
- **Tool surface is a placeholder** (`shell_run` → `tools.shell.run`, approval
  required). Validate the argument contract live and replace with the real
  connector tools (Gmail etc.) in Phase 1.
- **Long-running calls** already use the start→poll pattern (`run_goal` returns
  a handle; `get_status` polls). Progress-notification/resource streaming is a
  later nicety.
- Engine-side durability caveat for side-effecting tools: issue #750.
