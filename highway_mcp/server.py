"""Highway MCP server — the front door for Highway Agents (issue #749).

A remote MCP server that wraps the Highway Workflow Engine REST API so any MCP
host (Claude Desktop/Code, opencode, Cursor) can drive durable, always-on agents
with no infrastructure of their own:

  - run_goal(goal)              start a durable agent run; returns a run id
  - get_status(workflow_run_id) poll status/progress/result + pending approvals
  - approve(approval_key, ...)  resolve a human-in-the-loop approval to resume

The agent loop runs server-side on Highway, so it keeps going (listening,
acting, waiting for approval) even after this client disconnects. The user
re-attaches from any device to check status or approve.

Auth (Phase 0): the customer's Highway API key (hw_k1_...) is read from
HIGHWAY_API_KEY and forwarded as a Bearer token (the engine's RBAC middleware
accepts hw_k1_ keys inline). Per-connection auth (one key per customer, taken
from the MCP session) is the production follow-up — see README.

Run:
  HIGHWAY_BASE_URL=http://localhost:7822 HIGHWAY_API_KEY=hw_k1_xxx \
      python -m highway_mcp.server                      # stdio (default)
  TRANSPORT=http HIGHWAY_BASE_URL=... HIGHWAY_API_KEY=... \
      python -m highway_mcp.server                      # streamable-http
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP


BASE_URL = os.environ.get("HIGHWAY_BASE_URL", "http://localhost:7822").rstrip("/")
API_KEY = os.environ.get("HIGHWAY_API_KEY", "")
API_PREFIX = "/api/v1"
HTTP_PORT = int(os.environ.get("PORT", "8848"))

# Fixed durable agent workflow (Highway DSL generator). The goal is supplied at
# run time via `inputs`, so this source is identical for every run. Kept here as
# a self-contained string so the MCP server has no dependency on highway_dsl;
# the engine's dsl-compiler turns it into JSON on submit. Source of truth lives
# in the engine repo at api/dsl_templates/agent_run_goal.py — keep in sync.
AGENT_DSL = '''from highway_dsl import WorkflowBuilder


def get_workflow():
    builder = WorkflowBuilder(name="agent_run_goal", version="1.0.0")
    builder.task("init", "tools.agent.init_goal", kwargs={"goal": "{{goal}}"})

    def loop_body(b):
        return b.task(
            "agent_turn",
            "tools.agent.run_goal",
            kwargs={
                "goal": "{{goal}}",
                "provider": "ollama",
                "model": "gemma4:31b-cloud",
                "function_model": "functiongemma:270m",
                "max_turns": 12,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "shell_run",
                            "description": "Run a shell command and return its stdout.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "args": {"type": "array", "items": {"type": "string"}}
                                },
                                "required": ["args"],
                            },
                        },
                    }
                ],
                "tool_map": {"shell_run": "tools.shell.run"},
                "approval_required_tools": ["tools.shell.run"],
            },
            result_key="agent_step",
        )

    builder.while_loop(
        "agent_loop",
        condition="{{agent_done}} == 0",
        loop_body=loop_body,
        dependencies=["init"],
    )
    builder.task(
        "report",
        "tools.shell.run",
        args=["echo done"],
        dependencies=["agent_loop"],
    )
    return builder.build()


if __name__ == "__main__":
    print(get_workflow().to_json())
'''

mcp = FastMCP("highway-agents")


def _headers() -> dict[str, str]:
    if not API_KEY:
        raise RuntimeError("HIGHWAY_API_KEY is not set (expected an hw_k1_... key)")
    return {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def _client() -> httpx.Client:
    # Sync client on purpose: the engine's own lesson #721 is that async httpx in
    # workers causes zombie threads; sync is simpler and equally capable here.
    return httpx.Client(base_url=BASE_URL + API_PREFIX, headers=_headers(), timeout=30.0)


def _unwrap(payload: Any) -> Any:
    """Return the `data` field of a Highway success envelope, or the body itself."""
    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]
    return payload


@mcp.tool()
def run_goal(goal: str) -> dict[str, Any]:
    """Start a durable Highway agent that pursues `goal` to completion.

    Returns immediately with a workflow_run_id. The agent runs server-side and
    survives this client disconnecting. Use get_status to follow it and to see
    any actions awaiting your approval.
    """
    with _client() as c:
        r = c.post(
            "/workflows",
            json={"python_dsl": AGENT_DSL, "inputs": {"goal": goal}, "execute": True},
        )
    r.raise_for_status()
    data = _unwrap(r.json())
    return {"workflow_run_id": data.get("workflow_run_id"), "status": "started"}


@mcp.tool()
def get_status(workflow_run_id: str) -> dict[str, Any]:
    """Get an agent run's status, progress, result, and any pending approvals.

    `pending_approvals` lists actions the agent has paused on; pass an
    approval_key to the `approve` tool to let it continue.
    """
    with _client() as c:
        wf = c.get(f"/workflows/{workflow_run_id}")
        wf.raise_for_status()
        d = _unwrap(wf.json())
        ap = c.get("/approvals", params={"workflow_run_id": workflow_run_id, "status": "pending"})
        approvals_raw = _unwrap(ap.json()) if ap.status_code == 200 else {}

    if isinstance(approvals_raw, dict):
        approvals = approvals_raw.get("approvals", [])
    elif isinstance(approvals_raw, list):
        approvals = approvals_raw
    else:
        approvals = []

    return {
        "status": d.get("status"),
        "current_step": d.get("current_step"),
        "progress": d.get("progress"),
        "result": d.get("result"),
        "error": d.get("error"),
        "pending_approvals": [
            {
                "approval_key": a.get("approval_key"),
                "title": a.get("title"),
                "description": a.get("description"),
                "data": a.get("approval_data"),
            }
            for a in approvals
        ],
    }


@mcp.tool()
def approve(approval_key: str, approved: bool = True, comment: str | None = None) -> dict[str, Any]:
    """Approve (or reject) a pending agent action to resume the durable run.

    Set approved=False to reject. This works across sessions and devices: you
    can approve from one client an action a run started from another.
    """
    action = "approve" if approved else "reject"
    with _client() as c:
        r = c.post(f"/approvals/{approval_key}/{action}", json={"comment": comment or ""})
    r.raise_for_status()
    return _unwrap(r.json())


def main() -> None:
    transport = os.environ.get("TRANSPORT", "stdio")
    if transport in ("http", "streamable-http"):
        mcp.settings.port = HTTP_PORT
        mcp.run(transport="streamable-http")
    else:
        mcp.run()


if __name__ == "__main__":
    main()
