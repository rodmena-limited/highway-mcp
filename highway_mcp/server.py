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

Auth: each request carries the customer's Highway API key (hw_k1_...) as an
`Authorization: Bearer` header; the server reads it per request so every caller
acts as their own tenant (the tenant is derived from the key by Highway). For
local stdio use, HIGHWAY_API_KEY is the fallback.

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
from mcp.server.fastmcp import Context, FastMCP


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
                "function_model": "gemma4:31b-cloud",
                "function_base_url": "https://ollama.com",
                "max_turns": 12,
                "system_prompt": "You are an autonomous agent. Use the available tools (shell, HTTP, email) to accomplish the user's request. When done, give the final answer.",
                "tool_catalog": ["shell", "http", "email"],
                "approval_required_tools": ["tools.shell.run", "tools.http.request", "tools.email.send"],
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


def _api_key(ctx: Context | None) -> str:
    """Resolve the Highway API key for THIS request.

    Public/multi-tenant: each caller sends their own hw_k1_ key as an
    `Authorization: Bearer` header, so they act as their own tenant (Highway
    derives the tenant from the key). Falls back to the env key for local stdio.
    """
    auth = ""
    try:
        req = ctx.request_context.request if ctx is not None else None
        if req is not None:
            auth = req.headers.get("authorization", "") or ""
    except (ValueError, AttributeError):
        auth = ""
    key = auth[7:].strip() if auth[:7].lower() == "bearer " else auth.strip()
    key = key or API_KEY  # stdio/local fallback
    if not key:
        raise RuntimeError(
            "No Highway API key. Send 'Authorization: Bearer hw_k1_...' from your "
            "MCP client (or set HIGHWAY_API_KEY for local stdio use)."
        )
    if not key.startswith("hw_k1_"):
        raise RuntimeError("Invalid Highway API key (expected an hw_k1_... key).")
    return key


def _client(api_key: str) -> httpx.Client:
    # Sync client on purpose: the engine's own lesson #721 is that async httpx in
    # workers causes zombie threads; sync is simpler and equally capable here.
    return httpx.Client(
        base_url=BASE_URL + API_PREFIX,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=30.0,
    )


def _unwrap(payload: Any) -> Any:
    """Return the `data` field of a Highway success envelope, or the body itself."""
    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]
    return payload


@mcp.tool()
def run_goal(goal: str, ctx: Context) -> dict[str, Any]:
    """Start a durable Highway agent that pursues `goal` to completion NOW.

    Use this for IMMEDIATE actions the user is present to oversee: outbound/risky
    steps pause for approval (get_status surfaces them; resolve with `approve`).
    For anything DEFERRED ("in 10 minutes", "tomorrow", "at 3pm") use
    `run_goal_deferred`, and for RECURRING jobs use `schedule_goal` — those run
    pre-approved/unattended so they never block on an approval when nobody is
    around. Returns a workflow_run_id; the run survives this client disconnecting.
    """
    with _client(_api_key(ctx)) as c:
        r = c.post(
            "/workflows",
            json={"python_dsl": AGENT_DSL, "inputs": {"goal": goal}, "execute": True},
        )
    r.raise_for_status()
    data = _unwrap(r.json())
    return {"workflow_run_id": data.get("workflow_run_id"), "status": "started"}


@mcp.tool()
def get_status(workflow_run_id: str, ctx: Context) -> dict[str, Any]:
    """Get an agent run's status, progress, result, and any pending approvals.

    `pending_approvals` lists actions the agent has paused on; pass an
    approval_key to the `approve` tool to let it continue.
    """
    with _client(_api_key(ctx)) as c:
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
def approve(approval_key: str, ctx: Context, approved: bool = True, comment: str | None = None) -> dict[str, Any]:
    """Approve (or reject) a pending agent action to resume the durable run.

    Set approved=False to reject. This works across sessions and devices: you
    can approve from one client an action a run started from another.
    """
    action = "approve" if approved else "reject"
    with _client(_api_key(ctx)) as c:
        r = c.post(f"/approvals/{approval_key}/{action}", json={"comment": comment or ""})
    r.raise_for_status()
    return _unwrap(r.json())


# No-HITL agent definition for SCHEDULED (unattended) runs. The goal arrives via
# inputs; the schedule itself is the authorization, so there is no per-run approval.
SCHEDULED_AGENT_DSL = '''from highway_dsl import WorkflowBuilder


def get_workflow():
    builder = WorkflowBuilder(name="agent_scheduled", version="1.0.0")
    builder.task("init", "tools.agent.init_goal", kwargs={"goal": "{{goal}}"})

    def loop_body(b):
        return b.task("agent_turn", "tools.agent.run_goal", kwargs={
            "goal": "{{goal}}",
            "provider": "ollama",
            "model": "gemma4:31b-cloud",
            "function_model": "gemma4:31b-cloud",
            "function_base_url": "https://ollama.com",
            "system_prompt": "You are an autonomous scheduled agent running unattended. Use the available tools (shell, HTTP, email) to accomplish the request, then give the final answer.",
            "max_turns": 10,
            "tool_catalog": ["shell", "http", "email"],
            "approval_required_tools": [],
        }, result_key="agent_step")

    builder.while_loop("agent_loop", condition="{{agent_done}} == 0", loop_body=loop_body, dependencies=["init"])
    builder.task("report", "tools.shell.run", args=["echo done"], dependencies=["agent_loop"])
    return builder.build()


if __name__ == "__main__":
    print(get_workflow().to_json())
'''

# durable_cron wrapper that spawns the scheduled agent definition on a cron.
_CRON_DSL_TEMPLATE = '''from highway_dsl import WorkflowBuilder


def get_workflow():
    b = WorkflowBuilder(name="agent_cron", version="1.0.0")
    b.task("schedule", "tools.cron.durable_cron", kwargs={
        "job_name": %(job)s,
        "target_task_name": "tools.workflow.execute",
        "cron_expression": %(cron)s,
        "definition_id": %(def_id)s,
        "workflow_version": 1,
        "target_params": {"inputs": {"goal": %(goal)s}},
        "target_queue": "highway_default",
    })
    return b.build()


if __name__ == "__main__":
    print(get_workflow().to_json())
'''


def _cron_dsl(definition_id: str, cron_expression: str, goal: str) -> str:
    import json as _json

    return _CRON_DSL_TEMPLATE % {
        "job": _json.dumps("agent_sched_" + definition_id[:8]),
        "cron": _json.dumps(cron_expression),
        "def_id": _json.dumps(definition_id),
        "goal": _json.dumps(goal),
    }


@mcp.tool()
def schedule_goal(goal: str, cron_expression: str, ctx: Context) -> dict[str, Any]:
    """Schedule an agent to run `goal` repeatedly on a cron schedule (UNATTENDED).

    cron_expression is standard 5-field cron in UTC, e.g. '0 9 * * *' = daily 09:00,
    '*/30 * * * *' = every 30 minutes. The scheduled agent runs WITHOUT per-run human
    approval (creating the schedule is the authorization), so only schedule goals you
    trust. Uses Highway's durable cron (survives restarts; no history bloat). For a
    ONE-TIME future run (not recurring), use `run_goal_deferred` instead. Returns
    the schedule details.
    """
    with _client(_api_key(ctx)) as c:
        r1 = c.post("/workflows", json={"python_dsl": SCHEDULED_AGENT_DSL, "execute": False})
        r1.raise_for_status()
        def_id = _unwrap(r1.json()).get("definition_id")
        r2 = c.post(
            "/workflows",
            json={"python_dsl": _cron_dsl(def_id, cron_expression, goal), "execute": True},
        )
    r2.raise_for_status()
    d = _unwrap(r2.json())
    return {
        "scheduled": True,
        "cron": cron_expression,
        "goal": goal,
        "job_name": "agent_sched_" + (def_id or "")[:8],
        "definition_id": def_id,
        "cron_run_id": d.get("workflow_run_id"),
    }


# One-time DEFERRED agent: durably wait, then run the goal once, UNATTENDED.
# The wait uses Highway's durable timer (builder.wait -> WaitOperator) which
# suspends the workflow and releases the worker until the fire time — it survives
# restarts and does not depend on any client being online. The goal arrives via
# inputs; `delay` and the approval policy are baked in at schedule time.
_DEFERRED_DSL_TEMPLATE = '''from datetime import timedelta
from highway_dsl import WorkflowBuilder


def get_workflow():
    b = WorkflowBuilder(name="agent_deferred", version="1.0.0")
    b.wait("deferred_wait", wait_for=timedelta(seconds=%(delay)d))
    b.task("init", "tools.agent.init_goal", kwargs={"goal": "{{goal}}"}, dependencies=["deferred_wait"])

    def loop_body(bb):
        return bb.task("agent_turn", "tools.agent.run_goal", kwargs={
            "goal": "{{goal}}",
            "provider": "ollama",
            "model": "gemma4:31b-cloud",
            "function_model": "gemma4:31b-cloud",
            "function_base_url": "https://ollama.com",
            "system_prompt": "You are an autonomous deferred agent running unattended at the scheduled time. Use the available tools (shell, HTTP, email) to accomplish the request, then give the final answer.",
            "max_turns": 10,
            "tool_catalog": ["shell", "http", "email"],
            "approval_required_tools": %(approvals)s,
        }, result_key="agent_step")

    b.while_loop("agent_loop", condition="{{agent_done}} == 0", loop_body=loop_body, dependencies=["init"])
    b.task("report", "tools.shell.run", args=["echo done"], dependencies=["agent_loop"])
    return b.build()


if __name__ == "__main__":
    print(get_workflow().to_json())
'''

_HITL_TOOLS = ["tools.shell.run", "tools.http.request", "tools.email.send"]


def _deferred_dsl(delay_seconds: int, require_approval: bool) -> str:
    import json as _json

    return _DEFERRED_DSL_TEMPLATE % {
        "delay": int(delay_seconds),
        "approvals": _json.dumps(_HITL_TOOLS if require_approval else []),
    }


@mcp.tool()
def run_goal_deferred(
    goal: str,
    delay_seconds: int,
    ctx: Context,
    require_approval: bool = False,
) -> dict[str, Any]:
    """Run `goal` ONCE at a future time — durable, unattended, no machine needed.

    Use this for ANY "do X later" request: "send me an email in 10 minutes",
    "remind me tomorrow at 9am", "in 2 hours fetch the report and email it".
    Convert the user's time into `delay_seconds` from now (10 min = 600; for an
    absolute time like tomorrow 9am, compute the seconds until then).

    The job waits durably on Highway — it survives restarts and holds no worker —
    then runs WITHOUT any per-action approval. Scheduling it now (while the user is
    here) IS the authorization, so it fires even if the user's device is offline.
    This is the whole point of deferred jobs; do NOT use `run_goal` for them (its
    approval would block at the future time when nobody is around to approve).

    Set require_approval=True ONLY if the user explicitly says to ask before it
    acts. Returns the workflow_run_id (poll with get_status).
    """
    if delay_seconds < 0:
        raise ValueError("delay_seconds must be >= 0")
    if delay_seconds > 86400 * 30:
        raise ValueError("delay_seconds exceeds the 30-day maximum")
    with _client(_api_key(ctx)) as c:
        r = c.post(
            "/workflows",
            json={
                "python_dsl": _deferred_dsl(delay_seconds, require_approval),
                "inputs": {"goal": goal},
                "execute": True,
            },
        )
    r.raise_for_status()
    data = _unwrap(r.json())
    return {
        "workflow_run_id": data.get("workflow_run_id"),
        "status": "deferred",
        "delay_seconds": delay_seconds,
        "require_approval": require_approval,
        "goal": goal,
    }


def main() -> None:
    transport = os.environ.get("TRANSPORT", "stdio")
    if transport in ("http", "streamable-http"):
        from mcp.server.transport_security import TransportSecuritySettings

        # Bind to localhost only — nginx terminates TLS and fronts /mcp.
        mcp.settings.host = os.environ.get("HOST", "127.0.0.1")
        mcp.settings.port = HTTP_PORT
        # Binding to localhost makes FastMCP lock DNS-rebinding protection to
        # localhost. nginx forwards the real Host, so allow it explicitly (nginx
        # is the true host boundary). Keep protection on for defence in depth.
        allowed_hosts = [
            h.strip()
            for h in os.environ.get(
                "MCP_ALLOWED_HOSTS", "mcp.highway.rodmena.app,127.0.0.1:*,localhost:*"
            ).split(",")
            if h.strip()
        ]
        allowed_origins = [
            o.strip()
            for o in os.environ.get("MCP_ALLOWED_ORIGINS", "https://mcp.highway.rodmena.app").split(",")
            if o.strip()
        ]
        mcp.settings.transport_security = TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=allowed_hosts,
            allowed_origins=allowed_origins,
        )
        mcp.run(transport="streamable-http")
    else:
        mcp.run()


if __name__ == "__main__":
    main()
