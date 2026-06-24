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

import json
import os
from urllib.parse import quote
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
AGENT_DSL_TEMPLATE = '''from highway_dsl import WorkflowBuilder


def get_workflow():
    builder = WorkflowBuilder(name="agent_run_goal", version="1.0.0")
    builder.task("init", "tools.agent.init_goal", kwargs={"goal": "{{goal}}"})

    def loop_body(b):
        return b.task(
            "agent_turn",
            "tools.agent.run_goal",
            kwargs={
                "goal": "{{goal}}",
                # Model comes from the engine's gateway `agent` route (config), not hardcoded.
                "route": "agent",
                "max_turns": 12,
                "system_prompt": "You are an autonomous agent. Use the available tools to accomplish the user's request, then give the final answer. Guardrails: if you lack the information or a suitable tool, say so plainly; never fabricate, guess, or state unverifiable facts; do not use unrelated tools to manufacture an answer; take only the actions the request actually requires.",
                "tool_catalog": %(catalog)s,
                "tool_config": %(tool_config)s,
                "approval_required_tools": %(approvals)s,
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

# Research orchestrator (kept in sync with the engine repo's api/dsl_templates/agent_research.py):
# a delegate-only agent that fans a multi-part research goal out to parallel research-only
# (http) sub-agents, then synthesizes. Used by the research_goal tool.
RESEARCH_DSL_TEMPLATE = '''from highway_dsl import WorkflowBuilder


def get_workflow():
    builder = WorkflowBuilder(name="agent_research", version="1.0.0")
    builder.task("init", "tools.agent.init_goal", kwargs={"goal": "{{goal}}"})

    def loop_body(b):
        return b.task(
            "agent_turn",
            "tools.agent.run_goal",
            kwargs={
                "goal": "{{goal}}",
                # Model comes from the engine's gateway `agent` route (config), not hardcoded.
                "route": "agent",
                "max_turns": 8,
                "system_prompt": "You are a research orchestrator. Break the request into independent research sub-tasks and delegate them to parallel sub-agents with the delegate tool - each sub-agent researches the web on its own. Then combine their findings into your final answer. Even a single research task should be delegated. Only answer directly for a simple question you already know. Guardrails: never fabricate or state unverifiable facts; if something cannot be done, say so plainly.",
                "tool_catalog": ["delegate"],
                "approval_required_tools": ["tools.agent.spawn_subgoals"],
            },
            result_key="agent_step",
        )

    builder.while_loop("agent_loop", condition="{{agent_done}} == 0", loop_body=loop_body, dependencies=["init"])
    builder.task("report", "tools.shell.run", args=["echo done"], dependencies=["agent_loop"])
    return builder.build()
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


_BASE_CATALOG = ["shell", "http", "email", "gmail", "telegram"]
_BASE_APPROVALS = ["tools.shell.run", "tools.http.request", "tools.email.send", "apps.platform.gmail.send_email"]


def _agent_dsl_for(c: httpx.Client) -> str:
    """Build the run_goal DSL, injecting the tenant's connected MCP servers (if any) so
    the agent auto-discovers and can call their tools, each HITL-gated via tools.mcp.call.
    Tokens stay in Vault: only {name, url, auth_secret_path} are passed."""
    servers: list[dict[str, Any]] = []
    try:
        r = c.get("/mcp/servers")
        if r.status_code == 200:
            for s in (_unwrap(r.json()) or {}).get("servers", []):
                if s.get("enabled", True) and s.get("auth_secret_path"):
                    servers.append({"name": s["name"], "url": s["url"], "auth_secret_path": s["auth_secret_path"]})
    except Exception:  # noqa: BLE001 - a registry hiccup must not block run_goal
        servers = []
    catalog = list(_BASE_CATALOG) + (["mcp"] if servers else [])
    approvals = list(_BASE_APPROVALS) + (["tools.mcp.call"] if servers else [])
    tool_config = {"mcp": {"servers": servers}} if servers else {}
    return AGENT_DSL_TEMPLATE % {
        "catalog": json.dumps(catalog),
        "tool_config": json.dumps(tool_config),
        "approvals": json.dumps(approvals),
    }


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
        dsl = _agent_dsl_for(c)
        r = c.post(
            "/workflows",
            json={"python_dsl": dsl, "inputs": {"goal": goal}, "execute": True},
        )
    r.raise_for_status()
    data = _unwrap(r.json())
    return {"workflow_run_id": data.get("workflow_run_id"), "status": "started"}


@mcp.tool()
def research_goal(goal: str, ctx: Context) -> dict[str, Any]:
    """Research a multi-part question by fanning it out to parallel sub-agents NOW.

    Use this for BREADTH research: the agent breaks the goal into independent sub-tasks and
    runs them as parallel research sub-agents (each browses the web), then synthesizes their
    findings - faster and cleaner than one-at-a-time lookups. Unlike run_goal, this delegates
    rather than acting directly, so it does NOT take outbound actions (email/Gmail/Telegram);
    use run_goal for those. The single delegate step pauses once for approval. Returns a run id.
    """
    with _client(_api_key(ctx)) as c:
        r = c.post(
            "/workflows",
            json={"python_dsl": RESEARCH_DSL_TEMPLATE, "inputs": {"goal": goal}, "execute": True},
        )
    r.raise_for_status()
    data = _unwrap(r.json())
    return {"workflow_run_id": data.get("workflow_run_id"), "status": "started"}


@mcp.tool()
def connect_mcp(name: str, url: str, ctx: Context, auth_token: str | None = None) -> dict[str, Any]:
    """Connect an external MCP server so your agents can use its tools.

    name: a short handle ([a-zA-Z0-9_-], 1-64). url: the server's https Streamable-HTTP
    endpoint. auth_token: optional bearer token (stored in Vault, never in the workflow).
    After connecting, run_goal auto-discovers and can call this server's tools, each
    HITL-gated. Use list_mcp_servers / disconnect_mcp to manage them.
    """
    with _client(_api_key(ctx)) as c:
        r = c.post("/mcp/servers", json={"name": name, "url": url, "auth_token": auth_token})
    r.raise_for_status()
    return _unwrap(r.json())


@mcp.tool()
def list_mcp_servers(ctx: Context) -> dict[str, Any]:
    """List the external MCP servers connected for your tenant."""
    with _client(_api_key(ctx)) as c:
        r = c.get("/mcp/servers")
    r.raise_for_status()
    return _unwrap(r.json())


@mcp.tool()
def disconnect_mcp(name: str, ctx: Context) -> dict[str, Any]:
    """Disconnect an external MCP server by name (removes it and its stored token)."""
    with _client(_api_key(ctx)) as c:
        r = c.delete(f"/mcp/servers/{name}")
    r.raise_for_status()
    return _unwrap(r.json())


@mcp.tool()
def get_trace(workflow_run_id: str, ctx: Context) -> dict[str, Any]:
    """See WHAT an agent run actually did: a per-agent cost/latency/tool trace.

    Returns {totals, agents, spans}: the run's agent tree grouped into agent nodes (the
    orchestrator + each leaf sub-agent), each with its model calls, tool calls, delegation,
    token + USD cost, and latency - plus the individual spans. Use this to understand or
    debug a run: which sub-agent was slow or expensive, which model/route each call used
    (and whether it fell back), which tools ran. The way get_status shows progress, this
    shows the work.
    """
    with _client(_api_key(ctx)) as c:
        r = c.get(f"/workflows/{workflow_run_id}/trace")
    r.raise_for_status()
    return _unwrap(r.json())


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
            # Model comes from the engine's gateway `agent` route (config), not hardcoded.
            "route": "agent",
            "system_prompt": "You are an autonomous scheduled agent running unattended. Use the available tools to accomplish the request, then give the final answer. Guardrails: if you lack the information or a suitable tool, say so plainly; never fabricate, guess, or state unverifiable facts; do not use unrelated tools to manufacture an answer; take only the actions the request actually requires.",
            "max_turns": 10,
            "tool_catalog": ["shell", "http", "email", "gmail", "telegram"],
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


def _watch_goal(match: str, instruction: str) -> str:
    """Build the poll-and-react goal for a Gmail trigger. Dedup is stateless: we
    can't mark mail read (readonly scope), so the agent checks its own Sent mail
    before replying — self-correcting across the durable-cron ticks."""
    return (
        "You are a Gmail auto-responder; each run is a single poll. "
        f'STEP 1: call gmail_search with query "{match} newer_than:1d" and max_results 5. '
        "STEP 2: for each message returned, call gmail_get to read its sender, subject and body. "
        "STEP 3 (DEDUP — critical): before replying to a sender, call gmail_search with "
        '"in:sent to:<that exact sender address> newer_than:1d"; if it returns anything you have '
        "ALREADY replied to them today, so SKIP that message. Never reply to the same sender twice in a day. "
        f"STEP 4: for every message you have NOT already replied to, {instruction} — send it with "
        'gmail_send to the original sender, subject "Re: <their subject>". '
        "STEP 5: if there are no new, unreplied messages, do nothing. "
        "Act directly without asking for approval; keep replies short."
    )


@mcp.tool()
def watch_gmail(match: str, instruction: str, ctx: Context, mode: str = "push", interval_minutes: int = 2, agent: str = "") -> dict[str, Any]:
    """Create a Gmail trigger: when an email matching `match` arrives, automatically do `instruction`.

    Examples:
      watch_gmail("from:jack@example.com", "reply with a short programmer joke")
      watch_gmail("subject:invoice", "reply that we received it and will pay within 7 days")

    `match` is a simple Gmail filter: 'from:<addr>' or 'subject:<text>' (anything else is
    matched as a substring of subject/body). `instruction` is what to do for each new matching
    email (it replies to the sender via Gmail). Requires Gmail connected (connect_gmail).
    Returns a watch handle; stop it with stop_gmail_watch(watch_id).

    mode="push" (default) is REAL-TIME via Gmail push / Cloud Pub/Sub — fires within seconds,
    no polling, exactly-once. mode="poll" uses a durable cron every `interval_minutes` (a
    fallback for environments without Pub/Sub configured).
    """
    if mode == "push":
        m = match.strip()
        low = m.lower()
        if low.startswith("from:"):
            rule_match: dict[str, Any] = {"from": m[5:].strip()}
        elif low.startswith("subject:"):
            rule_match = {"contains": m[8:].strip()}
        else:
            rule_match = {"contains": m}
        rule_body: dict[str, Any] = {"channel": "gmail", "match": rule_match, "instruction": instruction}
        with _client(_api_key(ctx)) as c:
            w = c.post("/triggers/gmail/watch", json={})  # register/refresh the mailbox watch
            w.raise_for_status()
            if agent:
                ar = c.post("/agents", json={"name": agent})
                ar.raise_for_status()
                rule_body["agent_id"] = _unwrap(ar.json()).get("agent_id")
            r = c.post("/triggers/rules", json=rule_body)
        r.raise_for_status()
        rule = _unwrap(r.json())
        return {
            "mode": "push",
            "watching": match,
            "instruction": instruction,
            "agent": agent or None,
            "watch_id": rule.get("rule_id"),
            "note": "Real-time trigger active. Stop with stop_gmail_watch(watch_id).",
        }

    # mode == "poll": durable-cron fallback
    goal = _watch_goal(match, instruction)
    cron = "* * * * *" if int(interval_minutes) <= 1 else f"*/{int(interval_minutes)} * * * *"
    with _client(_api_key(ctx)) as c:
        r1 = c.post("/workflows", json={"python_dsl": SCHEDULED_AGENT_DSL, "execute": False})
        r1.raise_for_status()
        def_id = _unwrap(r1.json()).get("definition_id")
        r2 = c.post("/workflows", json={"python_dsl": _cron_dsl(def_id, cron, goal), "execute": True})
    r2.raise_for_status()
    return {
        "mode": "poll",
        "watching": match,
        "instruction": instruction,
        "interval_minutes": interval_minutes,
        "watch_id": "agent_sched_" + (def_id or "")[:8],
        "note": "Polling trigger active. Stop with stop_gmail_watch(watch_id).",
    }


@mcp.tool()
def stop_gmail_watch(watch_id: str, ctx: Context) -> dict[str, Any]:
    """Stop a Gmail trigger by its watch_id.

    Accepts either a real-time push rule id (UUID, from a push watch_gmail) or a
    polling schedule name ('agent_sched_...'); removes the rule / cancels the schedule.
    """
    with _client(_api_key(ctx)) as c:
        if watch_id.startswith("agent_sched_"):
            r = c.post(f"/workflows/schedules/{watch_id}/cancel", json={})
        else:
            r = c.request("DELETE", f"/triggers/rules/{watch_id}")
    r.raise_for_status()
    return _unwrap(r.json())


@mcp.tool()
def list_gmail_watches(ctx: Context) -> dict[str, Any]:
    """List the caller's active Gmail triggers — real-time push rules and polling schedules."""
    with _client(_api_key(ctx)) as c:
        rules = _unwrap(c.get("/triggers/rules", params={"channel": "gmail"}).json())
        scheds = _unwrap(c.get("/workflows/schedules").json())
    return {"push_rules": rules.get("rules", []), "poll_schedules": scheds.get("schedules", [])}


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
            # Model comes from the engine's gateway `agent` route (config), not hardcoded.
            "route": "agent",
            "system_prompt": "You are an autonomous deferred agent running unattended at the scheduled time. Use the available tools to accomplish the request, then give the final answer. Guardrails: if you lack the information or a suitable tool, say so plainly; never fabricate, guess, or state unverifiable facts; do not use unrelated tools to manufacture an answer; take only the actions the request actually requires.",
            "max_turns": 10,
            "tool_catalog": ["shell", "http", "email", "gmail", "telegram"],
            "approval_required_tools": %(approvals)s,
        }, result_key="agent_step")

    b.while_loop("agent_loop", condition="{{agent_done}} == 0", loop_body=loop_body, dependencies=["init"])
    b.task("report", "tools.shell.run", args=["echo done"], dependencies=["agent_loop"])
    return b.build()


if __name__ == "__main__":
    print(get_workflow().to_json())
'''

_HITL_TOOLS = ["tools.shell.run", "tools.http.request", "tools.email.send", "apps.platform.gmail.send_email"]


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


@mcp.tool()
def connect_gmail(ctx: Context) -> dict[str, Any]:
    """Get a one-time URL to connect the caller's Gmail account to Highway.

    Returns {authorize_url}: a Highway-hosted connect page (it wraps the Google consent
    screen with the "unverified app" guidance). Tell the user to open it in a browser and
    approve. After that, their agents can send/search/read their Gmail (the gmail_send /
    gmail_search / gmail_get tools). One-time per Gmail account; tokens are stored
    server-side per tenant, so scheduled/deferred jobs keep working unattended.
    """
    with _client(_api_key(ctx)) as c:
        r = c.post("/oauth/gmail/authorize", json={})
    r.raise_for_status()
    data = _unwrap(r.json())
    # Wrap the raw Google consent URL in the Highway-hosted landing page (friendlier UX +
    # the "unverified app" instructions). The page reads ?url= and builds the button, so
    # each caller gets their own fresh, signed-state consent link (expires_in_seconds).
    google_url = data.get("authorize_url") if isinstance(data, dict) else None
    if google_url:
        base = os.environ.get("MCP_PUBLIC_BASE_URL", "https://mcp.highway.rodmena.app").rstrip("/")
        data["authorize_url"] = f"{base}/connect-gmail.html?url={quote(google_url, safe='')}"
        data["google_consent_url"] = google_url
    return data


@mcp.tool()
def disconnect_gmail(ctx: Context) -> dict[str, Any]:
    """Disconnect Gmail for the caller (revoke + delete the stored tokens)."""
    with _client(_api_key(ctx)) as c:
        r = c.request("DELETE", "/oauth/gmail/disconnect")
    r.raise_for_status()
    return _unwrap(r.json())


@mcp.tool()
def connect_telegram(bot_token: str, ctx: Context) -> dict[str, Any]:
    """Connect a Telegram bot so your agents can send and react on Telegram.

    Create a bot with @BotFather (send /newbot) to get a token like '8123456789:AA...',
    then call this. It validates the token, stores it server-side, and registers the bot's
    webhook so incoming messages can trigger reactions (see watch_telegram). One-time per bot.
    Returns the bot @username.
    """
    with _client(_api_key(ctx)) as c:
        r = c.post("/integrations/telegram/connect", json={"bot_token": bot_token})
    r.raise_for_status()
    return _unwrap(r.json())


@mcp.tool()
def watch_telegram(match: str, instruction: str, ctx: Context, agent: str = "") -> dict[str, Any]:
    """Create a real-time Telegram trigger: when a message matching `match` arrives at your bot, do `instruction`.

    `match` is 'from:<username>' or any text (matched as a substring of the message). `instruction`
    is what to do for each matching message (it replies in the same chat). Requires connect_telegram
    first. Fires within seconds. Returns a watch handle; stop it with stop_telegram_watch(watch_id).
    """
    m = match.strip()
    rule_match: dict[str, Any] = {"from": m[5:].strip()} if m.lower().startswith("from:") else {"contains": m}
    rule_body: dict[str, Any] = {"channel": "telegram", "match": rule_match, "instruction": instruction}
    with _client(_api_key(ctx)) as c:
        if agent:
            ar = c.post("/agents", json={"name": agent})
            ar.raise_for_status()
            rule_body["agent_id"] = _unwrap(ar.json()).get("agent_id")
        r = c.post("/triggers/rules", json=rule_body)
    r.raise_for_status()
    rule = _unwrap(r.json())
    return {
        "channel": "telegram",
        "watching": match,
        "instruction": instruction,
        "agent": agent or None,
        "watch_id": rule.get("rule_id"),
        "note": "Real-time trigger active. Stop with stop_telegram_watch(watch_id).",
    }


@mcp.tool()
def stop_telegram_watch(watch_id: str, ctx: Context) -> dict[str, Any]:
    """Stop a Telegram trigger by its watch_id (the rule id returned by watch_telegram)."""
    with _client(_api_key(ctx)) as c:
        r = c.request("DELETE", f"/triggers/rules/{watch_id}")
    r.raise_for_status()
    return _unwrap(r.json())


@mcp.tool()
def list_automations(ctx: Context) -> dict[str, Any]:
    """List everything running for the caller: connected channels (Gmail/Telegram),
    trigger rules across all channels, and scheduled jobs. The "what's set up for me" view.
    """
    with _client(_api_key(ctx)) as c:
        r = c.get("/automations")
    r.raise_for_status()
    return _unwrap(r.json())


@mcp.tool()
def list_runs(ctx: Context, limit: int = 20) -> dict[str, Any]:
    """List the caller's recent agent runs (most recent first) — goal runs, reactions,
    and scheduled jobs — with status and a one-line result summary. The "what happened" view.
    """
    with _client(_api_key(ctx)) as c:
        r = c.get("/agent-runs", params={"limit": limit})
    r.raise_for_status()
    return _unwrap(r.json())


@mcp.tool()
def create_agent(name: str, ctx: Context, description: str = "") -> dict[str, Any]:
    """Create a named agent — a bundle of triggers you manage as one unit.

    After creating, attach triggers with watch_gmail(..., agent=name) or
    watch_telegram(..., agent=name). Pause/resume/delete the whole bundle with
    pause_agent / resume_agent / delete_agent. Use a simple name (e.g. "SupportBot").
    """
    with _client(_api_key(ctx)) as c:
        r = c.post("/agents", json={"name": name, "description": description or None})
    r.raise_for_status()
    return _unwrap(r.json())


@mcp.tool()
def list_agents(ctx: Context) -> dict[str, Any]:
    """List the caller's named agents with their status and trigger counts."""
    with _client(_api_key(ctx)) as c:
        r = c.get("/agents")
    r.raise_for_status()
    return _unwrap(r.json())


@mcp.tool()
def get_agent(agent: str, ctx: Context) -> dict[str, Any]:
    """Get a named agent's detail (its triggers). `agent` is the name or id."""
    with _client(_api_key(ctx)) as c:
        r = c.get(f"/agents/{agent}")
    r.raise_for_status()
    return _unwrap(r.json())


@mcp.tool()
def pause_agent(agent: str, ctx: Context) -> dict[str, Any]:
    """Pause a named agent — disables all its triggers. `agent` is the name or id."""
    with _client(_api_key(ctx)) as c:
        r = c.post(f"/agents/{agent}/pause", json={})
    r.raise_for_status()
    return _unwrap(r.json())


@mcp.tool()
def resume_agent(agent: str, ctx: Context) -> dict[str, Any]:
    """Resume a paused agent — re-enables all its triggers. `agent` is the name or id."""
    with _client(_api_key(ctx)) as c:
        r = c.post(f"/agents/{agent}/resume", json={})
    r.raise_for_status()
    return _unwrap(r.json())


@mcp.tool()
def delete_agent(agent: str, ctx: Context) -> dict[str, Any]:
    """Delete a named agent and all its triggers. `agent` is the name or id."""
    with _client(_api_key(ctx)) as c:
        r = c.request("DELETE", f"/agents/{agent}")
    r.raise_for_status()
    return _unwrap(r.json())


@mcp.tool()
def set_trigger_enabled(rule_id: str, enabled: bool, ctx: Context) -> dict[str, Any]:
    """Enable or disable a single trigger rule by its id (from list_automations / watch_*)."""
    with _client(_api_key(ctx)) as c:
        r = c.request("PATCH", f"/triggers/rules/{rule_id}", json={"enabled": enabled})
    r.raise_for_status()
    return _unwrap(r.json())


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
