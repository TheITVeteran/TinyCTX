"""
modules/heartbeat/__main__.py

Runs periodic agent turns on a configurable interval, isolated on their own
DB branch — never polluting the user's conversation thread.

Branch strategy (configured via "branch_from"):
  "root"    — branch off the global DB root, fully independent of the user session
  "session" — branch off the current tail of the agent's own session at the time
              heartbeat starts (inherits history up to that point, then diverges)

Reply handling:
  - "HEARTBEAT_OK" at start or end → silently dropped
    (if remaining content is ≤ ack_max_chars).
  - Any other reply → printed as a heartbeat alert, then the agent is
    re-prompted: "Continue the task, or reply HEARTBEAT_OK when done."
  - This continuation loop runs up to max_continuations times before giving up.
  - Errors are logged; the background task continues normally.

HEARTBEAT.md in the workspace is read by the agent via the normal filesystem
tools — this module doesn't inject it directly, the prompt tells the agent to.

Active hours: if configured, ticks outside the window are skipped.
The task still sleeps its normal interval; it just does nothing on waking
outside the allowed window.

Slash command:
  /heartbeat run  — fire one tick immediately (replaces /debug heartbeat)

Convention: register_agent(agent) — no imports from gateway or bridges.
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, time as dtime
from pathlib import Path

from TinyCTX.contracts import (
    InboundMessage, ContentType, UserIdentity, Platform,
    AgentTextFinal, AgentError
)

logger = logging.getLogger(__name__)

_HEARTBEAT_USER_ID = "heartbeat-system"
_HEARTBEAT_AUTHOR  = UserIdentity(
    platform=Platform.CRON,
    user_id=_HEARTBEAT_USER_ID,
    username="heartbeat",
)
_TOKEN = "HEARTBEAT_OK"


# ---------------------------------------------------------------------------
# Cursor bootstrap
# ---------------------------------------------------------------------------

def _get_or_create_cursor(agent, branch_from: str) -> tuple[str, str]:
    """
    Return (lane_node_id, tail_node_id) for the heartbeat branch.
    Both ids are stored on the agent instance so subsequent calls return cached values.
    """
    attr_lane = "_heartbeat_lane_node_id"
    attr_tail = "_heartbeat_cursor_node_id"
    if getattr(agent, attr_lane, None):
        return getattr(agent, attr_lane), getattr(agent, attr_tail)

    from TinyCTX.db import ConversationDB
    workspace = Path(agent.config.workspace.path).expanduser().resolve()
    db        = ConversationDB(workspace / "agent.db")

    if branch_from == "session":
        parent_id = agent._tail_node_id
    else:
        parent_id = db.get_root().id

    node = db.add_node(
        parent_id=parent_id,
        role="system",
        content="session:heartbeat",
    )
    setattr(agent, attr_lane, node.id)
    setattr(agent, attr_tail, node.id)
    logger.info(
        "[heartbeat] created branch cursor %s (branch_from=%s, parent=%s)",
        node.id, branch_from, parent_id,
    )
    return node.id, node.id

class _HeartbeatRunner:
    def __init__(self, runtime, cfg: dict) -> None:
        self.runtime = runtime
        self.cfg = cfg
        self.interval_secs = int(cfg.get("every_minutes", 30)) * 60
        self.cursor_node_id: str | None = None
        self._running = False

    def start(self):
        if self.interval_secs <= 0: return
        self._running = True
        asyncio.create_task(self._loop())

    async def _loop(self):
        # Initial delay to let the system stabilize
        await asyncio.sleep(10) 
        
        while self._running:
            if self._in_active_window():
                try:
                    await self._tick()
                except Exception:
                    logger.exception("[heartbeat] tick failed")
            
            await asyncio.sleep(self.interval_secs)

    async def _tick(self):
        # 1. Determine Parent (Root vs Session Branching)
        if not self.cursor_node_id:
            if self.cfg.get("branch_from") == "session":
                # Note: 'session' branching in a multi-user environment 
                # usually implies a specific thread. Here we default to Root 
                # if no specific session is provided to the runner.
                self.cursor_node_id = self.runtime.db.get_root().id
            else:
                self.cursor_node_id = self.runtime.db.get_root().id

        # 2. Continuation Loop
        current_prompt = self.cfg.get("prompt", "If nothing needs attention, reply HEARTBEAT_OK.")
        
        for turn in range(int(self.cfg.get("max_continuations", 5))):
            msg = InboundMessage(
                tail_node_id=self.cursor_node_id,
                author=_HEARTBEAT_AUTHOR,
                text=current_prompt,
                trigger=True
            )

            reply_event = asyncio.Event()
            full_text = []

            async def _collect(ev):
                if isinstance(ev, AgentTextFinal):
                    if ev.text: full_text.append(ev.text)
                    reply_event.set()
                elif isinstance(ev, AgentError):
                    reply_event.set()

            # Push and Listen
            new_node_id = await self.runtime.push(msg)
            self.runtime._cursor_handlers[new_node_id] = _collect
            
            try:
                await asyncio.wait_for(reply_event.wait(), timeout=120)
                # Advance cursor to the assistant's reply
                assistant_node = self.runtime.db.get_child_of(new_node_id)
                self.cursor_node_id = assistant_node.id if assistant_node else new_node_id
                
                # Parse Reply
                reply_content = "".join(full_text).strip()
                is_ok, alert = _parse_reply(reply_content, int(self.cfg.get("ack_max_chars", 300)))
                
                if is_ok:
                    break # Heartbeat satisfied
                
                logger.warning("[HEARTBEAT ALERT]\n%s", alert)
                current_prompt = self.cfg.get("continuation_prompt", "Continue...")
                
            finally:
                self.runtime._cursor_handlers.pop(new_node_id, None)

    def _in_active_window(self) -> bool:
        hours = self.cfg.get("active_hours")
        if not hours: return True
        now = datetime.now().time()
        start = _parse_hhmm(hours["start"])
        end = _parse_hhmm(hours["end"])
        return start <= now <= end if start < end else now >= start or now <= end

# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_global_runner: _HeartbeatRunner | None = None

def register_runtime(runtime) -> None:
    global _global_runner
    try:
        from TinyCTX.modules.heartbeat import EXTENSION_META
        cfg = EXTENSION_META.get("default_config", {})
    except ImportError:
        cfg = {}

    # Guard: check for HEARTBEAT.md
    workspace = Path(runtime.config.workspace.path).expanduser().resolve()
    if not (workspace / "HEARTBEAT.md").exists():
        logger.info("[heartbeat] HEARTBEAT.md missing, disabled.")
        return

    _global_runner = _HeartbeatRunner(runtime, cfg)
    _global_runner.start()

def register_agent(agent) -> None:
    """
    Commands move here to be available to the agent/user.
    """
    registry = getattr(agent, "commands", None)
    if registry and _global_runner:
        async def _cmd_run(args, context):
            asyncio.create_task(_global_runner._tick())
            if "console" in context:
                context["console"].print("[yellow]Heartbeat tick triggered manually.[/yellow]")

        registry.register("heartbeat", "run", _cmd_run, help="Manual heartbeat tick")


# ---------------------------------------------------------------------------
# Reply parsing
# ---------------------------------------------------------------------------

def _parse_reply(reply: str, ack_max: int) -> tuple[bool, str]:
    text    = reply
    matched = False

    if text == "":
        return True, ""

    if text.startswith(_TOKEN):
        text    = text[len(_TOKEN):].lstrip(" \n\r")
        matched = True
    elif text.endswith(_TOKEN):
        text    = text[: -len(_TOKEN)].rstrip(" \n\r")
        matched = True
    return matched and len(text) <= ack_max, text


def _emit_alert(text: str) -> None:
    logger.warning("[HEARTBEAT ALERT]\n%s", text)


# ---------------------------------------------------------------------------
# Active hours
# ---------------------------------------------------------------------------

def _parse_hhmm(s: str) -> dtime:
    h, m = s.strip().split(":")
    return dtime(int(h), int(m))


def _in_active_window(active_hours: dict | None) -> bool:
    if not active_hours:
        return True
    try:
        start = _parse_hhmm(active_hours["start"])
        end_  = _parse_hhmm(active_hours["end"])
    except (KeyError, ValueError):
        logger.warning("[heartbeat] invalid active_hours config — running anyway")
        return True
    if start == end_:
        return False
    now = datetime.now().time().replace(second=0, microsecond=0)
    if start < end_:
        return start <= now < end_
    return now >= start or now < end_
