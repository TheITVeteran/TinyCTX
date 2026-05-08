"""
bridges/matrix/__main__.py — Matrix bridge for TinyCTX.

Uses matrix-nio (pip install matrix-nio).

Config (in config.yaml under bridges.matrix.options):
  homeserver:        Full URL of your homeserver, e.g. https://matrix.org
  username:          Full Matrix ID, e.g. @yourbot:matrix.org
  password_env:      Name of the env var holding the account password.
                     Default: MATRIX_PASSWORD
  device_name:       Device display name registered with the server.
                     Default: TinyCTX
  store_path:        Path (relative to workspace) for nio's E2EE key store.
                     Default: matrix_store
  allowed_users:     Allowlist of Matrix user IDs (full MXIDs, e.g.
                     "@you:matrix.org") permitted to interact with the bot.
                     Empty list = open to everyone.
                     Default: []  (WARNING: open access — set this!)
  admin_users:       List of Matrix user IDs (full MXIDs) permitted to use
                     /reset in group rooms. Empty = nobody can reset.
                     Default: []
  dm_enabled:        Respond to 1-on-1 rooms. Default: true
  room_ids:          Whitelist of room IDs to respond in. Empty = all rooms
                     the bot is joined to. Default: []
  prefix_required:   In non-DM rooms, only respond when @mentioned or when
                     the message starts with command_prefix. Default: true
  command_prefix:    Text prefix that triggers the bot in rooms.
                     Default: "!"
  reset_command:     Command string that triggers a session reset in group rooms.
                     Default: "/reset"
  max_reply_length:  Max characters per Matrix message before chunking.
                     Default: 16000
  sync_timeout_ms:   Long-poll timeout per /sync call in ms. Default: 30000

Cursor persistence:
  All cursors (DMs and group rooms) are persisted to
  workspace/cursors/matrix.json so sessions survive bot restarts. The file maps
  cursor_key strings to DB node UUIDs:
    "dm:<sender_mxid>"    → node_id
    "group:<room_id>"     → node_id  (advances with each turn)

Password setup:
  export MATRIX_PASSWORD=your-password-here

Finding your MXID:
  Your full Matrix ID is shown in your client under Settings → Profile,
  in the format @username:homeserver.tld

Required:
  pip install matrix-nio
  For E2EE support: pip install matrix-nio[e2e]
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

from nio import (
    AsyncClient,
    AsyncClientConfig,
    LoginResponse,
    MatrixRoom,
    RoomMessageText,
    SyncError,
)
try:
    from nio import (
        RoomMessageAudio,
        RoomMessageFile,
        RoomMessageImage,
        RoomMessageVideo,
    )
    _HAS_MEDIA_EVENTS = True
except ImportError:
    RoomMessageAudio = RoomMessageFile = RoomMessageImage = None  # type: ignore
    RoomMessageVideo = None                                        # type: ignore
    _HAS_MEDIA_EVENTS = False

from TinyCTX.contracts import (
    AgentError,
    AgentThinkingChunk,
    AgentTextChunk,
    AgentTextFinal,
    AgentToolCall,
    AgentToolResult,
    Attachment,
    content_type_for,
    InboundMessage,
    Platform,
    UserIdentity,
)

if TYPE_CHECKING:
    from TinyCTX.runtime import Runtime

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULTS = {
    "password_env": "MATRIX_PASSWORD",
    "device_name": "TinyCTX",
    "store_path": "matrix_store",
    "allowed_users": [],
    "admin_users": [],
    "dm_enabled": True,
    "room_ids": [],
    "prefix_required": True,
    "command_prefix": "!",
    "reset_command": "/reset",
    "max_reply_length": 16000,
    "sync_timeout_ms": 30000,
    "typing_indicator": True,
    "typing_on_thinking": True,
    "typing_on_tools": True,
    "typing_on_reply": True,
}


# ---------------------------------------------------------------------------
# Mention humanization
# ---------------------------------------------------------------------------

_MATRIX_HTML_MENTION = re.compile(
    r'<a\s+href="https://matrix\.to/#/(@[^"]+)"[^>]*>([^<]*)</a>',
    re.IGNORECASE,
)
_MATRIX_PLAIN_MXID = re.compile(r"@[\w\-.]+:[\w\-.]+")


def _humanize_matrix_mentions(text: str, own_mxid: str) -> str:
    """Normalize Matrix mention formats to @localpart for LLM readability."""
    def _replace_html(m: re.Match) -> str:
        mxid = m.group(1)
        if mxid == own_mxid:
            return ""
        return f"@{mxid.split(':')[0].lstrip('@')}"

    text = _MATRIX_HTML_MENTION.sub(_replace_html, text)

    def _replace_plain(m: re.Match) -> str:
        mxid = m.group(0)
        if mxid == own_mxid:
            return ""
        return f"@{mxid.split(':')[0].lstrip('@')}"

    text = _MATRIX_PLAIN_MXID.sub(_replace_plain, text)
    return text.strip()


# ---------------------------------------------------------------------------
# Trigger detection (bridge-local, mirrors Discord's _is_group_trigger)
# ---------------------------------------------------------------------------

def _is_group_trigger(text: str, own_mxid: str, localpart: str,
                      prefix: str, prefix_required: bool) -> bool:
    """Return True if this group message should trigger an agent response."""
    if not prefix_required:
        return True
    if prefix and text.startswith(prefix):
        return True
    if own_mxid and own_mxid in text:
        return True
    if localpart and f"@{localpart}" in text:
        return True
    return False


# ---------------------------------------------------------------------------
# Cursor store
# ---------------------------------------------------------------------------

class CursorStore:
    """Persists matrix.json under workspace/cursors/."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._cursors: dict[str, str] = self._load()

    def _load(self) -> dict[str, str]:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except Exception:
                logger.warning("Matrix: corrupt cursor file %s — starting fresh", self._path)
        return {}

    def _save(self) -> None:
        try:
            self._path.write_text(
                json.dumps(self._cursors, indent=2), encoding="utf-8"
            )
        except Exception:
            logger.exception("Matrix: failed to save cursor file %s", self._path)

    def get(self, key: str) -> str | None:
        return self._cursors.get(key)

    def set(self, key: str, node_id: str) -> None:
        self._cursors[key] = node_id
        self._save()

    def delete(self, key: str) -> None:
        self._cursors.pop(key, None)
        self._save()


# ---------------------------------------------------------------------------
# Reply accumulator
# ---------------------------------------------------------------------------

class _ReplyAccumulator:
    def __init__(self, max_len: int) -> None:
        self._max_len = max_len
        self._buf: list[str] = []
        self._done = asyncio.Event()
        self._error: str | None = None

    def feed(self, chunk: str) -> None:
        self._buf.append(chunk)

    def finish(self, final_text: str) -> None:
        if final_text and not self._buf:
            self._buf.append(final_text)
        self._done.set()

    def error(self, message: str) -> None:
        self._error = message
        self._done.set()

    async def wait(self) -> list[str]:
        await self._done.wait()
        if self._error:
            return [f"⚠️ {self._error}"]
        text = "".join(self._buf).strip()
        if not text:
            return []
        return [text[i : i + self._max_len] for i in range(0, len(text), self._max_len)]


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _make_session_node(db, cursor_key: str) -> str:
    root = db.get_root()
    node = db.add_node(parent_id=root.id, role="system", content=f"session:{cursor_key}")
    return node.id


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------

class MatrixBridge:
    def __init__(self, runtime: "Runtime", options: dict) -> None:
        self._runtime = runtime
        self._opts = {**DEFAULTS, **options}

        self._homeserver: str  = str(self._opts["homeserver"])
        self._username: str    = str(self._opts["username"])
        self._max_len: int     = int(self._opts["max_reply_length"])
        self._prefix: str      = str(self._opts["command_prefix"])
        self._prefix_required: bool = bool(self._opts["prefix_required"])
        self._reset_command: str    = str(self._opts["reset_command"])
        self._dm_enabled: bool      = bool(self._opts["dm_enabled"])
        self._room_ids: set[str]    = set(self._opts["room_ids"])
        self._sync_timeout: int     = int(self._opts["sync_timeout_ms"])
        self._typing: bool           = bool(self._opts["typing_indicator"])
        self._typing_on_thinking: bool = bool(self._opts["typing_on_thinking"])
        self._typing_on_tools: bool    = bool(self._opts["typing_on_tools"])
        self._typing_on_reply: bool    = bool(self._opts["typing_on_reply"])

        self._allowed_users: set[str] = {str(u) for u in self._opts["allowed_users"]}
        self._admin_users:   set[str] = {str(u) for u in self._opts["admin_users"]}

        workspace = str(self._opts.get("workspace", runtime.config.workspace.path))
        raw_store = str(self._opts["store_path"])
        self._store_path = raw_store if os.path.isabs(raw_store) \
            else os.path.join(str(runtime.config.workspace.path), raw_store)
        os.makedirs(self._store_path, exist_ok=True)

        # node_id → _ReplyAccumulator
        self._accumulators: dict[str, _ReplyAccumulator] = {}
        # node_id → asyncio.Event signalling typing activity
        self._typing_active: dict[str, asyncio.Event] = {}
        # sender+room_id → pending Attachments (media arrives before text in Matrix)
        self._pending_attachments: dict[str, list[Attachment]] = {}

        # Persisted cursor store
        ws_path     = Path(runtime.config.workspace.path).expanduser().resolve()
        cursors_dir = ws_path / "cursors"
        cursors_dir.mkdir(parents=True, exist_ok=True)
        self._store = CursorStore(cursors_dir / "matrix.json")

        self._client: AsyncClient | None = None
        self._own_user_id: str = ""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_permission_level(self, room: MatrixRoom, sender: str) -> int:
        default   = int(self._opts.get("default_permission", 25))
        level_map = self._opts.get("power_level_map", {})
        if not level_map:
            return default
        power = room.power_levels.get_user_level(sender) if hasattr(room, "power_levels") else 0
        int_map = {int(k): int(v) for k, v in level_map.items()}
        if power in int_map:
            return int_map[power]
        lower_keys = [k for k in int_map if k <= power]
        if lower_keys:
            return int_map[max(lower_keys)]
        return default

    def _is_allowed(self, sender: str) -> bool:
        if not self._allowed_users:
            return True
        return sender in self._allowed_users

    def _is_admin(self, sender: str) -> bool:
        return sender in self._admin_users

    def _is_dm_room(self, room: MatrixRoom) -> bool:
        return room.member_count == 2

    def _display_name(self, room: MatrixRoom, sender: str) -> str:
        return room.user_name(sender) or sender.split(":")[0].lstrip("@")

    def _localpart(self) -> str:
        return self._username.split(":")[0].lstrip("@")

    # ------------------------------------------------------------------
    # Cursor management
    # ------------------------------------------------------------------

    def _get_or_create_cursor(self, cursor_key: str) -> str:
        node_id = self._store.get(cursor_key)
        if not node_id:
            node_id = _make_session_node(self._runtime.db, cursor_key)
            self._store.set(cursor_key, node_id)
            logger.info("Matrix: created cursor %s -> %s", cursor_key, node_id)
        return node_id

    def _advance_cursor(self, cursor_key: str, new_tail: str) -> None:
        if new_tail and new_tail != self._store.get(cursor_key):
            self._store.set(cursor_key, new_tail)

    # ------------------------------------------------------------------
    # Event handler registered with Runtime
    # ------------------------------------------------------------------

    async def handle_event(self, event) -> None:
        node_id = event.tail_node_id
        acc = self._accumulators.get(node_id)
        if acc is None:
            logger.debug("Matrix: received event for unknown cursor %s", node_id)
            return

        typing_ev = self._typing_active.get(node_id)

        if isinstance(event, AgentThinkingChunk):
            if typing_ev and self._typing_on_thinking:
                typing_ev.set()
        elif isinstance(event, AgentTextChunk):
            if typing_ev and self._typing_on_reply:
                typing_ev.set()
            acc.feed(event.text)
        elif isinstance(event, AgentTextFinal):
            acc._final_tail = event.tail_node_id
            acc.finish(event.text)
        elif isinstance(event, AgentToolCall):
            if typing_ev and self._typing_on_tools:
                typing_ev.set()
            logger.debug("Matrix: tool call %s for cursor %s", event.tool_name, node_id)
        elif isinstance(event, AgentToolResult):
            logger.debug(
                "Matrix: tool result %s (%s) for cursor %s",
                event.tool_name, "error" if event.is_error else "ok", node_id,
            )
        elif isinstance(event, AgentError):
            acc.error(event.message)

    # ------------------------------------------------------------------
    # nio message callbacks
    # ------------------------------------------------------------------

    async def _on_message(self, room: MatrixRoom, event: RoomMessageText) -> None:
        if event.sender == self._own_user_id:
            return

        # Drop replayed history from before startup.
        age_ms = getattr(event, "server_timestamp", 0)
        now_ms = int(time.time() * 1000)
        if age_ms and (now_ms - age_ms) > 60_000:
            return

        if not self._is_allowed(event.sender):
            logger.debug("Matrix: ignoring message from unauthorized user %s", event.sender)
            return

        is_dm = self._is_dm_room(room)
        if is_dm and not self._dm_enabled:
            return
        if self._room_ids and room.room_id not in self._room_ids:
            return

        body = event.body.strip()

        # ── DM path ───────────────────────────────────────────────────
        if is_dm:
            att_key     = f"{event.sender}:{room.room_id}"
            attachments = tuple(self._pending_attachments.pop(att_key, []))
            cursor_key  = f"dm:{event.sender}"
            node_id     = self._get_or_create_cursor(cursor_key)

            if body.startswith("/"):
                ctx = {
                    "room":    room,
                    "event":   event,
                    "bridge":  self,
                    "runtime": self._runtime,
                    "cursor":  node_id,
                    "send":    self._send,
                }
                handled = await self._runtime.commands.dispatch(body, ctx)
                if handled:
                    return

            author = UserIdentity(
                platform=Platform.MATRIX,
                user_id=event.sender,
                username=self._display_name(room, event.sender),
            )
            msg = InboundMessage(
                tail_node_id=node_id,
                author=author,
                content_type=content_type_for(body, bool(attachments)),
                text=body,
                message_id=event.event_id,
                timestamp=time.time(),
                attachments=attachments,
                permission_level=self._resolve_permission_level(room, event.sender),
                trigger=True,
            )
            acc = _ReplyAccumulator(self._max_len)
            self._accumulators[node_id] = acc
            asyncio.create_task(
                self._handle_turn(msg, room.room_id, node_id, acc, cursor_key)
            )
            return

        # ── Group room path ───────────────────────────────────────────
        cursor_key = f"group:{room.room_id}"

        # /reset — admin only
        if body == self._reset_command:
            if self._is_admin(event.sender):
                new_node_id = _make_session_node(self._runtime.db, cursor_key)
                self._store.set(cursor_key, new_node_id)
                await self._send(room.room_id, "✅ Session reset.")
                logger.info(
                    "Matrix: group room %s reset by admin %s",
                    room.room_id, event.sender,
                )
            else:
                await self._send(room.room_id, "⛔ Only admins can reset the session.")
            return

        # Module slash commands — before trigger gating
        if body.startswith("/"):
            node_id = self._get_or_create_cursor(cursor_key)
            ctx = {
                "room":    room,
                "event":   event,
                "bridge":  self,
                "runtime": self._runtime,
                "cursor":  node_id,
                "send":    self._send,
            }
            handled = await self._runtime.commands.dispatch(body, ctx)
            if handled:
                return

        # Humanize HTML mention markup (Matrix-specific formatting only)
        humanized_body = _humanize_matrix_mentions(body, self._own_user_id)

        display     = self._display_name(room, event.sender)
        att_key     = f"{event.sender}:{room.room_id}"
        attachments = tuple(self._pending_attachments.pop(att_key, []))
        node_id     = self._get_or_create_cursor(cursor_key)

        is_trigger = _is_group_trigger(
            humanized_body, self._own_user_id, self._localpart(),
            self._prefix, self._prefix_required,
        )

        author = UserIdentity(
            platform=Platform.MATRIX,
            user_id=event.sender,
            username=display,
        )
        msg = InboundMessage(
            tail_node_id=node_id,
            author=author,
            content_type=content_type_for(humanized_body, bool(attachments)),
            text=humanized_body,
            message_id=event.event_id,
            timestamp=time.time(),
            attachments=attachments,
            server_name=None,
            channel_name=getattr(room, "display_name", None) or room.room_id,
            permission_level=self._resolve_permission_level(room, event.sender),
            trigger=is_trigger,
        )

        if not is_trigger:
            # Persist the non-trigger node; Runtime.push() handles it.
            await self._runtime.push(msg)
            return

        acc = _ReplyAccumulator(self._max_len)
        self._accumulators[node_id] = acc
        asyncio.create_task(
            self._handle_turn(msg, room.room_id, node_id, acc, cursor_key)
        )

    async def _on_media(self, room: MatrixRoom, event) -> None:
        """Buffer media attachments — Matrix sends these separately from text."""
        if event.sender == self._own_user_id:
            return
        if not self._is_allowed(event.sender):
            return
        if self._client is None:
            return

        url: str = getattr(event, "url", "") or ""
        filename: str = (
            (event.source.get("content") or {}).get("body")
            or getattr(event, "body", None)
            or "attachment"
        )
        info: dict = getattr(event, "info", None) or {}
        mime: str = info.get("mimetype", "application/octet-stream")

        if not url:
            logger.warning("Matrix: media event from %s has no url", event.sender)
            return

        try:
            resp = await self._client.download(url)
            data: bytes = resp.body if hasattr(resp, "body") else bytes(resp)
        except Exception:
            logger.warning("Matrix: failed to download media from %s", event.sender)
            return

        att = Attachment(filename=filename, data=data, mime_type=mime)
        key = f"{event.sender}:{room.room_id}"
        self._pending_attachments.setdefault(key, []).append(att)
        logger.debug("Matrix: buffered attachment %s (%s) from %s", filename, mime, event.sender)

    # ------------------------------------------------------------------
    # Turn handling
    # ------------------------------------------------------------------

    async def _typing_keepalive(
        self,
        room_id: str,
        active_event: asyncio.Event,
        done_event: asyncio.Event,
    ) -> None:
        while not done_event.is_set():
            await active_event.wait()
            if done_event.is_set():
                break
            if self._client:
                try:
                    await self._client.room_typing(room_id, typing_state=True, timeout=30000)
                except Exception:
                    pass
            try:
                await asyncio.wait_for(done_event.wait(), timeout=25.0)
            except asyncio.TimeoutError:
                pass
        if self._client:
            try:
                await self._client.room_typing(room_id, typing_state=False)
            except Exception:
                pass

    async def _handle_turn(
        self,
        msg: InboundMessage,
        room_id: str,
        node_id: str,
        acc: _ReplyAccumulator,
        cursor_key: str | None = None,
    ) -> None:
        done_event = asyncio.Event()
        typing_ev  = asyncio.Event()
        self._typing_active[node_id] = typing_ev

        try:
            accepted = await self._runtime.push(msg)
            if not accepted:
                await self._send(room_id, "⏳ I'm busy — please try again in a moment.")
                return

            if self._typing:
                keepalive = asyncio.create_task(
                    self._typing_keepalive(room_id, typing_ev, done_event)
                )
                try:
                    chunks = await acc.wait()
                finally:
                    done_event.set()
                    typing_ev.set()
                    keepalive.cancel()
            else:
                chunks = await acc.wait()

            for chunk in chunks:
                await self._send(room_id, chunk)

            if cursor_key:
                new_tail = getattr(acc, "_final_tail", None)
                if new_tail:
                    self._advance_cursor(cursor_key, new_tail)

        except Exception:
            logger.exception("Matrix: error handling turn for cursor %s", node_id)
        finally:
            done_event.set()
            self._accumulators.pop(node_id, None)
            self._typing_active.pop(node_id, None)

    async def _send(self, room_id: str, text: str) -> None:
        if self._client is None:
            return
        await self._client.room_send(
            room_id=room_id,
            message_type="m.room.message",
            content={"msgtype": "m.text", "body": text},
        )

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        password_env = str(self._opts["password_env"])
        password = os.environ.pop(password_env, "")
        if not password:
            raise RuntimeError(
                f"Matrix bridge: env var '{password_env}' is not set. "
                "Export your Matrix account password before starting."
            )

        device_name = str(self._opts["device_name"])
        config = AsyncClientConfig(
            store_sync_tokens=True,
            encryption_enabled=False,
        )
        client = AsyncClient(
            homeserver=self._homeserver,
            user=self._username,
            store_path=self._store_path,
            config=config,
        )
        self._client = client

        logger.info("Matrix bridge: logging in as %s", self._username)
        resp = await client.login(password=password, device_name=device_name)
        if not isinstance(resp, LoginResponse):
            raise RuntimeError(f"Matrix login failed: {resp}")

        self._own_user_id = client.user_id
        logger.info("Matrix bridge: logged in, user_id=%s", self._own_user_id)

        if not self._allowed_users:
            logger.warning(
                "Matrix bridge: allowed_users is empty — the bot will respond "
                "to anyone. Set bridges.matrix.options.allowed_users in config.yaml."
            )
        if not self._admin_users:
            logger.warning(
                "Matrix bridge: admin_users is empty — nobody can use %s in group rooms.",
                self._reset_command,
            )

        self._runtime.register_platform_handler(Platform.MATRIX.value, self.handle_event)

        client.add_event_callback(self._on_message, RoomMessageText)
        if _HAS_MEDIA_EVENTS:
            for media_cls in (RoomMessageImage, RoomMessageFile,
                              RoomMessageAudio, RoomMessageVideo):
                client.add_event_callback(self._on_media, media_cls)
        else:
            logger.warning(
                "Matrix bridge: media event types not available in this nio version — "
                "file/image attachments will not be received. Upgrade matrix-nio."
            )

        logger.info("Matrix bridge: starting sync loop")
        try:
            await client.sync(timeout=0, full_state=True)
            await client.sync_forever(timeout=self._sync_timeout, full_state=False)
        finally:
            await client.close()
            logger.info("Matrix bridge: client closed")


# ---------------------------------------------------------------------------
# Loader entrypoint (called by main.py)
# ---------------------------------------------------------------------------

async def run(runtime: "Runtime") -> None:
    """Entry point called by main.py bridge loader."""
    bridge_cfg = runtime.config.bridges.get("matrix")
    options: dict = bridge_cfg.options if bridge_cfg else {}
    bridge = MatrixBridge(runtime, options)
    await bridge.run()