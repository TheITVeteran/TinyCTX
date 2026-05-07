"""
modules/knowledge/buffer.py

Writes conversation turns to the libbuffer/ session files.

Registered as a HOOK_POST_TURN on the main agent. After every completed
turn, appends:

    [username]: message text
    [assistant]: response text

One file per session, named session_<bridge>_<unix_timestamp>.txt.
Attachments are noted inline as [attachment: filename.ext].
Tool call / tool result turns are omitted.
"""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Regex to strip XML attachment markers from assistant responses so the buffer
# stays clean text. These are injected by the attachment pipeline.
_ATTACHMENT_TAG_RE = re.compile(r"<untrusted_file_content[^>]*>.*?</untrusted_file_content>", re.DOTALL)


class SessionBuffer:
    """
    One-file-per-session append-only buffer writer.
    Thread-safe for the single writer that is the main agent's post-turn hook.
    """

    def __init__(self, libbuffer_dir: Path, bridge_name: str) -> None:
        self._dir      = libbuffer_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._bridge   = bridge_name
        ts             = int(time.time())
        self._path     = self._dir / f"session_{bridge_name}_{ts}.txt"

    @property
    def path(self) -> Path:
        return self._path

    def append_turn(
        self,
        username: str,
        user_text: str,
        assistant_text: str,
        attachment_names: list[str] | None = None,
    ) -> None:
        """Append one user+assistant exchange to the buffer file."""
        lines: list[str] = []

        # User line — may have attachments
        user_line = str(user_text).strip() if user_text else ""
        if attachment_names:
            for name in attachment_names:
                user_line += f" [attachment: {name}]"
        if user_line:
            lines.append(f"[{username}]: {user_line}")

        # Assistant line — strip noise
        asst = _ATTACHMENT_TAG_RE.sub("", str(assistant_text or "")).strip()
        if asst:
            lines.append(f"[assistant]: {asst}")

        if not lines:
            return

        try:
            with self._path.open("a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except OSError as exc:
            logger.warning("[knowledge/buffer] write failed: %s", exc)
