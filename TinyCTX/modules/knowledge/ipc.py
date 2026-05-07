"""
modules/knowledge/ipc.py

Lightweight IPC between the main agent and the librarian process.

Protocol
--------
The librarian process listens on a Unix domain socket (or Windows named pipe).
Messages are newline-delimited JSON:

    {"type": "trigger"}                     # fire normal buffer ingest
    {"type": "targeted", "prompt": "..."}   # spawn targeted agent

On Windows, a named pipe \\.\pipe\tinyctx_librarian_<suffix> is used
instead of a Unix socket.

Client (main agent)
-------------------
    send_ipc(socket_path, message_dict)

Server (librarian process)
--------------------------
    IPCServer(socket_path, handler_fn).start_async()

The server accepts connections in its event loop and calls handler_fn(msg_dict)
for each parsed message. handler_fn may be async or sync.

Startup race
------------
If the librarian process is not yet listening, send_ipc raises IPCError.
call_librarian catches this and logs a warning rather than crashing.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_IS_WINDOWS = sys.platform == "win32"


class IPCError(Exception):
    pass


# ---------------------------------------------------------------------------
# Client — used by main agent tool
# ---------------------------------------------------------------------------

async def send_ipc(socket_path: Path, message: dict) -> None:
    """
    Send a JSON message to the librarian process and close the connection.
    Raises IPCError on failure (process not running, pipe broken, etc.).
    """
    data = (json.dumps(message) + "\n").encode()

    try:
        if _IS_WINDOWS:
            pipe_name = _pipe_name(socket_path)
            reader, writer = await asyncio.open_connection(
                None, None,
                open_mode=0,  # unused
                pipe=True,
                path=pipe_name,
            )
        else:
            reader, writer = await asyncio.open_unix_connection(str(socket_path))

        writer.write(data)
        await writer.drain()
        writer.close()
        await writer.wait_closed()
    except (ConnectionRefusedError, FileNotFoundError, OSError) as exc:
        raise IPCError(f"Cannot reach librarian process: {exc}") from exc


# ---------------------------------------------------------------------------
# Server — used by librarian process
# ---------------------------------------------------------------------------

class IPCServer:
    """
    Async Unix socket (or Windows named pipe) server.
    Calls handler(msg_dict) for each received JSON message.
    handler may be a coroutine function.
    """

    def __init__(self, socket_path: Path, handler) -> None:
        self._path    = socket_path
        self._handler = handler
        self._server  = None

    async def start(self) -> None:
        """Start listening. Returns immediately; runs in background."""
        if _IS_WINDOWS:
            self._server = await asyncio.start_server(
                self._handle_conn,
                None,
                None,
                pipe=True,
                path=_pipe_name(self._path),
            )
        else:
            sock = str(self._path)
            # Remove stale socket file left by a previous crash
            if os.path.exists(sock):
                os.unlink(sock)
            self._server = await asyncio.start_unix_server(
                self._handle_conn,
                path=sock,
            )
        logger.info("[knowledge/ipc] listening on %s", self._path)

    async def _handle_conn(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            line = await reader.readline()
            if not line:
                return
            msg = json.loads(line.decode())
            result = self._handler(msg)
            if asyncio.iscoroutine(result):
                await result
        except Exception as exc:
            logger.warning("[knowledge/ipc] handler error: %s", exc)
        finally:
            writer.close()

    def close(self) -> None:
        if self._server:
            self._server.close()
        if not _IS_WINDOWS:
            sock = str(self._path)
            if os.path.exists(sock):
                try:
                    os.unlink(sock)
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# Windows named pipe name helper
# ---------------------------------------------------------------------------

def _pipe_name(socket_path: Path) -> str:
    # Use the stem of the socket path as the pipe suffix
    return rf"\\.\pipe\tinyctx_{socket_path.stem}"
