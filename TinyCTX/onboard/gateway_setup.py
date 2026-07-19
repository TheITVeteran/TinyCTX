"""
onboard/gateway_setup.py — Step 4: Gateway port, API key, launch, and health check.

- Prompts for port (validates it's available).
- Auto-generates a gateway API key.
- Returns config dict (does NOT write config or launch — caller does that).
- launch() is called by __main__.py after write_config().
"""

from __future__ import annotations

import argparse
import asyncio
import secrets
import socket
import sys
from typing import Any
import questionary

from .helpers import (
    CONFIG_PATH,
    DEFAULT_GATEWAY_HOST,
    DEFAULT_GATEWAY_PORT,
    INSTANCE_DIR,
    GoBack,
    Mode,
    QSTYLE,
    c,
    section,
    success,
    warn,
)


def run(mode: Mode) -> dict[str, Any]:
    """
    Collect gateway config (host, port, api_key) from the user.

    Does NOT launch the gateway — call launch() after write_config().

    Returns a gateway config dict: { enabled, host, port, api_key }
    Raises GoBack if the user wants to return to the previous step.
    """
    if mode == "quickstart":
        section("Step 4 — Gateway Setup")
        c.print(
            "TinyCTX runs a local server so clients can connect to your agent.\n"
            "We'll auto-generate a secret key — keep it safe!\n"
        )
        host    = DEFAULT_GATEWAY_HOST
        port    = _pick_port(host)
        api_key = secrets.token_hex(16)
        success(f"Port [bold]{port}[/] is free. API key auto-generated.")
    else:
        section("Step 4 — Gateway (HTTP/SSE API)")
        c.print("Exposes TinyCTX to SillyTavern, curl, and other external clients.\n")

        raw_host = questionary.text(
            "Bind host:",
            default=DEFAULT_GATEWAY_HOST,
            style=QSTYLE,
        ).ask()
        if raw_host is None:
            raise GoBack
        host = raw_host.strip() or DEFAULT_GATEWAY_HOST

        port = _pick_port(host)

        api_key = "sk-" + secrets.token_hex(32)

        success(f"Gateway: http://{host}:{port}  key=[bold]{api_key}[/]")

    return {
        "enabled": True,
        "host":    host,
        "port":    port,
        "api_key": api_key,
    }


def launch(gateway: dict[str, Any]) -> None:
    """
    Start the gateway via `tinyctx start` (Docker Compose) and poll /v1/health.
    Call this AFTER write_config() so the daemon finds a valid config on disk.

    TinyCTX runs in Docker, not on bare metal — so onboarding must hand off to
    the same `tinyctx start` path as the CLI, rather than spawning main.py
    as a local process (which isn't how the gateway is meant to run).
    """
    host    = gateway["host"]
    port    = gateway["port"]
    api_key = gateway["api_key"]

    if not _launch_via_tinyctx_start():
        sys.exit(1)

    _launch_cli_bridge(host, port, api_key)


# ── private: launch & health check ───────────────────────────────────────────

def _launch_via_tinyctx_start() -> bool:
    """
    Delegate to `tinyctx start`'s Docker Compose launch + health check.

    Returns True on success. `commands.start.run` calls sys.exit(1) itself
    on failure (Docker missing, compose failure, or health check timeout),
    so we catch that and propagate a bool instead.
    """
    section("Launching Gateway")
    c.print("  Starting gateway via `tinyctx start` (Docker Compose) …\n")

    from TinyCTX.commands.start import run as start_run

    args = argparse.Namespace(dir=str(INSTANCE_DIR), config=str(CONFIG_PATH))
    try:
        start_run(args)
    except SystemExit as exc:
        if exc.code:
            return False
    return True


def _launch_cli_bridge(host: str, port: int, api_key: str) -> None:
    """Hand off to the CLI bridge via run_detached (blocks until user exits)."""
    from TinyCTX.bridges.cli.__main__ import run_detached

    gateway_url = f"http://{host}:{port}"
    try:
        asyncio.run(run_detached(gateway_url, api_key, {}))
    except KeyboardInterrupt:
        pass


# ── private helpers ───────────────────────────────────────────────────────────

def _is_port_available(port: int, host: str = "127.0.0.1") -> bool:
    """Return True if the given TCP port is not already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def _pick_port(host: str = DEFAULT_GATEWAY_HOST) -> int:
    """Prompt for a port number, validate it's in range and free. Loops until valid."""
    while True:
        raw = questionary.text(
            "Port to listen on:",
            default=str(DEFAULT_GATEWAY_PORT),
            style=QSTYLE,
        ).ask()
        if raw is None:
            raise GoBack
        raw = raw.strip()
        if not raw:
            port = DEFAULT_GATEWAY_PORT
        else:
            try:
                port = int(raw)
            except ValueError:
                warn(f"'{raw}' is not a valid port number. Try again.")
                continue
            if not (1 <= port <= 65535):
                warn("Port must be between 1 and 65535.")
                continue

        if _is_port_available(port, host):
            return port
        warn(f"Port {port} is already in use. Please choose another.")
