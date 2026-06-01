"""
commands/launch.py — `tinyctx launch <target>`

Currently supported targets: cli

Reads gateway host/port/api_key directly from config.yaml and calls
the bridge's run_detached() entry point.

Default config path: <repo_root>/config.yaml. Override with --config.

Flags
-----
  --config PATH    Path to config.yaml.
  --user USERNAME  TinyCTX username to log in as. If the user's
                   permission_level is below 100, you will be prompted
                   to elevate it (CLI is a trusted admin console — no
                   higher-level caller is required).

Docker
------
When TinyCTX is running inside a container, attach to the container and
run this command from within it:

    docker exec -it <container_name> python -m TinyCTX launch cli --user USERNAME

Or, if you have the TinyCTX CLI installed on the host and the gateway
port is published (e.g. -p 8085:8085), just run:

    tinyctx launch cli --user USERNAME

and point it at the published port — no docker exec needed because the
CLI bridge connects to the gateway over HTTP, not a Unix socket.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

_REPO_ROOT      = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CONFIG = _REPO_ROOT / "config.yaml"


def _prompt_elevate(username: str, current_level: int) -> bool:
    """Ask the user if they want to elevate to level 100. Returns True if yes."""
    print(
        f"\n  User '{username}' has permission_level {current_level}.\n"
        "  The CLI is a trusted admin console — you can elevate this user to\n"
        "  level 100 now. This grants full access to all agent capabilities.\n"
    )
    while True:
        try:
            answer = input("  Elevate to level 100? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        if answer in ("y", "yes"):
            return True
        if answer in ("", "n", "no"):
            return False
        print("  Please enter y or n.")


def run(args: argparse.Namespace) -> None:
    target = getattr(args, "target", "cli")

    if target != "cli":
        print(f"error: unknown launch target '{target}'", file=sys.stderr)
        sys.exit(1)

    config_path = Path(getattr(args, "config", None) or _DEFAULT_CONFIG).resolve()
    if not config_path.exists():
        print("error: no config.yaml found.", file=sys.stderr)
        print("  Run 'TinyCTX onboard' to set up TinyCTX, or manually create a config.yaml.", file=sys.stderr)
        sys.exit(1)

    from TinyCTX.config import load as load_config
    try:
        cfg = load_config(str(config_path))
    except Exception as exc:
        print(f"error: could not load config: {exc}", file=sys.stderr)
        sys.exit(1)

    gateway_url = f"http://{cfg.gateway.host}:{cfg.gateway.port}"
    api_key     = cfg.gateway.api_key or ""

    try:
        with urllib.request.urlopen(f"{gateway_url}/v1/health", timeout=2) as r:
            if r.status != 200:
                raise OSError(f"status {r.status}")
    except Exception as exc:
        print(f"error: gateway at {gateway_url} is not responding: {exc}", file=sys.stderr)
        sys.exit(1)

    # ── Resolve user ──────────────────────────────────────────────────────────
    username: str | None = getattr(args, "user", None)

    from TinyCTX.users import UserStore
    store = UserStore()

    if username is None:
        # No --user flag: prompt interactively.
        try:
            username = input("  TinyCTX username: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            sys.exit(0)

    if not username:
        print("error: username cannot be empty.", file=sys.stderr)
        sys.exit(1)

    user = store.get_user(username)
    if user is None:
        print(f"error: user '{username}' not found in users.db.", file=sys.stderr)
        print("  Check the username with: python -m TinyCTX.onboard.fix_permissions --user <name> --list", file=sys.stderr)
        sys.exit(1)

    # ── Offer elevation if level < 100 ────────────────────────────────────────
    if user.permission_level < 100:
        if _prompt_elevate(username, user.permission_level):
            from TinyCTX.onboard.fix_permissions import elevate_user
            user = elevate_user(username, 100, store)
            print(f"  ✓ '{username}' elevated to level 100.\n")
        else:
            print(f"  Continuing as level {user.permission_level}.\n")

    # ── Launch CLI ────────────────────────────────────────────────────────────
    options: dict = {}
    try:
        bridge_cfg = cfg.bridges.get("cli")
        if bridge_cfg:
            options = getattr(bridge_cfg, "options", {}) or {}
    except Exception:
        pass

    import asyncio
    from TinyCTX.bridges.cli.__main__ import run_detached
    try:
        asyncio.run(run_detached(gateway_url, api_key, options, username=username))
    except KeyboardInterrupt:
        pass
