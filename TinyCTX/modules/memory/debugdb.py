#!/usr/bin/env python3
"""
debugdb.py — Knowledge graph debug multitool.

Subcommands:
  dump    — list all entities with full edges  (default)
  pinned  — show only pinned entities, grouped by pinned_target
  entity  — inspect a single entity by UUID or name fragment
  stats   — graph statistics summary

Usage:
    python debugdb.py [subcommand] [--config path] [--db path] [options]

    python debugdb.py                         # dump all
    python debugdb.py pinned                  # pinned entities only
    python debugdb.py entity Kamie            # find + inspect by name
    python debugdb.py entity --uuid abc123    # inspect by UUID prefix
    python debugdb.py stats                   # graph stats

DB resolution (same for all subcommands):
  --db path      direct path to graph.lbug
  --config path  load workspace from config.yaml (default: config.yaml)
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(ts) -> str:
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ts)


def _get(e: dict, field: str) -> str:
    return str(e.get(f"e.{field}", e.get(field, "")) or "")


def _print_entity_full(e: dict) -> None:
    """Print a full entity dict (from get_entity) with all fields and edges."""
    pin = f" [pinned:{_get(e, 'pinned_target')}]" if _get(e, 'pinned_target') else ""
    print(f"[{_get(e, 'entity_type')}] {_get(e, 'name')}{pin}")
    print(f"  uuid:     {_get(e, 'uuid')}")
    print(f"  priority: {_get(e, 'priority')}")
    print(f"  created:  {_ts(_get(e, 'created_at'))}")
    print(f"  updated:  {_ts(_get(e, 'updated_at'))}")
    print(f"  mentions: {_get(e, 'mention_count')}")
    desc = _get(e, 'description')
    if desc:
        print(f"  desc:     {desc}")
    for edge in e.get("edges_out", []):
        w = f" w={edge['weight']:.2f}" if edge.get("weight") is not None else ""
        d = f" — {edge['description']}" if edge.get("description") else ""
        print(f"  → [{edge['relation']}] → {edge['target_name']} ({edge['target_uuid'][:8]}){w}{d}")
    for edge in e.get("edges_in", []):
        w = f" w={edge['weight']:.2f}" if edge.get("weight") is not None else ""
        d = f" — {edge['description']}" if edge.get("description") else ""
        print(f"  ← [{edge['relation']}] ← {edge['source_name']} ({edge['source_uuid'][:8]}){w}{d}")


# ---------------------------------------------------------------------------
# Subcommand: dump
# ---------------------------------------------------------------------------

def cmd_dump(gdb, args) -> None:
    all_entities = gdb.list_entities()
    if not all_entities:
        print("(no entities found)")
        return

    print(f"{len(all_entities)} entities\n")
    for e in all_entities:
        full = gdb.get_entity(e["uuid"])
        if full:
            _print_entity_full(full)
        print()


# ---------------------------------------------------------------------------
# Subcommand: pinned
# ---------------------------------------------------------------------------

def cmd_pinned(gdb, args) -> None:
    all_entities = gdb.list_entities()
    pinned = [e for e in all_entities if e.get("pinned_target")]

    if not pinned:
        print("(no pinned entities)")
        return

    # Group by pinned_target value
    groups: dict[str, list] = {}
    for e in pinned:
        target = e.get("pinned_target") or "unknown"
        groups.setdefault(target, []).append(e)

    total = len(pinned)
    print(f"{total} pinned entit{'y' if total == 1 else 'ies'} across {len(groups)} target(s)\n")

    for target, entities in sorted(groups.items()):
        print(f"══ pinned_target = '{target}' ({len(entities)}) ══")
        for e in sorted(entities, key=lambda x: -(x.get("priority") or 0)):
            full = gdb.get_entity(e["uuid"])
            if full:
                _print_entity_full(full)
            print()


# ---------------------------------------------------------------------------
# Subcommand: entity
# ---------------------------------------------------------------------------

def cmd_entity(gdb, args) -> None:
    uid = getattr(args, "uuid", None)
    name_frag = " ".join(args.name) if args.name else None

    if uid:
        # Match UUID prefix against all entities
        all_e = gdb.list_entities()
        matches = [e for e in all_e if e["uuid"].startswith(uid)]
        if not matches:
            print(f"[error] no entity with UUID starting with '{uid}'")
            return
        for e in matches:
            full = gdb.get_entity(e["uuid"])
            if full:
                _print_entity_full(full)
            print()

    elif name_frag:
        found = gdb.find_entity(name=name_frag)
        if not found:
            print(f"(no entity found matching '{name_frag}')")
            return
        print(f"{len(found)} match(es) for '{name_frag}':\n")
        for e in found:
            full = gdb.get_entity(e["uuid"])
            if full:
                _print_entity_full(full)
            print()

    else:
        print("[error] provide a name fragment or --uuid prefix")


# ---------------------------------------------------------------------------
# Subcommand: stats
# ---------------------------------------------------------------------------

def cmd_stats(gdb, args) -> None:
    s = gdb.get_stats()
    print(f"entities:         {s['entity_count']}")
    print(f"active edges:     {s['active_edge_count']}")
    print(f"superseded edges: {s['superseded_edge_count']}")
    print(f"pinned:           {s['pinned_count']}")
    print(f"embedded:         {s['embedded_count']}")
    print(f"avg priority:     {s['avg_priority']}")
    if s["by_type"]:
        print("\nby type:")
        for t, n in s["by_type"].items():
            print(f"  {t:<20} {n}")
    if s["top_mentioned"]:
        print("\ntop mentioned:")
        for m in s["top_mentioned"]:
            print(f"  {m['mention_count']:>4}x  [{m['entity_type']}] {m['name']}")


# ---------------------------------------------------------------------------
# DB open helper
# ---------------------------------------------------------------------------

def _find_config(given: str) -> Path:
    """Resolve config path: use given if it exists, else walk up from __file__ to find it."""
    p = Path(given)
    if p.exists():
        return p
    # Walk up from this file's directory looking for config.yaml
    here = Path(__file__).resolve().parent
    for parent in [here, *here.parents]:
        candidate = parent / "config.yaml"
        if candidate.exists():
            return candidate
    return p  # fall through to original error


def _open_db(args):
    if args.db:
        kg_path = Path(args.db).expanduser().resolve()
    else:
        config_path = _find_config(args.config)
        if not config_path.exists():
            print(f"[error] Config not found: {config_path.resolve()}")
            sys.exit(1)
        try:
            from TinyCTX.config import load as load_config
            cfg = load_config(str(config_path))
            memory_cfg = cfg.extra.get("memory", {})
            kg_path_raw = memory_cfg.get("db_path") if memory_cfg else None
            kg_path = (
                Path(kg_path_raw).expanduser().resolve()
                if kg_path_raw
                else Path(cfg.workspace.path) / "memory" / "graph.lbug"
            )
        except Exception as e:
            print(f"[error] Failed to load config: {e}")
            sys.exit(1)

    if not kg_path.exists():
        print(f"[error] Graph DB not found: {kg_path}")
        sys.exit(1)

    try:
        from TinyCTX.modules.memory.graph import GraphDatabase, GraphDB
    except ImportError:
        print("[error] ladybug not installed")
        sys.exit(1)

    try:
        graph_database = GraphDatabase(kg_path)
        gdb = GraphDB(graph_database)
    except Exception as e:
        print(f"[error] Could not open graph DB: {e}")
        sys.exit(1)

    return kg_path, graph_database, gdb


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

SUBCOMMANDS = {
    "dump":   cmd_dump,
    "pinned": cmd_pinned,
    "entity": cmd_entity,
    "stats":  cmd_stats,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Knowledge graph debug multitool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--db", default="", help="Direct path to graph.lbug")

    subparsers = parser.add_subparsers(dest="subcommand")

    subparsers.add_parser("dump",   help="List all entities with edges (default)")
    subparsers.add_parser("pinned", help="Show pinned entities grouped by pinned_target")
    subparsers.add_parser("stats",  help="Graph statistics summary")

    ep = subparsers.add_parser("entity", help="Inspect a single entity by name or UUID")
    ep.add_argument("name", nargs="*", help="Name fragment to search for")
    ep.add_argument("--uuid", default="", help="UUID prefix to match")

    args = parser.parse_args()

    cmd_fn = SUBCOMMANDS.get(args.subcommand or "dump", cmd_dump)

    kg_path, graph_database, gdb = _open_db(args)
    print(f"db: {kg_path}\n")
    try:
        cmd_fn(gdb, args)
    finally:
        gdb.close()
        graph_database.close()


if __name__ == "__main__":
    main()
