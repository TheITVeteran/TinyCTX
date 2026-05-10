"""
modules/rag/__main__.py

RAG pipeline wiring: indexing, hybrid search, auto-inject, memory_search tool,
and the context-nudge / consolidation hook.

Singletons (store, indexer, embedder) are created once via module-level lazy
state on the first register_agent call, so concurrent AgentCycles share the
same read-only store/indexer without re-opening files on every turn.

register_runtime registers the /memory consolidate command and wires the
post-turn consolidation hook.  It does NOT register tools — tools are
registered per-cycle in register_agent.

Config is read from the memory module's EXTENSION_META defaults merged with
workspace overrides under the "memory_search" key, keeping a single source of
truth for all memory/rag settings.
"""
from __future__ import annotations

import atexit
import logging
from pathlib import Path

from TinyCTX.context import HOOK_PRE_ASSEMBLE_ASYNC

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singleton state — initialized once on first register_agent call
# ---------------------------------------------------------------------------

_initialized = False
_store       = None
_indexer     = None
_embedder    = None
_cfg: dict   = {}
_workspace: Path | None = None

# Consolidation hook closure — set by register_runtime if nudge is configured.
_consolidation_hook = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    return len(text) // 4


def _format_results(results: list[dict], budget_tokens: int) -> str | None:
    if not results:
        return None

    header   = "<memory>"
    footer   = "</memory>"
    overhead = _estimate_tokens(header + "\n\n" + footer)

    blocks:      list[str] = []
    used_tokens: int       = overhead
    dropped:     int       = 0

    for i, r in enumerate(results):
        block = f"[{r['file']}]\n{r['text'].strip()}"
        cost  = _estimate_tokens(block + "\n\n")

        if i > 0 and budget_tokens > 0 and used_tokens + cost > budget_tokens:
            dropped += 1
            continue

        blocks.append(block)
        used_tokens += cost

    parts = [header] + blocks + [footer]
    if dropped:
        parts.insert(-1, f"[{dropped} chunk(s) omitted — memory budget reached]")

    return "\n\n".join(parts)


def _load_cfg(config) -> dict:
    """Merge EXTENSION_META defaults with workspace overrides."""
    try:
        from TinyCTX.modules.system_prompt import EXTENSION_META
        defaults: dict = EXTENSION_META.get("default_config", {})
    except ImportError:
        defaults = {}
    overrides: dict = {}
    if hasattr(config, "extra") and isinstance(config.extra, dict):
        overrides = config.extra.get("memory_search", {})
    return {**defaults, **overrides}


# ---------------------------------------------------------------------------
# Singleton initialization (called once from register_agent)
# ---------------------------------------------------------------------------

def _init_singletons(config) -> None:
    global _initialized, _store, _indexer, _embedder, _cfg, _workspace
    if _initialized:
        return

    workspace = Path(config.workspace.path).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    _workspace = workspace

    _cfg = _load_cfg(config)

    def _resolve(filename: str) -> Path:
        p = Path(filename)
        return p if p.is_absolute() else workspace / p

    memory_dir = _resolve(_cfg["memory_dir"])
    db_path    = _resolve(_cfg["db_file"])
    db_path.parent.mkdir(parents=True, exist_ok=True)

    from TinyCTX.modules.rag.store import MemoryStore
    _store = MemoryStore(db_path)
    atexit.register(_store.close)

    embedding_model = _cfg.get("embedding_model", "").strip()
    if embedding_model:
        try:
            from TinyCTX.ai import Embedder
            emb_cfg   = config.get_embedding_model(embedding_model)
            _embedder = Embedder.from_config(emb_cfg)
            logger.info("[rag] embedder: %s @ %s", emb_cfg.model, emb_cfg.base_url)
        except (KeyError, ValueError) as exc:
            logger.warning(
                "[rag] embedding_model '%s' not usable (%s) — BM25 only",
                embedding_model, exc,
            )

    model_name_str = (
        config.models[embedding_model].model
        if embedding_model and embedding_model in config.models
        else ""
    )

    from TinyCTX.modules.rag.chunkers import get_strategy
    chunk_kwargs: dict = _cfg.get("chunk_kwargs") or {}
    strategy = get_strategy(_cfg["chunk_strategy"], **chunk_kwargs)

    from TinyCTX.modules.rag.indexer import MemoryIndexer
    _indexer = MemoryIndexer(
        store           = _store,
        memory_dir      = memory_dir,
        strategy        = strategy,
        embedder        = _embedder,
        embedding_model = model_name_str,
    )

    _initialized = True
    logger.info(
        "[rag] singletons ready — dir: %s | db: %s | strategy: %s | embedder: %s",
        memory_dir, db_path, _cfg["chunk_strategy"], model_name_str or "BM25 only",
    )


# ---------------------------------------------------------------------------
# register_runtime — consolidation command + post-turn hook
# ---------------------------------------------------------------------------

def register_runtime(runtime) -> None:
    global _consolidation_hook

    cfg             = _load_cfg(runtime.config)
    nudge_threshold = float(cfg.get("nudge_threshold", 0.80))
    nudge_message   = cfg.get("nudge_message", "")
    token_limit     = runtime.config.context

    if nudge_threshold > 0.0 and nudge_message:
        nudge_delta = int(nudge_threshold * token_limit)

        async def _hook(tail_node_id: str) -> None:
            import json, datetime, time as _time
            from TinyCTX.contracts import InboundMessage, ContentType, UserIdentity, Platform

            state, _ = runtime.db.load_session_state(tail_node_id)
            tokens_now      = int(state.get("tokens_used", 0) or 0)
            tokens_at_nudge = int(state.get("memory_nudge_tokens_at_last", 0) or 0)
            if tokens_now - tokens_at_nudge < nudge_delta:
                return

            date_str = datetime.date.today().strftime("%d-%m-%Y")
            msg_text = nudge_message.format(date=date_str)

            opening = runtime.db.add_node(
                parent_id=tail_node_id, role="user", content=msg_text,
            )
            await runtime.push(InboundMessage(
                tail_node_id=opening.id,
                author=UserIdentity(platform=Platform.SYSTEM, user_id="system", username="system"),
                content_type=ContentType.TEXT,
                text=msg_text,
                message_id=f"consolidation-{_time.time_ns()}",
                timestamp=_time.time(),
                trigger=True,
                permission_level=100,
            ))
            runtime.db.update_node_state_delta(
                tail_node_id,
                json.dumps({"memory_nudge_tokens_at_last": tokens_now}),
            )
            logger.info(
                "[rag] consolidation spawned off tail=%s (delta %d/%d tokens)",
                tail_node_id, tokens_now - tokens_at_nudge, nudge_delta,
            )

        _consolidation_hook = _hook
        logger.info(
            "[rag] consolidation hook configured — threshold %.0f%% delta (%d tokens)",
            nudge_threshold * 100, nudge_delta,
        )

        async def _cmd_consolidate(args: list[str], context: dict) -> None:
            import time as _time, datetime
            from TinyCTX.contracts import InboundMessage, ContentType, UserIdentity, Platform
            console = context.get("console")
            c       = context.get("theme_c", lambda k: "")
            tail    = context.get("tail_node_id")
            if not tail:
                if console:
                    console.print(f"[{c('error')}]  ✗  rag: no active session[/{c('error')}]")
                return
            date_str = datetime.date.today().strftime("%d-%m-%Y")
            msg_text = nudge_message.format(date=date_str)
            opening  = runtime.db.add_node(parent_id=tail, role="user", content=msg_text)
            await runtime.push(InboundMessage(
                tail_node_id=opening.id,
                author=UserIdentity(platform=Platform.SYSTEM, user_id="system", username="system"),
                content_type=ContentType.TEXT,
                text=msg_text,
                message_id=f"consolidation-cmd-{_time.time_ns()}",
                timestamp=_time.time(),
                trigger=True,
                permission_level=100,
            ))
            if console:
                console.print(f"[{c('tool_ok')}]  ✓  consolidation started (tail={tail[:8]}…)[/{c('tool_ok')}]")

        runtime.commands.register(
            "memory", "consolidate", _cmd_consolidate,
            help="Spawn a memory consolidation branch immediately",
        )
    else:
        logger.info("[rag] consolidation disabled")


# ---------------------------------------------------------------------------
# register_agent — RAG wiring per cycle
# ---------------------------------------------------------------------------

def register_agent(cycle) -> None:
    _init_singletons(cycle.config)

    cfg                 = _cfg
    store               = _store
    indexer             = _indexer
    embedder            = _embedder
    budget_tokens       = int(cfg["memory_budget_tokens"])
    top_k               = int(cfg["top_k"])
    bm25_weight         = float(cfg["bm25_weight"])
    decay_halflife_days = float(cfg.get("decay_halflife_days", 30.0))
    decay_weight        = float(cfg.get("decay_weight", 0.0))
    auto_inject         = bool(cfg["auto_inject"])
    ms_vis = str(cfg.get("tools", {}).get("memory_search", "always_on")).lower().strip()

    # 1. Async pre-assemble hook — ephemeral results shared via closure
    results: list = []

    async def _pre_assemble_async(ctx) -> None:
        if ctx.dialogue:
            if ctx.dialogue[-1].role in ("tool", "assistant"):
                return

        await indexer.sync()

        query = ""
        for entry in reversed(ctx.dialogue):
            if entry.role == "user":
                content = entry.content
                if isinstance(content, list):
                    query = " ".join(
                        p.get("text", "") for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    ).strip()
                else:
                    query = content
                if query.strip():
                    break

        if not query.strip():
            results[:] = []
            return

        if budget_tokens > 0:
            total = store.total_chunks_text_tokens()
            if total <= budget_tokens:
                found = store.hybrid_search(
                    query, None, top_k=999, bm25_weight=1.0,
                    decay_halflife_days=decay_halflife_days,
                    decay_weight=decay_weight,
                ) if total > 0 else []
                results[:] = found
                return

        q_vec = None
        if embedder is not None:
            try:
                q_vec = await embedder.embed_one(query)
            except Exception as exc:
                logger.warning("[rag] embed failed: %s — BM25 only", exc)

        found = store.hybrid_search(
            query, q_vec, top_k, bm25_weight,
            decay_halflife_days=decay_halflife_days,
            decay_weight=decay_weight,
        )
        results[:] = found
        if found:
            logger.debug("[rag] '%s…' → %d result(s)", query[:40], len(found))

    cycle.context.register_hook(HOOK_PRE_ASSEMBLE_ASYNC, _pre_assemble_async, priority=0)

    # 2. Auto-inject prompt
    if auto_inject:
        cycle.context.register_prompt(
            "memory_search",
            lambda ctx: _format_results(results, budget_tokens),
            role="system",
            priority=int(cfg["search_priority"]),
        )

    # 3. memory_search tool
    async def memory_search(query: str) -> str:
        """
        Search the memory store for information relevant to a query.
        Use this to explicitly recall facts, notes, or context that may
        not have been automatically injected into the current turn.

        Args:
            query: The topic, question, or keywords to search for.
        """
        await indexer.sync()
        q_vec = None
        if embedder is not None:
            try:
                q_vec = await embedder.embed_one(query)
            except Exception as exc:
                logger.warning("[rag] tool embed failed: %s — BM25 only", exc)
        found = store.hybrid_search(
            query, q_vec, top_k, bm25_weight,
            decay_halflife_days=decay_halflife_days,
            decay_weight=decay_weight,
        )
        if not found:
            return "[no memory found for that query]"
        return _format_results(found, budget_tokens) or "[no memory found for that query]"

    if ms_vis != "disabled":
        cycle.tool_handler.register_tool(
            memory_search,
            always_on=(ms_vis != "deferred"),
            min_permission=25,
        )

    # 4. Post-turn consolidation hook (if configured by register_runtime)
    if _consolidation_hook is not None:
        cycle.post_turn_hooks.append(_consolidation_hook)
