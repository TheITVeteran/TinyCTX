"""
modules/memory/__main__.py

All wiring happens in register_agent(cycle). Singletons (store, indexer,
embedder) are created once via module-level lazy state the first time
register_agent is called, so concurrent AgentCycles share the same
read-only store/indexer without re-opening files on every turn.

register_runtime is used only for the /memory consolidate command, which
needs a reference to runtime.push and runtime.db. It does NOT register
tools — tools are registered per-cycle in register_agent.

The consolidation post-turn hook is appended to cycle.post_turn_hooks so
it runs after the cycle completes. It needs runtime to call push(), so it
captures runtime via closure from register_runtime.
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
# Called by register_agent to append onto each cycle's post_turn_hooks.
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

    try:
        from TinyCTX.modules.memory import EXTENSION_META
        defaults: dict = EXTENSION_META.get("default_config", {})
    except ImportError:
        defaults = {}

    overrides: dict = {}
    if hasattr(config, "extra") and isinstance(config.extra, dict):
        overrides = config.extra.get("memory_search", {})

    _cfg = {**defaults, **overrides}

    def _resolve(filename: str) -> Path:
        p = Path(filename)
        return p if p.is_absolute() else workspace / p

    memory_dir = _resolve(_cfg["memory_dir"])
    db_path    = _resolve(_cfg["db_file"])
    db_path.parent.mkdir(parents=True, exist_ok=True)

    from TinyCTX.modules.memory.store import MemoryStore
    _store = MemoryStore(db_path)
    atexit.register(_store.close)

    embedding_model = _cfg.get("embedding_model", "").strip()
    if embedding_model:
        try:
            from TinyCTX.ai import Embedder
            emb_cfg  = config.get_embedding_model(embedding_model)
            _embedder = Embedder.from_config(emb_cfg)
            logger.info("[memory] embedder: %s @ %s", emb_cfg.model, emb_cfg.base_url)
        except (KeyError, ValueError) as exc:
            logger.warning(
                "[memory] embedding_model '%s' not usable (%s) — BM25 only",
                embedding_model, exc,
            )

    model_name_str = (
        config.models[embedding_model].model
        if embedding_model and embedding_model in config.models
        else ""
    )

    from TinyCTX.modules.memory.chunkers import get_strategy
    chunk_kwargs: dict = _cfg.get("chunk_kwargs") or {}
    strategy = get_strategy(_cfg["chunk_strategy"], **chunk_kwargs)

    from TinyCTX.modules.memory.indexer import MemoryIndexer
    _indexer = MemoryIndexer(
        store           = _store,
        memory_dir      = memory_dir,
        strategy        = strategy,
        embedder        = _embedder,
        embedding_model = model_name_str,
    )

    _initialized = True
    logger.info(
        "[memory] singletons ready — dir: %s | db: %s | strategy: %s | embedder: %s",
        memory_dir, db_path, _cfg["chunk_strategy"], model_name_str or "BM25 only",
    )


# ---------------------------------------------------------------------------
# register_runtime — command registration only
# ---------------------------------------------------------------------------

def register_runtime(runtime) -> None:
    global _consolidation_hook

    # We need config to read nudge settings but can't call _init_singletons
    # here (no cycle yet). Read config directly.
    try:
        from TinyCTX.modules.memory import EXTENSION_META
        defaults: dict = EXTENSION_META.get("default_config", {})
    except ImportError:
        defaults = {}
    overrides: dict = {}
    if hasattr(runtime.config, "extra") and isinstance(runtime.config.extra, dict):
        overrides = runtime.config.extra.get("memory_search", {})
    cfg = {**defaults, **overrides}

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
                "[memory] consolidation spawned off tail=%s (delta %d/%d tokens)",
                tail_node_id, tokens_now - tokens_at_nudge, nudge_delta,
            )

        _consolidation_hook = _hook
        logger.info(
            "[memory] consolidation hook configured — threshold %.0f%% delta (%d tokens)",
            nudge_threshold * 100, nudge_delta,
        )

        # /memory consolidate command
        async def _cmd_consolidate(args: list[str], context: dict) -> None:
            import time as _time, datetime
            from TinyCTX.contracts import InboundMessage, ContentType, UserIdentity, Platform
            console = context.get("console")
            c       = context.get("theme_c", lambda k: "")
            tail    = context.get("tail_node_id")
            if not tail:
                if console:
                    console.print(f"[{c('error')}]  ✗  memory: no active session[/{c('error')}]")
                return
            date_str = datetime.date.today().strftime("%d-%m-%Y")
            msg_text = nudge_message.format(date=date_str)
            opening = runtime.db.add_node(parent_id=tail, role="user", content=msg_text)
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
        logger.info("[memory] consolidation disabled")


# ---------------------------------------------------------------------------
# register_agent — all per-cycle wiring
# ---------------------------------------------------------------------------

def register_agent(cycle) -> None:
    # Initialize singletons on first call.
    _init_singletons(cycle.config)

    cfg          = _cfg
    store        = _store
    indexer      = _indexer
    embedder     = _embedder
    workspace    = _workspace
    budget_tokens = int(cfg["memory_budget_tokens"])
    top_k         = int(cfg["top_k"])
    bm25_weight   = float(cfg["bm25_weight"])
    decay_halflife_days = float(cfg.get("decay_halflife_days", 30.0))
    decay_weight        = float(cfg.get("decay_weight", 0.0))
    auto_inject         = bool(cfg["auto_inject"])
    ms_vis = str(cfg.get("tools", {}).get("memory_search", "always_on")).lower().strip()

    def _resolve(filename: str) -> Path:
        p = Path(filename)
        return p if p.is_absolute() else workspace / p

    soul_path   = _resolve(cfg["soul_file"])
    agents_path = _resolve(cfg["agents_file"])
    memory_path = _resolve(cfg["memory_file"])
    tools_path  = _resolve(cfg["tools_file"])

    from TinyCTX.modules.memory.inject import MacroResolver, make_provider
    resolver = MacroResolver()

    # 1. Static prompt providers
    cycle.context.register_prompt(
        "soul",
        make_provider(soul_path, workspace, extra_macros=resolver),
        role="system",
        priority=int(cfg["soul_priority"]),
    )
    cycle.context.register_prompt(
        "agents",
        make_provider(agents_path, workspace, extra_macros=resolver),
        role="system",
        priority=int(cfg["agents_priority"]),
    )
    cycle.context.register_prompt(
        "memory",
        make_provider(memory_path, workspace, extra_macros=resolver),
        role="system",
        priority=int(cfg["memory_priority"]),
    )
    cycle.context.register_prompt(
        "tools",
        make_provider(tools_path, workspace, extra_macros=resolver),
        role="system",
        priority=int(cfg["tools_priority"]),
    )

    # 2. Async pre-assemble hook — ephemeral results shared via closure
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
                logger.warning("[memory] embed failed: %s — BM25 only", exc)

        found = store.hybrid_search(
            query, q_vec, top_k, bm25_weight,
            decay_halflife_days=decay_halflife_days,
            decay_weight=decay_weight,
        )
        results[:] = found
        if found:
            logger.debug("[memory] '%s…' → %d result(s)", query[:40], len(found))

    cycle.context.register_hook(HOOK_PRE_ASSEMBLE_ASYNC, _pre_assemble_async, priority=0)

    # 3. Auto-inject prompt
    if auto_inject:
        cycle.context.register_prompt(
            "memory_search",
            lambda ctx: _format_results(results, budget_tokens),
            role="system",
            priority=int(cfg["search_priority"]),
        )

    # 4. memory_search tool
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
                logger.warning("[memory] tool embed failed: %s — BM25 only", exc)
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

    # 5. Post-turn consolidation hook (if configured by register_runtime)
    if _consolidation_hook is not None:
        cycle.post_turn_hooks.append(_consolidation_hook)
