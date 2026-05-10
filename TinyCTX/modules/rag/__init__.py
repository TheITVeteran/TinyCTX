EXTENSION_META = {
    "name":    "rag",
    "version": "2.1",
    "description": (
        "Hybrid BM25+vector search index over *.md files under workspace/memory/. "
        "Provides a memory_search tool and an auto-inject system prompt block. "
        "Load alongside modules/system_prompt, which handles static prompt injection."
    ),
    "default_config": {
        # --- Search index ---
        # Directory scanned recursively for *.md files, relative to workspace
        "memory_dir": "memory",
        # SQLite cache DB, relative to workspace
        "db_file":    "memory/cache.db",

        # --- Chunking ---
        # Strategy name: "markdown" | "tokens" | "chars" | "delimiter"
        "chunk_strategy": "markdown",
        # Strategy kwargs — passed through to get_strategy(); leave empty for defaults.
        # e.g. for tokens:    {"chunk_tokens": 256, "overlap_tokens": 32}
        # e.g. for delimiter: {"delimiter": "---"}
        "chunk_kwargs": {},

        # --- Embedding ---
        # Key from models: with kind: embedding, or "" for BM25-only mode.
        "embedding_model": "",

        # --- Retrieval ---
        # Maximum chunks fetched from the index before budget trimming.
        "top_k": 5,
        # BM25 share of hybrid score (vector weight = 1 - bm25_weight).
        "bm25_weight": 0.3,
        # Temporal decay — down-weights chunks from older files.
        # decay_weight=0.0 disables decay entirely (default, preserves existing behaviour).
        # decay_weight=1.0 multiplies score fully by the decay factor.
        # decay_halflife_days: age in days at which a file's score is halved.
        "decay_halflife_days": 30.0,
        "decay_weight": 0.0,

        # --- Memory budget ---
        # Maximum tokens the injected <memory> block may occupy in the system
        # prompt. Chunks are included highest-score-first; once adding the next
        # chunk would exceed the budget it (and all remaining chunks) are dropped
        # and a truncation note is appended.
        # Set to 0 to disable budget enforcement (inject all top_k results).
        # Rule of thumb: keep this well below context / 4 so search results
        # don't crowd out conversation history.
        "memory_budget_tokens": 2048,

        # --- Auto-inject ---
        # true:  inject the budgeted results as a system prompt block every turn.
        # false: retrieval only via memory_search tool.
        "auto_inject": True,
        # System prompt priority for the injected block (after MEMORY.md at 20).
        "search_priority": 25,

        # --- Context nudge ---
        # Inject a reminder to write important info to memory when the tokens
        # accumulated *since the last nudge* exceed this fraction of the context
        # window. Tracks delta, not absolute fill, so the nudge recurs only when
        # enough new content has arrived since the agent last saved.
        # Set to 0.0 to disable nudging entirely.
        "nudge_threshold": 0.80,
        # The nudge message injected as a user turn at the end of the context.
        "nudge_message": (
            "<context_sentinel>"
            "Your context window is getting full. "
            "Please write any important information, decisions, or ongoing tasks "
            "to memory/session-{date}.md or MEMORY.md before continuing."
            "</context_sentinel>"
        ),
    },
}
