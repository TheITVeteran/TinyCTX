from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path
from typing import AsyncIterator

from TinyCTX.contracts import (
    AgentError, AgentEvent, AgentTextChunk, AgentTextFinal,
    AgentThinkingChunk, AgentToolCall, AgentToolResult,
    InboundMessage, ToolCall, ToolResult, IMAGE_BLOCK_PREFIX
)
from TinyCTX.context import Context, HistoryEntry, HOOK_PRE_ASSEMBLE_ASYNC
from TinyCTX.ai import LLM, TextDelta, ThinkingDelta, ToolCallAssembled, LLMError
from TinyCTX.utils.tool_handler import ToolCallHandler

logger = logging.getLogger(__name__)

class AgentCycle:
    """
    A single execution turn. 
    Initializes with core config, but waits until .run() to load DB and state.
    """

    def __init__(self, config, module_registry) -> None:
        self.config = config
        self.module_registry = module_registry
        self.trace_id = str(uuid.uuid4())
        
        # Resources initialized during .run()
        self.db = None
        self.context = None
        self.models: dict[str, LLM] = {}
        self.tool_handler = None
        self.permission_level = 0

    async def run(
        self,
        node_id: str,
        permission_level: int,
        abort_event: asyncio.Event | None = None,
    ) -> AsyncIterator[AgentEvent]:
        if abort_event is None:
            abort_event = asyncio.Event()

        self.permission_level = permission_level

        # --- 1. Resource Setup (Lazy Loading) ---
        if not self.db:
            from TinyCTX.db import ConversationDB
            workspace = Path(self.config.workspace.path).expanduser().resolve()
            workspace.mkdir(parents=True, exist_ok=True)
            self.db = ConversationDB(workspace / "agent.db")

        # Load session state (model choice, enabled tools, etc.)
        state, _ = self.db.load_session_state(node_id)
        
        # Build LLMs based on primary + fallbacks
        primary_name = state.get("model") or self.config.llm.primary
        model_chain = [primary_name] + list(self.config.llm.fallback)
        
        self.models = {
            name: self._build_llm(self.config.models[name])
            for name in model_chain if name in self.config.models
        }

        # Build Tools
        self.tool_handler = ToolCallHandler()
        self.tool_handler.register_tool(self.tool_handler.tools_search, always_on=True)
        enabled_tools = state.get("enabled_tools")
        if enabled_tools:
            for t in enabled_tools:
                if t in self.tool_handler.tools:
                    self.tool_handler.enabled.add(t)

        # Build Context
        primary_mc = self.config.models.get(primary_name)
        self.context = Context(
            db=self.db,
            tail_node_id=node_id,
            token_limit=self.config.context,
            image_tokens_per_block=getattr(primary_mc, "tokens_per_image", 280),
        )

        # Wire modules into this cycle turn
        self.module_registry.register_agent(self)

        # --- 2. Generation Loop ---
        # Tracker for metadata yielded in events
        meta = {
            "trace_id": self.trace_id,
            "message_id": "synthetic",
            "tail_node_id": node_id
        }

        max_cycles = self.config.max_tool_cycles
        final_text = ""
        streaming_active = False

        for cycle_num in range(max_cycles):
            if abort_event.is_set():
                yield AgentError(message="[aborted]", **meta)
                return

            # Context Assembly
            await self.context.run_async_hooks(HOOK_PRE_ASSEMBLE_ASYNC)
            tools = self.tool_handler.get_tool_definitions(
                caller_level=self.permission_level,
                minimal_tokens=self.config.permissions.minimal_tokens,
            ) or None
            
            messages, _ = self.context.assemble(tools=tools)

            # Inference with Fallback logic
            text_chunks, tool_calls_list, error = await self._perform_inference(
                messages, tools, model_chain, abort_event, meta
            )

            if error:
                yield AgentError(message=f"[LLM error: {error}]", **meta)
                return

            # Record Assistant response in Context
            response_text = "".join(text_chunks)
            self.context.add(HistoryEntry.assistant(
                content=response_text,
                tool_calls=tool_calls_list or None,
            ))
            meta["tail_node_id"] = self.context.tail_node_id

            if not tool_calls_list:
                final_text = response_text
                break

            # Tool Execution
            is_last_cycle = (cycle_num == max_cycles - 1)
            for tc in tool_calls_list:
                yield AgentToolCall(call_id=tc.call_id, tool_name=tc.tool_name, args=tc.args, **meta)
                
                result = await self._execute_tool(tc)
                
                if is_last_cycle:
                    result.output = "[Tool Limit Reached] Summarize now."
                    result.is_error = True

                self.context.add(HistoryEntry.tool_result(result))
                meta["tail_node_id"] = self.context.tail_node_id
                
                yield AgentToolResult(
                    call_id=result.call_id,
                    tool_name=result.tool_name,
                    output="[image]" if result.is_image else result.output,
                    is_error=result.is_error,
                    **meta
                )

        yield AgentTextFinal(text=final_text if not streaming_active else "", **meta)

    # --- Internal Helpers ---

    def _build_llm(self, mc) -> LLM:
        return LLM(
            base_url=mc.base_url,
            api_key=getattr(mc, "api_key", "no-key"),
            model=mc.model,
            max_tokens=mc.max_tokens,
            temperature=mc.temperature,
        )

    async def _perform_inference(self, messages, tools, model_chain, abort_event, meta):
        """Walks the model chain until success or exhaustion."""
        for model_name in model_chain:
            llm = self.models[model_name]
            chunks, calls, error = [], [], None

            async for ev in llm.stream(messages, tools=tools):
                if abort_event.is_set(): return [], [], "aborted"
                
                if isinstance(ev, ThinkingDelta):
                    yield AgentThinkingChunk(text=ev.text, **meta)
                elif isinstance(ev, TextDelta):
                    chunks.append(ev.text)
                    yield AgentTextChunk(text=ev.text, **meta)
                elif isinstance(ev, ToolCallAssembled):
                    calls.append(ToolCall(ev.call_id, ev.tool_name, ev.args))
                elif isinstance(ev, LLMError):
                    error = ev.message
                    break
            
            if not error: return chunks, calls, None
            logger.warning("Model %s failed: %s", model_name, error)
        
        return [], [], error

    async def _execute_tool(self, call: ToolCall) -> ToolResult:
        proxy = {"function": {"name": call.tool_name, "arguments": call.args}, "id": call.call_id}
        res = await self.tool_handler.execute_tool_call(proxy, caller_level=self.permission_level)
        
        raw = str(res.get("result", res.get("error", "[no output]")))
        is_err = (not res.get("success", False))
        
        # Handle Image outputs
        if not is_err and raw.startswith(IMAGE_BLOCK_PREFIX):
            # ... (Existing image parsing logic) ...
            pass

        return ToolResult(call.call_id, call.tool_name, output=raw, is_error=is_err)