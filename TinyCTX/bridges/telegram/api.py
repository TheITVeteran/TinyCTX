"""
bridges/telegram/api.py — thin Telegram Bot API client (aiohttp, no extra
dependency). Long-polling via getUpdates; only the handful of methods the
bridge needs. `api_base` is overridable for tests (mock server).
"""

from __future__ import annotations

import logging

import aiohttp

logger = logging.getLogger(__name__)

DEFAULT_API_BASE = "https://api.telegram.org"


class TelegramAPIError(Exception):
    pass


class TelegramAPI:
    def __init__(self, token: str, api_base: str = DEFAULT_API_BASE) -> None:
        self._base = f"{api_base.rstrip('/')}/bot{token}"
        self._session: aiohttp.ClientSession | None = None

    async def _call(self, method: str, *, http_timeout: float = 30.0, **params) -> dict | list:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        async with self._session.post(
            f"{self._base}/{method}", json=params,
            timeout=aiohttp.ClientTimeout(total=http_timeout),
        ) as resp:
            data = await resp.json()
        if not data.get("ok"):
            raise TelegramAPIError(f"{method}: {data.get('description', data)}")
        return data["result"]

    async def get_me(self) -> dict:
        return await self._call("getMe")

    async def get_updates(self, offset: int, poll_timeout: int = 50) -> list[dict]:
        # HTTP timeout must outlast Telegram's long-poll hold time.
        return await self._call(
            "getUpdates",
            http_timeout=poll_timeout + 10,
            offset=offset,
            timeout=poll_timeout,
            allowed_updates=["message", "edited_message"],
        )

    async def send_message(self, chat_id: int | str, text: str) -> dict:
        return await self._call("sendMessage", chat_id=chat_id, text=text)

    async def send_chat_action(self, chat_id: int | str, action: str = "typing") -> None:
        try:
            await self._call("sendChatAction", chat_id=chat_id, action=action)
        except Exception:
            pass  # cosmetic only

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
