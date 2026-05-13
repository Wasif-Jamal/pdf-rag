import json
from typing import Any

from redis.asyncio import Redis

from app.config.redis_config import get_redis_config


class ShortMemoryService:
    """Handles short-term conversational memory."""

    def __init__(
        self,
        redis_client: Redis,
        ttl_seconds: int = 3600,
    ) -> None:
        self.redis_client = redis_client
        self.ttl_seconds = ttl_seconds

    def _build_key(self, session_id: str) -> str:
        return f"chat:{session_id}"

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> None:
        key = self._build_key(session_id)

        message = {
            "role": role,
            "content": content,
        }

        existing_messages = await self.get_history(session_id)
        existing_messages.append(message)

        await self.redis_client.set(
            key,
            json.dumps(existing_messages),
            ex=self.ttl_seconds,
        )

    async def get_history(self, session_id: str) -> list[dict[str, Any]]:
        key = self._build_key(session_id)

        data = await self.redis_client.get(key)

        if not data:
            return []

        return json.loads(data)

    async def clear_history(self, session_id: str) -> None:
        key = self._build_key(session_id)
        await self.redis_client.delete(key)

redis_client = get_redis_config().get_client()

short_memory_service = ShortMemoryService(
    redis_client=redis_client
)