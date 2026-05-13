from app.services.short_memory_service import ShortMemoryService


class MemoryManagerService:
    """Combines short-term and long-term memory."""

    def __init__(
        self,
        short_memory_service: ShortMemoryService,
    ) -> None:
        self.short_memory_service = short_memory_service

    async def build_context(
        self,
        session_id: str,
    ) -> dict:
        chat_history = await self.short_memory_service.get_history(session_id)

        return {
            "chat_history": chat_history,
        }