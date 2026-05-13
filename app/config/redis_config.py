from functools import lru_cache

from redis.asyncio import Redis


class RedisConfig:
    """Centralized Redis configuration."""

    def __init__(self) -> None:
        self.redis_url = "redis://localhost:6379/0"

    def get_client(self) -> Redis:
        return Redis.from_url(
            self.redis_url,
            decode_responses=True,
        )


@lru_cache
def get_redis_config() -> RedisConfig:
    return RedisConfig()