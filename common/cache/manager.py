import redis
from typing_extensions import Self

from common.utils.logging import logger


class RedisManager:
    """A singleton class for creating redis client."""

    _instance = None

    def __new__(cls, host="localhost", port=6379, password="", decode_responses=True) -> Self:
        if cls._instance is None:
            cls._instance = super(RedisManager, cls).__new__(cls)
            logger.info(f"{type(cls._instance).__name__} instance initialized...")
            try:
                cls._instance.client = redis.StrictRedis(
                    host=host,
                    port=port,
                    password=password,
                    decode_responses=decode_responses,
                )
                logger.info(f"Initialized {type(cls._instance.client).__name__} successfully...")
            except redis.RedisError as e:
                logger.error(f"Error when initializing {type(cls._instance).__name__}...\n\n{e}")
                cls._instance = None
        return cls._instance

    def __repr__(self) -> str:
        return f"<{type(self._instance).__name__} with client={self._instance.client}>"

    def __getattr__(self, name):
        """Delegate attribute access to the underlying redis client."""
        return getattr(self.client, name)

    def get_connection(self) -> redis.StrictRedis:
        return self._instance.client


if __name__ == "__main__":
    r = RedisManager()
    print(r)