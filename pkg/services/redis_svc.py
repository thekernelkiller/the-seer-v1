from typing import Awaitable, Optional, Union

from typing_extensions import Any, Self

from common.cache.manager import RedisManager


class RedisService:
    _instance = None

    def __new__(cls, *args, **kwargs) -> Self:
        if cls._instance is None:
            cls._instance = super(RedisService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, client: RedisManager) -> None:
        if self._initialized:
            return
        self._initalized = True
        self.client = client

    def set(self, key: str, value: str) -> Union[Awaitable[int], int]:
        return self.client.get_connection().set(key, value)

    def get(self, key: str) -> Optional[str]:
        resp = self.client.get_connection().get(key)
        return str(resp)

    def delete(self, *keys) -> Union[Awaitable[Any], Any]:
        return self.client.get_connection().delete(keys)