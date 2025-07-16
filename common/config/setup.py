import os

import dotenv
from qdrant_client import models
from typing_extensions import Self

from common.schemas.config import QdrantCollectionConfig


class Config:
    _instance = None

    def __new__(cls, *args, **kwargs) -> Self:
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance.__initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self.__initialized:
            return
        self.__initialized = True
        dotenv.load_dotenv()

        self.QDRANT_URL = os.environ.get("QDRANT_URL", None)
        self.REDIS_HOST = os.environ.get("REDIS_HOST", None)
        self.REDIS_PORT = os.environ.get("REDIS_PORT", None)
        self.REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", None)
        self.QDRANT_QA_INDEX = os.environ.get("QDRANT_QA_INDEX", None)
        self.GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", None)
        self.QDRANT_CACHE_INDEX = os.environ.get("QDRANT_CACHE_INDEX", None)

        assert self.QDRANT_URL is not None, "QDRANT_URL not specified."
        assert self.REDIS_PORT is not None, "REDIS_PORT not specified."
        assert self.REDIS_HOST is not None, "REDIS_HOST not specified."
        assert self.REDIS_PASSWORD is not None, "REDIS_PASSWORD not specified."
        assert self.QDRANT_QA_INDEX is not None, "QDRANT_QA_INDEX not specified."
        assert self.GEMINI_API_KEY is not None, "GEMINI_API_KEY not specified."
        assert self.QDRANT_CACHE_INDEX is not None, "QDRANT_CACHE_INDEX not specified."

        self.QDRANT_COLLECTIONS = [
            QdrantCollectionConfig(
                collection_name=self.QDRANT_QA_INDEX,
                embedding_dim=384,
                distance_metric=models.Distance.DOT,
            ),
            QdrantCollectionConfig(
                collection_name=self.QDRANT_CACHE_INDEX,
                embedding_dim=384,
                distance_metric=models.Distance.DOT,
            )
        ]


if __name__ == "__main__":
    conf = Config()
    print(conf.QDRANT_URL, conf.REDIS_HOST, conf.REDIS_PORT, conf.REDIS_PASSWORD)