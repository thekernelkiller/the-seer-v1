from typing import List

from qdrant_client import QdrantClient, models
from tqdm.auto import tqdm
from typing_extensions import Self

from common.schemas.config import QdrantCollectionConfig
from common.utils.logging import logger


class QdrantManager:
    _instance = None

    def __new__(
        cls,
        url: str = "http://localhost:6333",
        collection_config: List[QdrantCollectionConfig] = [],
    ) -> Self:
        if cls._instance is None:
            cls._instance = super(QdrantManager, cls).__new__(cls)
            logger.info(f"{type(cls._instance).__name__} instance initialized...")
            try:
                cls._instance.client = QdrantClient(url=url)
                for conf in tqdm(collection_config, total=len(collection_config)):
                    try:
                        cls._instance.client.create_collection(
                            collection_name=conf.collection_name,
                            vectors_config=models.VectorParams(
                                size=conf.embedding_dim,
                                distance=conf.distance_metric,
                            ),
                        )
                        logger.info(f"Collection {conf.collection_name} created...")
                    except Exception as e:
                        logger.error(f"Failed to create collection...\n\n{e}")
            except Exception as e:
                logger.error(f"Error when initializing {type(cls._instance).__name__}...\n\n{e}")
                cls._instance = None
        return cls._instance

    def __repr__(self) -> str:
        return f"<{type(self._instance).__name__} with client={self._instance.client}>"

    def get_connection(self) -> QdrantClient:
        return self._instance.client


if __name__ == "__main__":
    collection_config = [
        QdrantCollectionConfig(
            collection_name="test1",
            embedding_dim=1,
            distance_metric=models.Distance.DOT,
        ),
        QdrantCollectionConfig(
            collection_name="test2",
            embedding_dim=1,
            distance_metric=models.Distance.DOT,
        ),
    ]

    db = QdrantManager(collection_config=collection_config)
    print(db)
