from typing import List

import numpy as np
from qdrant_client import models
from qdrant_client.conversions.common_types import ScoredPoint, UpdateResult
from typing_extensions import Self

from common.schemas.entities import SampleRequestPayload
from common.vector_index.manager import QdrantManager


class QdrantService:
    """Implementation of Semantic Cache"""

    _instance = None

    def __new__(cls, *args, **kwargs) -> Self:
        if cls._instance is None:
            cls._instance = super(QdrantService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, client: QdrantManager, threshold: float = 0.95) -> None:
        if self._initialized:
            return
        self._initialized = True
        self.client = client
        self.threshold = threshold

    def upsert(self, collection_name: str, vector: np.ndarray, payload: QuestionPayload) -> UpdateResult:
        info = self.client.get_connection().upsert(
            collection_name=collection_name,
            wait=True,
            points=[models.PointStruct(id=payload.id, vector=vector, payload=payload.dict())],
        )
        return info

    def get_topk_similar_vectors(
        self, collection_name: str, vector: np.ndarray, limit: int, threshold: float
    ) -> List[ScoredPoint]:
        result = (
            self.client.get_connection()
            .query_points(
                collection_name=collection_name,
                query=vector,
                limit=limit,
                score_threshold=threshold,
                with_payload=True,
            )
            .points
        )
        return result

    def get_topk_similar_vectors_with_filters(
        self,
        collection_name: str,
        vector: np.ndarray,
        limit: int,
        threshold: float,
        filters: models.Filter,
    ) -> List[ScoredPoint]:
        result = (
            self.client.get_connection()
            .query_points(
                collection_name=collection_name,
                query=vector,
                limit=limit,
                score_threshold=threshold,
                query_filter=filters,
                with_payload=True,
            )
            .points
        )
        return result