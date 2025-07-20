import time
import uuid

from qdrant_client.http.models import PointStruct, models
from typing_extensions import Self

from common.config.setup import Config
# from common.utils.embedding import get_embeddings_for_text
from common.vector_index.manager import QdrantManager


def get_embedding(question: str) -> list[float]:
    # return list(get_embeddings_for_text(text=question))
    pass


class SemanticCacheManager:
    _instance = None

    def __new__(cls, threshold: float) -> Self:
        if cls._instance is None:
            cls._instance = super(SemanticCacheManager, cls).__new__(cls)
            cls._instance.qdrant = QdrantManager(
                url=Config().QDRANT_URL,
                collection_config=Config().QDRANT_COLLECTIONS,
            )
            cls._instance.__initialized = False
        return cls._instance

    def __init__(self, threshold: float = 0.80) -> None:
        if self.__initialized:
            return
        self.__initialized = True
        self.threshold = threshold

    # def __repr__(self) -> str:
    #     return f"<{type(self._instance).__name__} with client={self._instance}>"

    def search_cache(self, embedding: list[float], filters: models.Filter = None):
        search_result = self.qdrant.get_connection().search(
            collection_name=Config().QDRANT_CACHE_INDEX, query_vector=embedding, limit=1, query_filter=filters
        )
        return search_result

    def add_to_cache(self, question: str, question_id: str, response_text: str) -> None:
        point_id = str(uuid.uuid4())
        vector = get_embedding(question=question)
        point = PointStruct(
            id=point_id,
            vector=vector,
            payload={
                "response_text": response_text,
                "question": question,
                "question_id": question_id
            },
        )
        self.qdrant.get_connection().upload_points(
            collection_name=Config().QDRANT_CACHE_INDEX, points=[point]
        )

    def search(self, question: str, filters: models.Filter = None):
        start = time.time()
        vector = get_embedding(question=question)
        search_result = self.search_cache(vector, filters)
        if search_result:
            for s in search_result:
                print(s.id, s.score)
                if s.score >= self.threshold:
                    print(f"Answer present in the cache [score={s.score:.4f}]")
                    total_time = time.time() - start
                    return {"payload": s.payload, "time": total_time}
        return {"payload": None, "time": time.time() - start}