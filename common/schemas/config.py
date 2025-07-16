from dataclasses import dataclass

from qdrant_client.http.models.models import Distance


@dataclass
class QdrantCollectionConfig:
    collection_name: str
    embedding_dim: int
    distance_metric: Distance


@dataclass
class Config:
    qdrant_url: str
    redis_host: str
    redis_port: int
    redis_password: str
