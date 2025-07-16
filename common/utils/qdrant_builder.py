from typing import Any, Dict

from qdrant_client import models


def build_filters(must: Dict[str, Any]) -> models.Filter:
    must_filters = []
    for k, it in must.items():
        if isinstance(it, str) or isinstance(it, int) or isinstance(it, float) or isinstance(it, bool):
            must_filters.append(models.FieldCondition(key=k, match=models.MatchValue(value=it)))
        elif isinstance(it, list):
            must_filters.append(models.FieldCondition(key=k, match=models.MatchAny(any=it)))

    return models.Filter(must=must_filters)