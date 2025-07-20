# from fastapi import APIRouter
# from fastapi.responses import JSONResponse

# from common.cache.manager import RedisManager
# from common.config.setup import Config
# from common.vector_index.cache import SemanticCacheManager
# from pkg.services.redis_svc import RedisService

# router = APIRouter(prefix="/qa", tags=["bot"])
# redis = RedisService(
#     client=RedisManager(host=Config().REDIS_HOST, port=Config().REDIS_PORT, password=Config().REDIS_PASSWORD)
# )
# cache = SemanticCacheManager(threshold=0.95)

# @router.post("/", responses=JSONResponse)
# async def function(payload):
#     # do something
#     return {}
