from fastapi import FastAPI

from .routers.sample import router as sample


app = FastAPI(
    title="Trademarkia AI Service",
)

app.include_router(router=sample)