import uvicorn

from src.routes import app


@app.get("/")
async def say_hi():
    return {"message": "hello world"}


if __name__ == "__main__":
    uvicorn.run(app=app, host="127.0.0.1", port=8080)
