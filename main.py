import uvicorn
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from src.routers.analysis import router as analysis_router

# Setup logging
logger = logging.getLogger(__name__)


app = FastAPI(
    title="The Seer v1 - Financial Research Agent",
    description="AI-powered financial analysis agent for Indian stock markets using LangGraph and advanced analytics",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors gracefully"""
    logger.error(f"Validation error: {str(exc)}")
    return JSONResponse(
        status_code=400,
        content={"detail": f"Validation error: {str(exc)}"}
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions gracefully"""
    # This avoids logging 404s as errors, which can be noisy.
    if exc.status_code != 404:
        logger.error(f"HTTP exception: {exc.detail}", exc_info=True)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors gracefully"""
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again later."}
    )


# Include routers from other files
app.include_router(router=analysis_router)


@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "The Seer v1 - Financial Research Agent",
        "version": "1.0.0",
        "description": "AI-powered financial analysis for Indian stock markets",
        "features": [
            "Comprehensive fundamental analysis",
            "Technical analysis with indicators",
            "News and sentiment analysis",
            "Market context and sector analysis",
            "Risk assessment and price targets",
            "Multi-perspective synthesis"
        ],
        "endpoints": {
            "docs": "/docs",
            "analysis": "/api/v1/analysis/",
            "health": "/api/v1/analysis/health"
        }
    }


@app.get("/health")
async def health_check():
    """Global health check endpoint"""
    return {
        "status": "healthy",
        "service": "the_seer_v1",
        "message": "Financial Research Agent is operational"
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8080, reload=True)
