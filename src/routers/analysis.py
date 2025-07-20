"""
Financial Analysis API Router
Provides endpoints for running and retrieving financial analysis reports
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any
import logging
from datetime import datetime

from common.schemas.entities import (
    AnalysisRequest, AnalysisResponse, AnalysisStatus,
    FinancialAnalysisReport, AnalysisType, AnalysisTimeHorizon
)
from src.services.agent_runner import (
    start_financial_analysis, get_analysis_status, get_analysis_result,
    analysis_runner
)
from common.config.setup import Config


# Setup logging
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/v1/analysis", tags=["Financial Analysis"])

# Initialize config
config = Config()


@router.post("/start", response_model=Dict[str, str])
async def start_analysis(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, str]:
    """
    Start a new financial analysis for a given ticker
    
    Args:
        request: Analysis request with ticker and analysis parameters
        background_tasks: FastAPI background tasks for async processing
    
    Returns:
        Dict with session_id for tracking the analysis
        
    Example:
        POST /api/v1/analysis/start
        {
            "ticker": "RELIANCE.NS",
            "analysis_type": "comprehensive",
            "time_horizon": "medium_term",
            "include_technical": true,
            "include_fundamental": true,
            "include_news": true,
            "include_sector": true
        }
    """
    try:
        # Validate ticker format for Indian stocks
        if not _validate_indian_ticker(request.ticker):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid ticker format: {request.ticker}. Expected format: SYMBOL.NS"
            )
        
        # Start the analysis workflow
        session_id = await start_financial_analysis(request)
        
        logger.info(f"Started financial analysis for {request.ticker} with session {session_id}")
        
        return {
            "session_id": session_id,
            "message": f"Analysis started for {request.ticker}",
            "status": "initiated"
        }
        
    except Exception as e:
        logger.error(f"Error starting analysis for {request.ticker}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start analysis: {str(e)}"
        )


@router.get("/status/{session_id}", response_model=AnalysisStatus)
async def get_status(session_id: str) -> AnalysisStatus:
    """
    Get the current status of a financial analysis
    
    Args:
        session_id: Unique session identifier from start_analysis
    
    Returns:
        Current analysis status including progress and current step
        
    Example:
        GET /api/v1/analysis/status/123e4567-e89b-12d3-a456-426614174000
    """
    try:
        status = await get_analysis_status(session_id)
        
        if not status:
            raise HTTPException(
                status_code=404,
                detail=f"Analysis session {session_id} not found"
            )
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting status for session {session_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get analysis status: {str(e)}"
        )


@router.get("/result/{session_id}", response_model=AnalysisResponse)
async def get_result(session_id: str) -> AnalysisResponse:
    """
    Get the completed financial analysis result
    
    Args:
        session_id: Unique session identifier from start_analysis
    
    Returns:
        Complete analysis response with financial report (if completed)
        
    Example:
        GET /api/v1/analysis/result/123e4567-e89b-12d3-a456-426614174000
    """
    try:
        result = await get_analysis_result(session_id)
        
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Analysis session {session_id} not found"
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting result for session {session_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get analysis result: {str(e)}"
        )


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_stock(
    request: AnalysisRequest,
    wait_for_completion: bool = False,
    timeout_seconds: int = 300
) -> AnalysisResponse:
    """
    Start analysis and optionally wait for completion (synchronous mode)
    
    Args:
        request: Analysis request with ticker and parameters
        wait_for_completion: Whether to wait for analysis completion
        timeout_seconds: Maximum time to wait for completion (default 300s)
    
    Returns:
        Analysis response (immediate session info or completed analysis)
        
    Example:
        POST /api/v1/analysis/analyze?wait_for_completion=true
        {
            "ticker": "TCS.NS",
            "analysis_type": "comprehensive"
        }
    """
    try:
        # Start the analysis
        session_id = await start_financial_analysis(request)
        
        if not wait_for_completion:
            # Return session info immediately
            return AnalysisResponse(
                session_id=session_id,
                status="initiated",
                analysis=None,
                metadata=None,
                error_message=None
            )
        
        # Wait for completion with timeout
        import asyncio
        
        async def wait_for_result():
            max_attempts = timeout_seconds // 5  # Check every 5 seconds
            
            for attempt in range(max_attempts):
                await asyncio.sleep(5)  # Wait 5 seconds between checks
                
                result = await get_analysis_result(session_id)
                if result and result.status in ["completed", "failed"]:
                    return result
            
            # Timeout reached - return a 'failed' status
            return AnalysisResponse(
                session_id=session_id,
                status="failed",
                analysis=None,
                metadata=None,
                error_message=f"Analysis timed out after {timeout_seconds} seconds"
            )
        
        result = await wait_for_result()
        return result
        
    except Exception as e:
        logger.error(f"Error in analyze_stock for {request.ticker}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@router.get("/sessions", response_model=Dict[str, Any])
async def list_active_sessions() -> Dict[str, Any]:
    """
    List all active analysis sessions (for monitoring/debugging)
    
    Returns:
        Dictionary of active sessions with their metadata
        
    Example:
        GET /api/v1/analysis/sessions
    """
    try:
        active_sessions = await analysis_runner.list_active_sessions()
        
        return {
            "total_sessions": len(active_sessions),
            "sessions": active_sessions,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error listing active sessions: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list sessions: {str(e)}"
        )


@router.delete("/session/{session_id}")
async def cancel_analysis(session_id: str) -> Dict[str, str]:
    """
    Cancel an ongoing analysis session
    
    Args:
        session_id: Session identifier to cancel
    
    Returns:
        Confirmation message
        
    Example:
        DELETE /api/v1/analysis/session/123e4567-e89b-12d3-a456-426614174000
    """
    try:
        # Check if session exists
        status = await get_analysis_status(session_id)
        
        if not status:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found"
            )
        
        if status.status in ["completed", "failed"]:
            return {
                "session_id": session_id,
                "message": f"Session already {status.status}, nothing to cancel"
            }
        
        # Mark session as cancelled (simplified implementation)
        # In production, you'd want to actually interrupt the workflow
        if session_id in analysis_runner.active_sessions:
            analysis_runner.active_sessions[session_id]["status"] = "cancelled"
        
        logger.info(f"Cancelled analysis session {session_id}")
        
        return {
            "session_id": session_id,
            "message": "Analysis session cancelled successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling session {session_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cancel session: {str(e)}"
        )


@router.post("/batch", response_model=Dict[str, Any])
async def batch_analysis(
    tickers: list[str],
    analysis_type: AnalysisType = AnalysisType.QUICK,
    time_horizon: AnalysisTimeHorizon = AnalysisTimeHorizon.MEDIUM_TERM
) -> Dict[str, Any]:
    """
    Start batch analysis for multiple tickers
    
    Args:
        tickers: List of ticker symbols to analyze
        analysis_type: Type of analysis for all tickers
        time_horizon: Time horizon for all analyses
    
    Returns:
        Dictionary with session IDs for each ticker
        
    Example:
        POST /api/v1/analysis/batch
        {
            "tickers": ["RELIANCE.NS", "TCS.NS", "INFY.NS"],
            "analysis_type": "quick",
            "time_horizon": "medium_term"
        }
    """
    try:
        if len(tickers) > 10:  # Limit batch size
            raise HTTPException(
                status_code=400,
                detail="Batch size limited to 10 tickers maximum"
            )
        
        session_ids = {}
        errors = {}
        
        for ticker in tickers:
            try:
                # Validate ticker
                if not _validate_indian_ticker(ticker):
                    errors[ticker] = f"Invalid ticker format: {ticker}"
                    continue
                
                # Create analysis request
                request = AnalysisRequest(
                    ticker=ticker,
                    analysis_type=analysis_type,
                    time_horizon=time_horizon
                )
                
                # Start analysis
                session_id = await start_financial_analysis(request)
                session_ids[ticker] = session_id
                
            except Exception as e:
                errors[ticker] = str(e)
        
        logger.info(f"Started batch analysis for {len(session_ids)} tickers")
        
        return {
            "success_count": len(session_ids),
            "error_count": len(errors),
            "session_ids": session_ids,
            "errors": errors,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch analysis: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch analysis failed: {str(e)}"
        )


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint for the analysis service
    
    Returns:
        Service health status
    """
    try:
        # Basic health checks
        active_sessions = await analysis_runner.list_active_sessions()
        
        return {
            "status": "healthy",
            "service": "financial_analysis",
            "active_sessions": str(len(active_sessions)),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Service unhealthy: {str(e)}"
        )


# Helper functions

def _validate_indian_ticker(ticker: str) -> bool:
    """
    Validate Indian stock ticker format
    
    Args:
        ticker: Ticker symbol to validate
        
    Returns:
        True if valid Indian ticker format
    """
    if not ticker:
        return False
    
    # Indian NSE tickers typically end with .NS
    if not ticker.endswith('.NS'):
        # Allow tickers without .NS suffix for flexibility
        return len(ticker) >= 2 and ticker.isalnum()
    
    # Remove .NS suffix and check base ticker
    base_ticker = ticker[:-3]
    return len(base_ticker) >= 2 and base_ticker.isalnum() 