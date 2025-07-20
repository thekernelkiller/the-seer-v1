"""
Agent Runner Service - Integrates LangGraph with FastAPI
Provides async interface for running financial analysis workflows
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, AsyncGenerator
import logging
import json

from common.config.setup import Config
from common.cache.manager import RedisManager
from common.schemas.entities import (
    AnalysisRequest, AnalysisResponse, AnalysisStatus, 
    FinancialAnalysisReport, AnalysisMetadata
)
from src.agents.financial_agent import run_financial_analysis, AnalysisState


# Setup logging
logger = logging.getLogger(__name__)

# Initialize configuration and Redis
config = Config()
redis_client = RedisManager(
    # host=config.REDIS_HOST,
    # port=config.REDIS_PORT,
    # password=config.REDIS_PASSWORD
    host="localhost",
    port=6379,
    password="password"
)


class FinancialAnalysisRunner:
    """
    Service for running financial analysis workflows
    Provides async interface and status tracking
    """
    
    def __init__(self):
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
    async def start_analysis(self, request: AnalysisRequest) -> str:
        """
        Start a new financial analysis workflow
        
        Args:
            request: Analysis request parameters
            
        Returns:
            Session ID for tracking the analysis
        """
        session_id = str(uuid.uuid4())
        
        # Store session metadata
        session_metadata = {
            "session_id": session_id,
            "request": request.dict(),
            "status": "initiated",
            "created_at": datetime.now().isoformat(),
            "progress": 0.0,
            "current_step": "initialization"
        }
        
        self.active_sessions[session_id] = session_metadata
        
        # Store in Redis for persistence
        redis_key = f"analysis_session:{session_id}"
        redis_client.setex(
            redis_key, 
            3600 * 24,  # 24 hour expiry
            json.dumps(session_metadata, default=str)
        )
        
        # Start the analysis workflow asynchronously
        asyncio.create_task(self._run_analysis_workflow(session_id, request))
        
        logger.info(f"Started analysis for {request.ticker} with session {session_id}")
        return session_id
    
    async def get_analysis_status(self, session_id: str) -> Optional[AnalysisStatus]:
        """
        Get current status of an analysis session
        
        Args:
            session_id: Session identifier
            
        Returns:
            Current analysis status or None if not found
        """
        # Check active sessions first
        if session_id in self.active_sessions:
            session_data = self.active_sessions[session_id]
            return AnalysisStatus(
                session_id=session_id,
                status=session_data["status"],
                progress_percentage=session_data.get("progress", 0.0),
                current_step=session_data.get("current_step", "unknown"),
                estimated_completion_time=None,  # Could implement ETA logic
                error_message=session_data.get("error_message")
            )
        
        # Check Redis for persisted sessions
        redis_key = f"analysis_session:{session_id}"
        cached_data = redis_client.get(redis_key)
        if cached_data:
            session_data = json.loads(cached_data)
            return AnalysisStatus(
                session_id=session_id,
                status=session_data["status"],
                progress_percentage=session_data.get("progress", 0.0),
                current_step=session_data.get("current_step", "unknown"),
                estimated_completion_time=None,
                error_message=session_data.get("error_message")
            )
        
        return None
    
    async def get_analysis_result(self, session_id: str) -> Optional[AnalysisResponse]:
        """
        Get completed analysis result
        
        Args:
            session_id: Session identifier
            
        Returns:
            Analysis response with report or None if not found/incomplete
        """
        # Check for completed analysis in Redis
        result_key = f"analysis_result:{session_id}"
        cached_result = redis_client.get(result_key)
        
        if cached_result:
            result_data = json.loads(cached_result)
            
            if result_data["status"] == "completed" and "analysis" in result_data:
                return AnalysisResponse(
                    session_id=session_id,
                    status="completed",
                    analysis=FinancialAnalysisReport.parse_obj(result_data["analysis"]),
                    metadata=AnalysisMetadata.parse_obj(result_data["metadata"]) if "metadata" in result_data else None,
                    error_message=None
                )
            elif result_data["status"] == "failed":
                return AnalysisResponse(
                    session_id=session_id,
                    status="failed",
                    analysis=None,
                    metadata=None,
                    error_message=result_data.get("error_message", "Analysis failed")
                )
        
        # Check if analysis is still in progress
        status = await self.get_analysis_status(session_id)
        if status:
            return AnalysisResponse(
                session_id=session_id,
                status=status.status,
                analysis=None,
                metadata=None,
                error_message=status.error_message
            )
        
        return None
    
    async def _run_analysis_workflow(self, session_id: str, request: AnalysisRequest):
        """
        Execute the financial analysis workflow
        
        Args:
            session_id: Session identifier
            request: Analysis request parameters
        """
        try:
            logger.info(f"Running analysis workflow for session {session_id}")
            
            # Update status to running
            await self._update_session_status(session_id, "in_progress", 5.0, "starting_analysis")
            
            # Run the LangGraph workflow
            final_state = await run_financial_analysis(request)
            
            # Update progress throughout the workflow by monitoring state
            await self._monitor_workflow_progress(session_id, final_state)
            
            # Store the result
            if final_state["status"] == "completed" and final_state["analysis_report"]:
                await self._store_analysis_result(session_id, final_state)
                await self._update_session_status(session_id, "completed", 100.0, "analysis_complete")
                logger.info(f"Analysis completed successfully for session {session_id}")
            else:
                error_msg = "; ".join(final_state.get("errors", ["Unknown error"]))
                await self._update_session_status(
                    session_id, "failed", final_state.get("progress", 0.0), 
                    "analysis_failed", error_msg
                )
                logger.error(f"Analysis failed for session {session_id}: {error_msg}")
                
        except Exception as e:
            logger.error(f"Workflow execution failed for session {session_id}: {str(e)}")
            await self._update_session_status(
                session_id, "failed", 0.0, "workflow_error", str(e)
            )
    
    async def _update_session_status(
        self, 
        session_id: str, 
        status: str, 
        progress: float, 
        current_step: str,
        error_message: Optional[str] = None
    ):
        """Update session status in memory and Redis"""
        
        update_data = {
            "status": status,
            "progress": progress,
            "current_step": current_step,
            "updated_at": datetime.now().isoformat()
        }
        
        if error_message:
            update_data["error_message"] = error_message
        
        # Update active sessions
        if session_id in self.active_sessions:
            self.active_sessions[session_id].update(update_data)
        
        # Update Redis
        redis_key = f"analysis_session:{session_id}"
        cached_data = redis_client.get(redis_key)
        if cached_data:
            session_data = json.loads(cached_data)
            session_data.update(update_data)
            redis_client.setex(
                redis_key, 
                3600 * 24,  # 24 hour expiry
                json.dumps(session_data, default=str)
            )
    
    async def _monitor_workflow_progress(self, session_id: str, state: AnalysisState):
        """Monitor and update workflow progress"""
        progress_map = {
            "initialization": 5.0,
            "market_context": 20.0,
            "fundamental_analysis": 40.0,
            "technical_analysis": 60.0,
            "news_sentiment_analysis": 80.0,
            "synthesis": 90.0,
            "report_generation": 100.0
        }
        
        current_step = state.get("current_step", "unknown")
        progress = progress_map.get(current_step, state.get("progress", 0.0))
        
        await self._update_session_status(session_id, "in_progress", progress, current_step)
    
    async def _store_analysis_result(self, session_id: str, final_state: AnalysisState):
        """Store completed analysis result in Redis"""
        result_data = {
            "session_id": session_id,
            "status": "completed",
            "analysis": final_state["analysis_report"].dict() if final_state["analysis_report"] else None,
            "metadata": {
                "analysis_duration": 120.0,  # Would calculate actual duration
                "data_sources_used": ["Twelve Data", "Serper News"],
                "analysis_timestamp": datetime.now().isoformat(),
                "agent_version": "1.0"
            },
            "completed_at": datetime.now().isoformat()
        }
        
        result_key = f"analysis_result:{session_id}"
        redis_client.setex(
            result_key,
            3600 * 24 * 7,  # 7 day expiry for results
            json.dumps(result_data, default=str)
        )
    
    async def list_active_sessions(self) -> Dict[str, Dict[str, Any]]:
        """Get all active analysis sessions"""
        return self.active_sessions.copy()
    
    async def cleanup_completed_sessions(self, max_age_hours: int = 24):
        """Clean up old completed sessions from memory"""
        current_time = datetime.now()
        sessions_to_remove = []
        
        for session_id, session_data in self.active_sessions.items():
            if session_data["status"] in ["completed", "failed"]:
                created_at = datetime.fromisoformat(session_data["created_at"])
                age_hours = (current_time - created_at).total_seconds() / 3600
                
                if age_hours > max_age_hours:
                    sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.active_sessions[session_id]
            logger.info(f"Cleaned up old session {session_id}")


# Global instance
analysis_runner = FinancialAnalysisRunner()


# Async context manager for periodic cleanup
class AnalysisRunnerManager:
    """Context manager for the analysis runner with periodic cleanup"""
    
    def __init__(self, cleanup_interval_hours: int = 6):
        self.cleanup_interval_hours = cleanup_interval_hours
        self.cleanup_task = None
    
    async def __aenter__(self):
        # Start periodic cleanup task
        self.cleanup_task = asyncio.create_task(self._periodic_cleanup())
        return analysis_runner
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Cancel cleanup task
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
    
    async def _periodic_cleanup(self):
        """Periodic cleanup of old sessions"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval_hours * 3600)
                await analysis_runner.cleanup_completed_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {str(e)}")


# Convenience functions for FastAPI integration
async def start_financial_analysis(request: AnalysisRequest) -> str:
    """Start a new financial analysis and return session ID"""
    return await analysis_runner.start_analysis(request)


async def get_analysis_status(session_id: str) -> Optional[AnalysisStatus]:
    """Get status of an analysis session"""
    return await analysis_runner.get_analysis_status(session_id)


async def get_analysis_result(session_id: str) -> Optional[AnalysisResponse]:
    """Get completed analysis result"""
    return await analysis_runner.get_analysis_result(session_id) 