from dataclasses import asdict, dataclass
from typing import List, Optional, Dict, Literal
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


@dataclass
class SampleRequestPayload:
    sample_field: str


# Financial Analysis Schemas

class AnalysisTimeHorizon(str, Enum):
    """Time horizons for investment analysis"""
    SHORT_TERM = "short_term"  # 1-3 months
    MEDIUM_TERM = "medium_term"  # 6-12 months
    LONG_TERM = "long_term"  # 2+ years


class AnalysisType(str, Enum):
    """Types of analysis to perform"""
    COMPREHENSIVE = "comprehensive"  # All dimensions
    QUICK = "quick"  # Summary analysis
    TECHNICAL_ONLY = "technical_only"  # Technical analysis focus
    FUNDAMENTAL_ONLY = "fundamental_only"  # Fundamental analysis focus


class ConfidenceLevel(BaseModel):
    """Confidence scoring for analysis components"""
    score: float = Field(..., ge=1.0, le=10.0, description="Confidence score from 1-10")
    reasoning: str = Field(..., description="Explanation for confidence level")


class DataFreshness(BaseModel):
    """Tracking data freshness for quality assurance"""
    source: str = Field(..., description="Data source name")
    last_updated: datetime = Field(..., description="When data was last refreshed")
    quality_score: float = Field(..., ge=0.0, le=1.0, description="Data quality score 0-1")


class AnalysisRequest(BaseModel):
    """Request schema for financial analysis"""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., RELIANCE.NS)")
    analysis_type: AnalysisType = Field(default=AnalysisType.COMPREHENSIVE)
    time_horizon: AnalysisTimeHorizon = Field(default=AnalysisTimeHorizon.MEDIUM_TERM)
    include_technical: bool = Field(default=True, description="Include technical analysis")
    include_fundamental: bool = Field(default=True, description="Include fundamental analysis")
    include_news: bool = Field(default=True, description="Include news and sentiment")
    include_sector: bool = Field(default=True, description="Include sector analysis")


class MarketContextAnalysis(BaseModel):
    """Market context and macro environment analysis"""
    nifty_trend: str = Field(..., description="Current Nifty 50 trend assessment")
    sector_rotation: str = Field(..., description="Current sector rotation patterns")
    fii_dii_flows: str = Field(..., description="FII/DII flow analysis")
    market_sentiment: str = Field(..., description="Overall market sentiment")
    macro_factors: List[str] = Field(..., description="Key macro factors affecting market")
    confidence: ConfidenceLevel


class FundamentalAnalysis(BaseModel):
    """Fundamental analysis results"""
    revenue_growth: str = Field(..., description="Revenue growth analysis")
    profitability: str = Field(..., description="Profitability and margin analysis")
    balance_sheet_strength: str = Field(..., description="Balance sheet health assessment")
    cash_generation: str = Field(..., description="Cash flow generation analysis")
    valuation_metrics: Dict[str, str] = Field(..., description="Key valuation ratios and analysis")
    competitive_position: str = Field(..., description="Competitive moat and positioning")
    management_quality: str = Field(..., description="Management and capital allocation assessment")
    confidence: ConfidenceLevel


class TechnicalAnalysis(BaseModel):
    """Technical analysis results"""
    trend_analysis: str = Field(..., description="Price trend and momentum analysis")
    support_resistance: str = Field(..., description="Key support and resistance levels")
    volume_analysis: str = Field(..., description="Volume patterns and confirmation")
    technical_indicators: Dict[str, str] = Field(..., description="Key technical indicators analysis")
    chart_patterns: str = Field(..., description="Identified chart patterns")
    relative_strength: str = Field(..., description="Relative strength vs sector/market")
    confidence: ConfidenceLevel


class NewsSentimentAnalysis(BaseModel):
    """News and sentiment analysis"""
    recent_news_impact: str = Field(..., description="Analysis of recent news impact")
    management_commentary: str = Field(..., description="Latest management commentary analysis")
    analyst_sentiment: str = Field(..., description="Analyst recommendations and sentiment")
    social_sentiment: str = Field(..., description="Social media and retail sentiment")
    upcoming_catalysts: List[str] = Field(..., description="Upcoming events and catalysts")
    news_sources: List[str] = Field(..., description="Key news sources analyzed")
    confidence: ConfidenceLevel


class RiskAssessment(BaseModel):
    """Risk analysis and assessment"""
    company_specific_risks: List[str] = Field(..., description="Company-specific risk factors")
    sector_risks: List[str] = Field(..., description="Sector and industry risks")
    macro_risks: List[str] = Field(..., description="Macroeconomic risks")
    technical_risks: List[str] = Field(..., description="Technical analysis risks")
    risk_probability: Dict[str, float] = Field(..., description="Risk probability assessments")
    mitigation_strategies: List[str] = Field(..., description="Risk mitigation suggestions")


class PriceTargets(BaseModel):
    """Price target analysis"""
    bull_case_target: float = Field(..., description="Bull case price target")
    base_case_target: float = Field(..., description="Base case price target")
    bear_case_target: float = Field(..., description="Bear case price target")
    time_horizon: str = Field(..., description="Time horizon for targets")
    target_reasoning: str = Field(..., description="Reasoning behind price targets")
    current_price: float = Field(..., description="Current market price")


class ConflictResolution(BaseModel):
    """Analysis of conflicting signals and resolution"""
    conflicting_signals: List[str] = Field(..., description="Identified conflicting signals")
    bull_case_summary: str = Field(..., description="Summary of bullish arguments")
    bear_case_summary: str = Field(..., description="Summary of bearish arguments")
    resolution_approach: str = Field(..., description="How conflicts were resolved")
    confidence_weighted_conclusion: str = Field(..., description="Final confidence-weighted conclusion")


class AnalysisMetadata(BaseModel):
    """Metadata about the analysis process"""
    analysis_duration: float = Field(..., description="Analysis duration in seconds")
    data_sources_used: List[str] = Field(..., description="Data sources consulted")
    data_freshness: Optional[List[DataFreshness]] = Field(None, description="Data freshness tracking")
    analysis_timestamp: datetime = Field(..., description="When analysis was performed")
    agent_version: str = Field(default="1.0", description="Analysis agent version")


class FinancialAnalysisReport(BaseModel):
    """Complete financial analysis report"""
    session_id: str = Field(..., description="Unique session identifier")
    ticker: str = Field(..., description="Analyzed ticker symbol")
    company_name: Optional[str] = Field(None, description="Company name")
    
    # Analysis sections
    executive_summary: str = Field(..., description="Executive summary and key takeaways")
    market_context: MarketContextAnalysis
    fundamental_analysis: FundamentalAnalysis
    technical_analysis: TechnicalAnalysis
    news_sentiment: NewsSentimentAnalysis
    risk_assessment: RiskAssessment
    price_targets: PriceTargets
    
    # Meta-analysis
    conflict_resolution: ConflictResolution
    confidence_scores: Dict[str, float] = Field(..., description="Overall confidence by analysis type")
    investment_recommendation: str = Field(..., description="Final investment recommendation")
    key_catalysts: List[str] = Field(..., description="Key catalysts to monitor")
    
    # Metadata
    metadata: AnalysisMetadata


class AnalysisStatus(BaseModel):
    """Status tracking for analysis progress"""
    session_id: str
    status: Literal["initiated", "in_progress", "completed", "failed"]
    progress_percentage: float = Field(ge=0.0, le=100.0)
    current_step: str
    estimated_completion_time: Optional[datetime] = None
    error_message: Optional[str] = None


class AnalysisResponse(BaseModel):
    """API response wrapper for analysis"""
    session_id: str
    status: Literal["initiated", "in_progress", "completed", "failed"]
    analysis: Optional[FinancialAnalysisReport] = None
    metadata: Optional[AnalysisMetadata] = None
    error_message: Optional[str] = None
