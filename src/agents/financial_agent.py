"""
Financial Analysis Agent - Main LangGraph orchestrator
Implements the Market Realist personality with comprehensive analysis workflow
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, TypedDict, Annotated
import logging
import json
import re

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

from common.config.setup import Config
from common.schemas.entities import (
    AnalysisRequest, FinancialAnalysisReport, AnalysisStatus,
    MarketContextAnalysis, FundamentalAnalysis, TechnicalAnalysis,
    NewsSentimentAnalysis, RiskAssessment, PriceTargets,
    ConflictResolution, AnalysisMetadata, ConfidenceLevel, DataFreshness
)
from src.tools.twelve_data_tools import TWELVE_DATA_TOOLS
from src.tools.news_sentiment_tools import NEWS_SENTIMENT_TOOLS


# Setup logging
logger = logging.getLogger(__name__)

# Initialize configuration
config = Config()

# Configure Gemini model to return strict JSON objects using the official structured
# output mechanism. See https://ai.google.dev/gemini-api/docs/structured-output
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=config.GEMINI_API_KEY,
    temperature=0.4,  # Low temperature for analytical consistency
    # Instruct Gemini to always respond with a single JSON object so that the
    # `_parse_llm_json` helper can reliably parse the response without needing
    # to strip markdown fences.
    response_format={"type": "json_object"},
)

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

# Helper to safely validate Pydantic models provided by the LLM. If validation
# fails we fall back to a sensible default structure that satisfies the schema
# preventing runtime exceptions while still surfacing the original validation
# error in the logs.
from pydantic import ValidationError


def _safe_model_validate(model_cls, data: dict | None, fallback: dict):
    """Validate *data* against *model_cls* returning a model instance.

    If validation fails, *fallback* is merged with the *data* values whose keys
    exist in the model schema. This guarantees that **all** required fields are
    present so that model instantiation never throws inside the workflow.
    """

    data = data or {}

    try:
        return model_cls.model_validate(data)
    except ValidationError as exc:
        logger.warning(
            "Validation failed for %s – falling back to default. Error: %s",
            model_cls.__name__,
            exc,
        )

    # Merge fallback with any valid keys coming from *data* so that we preserve
    # as much information as possible while still respecting the schema.
    merged: dict = {**fallback}
    for key in model_cls.model_fields:  # type: ignore[attr-defined]
        if key in data and data[key] is not None:
            merged[key] = data[key]

    return model_cls.model_validate(merged)


class AnalysisState(TypedDict):
    """State for the financial analysis workflow"""
    session_id: str
    request: AnalysisRequest
    messages: Annotated[list, add_messages]
    
    # Raw data and analysis components
    company_profile: Optional[Dict[str, Any]]
    market_context: Optional[MarketContextAnalysis]
    fundamental_data: Optional[FundamentalAnalysis]
    technical_data: Optional[TechnicalAnalysis]
    news_sentiment_data: Optional[NewsSentimentAnalysis]
    synthesis: Optional[Dict[str, Any]]

    # Data freshness tracking
    data_freshness: List[Dict[str, Any]]
    
    # Timing
    start_time: Optional[datetime]
    
    # Processing flags
    current_step: str
    progress: float
    errors: List[str]
    
    # Final analysis
    analysis_report: Optional[FinancialAnalysisReport]
    status: str  # "initiated", "in_progress", "completed", "failed"


# Helper to parse LLM JSON output
def _parse_llm_json(llm_output: str) -> Dict[str, Any]:
    """Safely parse LLM JSON output, handling markdown code blocks."""
    try:
        # Strip markdown ```json ... ```
        match = re.search(r"```json\n(.*?)\n```", llm_output, re.DOTALL)
        if match:
            clean_json = match.group(1)
        else:
            clean_json = llm_output
        return json.loads(clean_json)
    except (json.JSONDecodeError, TypeError) as e:
        logger.error(f"Error parsing LLM JSON: {e}\nRaw output: {llm_output}")
        return {"error": "Failed to parse LLM response"}


# Market Realist Personality System Prompt
MARKET_REALIST_PROMPT = """
You are "The Market Realist" - an expert financial analyst for Indian stocks.
Your analysis must be data-driven, balanced, and transparent.
Always return your analysis in a structured JSON format.
"""


def create_financial_analysis_agent() -> StateGraph:
    """
    Create the main financial analysis agent workflow
    
    Returns:
        Configured StateGraph for financial analysis
    """
    
    # Define the workflow graph
    workflow = StateGraph(AnalysisState)
    
    # Add nodes to the workflow
    workflow.add_node("initialize_analysis", initialize_analysis)
    workflow.add_node("gather_market_context", gather_market_context)
    workflow.add_node("analyze_fundamentals", analyze_fundamentals)
    workflow.add_node("analyze_technical", analyze_technical)
    workflow.add_node("analyze_news_sentiment", analyze_news_sentiment)
    workflow.add_node("synthesize_analysis", synthesize_analysis)
    workflow.add_node("generate_report", generate_report)
    
    # Define the workflow edges
    workflow.add_edge(START, "initialize_analysis")
    workflow.add_edge("initialize_analysis", "gather_market_context")
    workflow.add_edge("gather_market_context", "analyze_fundamentals")
    workflow.add_edge("analyze_fundamentals", "analyze_technical")
    workflow.add_edge("analyze_technical", "analyze_news_sentiment")
    workflow.add_edge("analyze_news_sentiment", "synthesize_analysis")
    workflow.add_edge("synthesize_analysis", "generate_report")
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()


async def initialize_analysis(state: AnalysisState) -> AnalysisState:
    """Initialize the analysis workflow"""
    logger.info(f"Initializing analysis for {state['request'].ticker}")
    
    state["session_id"] = str(uuid.uuid4())
    state["current_step"] = "initialization"
    state["progress"] = 5.0
    state["status"] = "in_progress"
    state["errors"] = []
    state["data_freshness"] = []
    state["start_time"] = datetime.now()
    
    # Initialize all state components to None
    state["company_profile"] = None
    state["market_context"] = None
    state["fundamental_data"] = None
    state["technical_data"] = None
    state["news_sentiment_data"] = None
    state["synthesis"] = None
    
    # Add initial system message
    system_message = SystemMessage(content=MARKET_REALIST_PROMPT)
    
    initial_message = HumanMessage(
        content=f"""
        I need a comprehensive financial analysis for {state['request'].ticker}.
        
        Analysis Requirements:
        - Ticker: {state['request'].ticker}
        - Analysis Type: {state['request'].analysis_type}
        - Time Horizon: {state['request'].time_horizon}
        - Include Technical: {state['request'].include_technical}
        - Include Fundamental: {state['request'].include_fundamental}
        - Include News: {state['request'].include_news}
        - Include Sector: {state['request'].include_sector}
        
        Please conduct a thorough analysis following the Market Realist methodology.
        """
    )
    
    state["messages"] = [system_message, initial_message]
    
    return state


async def gather_market_context(state: AnalysisState) -> AnalysisState:
    """Gather market context and macro environment data"""
    logger.info("Gathering market context")
    state["current_step"] = "market_context"
    state["progress"] = 20.0
    
    try:
        from src.tools.twelve_data_tools import get_company_profile
        from src.tools.news_sentiment_tools import search_market_sentiment, search_sector_news
        
        ticker = state["request"].ticker
        
        # Gather context data
        company_profile_task = get_company_profile.ainvoke({"ticker": ticker})
        market_sentiment_task = search_market_sentiment.ainvoke({"query_type": "general", "days_back": 3})
        
        company_profile, market_sentiment = await asyncio.gather(company_profile_task, market_sentiment_task)
        state["company_profile"] = company_profile

        sector = "General"
        if company_profile and company_profile.get("data") and isinstance(company_profile["data"], dict):
            sector = company_profile["data"].get("sector", "General")
        
        sector_news = await search_sector_news.ainvoke({"sector": sector, "days_back": 7})
        
        context_analysis_prompt = f"""
        <thinking>
        I need to analyze the current market context for {ticker}. Let me examine:
        1. Company profile data: {company_profile}
        2. Overall market sentiment: {market_sentiment}
        3. Sector-specific news: {sector_news}
        
        What's the macro environment telling me about this investment?
        </thinking>
        
        Based on the market data provided, analyze the current market context for {ticker}.
        
        Company Profile: {company_profile}
        Market Sentiment: {market_sentiment}
        Sector News: {sector_news}
        
        Provide analysis of:
        1. Current Nifty 50 trend and market environment
        2. Sector rotation patterns affecting this stock
        3. FII/DII flow impact on the sector
        4. Overall market sentiment 
        5. Key macro factors affecting this investment
        
        **Output Format**: Return a single JSON object strictly matching this Pydantic JSON schema:
        {json.dumps(MarketContextAnalysis.model_json_schema(), indent=2)}
        """
        
        context_message = HumanMessage(content=context_analysis_prompt)
        response = await llm.ainvoke([state["messages"][0], context_message])
        
        # Parse and store the structured analysis
        parsed_analysis = _parse_llm_json(response.content)
        if "error" not in parsed_analysis:
            state["market_context"] = MarketContextAnalysis.model_validate(parsed_analysis)
        else:
            state["errors"].append("Failed to parse market context analysis.")
            state["market_context"] = None
        
        # Update data freshness
        if company_profile and "timestamp" in company_profile:
            state["data_freshness"].append({"source": "company_profile", "last_updated": company_profile["timestamp"]})

    except Exception as e:
        logger.error(f"Error gathering market context: {str(e)}")
        state["errors"].append(f"Market context error: {str(e)}")
        state["market_context"] = None

    return state


async def analyze_fundamentals(state: AnalysisState) -> AnalysisState:
    """Analyze fundamental data for the company"""
    logger.info("Analyzing fundamentals")
    state["current_step"] = "fundamental_analysis"
    state["progress"] = 40.0

    if not state["request"].include_fundamental:
        state["fundamental_data"] = {"status": "skipped"}
        return state

    try:
        from src.tools.twelve_data_tools import get_income_statement, get_balance_sheet, get_cash_flow
        
        ticker = state["request"].ticker
        
        # Gather fundamental data
        income_statement_task = get_income_statement.ainvoke({"ticker": ticker, "period": "quarterly", "limit": 4})
        balance_sheet_task = get_balance_sheet.ainvoke({"ticker": ticker, "period": "quarterly", "limit": 4})
        cash_flow_task = get_cash_flow.ainvoke({"ticker": ticker, "period": "quarterly", "limit": 4})
        
        income_statement, balance_sheet, cash_flow = await asyncio.gather(
            income_statement_task, balance_sheet_task, cash_flow_task
        )
        
        fundamental_analysis_prompt = f"""
        Analyze the fundamental health of {ticker} using the provided financial statements.
        Income Statement: {json.dumps(income_statement)}
        Balance Sheet: {json.dumps(balance_sheet)}
        Cash Flow: {json.dumps(cash_flow)}
        Focus on revenue growth, profitability, balance sheet strength, and cash generation.
        
        **Output Format**: Return a single JSON object strictly matching this Pydantic JSON schema:
        {json.dumps(FundamentalAnalysis.model_json_schema(), indent=2)}
        """
        
        fundamental_message = HumanMessage(content=fundamental_analysis_prompt)
        response = await llm.ainvoke([state["messages"][0], fundamental_message])
        
        # Parse and store structured analysis
        parsed_analysis = _parse_llm_json(response.content)
        if "error" not in parsed_analysis:
            state["fundamental_data"] = FundamentalAnalysis.model_validate(parsed_analysis)
        else:
            state["errors"].append("Failed to parse fundamental analysis.")
            state["fundamental_data"] = None
        
        if income_statement and "timestamp" in income_statement:
            state["data_freshness"].append({"source": "income_statement", "last_updated": income_statement["timestamp"]})

    except Exception as e:
        logger.error(f"Error in fundamental analysis: {str(e)}")
        state["errors"].append(f"Fundamental analysis error: {str(e)}")
        state["fundamental_data"] = None

    return state


async def analyze_technical(state: AnalysisState) -> AnalysisState:
    """Analyze technical indicators for the stock"""
    logger.info("Analyzing technical indicators")
    state["current_step"] = "technical_analysis"
    state["progress"] = 60.0

    if not state["request"].include_technical:
        state["technical_data"] = {"status": "skipped"}
        return state

    try:
        from src.tools.twelve_data_tools import get_historical_prices, get_technical_indicators
        
        ticker = state["request"].ticker
        
        # Gather technical data
        historical_prices_task = get_historical_prices.ainvoke({"ticker": ticker, "interval": "1day", "outputsize": 100})
        sma_task = get_technical_indicators.ainvoke({"ticker": ticker, "indicator": "sma", "interval": "1day", "time_period": 50})
        rsi_task = get_technical_indicators.ainvoke({"ticker": ticker, "indicator": "rsi", "interval": "1day", "time_period": 14})
        
        historical_prices, sma, rsi = await asyncio.gather(historical_prices_task, sma_task, rsi_task)
        
        technical_analysis_prompt = f"""
        Analyze the technical outlook for {ticker} based on price action and indicators.
        Historical Prices: {json.dumps(historical_prices)}
        50-day SMA: {json.dumps(sma)}
        14-day RSI: {json.dumps(rsi)}
        Focus on trend, support/resistance, volume, and key indicator levels.

        **Output Format**: Return a single JSON object strictly matching this Pydantic JSON schema:
        {json.dumps(TechnicalAnalysis.model_json_schema(), indent=2)}
        """
        
        technical_message = HumanMessage(content=technical_analysis_prompt)
        response = await llm.ainvoke([state["messages"][0], technical_message])
        
        # Parse and store structured analysis
        parsed_analysis = _parse_llm_json(response.content)
        if "error" not in parsed_analysis:
            state["technical_data"] = TechnicalAnalysis.model_validate(parsed_analysis)
        else:
            state["errors"].append("Failed to parse technical analysis.")
            state["technical_data"] = None
        
        if historical_prices and "timestamp" in historical_prices:
            state["data_freshness"].append({"source": "historical_prices", "last_updated": historical_prices["timestamp"]})

    except Exception as e:
        logger.error(f"Error in technical analysis: {str(e)}")
        state["errors"].append(f"Technical analysis error: {str(e)}")
        state["technical_data"] = None

    return state


async def analyze_news_sentiment(state: AnalysisState) -> AnalysisState:
    """Analyze news and market sentiment"""
    logger.info("Analyzing news and sentiment")
    state["current_step"] = "news_sentiment"
    state["progress"] = 80.0

    if not state["request"].include_news:
        state["news_sentiment_data"] = {"status": "skipped"}
        return state

    try:
        from src.tools.news_sentiment_tools import search_company_news
        
        company_name = state["request"].ticker
        if state.get("company_profile") and isinstance(state["company_profile"], dict) and state["company_profile"].get("data"):
            if isinstance(state["company_profile"]["data"], dict):
                company_name = state["company_profile"]["data"].get("name", state["request"].ticker)

        # Gather news data
        company_news = await search_company_news.ainvoke({"company_name": company_name, "ticker": state["request"].ticker, "days_back": 7})
        
        news_analysis_prompt = f"""
        Analyze the recent news and sentiment for {company_name}.
        Recent News: {json.dumps(company_news)}
        Focus on the impact of recent news, management commentary, and overall sentiment.
        Identify any upcoming catalysts.

        **Output Format**: Return a single JSON object strictly matching this Pydantic JSON schema:
        {json.dumps(NewsSentimentAnalysis.model_json_schema(), indent=2)}
        """
        
        news_message = HumanMessage(content=news_analysis_prompt)
        response = await llm.ainvoke([state["messages"][0], news_message])
        
        # Parse and store structured analysis
        parsed_analysis = _parse_llm_json(response.content)
        if "error" not in parsed_analysis:
            state["news_sentiment_data"] = NewsSentimentAnalysis.model_validate(parsed_analysis)
        else:
            state["errors"].append("Failed to parse news/sentiment analysis.")
            state["news_sentiment_data"] = None
        
        if company_news and "timestamp" in company_news:
            state["data_freshness"].append({"source": "company_news", "last_updated": company_news["timestamp"]})

    except Exception as e:
        logger.error(f"Error in news/sentiment analysis: {str(e)}")
        state["errors"].append(f"News/sentiment analysis error: {str(e)}")
        state["news_sentiment_data"] = None

    return state


async def synthesize_analysis(state: AnalysisState) -> AnalysisState:
    """Synthesize all analysis components and resolve conflicts"""
    logger.info("Synthesizing analysis")
    
    state["current_step"] = "synthesis"
    state["progress"] = 90.0
    
    try:
        # Use LLM to synthesize all findings
        # Gemini structured output – embed **full JSON schemas** so the model can
        # return data that passes Pydantic validation without additional
        # post-processing.  This significantly reduces runtime failures coming
        # from mismatched keys / missing fields.

        synthesis_prompt = f"""
        <thinking>
        Now I need to synthesize all my analysis components for {state['request'].ticker}:
        
        1. Market Context: {state.get('market_context').model_dump_json(indent=2) if state.get('market_context') else 'No data'}
        2. Fundamental Analysis: {state.get('fundamental_data').model_dump_json(indent=2) if state.get('fundamental_data') else 'No data'}
        3. Technical Analysis: {state.get('technical_data').model_dump_json(indent=2) if state.get('technical_data') else 'No data'}
        4. News/Sentiment: {state.get('news_sentiment_data').model_dump_json(indent=2) if state.get('news_sentiment_data') else 'No data'}
        
        What are the key conflicts between these perspectives?
        How do I resolve them using confidence-weighted analysis?
        What's my overall investment recommendation?
        </thinking>
        
        You are the Market Realist. Synthesize your complete analysis for {state['request'].ticker}.
        
        PREVIOUS ANALYSIS COMPONENTS:
        
        Market Context Analysis:
        {state.get('market_context').model_dump_json(indent=2) if state.get('market_context') else 'No market context data available'}
        
        Fundamental Analysis:
        {state.get('fundamental_data').model_dump_json(indent=2) if state.get('fundamental_data') else 'No fundamental data available'}
        
        Technical Analysis:
        {state.get('technical_data').model_dump_json(indent=2) if state.get('technical_data') else 'No technical data available'}
        
        News & Sentiment Analysis:
        {state.get('news_sentiment_data').model_dump_json(indent=2) if state.get('news_sentiment_data') else 'No news/sentiment data available'}
        
        **Output Format**: Return **one** JSON object that strictly matches the
        following composite schema. Make sure every required field is present –
        omit *none*.

        ```json
        {{
          "executive_summary": "string",
          "risk_assessment": {json.dumps(RiskAssessment.model_json_schema(), indent=2)},
          "price_targets": {json.dumps(PriceTargets.model_json_schema(), indent=2)},
          "conflict_resolution": {json.dumps(ConflictResolution.model_json_schema(), indent=2)},
          "confidence_scores": {{"component_name": 7.5}},
          "investment_recommendation": "string",
          "key_catalysts": ["string"]
        }}
        ```

        Base your synthesis strictly on the provided analysis components. **Do
        not** wrap the JSON in markdown fences – the response must be a raw
        JSON object because the client has `response_format: json_object` set.
        """
        
        synthesis_message = HumanMessage(content=synthesis_prompt)
        response = await llm.ainvoke([state["messages"][0], synthesis_message])
        
        # Parse synthesis and store it in state
        state["synthesis"] = _parse_llm_json(response.content)
        
    except Exception as e:
        logger.error(f"Error in synthesis: {str(e)}")
        state["errors"].append(f"Synthesis error: {str(e)}")
    
    return state


async def generate_report(state: AnalysisState) -> AnalysisState:
    """Generate the final analysis report"""
    logger.info("Generating final report")
    
    state["current_step"] = "report_generation"
    
    synthesis_data = state.get("synthesis") or {}

    # Get company name and current price safely
    company_name = state["request"].ticker
    if state.get("company_profile") and isinstance(state["company_profile"], dict) and state["company_profile"].get("data"):
        if isinstance(state["company_profile"]["data"], dict):
            company_name = state["company_profile"]["data"].get("name", state["request"].ticker)

    current_price = 0.0 # Placeholder
    
    # Generate metadata
    end_time = datetime.now()
    analysis_duration = (end_time - state["start_time"]).total_seconds() if state.get("start_time") else 0.0
    
    data_sources = list(set(item.get("source") for item in state.get("data_freshness", [])))
    
    freshness_list = []
    for item in state.get("data_freshness", []):
        if "source" in item and "last_updated" in item:
            # Convert common timestamp string → datetime if necessary to satisfy
            # Pydantic validation.
            last_updated_val = item["last_updated"]
            if isinstance(last_updated_val, str):
                try:
                    last_updated_val = datetime.fromisoformat(last_updated_val.replace("Z", ""))
                except ValueError:
                    last_updated_val = datetime.utcnow()

            freshness_list.append(
                DataFreshness(
                    source=item["source"],
                    last_updated=last_updated_val,
                    quality_score=item.get("quality_score", 0.9),
                )
            )

    metadata = AnalysisMetadata(
        analysis_duration=analysis_duration,
        data_sources_used=data_sources,
        data_freshness=freshness_list,
        analysis_timestamp=end_time,
        agent_version="1.3.0"
    )
    
    # Build the final report from state
    analysis_report = FinancialAnalysisReport(
        session_id=state["session_id"],
        ticker=state["request"].ticker,
        company_name=company_name,
        
        executive_summary=synthesis_data.get("executive_summary", "Synthesis failed."),
        
        market_context=state.get("market_context") or MarketContextAnalysis.model_validate({
            "nifty_trend": "Data unavailable", "sector_rotation": "Data unavailable", 
            "fii_dii_flows": "Data unavailable", "market_sentiment": "Data unavailable",
            "macro_factors": [], "confidence": {"score": 0, "reasoning": "Data unavailable"}
        }),
        
        fundamental_analysis=state.get("fundamental_data") or FundamentalAnalysis.model_validate({
             "revenue_growth": "Data unavailable", "profitability": "Data unavailable", 
             "balance_sheet_strength": "Data unavailable", "cash_generation": "Data unavailable",
             "valuation_metrics": {}, "competitive_position": "Data unavailable",
             "management_quality": "Data unavailable", "confidence": {"score": 0, "reasoning": "Data unavailable"}
        }),
        
        technical_analysis=state.get("technical_data") or TechnicalAnalysis.model_validate({
            "trend_analysis": "Data unavailable", "support_resistance": "Data unavailable",
            "volume_analysis": "Data unavailable", "technical_indicators": {},
            "chart_patterns": "Data unavailable", "relative_strength": "Data unavailable",
            "confidence": {"score": 0, "reasoning": "Data unavailable"}
        }),
        
        news_sentiment=state.get("news_sentiment_data") or NewsSentimentAnalysis.model_validate({
            "recent_news_impact": "Data unavailable", "management_commentary": "Data unavailable",
            "analyst_sentiment": "Data unavailable", "social_sentiment": "Data unavailable",
            "upcoming_catalysts": [], "news_sources": [], "confidence": {"score": 0, "reasoning": "Data unavailable"}
        }),
        
        risk_assessment=_safe_model_validate(
            RiskAssessment,
            synthesis_data.get("risk_assessment"),
            {
                "company_specific_risks": [],
                "sector_risks": [],
                "macro_risks": [],
                "technical_risks": [],
                "risk_probability": {},
                "mitigation_strategies": [],
            },
        ),

        price_targets=_safe_model_validate(
            PriceTargets,
            synthesis_data.get("price_targets"),
            {
                "bull_case_target": 0,
                "base_case_target": 0,
                "bear_case_target": 0,
                "time_horizon": "N/A",
                "target_reasoning": "Data unavailable",
                "current_price": current_price,
            },
        ),

        conflict_resolution=_safe_model_validate(
            ConflictResolution,
            synthesis_data.get("conflict_resolution"),
            {
                "conflicting_signals": [],
                "bull_case_summary": "Data unavailable",
                "bear_case_summary": "Data unavailable",
                "resolution_approach": "Data unavailable",
                "confidence_weighted_conclusion": "Data unavailable",
            },
        ),
        
        confidence_scores=synthesis_data.get("confidence_scores", {}),
        investment_recommendation=synthesis_data.get("investment_recommendation", "Recommendation unavailable."),
        key_catalysts=synthesis_data.get("key_catalysts", []),
        
        metadata=metadata
    )
    
    state["analysis_report"] = analysis_report
    state["status"] = "completed"
    state["progress"] = 100.0
    
    return state


# Main agent instance
financial_agent = create_financial_analysis_agent()


async def run_financial_analysis(request: AnalysisRequest) -> AnalysisState:
    """
    Run the complete financial analysis workflow
    
    Args:
        request: Analysis request with ticker and parameters
        
    Returns:
        Final analysis state with report
    """
    initial_state = AnalysisState(
        session_id="",
        request=request,
        messages=[],
        company_profile=None,
        market_context=None,
        fundamental_data=None,
        technical_data=None,
        news_sentiment_data=None,
        synthesis=None,
        current_step="",
        progress=0.0,
        errors=[],
        analysis_report=None,
        status="initiated"
    )
    
    try:
        # Run the workflow
        final_state = await financial_agent.ainvoke(initial_state)
        return final_state
    except Exception as e:
        logger.error(f"Error running financial analysis: {str(e)}")
        initial_state["status"] = "failed"
        initial_state["errors"].append(str(e))
        return initial_state 