"""
Twelve Data API Tools for Financial Analysis
Provides tools for fetching fundamental and technical data with intelligent caching
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import logging
from functools import wraps

from langchain_core.tools import tool
from twelvedata import TDClient
import json

from common.config.setup import Config
from common.cache.manager import RedisManager


# Setup logging
logger = logging.getLogger(__name__)

# Initialize Twelve Data client
config = Config()
td_client = TDClient(apikey=config.TWELVE_DATA_API_KEY)

# ---------------------------------------------------------------------------
# Symbol conversion helpers
# ---------------------------------------------------------------------------


def _convert_to_td_symbol(ticker: str) -> str:
    """Convert human-readable ticker (e.g. `TCS.NS`) to TwelveData symbol.

    The official MIC code for National Stock Exchange of India is **XNSE**, but
    Twelve Data also accepts the shorthand `:NSE`. Using the latter keeps URLs
    compact while still guaranteeing the lookup hits the Indian listing - this
    avoids collisions such as *TCS* (Tecsys Inc.) on the Toronto exchange.

    Supported mappings:
    * `.NS`  → `:<NSE>`
    * `.BO`  → `:<BSE>`
    * Already-qualified tickers (containing `:`) are passed through unchanged.
    """

    if ":" in ticker:
        return ticker  # Already qualified (e.g. INFY:NSE)

    if ticker.endswith(".NS"):
        return f"{ticker[:-3]}:NSE"

    if ticker.endswith(".BO"):
        return f"{ticker[:-3]}:BSE"

    # Fall back to original - some indices/ETFs may not follow the pattern.
    return ticker


# Initialize Redis for caching
redis_client = RedisManager(
    # host=config.REDIS_HOST,
    # port=config.REDIS_PORT,
    # password=config.REDIS_PASSWORD
    host="localhost",
    port=6379,
    password=""
)


def cached_api_call(cache_duration_hours: int = 24):
    """
    Decorator for caching Twelve Data API calls based on data volatility
    
    Args:
        cache_duration_hours: How long to cache the data (varies by data type)
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"td_api:{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            try:
                # Try to get cached result
                cached_result = redis_client.get(cache_key)
                if cached_result:
                    logger.info(f"Cache hit for {func.__name__}")
                    return json.loads(cached_result)
                
                # If not cached, call the API
                logger.info(f"Cache miss for {func.__name__}, calling API")
                result = await func(*args, **kwargs)
                
                # Cache the result
                redis_client.setex(
                    cache_key, 
                    cache_duration_hours * 3600, 
                    json.dumps(result, default=str)
                )
                
                return result
                
            except Exception as e:
                logger.error(f"Error in cached API call for {func.__name__}: {str(e)}")
                # Try to return stale cache if available
                stale_cache = redis_client.get(f"{cache_key}:stale")
                if stale_cache:
                    logger.warning(f"Returning stale cache for {func.__name__}")
                    return json.loads(stale_cache)
                raise e
                
        return wrapper
    return decorator


@tool
@cached_api_call(cache_duration_hours=24 * 7)  # Cache fundamentals for 7 days
async def get_company_profile(ticker: str, **kwargs) -> Dict[str, Any]:
    """
    Get comprehensive company profile information for Indian stock analysis.
    
    Args:
        ticker: NSE ticker symbol (e.g., "RELIANCE.NS", "TCS.NS")
    
    Returns:
        Dictionary containing company profile data including business description,
        sector, industry, market cap, and other key company information
        
    Example:
        get_company_profile("TCS.NS")
        # Returns detailed TCS company profile
    """
    try:
        # Convert NSE/BSE tickers to explicit MIC-qualified symbol to prevent
        # cross-exchange collisions (e.g. TCS - Tecsys vs Tata Consultancy).
        sanitized_ticker = _convert_to_td_symbol(ticker)
        response = await asyncio.to_thread(td_client.get_profile(symbol=sanitized_ticker).as_json)
        return {
            "ticker": ticker,
            "data": response,
            "timestamp": datetime.now().isoformat(),
            "source": "twelve_data_profile"
        }
    except Exception as e:
        logger.error(f"Error fetching company profile for {ticker}: {str(e)}")
        return {
            "ticker": ticker,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "source": "twelve_data_profile"
        }


@tool
@cached_api_call(cache_duration_hours=24 * 7)  # Cache income statements for 7 days
async def get_income_statement(
    ticker: str, 
    period: str = "quarterly", 
    limit: int = 4,
    **kwargs
) -> Dict[str, Any]:
    """
    Retrieve income statement data for fundamental analysis.
    
    Args:
        ticker: NSE ticker symbol (e.g., "RELIANCE.NS", "TCS.NS")
        period: "quarterly" or "annual" data
        limit: Number of periods to retrieve (max 8)
    
    Returns:
        Structured income statement data with revenue, expenses, and profitability metrics
        
    Example:
        get_income_statement("TCS.NS", "quarterly", 4)
        # Returns last 4 quarters of TCS income statements
    """
    try:
        sanitized_ticker = _convert_to_td_symbol(ticker)
        response = await asyncio.to_thread(
            td_client.get_income_statement(symbol=sanitized_ticker, period=period).as_json
        )
        
        return {
            "ticker": ticker,
            "period": period,
            "data": response,
            "timestamp": datetime.now().isoformat(),
            "source": "twelve_data_income_statement"
        }
    except Exception as e:
        logger.error(f"Error fetching income statement for {ticker}: {str(e)}")
        return {
            "ticker": ticker,
            "period": period,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "source": "twelve_data_income_statement"
        }


@tool
@cached_api_call(cache_duration_hours=24 * 7)  # Cache balance sheets for 7 days
async def get_balance_sheet(
    ticker: str, 
    period: str = "quarterly", 
    limit: int = 4,
    **kwargs
) -> Dict[str, Any]:
    """
    Retrieve balance sheet data for financial health analysis.
    
    Args:
        ticker: NSE ticker symbol (e.g., "RELIANCE.NS", "TCS.NS")
        period: "quarterly" or "annual" data
        limit: Number of periods to retrieve (max 8)
    
    Returns:
        Structured balance sheet data with assets, liabilities, and equity information
        
    Example:
        get_balance_sheet("RELIANCE.NS", "quarterly", 4)
        # Returns last 4 quarters of Reliance balance sheets
    """
    try:
        sanitized_ticker = _convert_to_td_symbol(ticker)
        response = await asyncio.to_thread(
            td_client.get_balance_sheet(symbol=sanitized_ticker, period=period).as_json
        )
        
        return {
            "ticker": ticker,
            "period": period,
            "data": response,
            "timestamp": datetime.now().isoformat(),
            "source": "twelve_data_balance_sheet"
        }
    except Exception as e:
        logger.error(f"Error fetching balance sheet for {ticker}: {str(e)}")
        return {
            "ticker": ticker,
            "period": period,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "source": "twelve_data_balance_sheet"
        }


@tool
@cached_api_call(cache_duration_hours=24 * 7)  # Cache cash flow for 7 days
async def get_cash_flow(
    ticker: str, 
    period: str = "quarterly", 
    limit: int = 4,
    **kwargs
) -> Dict[str, Any]:
    """
    Retrieve cash flow statement data for cash generation analysis.
    
    Args:
        ticker: NSE ticker symbol (e.g., "RELIANCE.NS", "TCS.NS")
        period: "quarterly" or "annual" data
        limit: Number of periods to retrieve (max 8)
    
    Returns:
        Structured cash flow data with operating, investing, and financing activities
        
    Example:
        get_cash_flow("TCS.NS", "quarterly", 4)
        # Returns last 4 quarters of TCS cash flow statements
    """
    try:
        sanitized_ticker = _convert_to_td_symbol(ticker)
        response = await asyncio.to_thread(
            td_client.get_cash_flow(symbol=sanitized_ticker, period=period).as_json
        )
        
        return {
            "ticker": ticker,
            "period": period,
            "data": response,
            "timestamp": datetime.now().isoformat(),
            "source": "twelve_data_cash_flow"
        }
    except Exception as e:
        logger.error(f"Error fetching cash flow for {ticker}: {str(e)}")
        return {
            "ticker": ticker,
            "period": period,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "source": "twelve_data_cash_flow"
        }


@tool
@cached_api_call(cache_duration_hours=24)  # Cache daily prices for 1 day
async def get_historical_prices(
    ticker: str,
    interval: str = "1day",
    outputsize: int = 100,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Retrieve historical price data (OHLCV).
    
    Args:
        ticker: NSE ticker symbol (e.g., "RELIANCE.NS")
        interval: Time interval (e.g., "1day", "1h")
        outputsize: Number of data points to return
        start_date: Start date for historical data (YYYY-MM-DD)
        end_date: End date for historical data (YYYY-MM-DD)
        
    Returns:
        Historical time series data
    """
    try:
        sanitized_ticker = _convert_to_td_symbol(ticker)
        
        # Build query parameters
        params = {
            "symbol": sanitized_ticker,
            "interval": interval,
            "outputsize": outputsize
        }
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
            
        # Correctly use the time_series builder
        ts = td_client.time_series(**params)
        response = await asyncio.to_thread(ts.as_json)
        
        return {
            "ticker": ticker,
            "params": params,
            "data": response,
            "timestamp": datetime.now().isoformat(),
            "source": "twelve_data_historical_prices"
        }
    except Exception as e:
        logger.error(f"Error fetching historical prices for {ticker}: {str(e)}")
        return {
            "ticker": ticker,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "source": "twelve_data_historical_prices"
        }


@tool
@cached_api_call(cache_duration_hours=1)  # Cache real-time quotes for 1 hour
async def get_real_time_quote(ticker: str, **kwargs) -> Dict[str, Any]:
    """
    Get real-time stock quote with current price, change, and volume.
    
    Args:
        ticker: NSE ticker symbol (e.g., "RELIANCE.NS", "TCS.NS")
    
    Returns:
        Real-time quote data including current price, volume, and market status
        
    Example:
        get_real_time_quote("TCS.NS")
        # Returns current TCS quote data
    """
    try:
        sanitized_ticker = _convert_to_td_symbol(ticker)
        response = await asyncio.to_thread(td_client.quote(symbol=sanitized_ticker).as_json)
        
        return {
            "ticker": ticker,
            "data": response,
            "timestamp": datetime.now().isoformat(),
            "source": "twelve_data_quote"
        }
    except Exception as e:
        logger.error(f"Error fetching real-time quote for {ticker}: {str(e)}")
        return {
            "ticker": ticker,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "source": "twelve_data_quote"
        }


@tool
@cached_api_call(cache_duration_hours=24)  # Cache statistics for 1 day
async def get_company_statistics(ticker: str, **kwargs) -> Dict[str, Any]:
    """
    Get key company statistics including valuation, dividend, and performance metrics.
    
    Args:
        ticker: NSE ticker symbol (e.g., "RELIANCE.NS", "TCS.NS")
    
    Returns:
        Key statistics including P/E ratio, market cap, EPS, dividend yield, etc.
        
    Example:
        get_company_statistics("RELIANCE.NS")
        # Returns Reliance key financial statistics
    """
    try:
        sanitized_ticker = _convert_to_td_symbol(ticker)
        response = await asyncio.to_thread(td_client.get_statistics(symbol=sanitized_ticker).as_json)
        
        return {
            "ticker": ticker,
            "data": response,
            "timestamp": datetime.now().isoformat(),
            "source": "twelve_data_statistics"
        }
    except Exception as e:
        logger.error(f"Error fetching statistics for {ticker}: {str(e)}")
        return {
            "ticker": ticker,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "source": "twelve_data_statistics"
        }


@tool
@cached_api_call(cache_duration_hours=24)  # Cache technical indicators for 1 day
async def get_technical_indicators(
    ticker: str,
    indicator: str,
    interval: str = "1day",
    time_period: int = 20,
    outputsize: int = 100,
    **kwargs
) -> Dict[str, Any]:
    """
    Retrieve technical indicator data for a stock.
    Supports: sma, ema, rsi, macd, bbands
    
    Args:
        ticker: NSE ticker symbol (e.g., "RELIANCE.NS")
        indicator: Technical indicator to calculate (e.g., "sma")
        interval: Time interval for calculation
        time_period: Lookback period for the indicator
        outputsize: Number of data points
        
    Returns:
        Technical indicator data series
    """
    try:
        sanitized_ticker = _convert_to_td_symbol(ticker)
        
        # Create a time series object first
        ts = td_client.time_series(
            symbol=sanitized_ticker,
            interval=interval,
            outputsize=outputsize
        )
        
        # Apply the technical indicator using the correct builder pattern
        indicator_map = {
            "sma": ts.with_sma(time_period=time_period),
            "ema": ts.with_ema(time_period=time_period),
            "rsi": ts.with_rsi(time_period=time_period),
            "macd": ts.with_macd(),
            "bbands": ts.with_bbands(time_period=time_period)
        }
        
        if indicator.lower() not in indicator_map:
            raise ValueError(f"Unsupported indicator: {indicator}")
            
        # Execute the request
        indicator_builder = indicator_map[indicator.lower()]
        response = await asyncio.to_thread(indicator_builder.as_json)
        
        return {
            "ticker": ticker,
            "indicator": indicator,
            "data": response,
            "timestamp": datetime.now().isoformat(),
            "source": f"twelve_data_{indicator}"
        }
    except Exception as e:
        logger.error(f"Error fetching {indicator} for {ticker}: {str(e)}")
        return {
            "ticker": ticker,
            "indicator": indicator,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "source": f"twelve_data_{indicator}"
        }


@tool
async def get_api_usage(**kwargs) -> Dict[str, Any]:
    """Get current Twelve Data API usage"""
    try:
        response = await asyncio.to_thread(td_client.api_usage().as_json)
        
        return {
            "data": response,
            "timestamp": datetime.now().isoformat(),
            "source": "twelve_data_api_usage"
        }
    except Exception as e:
        logger.error(f"Error fetching API usage: {str(e)}")
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "source": "twelve_data_api_usage"
        }


# List of all available tools for easy import
TWELVE_DATA_TOOLS = [
    get_company_profile,
    get_income_statement,
    get_balance_sheet,
    get_cash_flow,
    get_historical_prices,
    get_real_time_quote,
    get_company_statistics,
    get_technical_indicators,
    get_api_usage
] 