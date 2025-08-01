"""
News and Sentiment Analysis Tools for Financial Analysis
Provides tools for gathering news, sentiment, and market context using Serper API
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import json
import re
from functools import wraps

import httpx
from langchain_core.tools import tool

from common.config.setup import Config
from common.cache.manager import RedisManager


# Setup logging
logger = logging.getLogger(__name__)

# Initialize configuration
config = Config()

# Initialize Redis for caching
redis_client = RedisManager(
    # host=config.REDIS_HOST,
    # port=config.REDIS_PORT,
    # password=config.REDIS_PASSWORD
    host="localhost",
    port=6379,
    password="password"
)

# Serper API configuration
SERPER_API_URL = "https://google.serper.dev/search"
SERPER_NEWS_API_URL = "https://google.serper.dev/news"

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


from datetime import date


def _build_after_date(days_back: int) -> str:
    """Return YYYY-MM-DD string *days_back* ago from today (inclusive).

    Used to build `after:` filters in Google-style queries so that Serper only
    returns **fresh** articles.
    """

    days_back = max(days_back, 1)
    target = date.today() - timedelta(days=days_back)
    return target.strftime("%Y-%m-%d")


def cached_news_call(cache_duration_hours: int = 2):
    """
    Decorator for caching news API calls with shorter duration
    
    Args:
        cache_duration_hours: How long to cache news data (default 2 hours)
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"news_api:{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
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
                logger.error(f"Error in cached news call for {func.__name__}: {str(e)}")
                # Try to return stale cache if available
                stale_cache = redis_client.get(f"{cache_key}:stale")
                if stale_cache:
                    logger.warning(f"Returning stale cache for {func.__name__}")
                    return json.loads(stale_cache)
                raise e
                
        return wrapper
    return decorator


async def _make_serper_request(
    url: str, 
    query: str, 
    **kwargs
) -> Dict[str, Any]:
    """
    Make HTTP request to Serper API with proper error handling
    
    Args:
        url: Serper API endpoint URL
        query: Search query
        **kwargs: Additional parameters for the request
    
    Returns:
        API response data
    """
    headers = {
        "X-API-KEY": config.SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    
    payload = {
        "q": query,
        **kwargs
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"HTTP error in Serper request: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in Serper request: {str(e)}")
            raise


@tool
@cached_news_call(cache_duration_hours=4)
async def search_company_news(
    company_name: str,
    ticker: str,
    days_back: int = 7,
    num_results: int = 10,
    **kwargs
) -> Dict[str, Any]:
    """
    Search for recent news about a specific company focusing on Indian financial sources.
    
    Args:
        company_name: Full company name (e.g., "Tata Consultancy Services")
        ticker: Stock ticker (e.g., "TCS.NS")
        days_back: Number of days to look back for news
        num_results: Number of news results to return
    
    Returns:
        Recent news articles with titles, snippets, sources, and dates
        
    Example:
        search_company_news("Reliance Industries", "RELIANCE.NS", 7, 10)
        # Returns recent Reliance news from Indian financial sources
    """
    try:
        # Build a *fresh* query. Do **not** overly restrict to a fixed site list
        # – that caused stale and sparse results. We still bias towards Indian
        # finance portals via `gl='in'` but let Google News ranking surface the
        # most relevant links.

        after_date = _build_after_date(days_back)

        # For Indian context add a light site preference (without strict filter)
        preferred_sites = "(economictimes.indiatimes.com OR moneycontrol.com OR livemint.com OR business-standard.com OR zeebiz.com)"

        query = (
            f'("{company_name}" OR "{ticker}") '
            f'{preferred_sites} '
            f'after:{after_date}'
        )
        
        response = await _make_serper_request(
            SERPER_NEWS_API_URL,
            query,
            num=num_results,
            hl="en",
            gl="in",  # Geographic location: India
            type="news",
        )
        
        # Process and enrich the response
        processed_news = []
        for article in response.get("news", []):
            processed_article = {
                "title": article.get("title", ""),
                "snippet": article.get("snippet", ""),
                "link": article.get("link", ""),
                "source": article.get("source", ""),
                "date": article.get("date", ""),
                "position": article.get("position", 0),
                "relevance_score": _calculate_relevance_score(
                    article.get("title", "") + " " + article.get("snippet", ""),
                    company_name,
                    ticker
                )
            }
            processed_news.append(processed_article)
        
        # Sort by relevance score
        processed_news.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return {
            "company_name": company_name,
            "ticker": ticker,
            "query": query,
            "total_results": len(processed_news),
            "news": processed_news,
            "timestamp": datetime.now().isoformat(),
            "source": "serper_company_news"
        }
        
    except Exception as e:
        logger.error(f"Error searching company news for {company_name}: {str(e)}")
        return {
            "company_name": company_name,
            "ticker": ticker,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "source": "serper_company_news"
        }


@tool
@cached_news_call(cache_duration_hours=6)
async def search_sector_news(
    sector: str,
    days_back: int = 7,
    num_results: int = 15,
    **kwargs
) -> Dict[str, Any]:
    """
    Search for sector-specific news and trends in Indian markets.
    
    Args:
        sector: Sector name (e.g., "IT", "Banking", "Pharma", "Auto")
        days_back: Number of days to look back for news
        num_results: Number of news results to return
    
    Returns:
        Sector-specific news articles and trend analysis
        
    Example:
        search_sector_news("IT", 7, 15)
        # Returns recent IT sector news and trends
    """
    try:
        # Construct sector-specific search query
        sector_keywords = {
            "IT": "information technology software services outsourcing",
            "Banking": "banking financial services NBFC loans",
            "Pharma": "pharmaceutical drugs healthcare medicine",
            "Auto": "automobile automotive cars commercial vehicles",
            "FMCG": "consumer goods retail brands",
            "Energy": "oil gas petroleum refineries power",
            "Metals": "steel iron ore mining metals",
            "Telecom": "telecommunications mobile telecom digital"
        }
        
        keywords = sector_keywords.get(sector, sector)

        after_date = _build_after_date(days_back)

        query = (
            f'India "{sector}" OR "{keywords}" (nifty OR sensex OR stock market) '
            f'after:{after_date}'
        )
        
        response = await _make_serper_request(
            SERPER_NEWS_API_URL,
            query,
            num=num_results,
            hl="en",
            gl="in"
        )
        
        # Process sector news
        processed_news = []
        for article in response.get("news", []):
            processed_article = {
                "title": article.get("title", ""),
                "snippet": article.get("snippet", ""),
                "link": article.get("link", ""),
                "source": article.get("source", ""),
                "date": article.get("date", ""),
                "sentiment": _analyze_sentiment(article.get("title", "") + " " + article.get("snippet", "")),
                "keywords": _extract_keywords(article.get("title", "") + " " + article.get("snippet", ""), sector)
            }
            processed_news.append(processed_article)
        
        return {
            "sector": sector,
            "query": query,
            "total_results": len(processed_news),
            "news": processed_news,
            "sentiment_summary": _summarize_sentiment(processed_news),
            "timestamp": datetime.now().isoformat(),
            "source": "serper_sector_news"
        }
        
    except Exception as e:
        logger.error(f"Error searching sector news for {sector}: {str(e)}")
        return {
            "sector": sector,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "source": "serper_sector_news"
        }


@tool
@cached_news_call(cache_duration_hours=3)
async def search_market_sentiment(
    query_type: str = "general",
    days_back: int = 3,
    num_results: int = 20,
    **kwargs
) -> Dict[str, Any]:
    """
    Analyze overall market sentiment and investor mood in Indian markets.
    
    Args:
        query_type: Type of sentiment analysis ("general", "fii_flows", "policy", "global")
        days_back: Number of days to look back
        num_results: Number of results to analyze
    
    Returns:
        Market sentiment analysis with key themes and investor mood
        
    Example:
        search_market_sentiment("fii_flows", 3, 20)
        # Returns analysis of FII flow impact on market sentiment
    """
    try:
        # Define query templates for different sentiment types
        query_templates = {
            "general": "India stock market sentiment investor mood nifty sensex bullish bearish",
            "fii_flows": "FII DII flows foreign institutional investors India stock market",
            "policy": "RBI policy government budget India market impact economic policy",
            "global": "global markets impact India nifty dollar index crude oil"
        }
        
        base_query = query_templates.get(query_type, query_templates["general"])
        after_date = _build_after_date(days_back)

        query = f'{base_query} after:{after_date}'
        
        response = await _make_serper_request(
            SERPER_NEWS_API_URL,
            query,
            num=num_results,
            hl="en",
            gl="in"
        )
        
        # Analyze sentiment across all articles
        articles = response.get("news", [])
        sentiment_data = []
        
        for article in articles:
            content = article.get("title", "") + " " + article.get("snippet", "")
            sentiment = _analyze_sentiment(content)
            
            sentiment_data.append({
                "title": article.get("title", ""),
                "source": article.get("source", ""),
                "date": article.get("date", ""),
                "sentiment": sentiment,
                "key_themes": _extract_themes(content)
            })
        
        # Aggregate sentiment analysis
        overall_sentiment = _calculate_overall_sentiment(sentiment_data)
        key_themes = _aggregate_themes(sentiment_data)
        
        return {
            "query_type": query_type,
            "total_articles_analyzed": len(sentiment_data),
            "overall_sentiment": overall_sentiment,
            "key_themes": key_themes,
            "articles": sentiment_data[:10],  # Return top 10 for detailed view
            "sentiment_distribution": _get_sentiment_distribution(sentiment_data),
            "timestamp": datetime.now().isoformat(),
            "source": "serper_market_sentiment"
        }
        
    except Exception as e:
        logger.error(f"Error analyzing market sentiment: {str(e)}")
        return {
            "query_type": query_type,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "source": "serper_market_sentiment"
        }


@tool
@cached_news_call(cache_duration_hours=8)
async def search_regulatory_news(
    regulator: str = "all",
    days_back: int = 14,
    num_results: int = 10,
    **kwargs
) -> Dict[str, Any]:
    """
    Search for regulatory updates and policy changes affecting Indian markets.
    
    Args:
        regulator: Specific regulator ("SEBI", "RBI", "all")
        days_back: Number of days to look back
        num_results: Number of results to return
    
    Returns:
        Regulatory news and policy updates with impact analysis
        
    Example:
        search_regulatory_news("SEBI", 14, 10)
        # Returns recent SEBI regulatory updates
    """
    try:
        # Define regulator-specific queries
        regulator_queries = {
            "SEBI": "SEBI securities market regulation policy guidelines",
            "RBI": "RBI monetary policy repo rate banking regulation",
            "all": "SEBI RBI government policy regulation India financial markets"
        }
        
        after_date = _build_after_date(days_back)

        base_query = regulator_queries.get(regulator, regulator_queries["all"])
        query = f'{base_query} after:{after_date}'
        
        response = await _make_serper_request(
            SERPER_NEWS_API_URL,
            query,
            num=num_results,
            hl="en",
            gl="in"
        )
        
        # Process regulatory news
        regulatory_updates = []
        for article in response.get("news", []):
            content = article.get("title", "") + " " + article.get("snippet", "")
            
            update = {
                "title": article.get("title", ""),
                "snippet": article.get("snippet", ""),
                "link": article.get("link", ""),
                "source": article.get("source", ""),
                "date": article.get("date", ""),
                "regulator": _identify_regulator(content),
                "impact_level": _assess_market_impact(content),
                "affected_sectors": _identify_affected_sectors(content)
            }
            regulatory_updates.append(update)
        
        return {
            "regulator": regulator,
            "total_updates": len(regulatory_updates),
            "updates": regulatory_updates,
            "high_impact_updates": [u for u in regulatory_updates if u["impact_level"] == "high"],
            "timestamp": datetime.now().isoformat(),
            "source": "serper_regulatory_news"
        }
        
    except Exception as e:
        logger.error(f"Error searching regulatory news: {str(e)}")
        return {
            "regulator": regulator,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "source": "serper_regulatory_news"
        }


# Helper functions for sentiment and content analysis

def _calculate_relevance_score(content: str, company_name: str, ticker: str) -> float:
    """Calculate relevance score for news articles"""
    content_lower = content.lower()
    company_lower = company_name.lower()
    # Remove NSE/BSE qualifiers (dot or colon) for relevance matching
    ticker_clean = re.sub(r'(:?\.?NS|:?\.?NSE|:?\.?BSE)$', "", ticker, flags=re.I).lower()
    
    score = 0.0
    
    # Direct company name mentions
    if company_lower in content_lower:
        score += 0.4
    
    # Ticker mentions
    if ticker_clean in content_lower:
        score += 0.3
    
    # Financial keywords
    financial_keywords = ["earnings", "revenue", "profit", "loss", "results", "guidance", "outlook"]
    for keyword in financial_keywords:
        if keyword in content_lower:
            score += 0.1
    
    return min(score, 1.0)


def _analyze_sentiment(content: str) -> Dict[str, Any]:
    """Simple sentiment analysis based on keywords"""
    content_lower = content.lower()
    
    positive_words = ["growth", "profit", "gain", "rise", "up", "bullish", "positive", "strong", "outperform", "buy"]
    negative_words = ["loss", "fall", "down", "bearish", "negative", "weak", "underperform", "sell", "decline", "crash"]
    
    positive_count = sum(1 for word in positive_words if word in content_lower)
    negative_count = sum(1 for word in negative_words if word in content_lower)
    
    if positive_count > negative_count:
        sentiment = "positive"
        confidence = min((positive_count - negative_count) / max(positive_count + negative_count, 1), 1.0)
    elif negative_count > positive_count:
        sentiment = "negative"
        confidence = min((negative_count - positive_count) / max(positive_count + negative_count, 1), 1.0)
    else:
        sentiment = "neutral"
        confidence = 0.5
    
    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "positive_signals": positive_count,
        "negative_signals": negative_count
    }


def _extract_keywords(content: str, sector: str) -> List[str]:
    """Extract relevant keywords from content"""
    content_lower = content.lower()
    
    # Sector-specific keywords
    sector_keywords = {
        "IT": ["digital", "technology", "software", "outsourcing", "cloud", "AI"],
        "Banking": ["loans", "deposits", "NPA", "credit", "digital banking"],
        "Pharma": ["drug approval", "clinical trials", "FDA", "patent", "biosimilar"],
        "Auto": ["sales", "production", "EV", "electric vehicle", "export"]
    }
    
    keywords = []
    for keyword in sector_keywords.get(sector, []):
        if keyword in content_lower:
            keywords.append(keyword)
    
    return keywords


def _extract_themes(content: str) -> List[str]:
    """Extract key themes from content"""
    content_lower = content.lower()
    themes = []
    
    theme_patterns = {
        "earnings": ["earnings", "results", "profit", "revenue"],
        "policy": ["policy", "regulation", "government", "budget"],
        "global_impact": ["global", "international", "US", "China", "trade"],
        "market_trends": ["trend", "rally", "correction", "volatility"]
    }
    
    for theme, patterns in theme_patterns.items():
        if any(pattern in content_lower for pattern in patterns):
            themes.append(theme)
    
    return themes


def _summarize_sentiment(news_list: List[Dict]) -> Dict[str, Any]:
    """Summarize sentiment across multiple news articles"""
    if not news_list:
        return {"overall": "neutral", "confidence": 0.0}
    
    sentiments = [article.get("sentiment", {}) for article in news_list if "sentiment" in article]
    
    positive_count = sum(1 for s in sentiments if s.get("sentiment") == "positive")
    negative_count = sum(1 for s in sentiments if s.get("sentiment") == "negative")
    neutral_count = len(sentiments) - positive_count - negative_count
    
    total = len(sentiments)
    if total == 0:
        return {"overall": "neutral", "confidence": 0.0}
    
    positive_ratio = positive_count / total
    negative_ratio = negative_count / total
    
    if positive_ratio > 0.6:
        overall = "positive"
    elif negative_ratio > 0.6:
        overall = "negative"
    else:
        overall = "mixed"
    
    return {
        "overall": overall,
        "positive_ratio": positive_ratio,
        "negative_ratio": negative_ratio,
        "neutral_ratio": neutral_count / total,
        "total_analyzed": total
    }


def _calculate_overall_sentiment(sentiment_data: List[Dict]) -> Dict[str, Any]:
    """Calculate overall market sentiment from multiple data points"""
    if not sentiment_data:
        return {"sentiment": "neutral", "confidence": 0.0}
    
    sentiments = [item["sentiment"] for item in sentiment_data]
    
    positive_count = sum(1 for s in sentiments if s["sentiment"] == "positive")
    negative_count = sum(1 for s in sentiments if s["sentiment"] == "negative")
    total = len(sentiments)
    
    if positive_count > negative_count * 1.5:
        overall_sentiment = "bullish"
    elif negative_count > positive_count * 1.5:
        overall_sentiment = "bearish"
    else:
        overall_sentiment = "mixed"
    
    confidence = abs(positive_count - negative_count) / total if total > 0 else 0.0
    
    return {
        "sentiment": overall_sentiment,
        "confidence": confidence,
        "distribution": {
            "positive": positive_count / total if total > 0 else 0,
            "negative": negative_count / total if total > 0 else 0,
            "neutral": (total - positive_count - negative_count) / total if total > 0 else 0
        }
    }


def _aggregate_themes(sentiment_data: List[Dict]) -> List[Dict[str, Any]]:
    """Aggregate and rank themes across articles"""
    theme_counts = {}
    
    for item in sentiment_data:
        for theme in item.get("key_themes", []):
            theme_counts[theme] = theme_counts.get(theme, 0) + 1
    
    # Sort themes by frequency
    sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
    
    return [{"theme": theme, "frequency": count} for theme, count in sorted_themes[:10]]


def _get_sentiment_distribution(sentiment_data: List[Dict]) -> Dict[str, int]:
    """Get distribution of sentiment across articles"""
    distribution = {"positive": 0, "negative": 0, "neutral": 0}
    
    for item in sentiment_data:
        sentiment = item["sentiment"]["sentiment"]
        if sentiment in distribution:
            distribution[sentiment] += 1
    
    return distribution


def _identify_regulator(content: str) -> str:
    """Identify which regulator is mentioned in the content"""
    content_lower = content.lower()
    
    if "sebi" in content_lower:
        return "SEBI"
    elif "rbi" in content_lower:
        return "RBI"
    elif "government" in content_lower or "ministry" in content_lower:
        return "Government"
    else:
        return "Other"


def _assess_market_impact(content: str) -> str:
    """Assess potential market impact of regulatory news"""
    content_lower = content.lower()
    
    high_impact_words = ["policy change", "new regulation", "ban", "restriction", "major"]
    medium_impact_words = ["guideline", "clarification", "modification"]
    
    if any(word in content_lower for word in high_impact_words):
        return "high"
    elif any(word in content_lower for word in medium_impact_words):
        return "medium"
    else:
        return "low"


def _identify_affected_sectors(content: str) -> List[str]:
    """Identify sectors potentially affected by regulatory changes"""
    content_lower = content.lower()
    sectors = []
    
    sector_indicators = {
        "Banking": ["bank", "banking", "financial institution", "nbfc"],
        "IT": ["technology", "software", "data", "digital"],
        "Pharma": ["pharmaceutical", "drug", "healthcare", "medicine"],
        "Capital Markets": ["stock", "equity", "mutual fund", "investment"]
    }
    
    for sector, indicators in sector_indicators.items():
        if any(indicator in content_lower for indicator in indicators):
            sectors.append(sector)
    
    return sectors


# List of all available news and sentiment tools
NEWS_SENTIMENT_TOOLS = [
    search_company_news,
    search_sector_news,
    search_market_sentiment,
    search_regulatory_news
] 