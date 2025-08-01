import streamlit as st
import requests
import time
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

# --- Configuration ---
API_BASE_URL = "http://127.0.0.1:8000/api/v1/analysis"
POLL_INTERVAL_SECONDS = 5
MAX_WAIT_SECONDS = 600

# --- Pydantic Models (for type hinting and reference) ---
# These are simplified versions for the frontend.
# In a larger app, you might share code between backend and frontend.

@dataclass
class AnalysisRequest:
    ticker: str
    analysis_type: str = "comprehensive"
    time_horizon: str = "medium_term"
    include_technical: bool = True
    include_fundamental: bool = True
    include_news: bool = True
    include_sector: bool = True

@dataclass
class AnalysisStatus:
    session_id: str
    status: str
    progress_percentage: float
    current_step: str
    estimated_completion_time: Optional[str] = None
    error_message: Optional[str] = None

@dataclass
class AnalysisResponse:
    session_id: str
    status: str
    analysis: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


# --- API Communication ---

def start_analysis(request: AnalysisRequest) -> Optional[str]:
    """Sends a request to start a financial analysis."""
    try:
        response = requests.post(f"{API_BASE_URL}/start", json=asdict(request))
        response.raise_for_status()
        return response.json().get("session_id")
    except requests.exceptions.RequestException as e:
        st.error(f"Error starting analysis: {e}")
        return None

def get_analysis_status(session_id: str) -> Optional[AnalysisStatus]:
    """Polls the status of an ongoing analysis."""
    try:
        response = requests.get(f"{API_BASE_URL}/status/{session_id}")
        response.raise_for_status()
        data = response.json()
        return AnalysisStatus(**data)
    except requests.exceptions.RequestException as e:
        st.error(f"Error getting status for session {session_id}: {e}")
        return None
    except Exception as e:
        st.error(f"Error processing status response: {e}")
        return None


def get_analysis_result(session_id: str) -> Optional[AnalysisResponse]:
    """Fetches the final analysis result."""
    try:
        response = requests.get(f"{API_BASE_URL}/result/{session_id}")
        response.raise_for_status()
        data = response.json()
        return AnalysisResponse(**data)
    except requests.exceptions.RequestException as e:
        st.error(f"Error getting result for session {session_id}: {e}")
        return None
    except Exception as e:
        st.error(f"Error processing result response: {e}")
        return None

# --- UI Components ---

def render_sidebar():
    """Renders the sidebar for user inputs."""
    st.sidebar.title("The Seer")
    st.sidebar.header("Financial Analysis")

    st.sidebar.subheader("Stock Ticker")
    ticker = st.sidebar.text_input("Enter Ticker (e.g., RELIANCE.NS)", "RELIANCE.NS")

    st.sidebar.subheader("Analysis Type")
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        ("comprehensive", "quick", "technical_only", "fundamental_only"),
        index=0,
    )

    st.sidebar.subheader("Time Horizon")
    time_horizon = st.sidebar.selectbox(
        "Select Time Horizon",
        ("short_term", "medium_term", "long_term"),
        index=1,
    )

    st.sidebar.subheader("Analysis Options")
    include_technical = st.sidebar.checkbox("Technical Analysis", True)
    include_fundamental = st.sidebar.checkbox("Fundamental Analysis", True)
    include_news = st.sidebar.checkbox("News & Sentiment", True)
    include_sector = st.sidebar.checkbox("Sector Analysis", True)

    run_analysis = st.sidebar.button("Run Analysis")

    return AnalysisRequest(
        ticker=ticker,
        analysis_type=analysis_type,
        time_horizon=time_horizon,
        include_technical=include_technical,
        include_fundamental=include_fundamental,
        include_news=include_news,
        include_sector=include_sector,
    ), run_analysis


def display_results(result: Dict[str, Any]):
    """Renders the final analysis report."""
    st.header("Financial Analysis Report")
    
    st.subheader(f"Ticker: {result.get('ticker', 'N/A')} ({result.get('company_name', 'N/A')})")
    
    st.markdown(f"**Investment Recommendation:** {result.get('investment_recommendation', 'N/A')}")

    with st.expander("Executive Summary", expanded=True):
        st.markdown(result.get('executive_summary', 'Not available.'))

    with st.expander("Price Targets"):
        targets = result.get('price_targets', {})
        st.metric("Bull Case", f"₹{targets.get('bull_case_target', 0):.2f}")
        st.metric("Base Case", f"₹{targets.get('base_case_target', 0):.2f}")
        st.metric("Bear Case", f"₹{targets.get('bear_case_target', 0):.2f}")
        st.write(f"**Time Horizon:** {targets.get('time_horizon', 'N/A')}")
        st.write(f"**Current Price:** ₹{targets.get('current_price', 0):.2f}")

    # Display other sections
    display_section("Market Context", result.get('market_context'))
    display_section("Fundamental Analysis", result.get('fundamental_analysis'))
    display_section("Technical Analysis", result.get('technical_analysis'))
    display_section("News & Sentiment", result.get('news_sentiment'))
    display_section("Risk Assessment", result.get('risk_assessment'))
    display_section("Conflict Resolution", result.get('conflict_resolution'))

    with st.expander("Metadata"):
        st.json(result.get('metadata', {}))


def display_section(title: str, data: Optional[Dict[str, Any]]):
    """Generic function to display a section of the report."""
    if not data:
        return
    
    with st.expander(title):
        for key, value in data.items():
            if isinstance(value, dict):
                st.subheader(key.replace("_", " ").title())
                st.json(value)
            elif isinstance(value, list):
                st.subheader(key.replace("_", " ").title())
                for item in value:
                    st.markdown(f"- {item}")
            else:
                st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")


# --- Main Application Logic ---

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="The Seer - Financial Analysis", layout="wide")
    
    request, run_analysis = render_sidebar()

    # Placeholders for dynamic content
    status_placeholder = st.empty()
    results_placeholder = st.empty()

    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    
    if run_analysis:
        st.session_state.session_id = None # Reset on new run
        with st.spinner("Starting analysis..."):
            session_id = start_analysis(request)
            if session_id:
                st.session_state.session_id = session_id
                st.success(f"Analysis started with session ID: {session_id}")
            else:
                st.error("Failed to start analysis.")

    if st.session_state.session_id:
        session_id = st.session_state.session_id
        
        with status_placeholder.container():
            st.info(f"Monitoring analysis for session: {session_id}")
            progress_bar = st.progress(0)
            status_text = st.empty()

        start_time = time.time()
        while time.time() - start_time < MAX_WAIT_SECONDS:
            status = get_analysis_status(session_id)
            if not status:
                status_placeholder.error("Could not retrieve analysis status.")
                break
            
            progress = status.progress_percentage
            progress_bar.progress(progress / 100.0)
            status_text.text(f"Status: {status.status} - {status.current_step}")

            if status.status == "completed":
                status_placeholder.success("Analysis complete!")
                result_data = get_analysis_result(session_id)
                if result_data and result_data.analysis:
                    with results_placeholder.container():
                        display_results(result_data.analysis)
                else:
                    results_placeholder.error("Failed to retrieve or parse analysis result.")
                break

            if status.status == "failed":
                status_placeholder.error(f"Analysis failed: {status.error_message or 'No details'}")
                break

            time.sleep(POLL_INTERVAL_SECONDS)
        else:
            status_placeholder.warning(f"Stopped monitoring after {MAX_WAIT_SECONDS} seconds.")


if __name__ == "__main__":
    main()
