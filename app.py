import streamlit as st
import requests
import time
import os
from typing import Optional, Dict, Any

# Configuration
# Use environment variable for Docker networking, fallback to localhost for local testing
API_URL = os.getenv("API_URL", "http://localhost:8000")
SEARCH_ENDPOINT = f"{API_URL}/search"
HEALTH_ENDPOINT = f"{API_URL}/"

# Page config
st.set_page_config(
    page_title="NCO 2015 AI Career Advisor",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for pitch black background and consistent fonts
st.markdown("""
<style>
    /* Main background and text */
    .stApp {
        background-color: #000000;
        color: #ffffff;
    }
    
    /* All text elements */
    body, p, h1, h2, h3, h4, h5, h6, span, div {
        color: #ffffff !important;
        font-size: 16px !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }
    
    /* Headers with consistent sizing */
    h1 {
        font-size: 32px !important;
        font-weight: 600 !important;
        margin-bottom: 1rem !important;
    }
    
    h2 {
        font-size: 24px !important;
        font-weight: 500 !important;
        margin-bottom: 0.8rem !important;
    }
    
    h3 {
        font-size: 20px !important;
        font-weight: 500 !important;
        margin-bottom: 0.6rem !important;
    }
    
    /* Text area styling */
    .stTextArea textarea {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
        font-size: 16px !important;
        border-radius: 8px;
    }
    
    .stTextArea textarea:focus {
        border-color: #666666 !important;
    }
    
    /* Input labels */
    .stTextArea label, .stSlider label {
        color: #cccccc !important;
        font-size: 16px !important;
        font-weight: normal !important;
    }
    
    /* Button styling */
    .stButton button {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
        font-size: 16px !important;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    
    .stButton button:hover {
        background-color: #333333 !important;
        border-color: #666666 !important;
    }
    
    .stButton button:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }
    
    /* Primary button */
    .stButton button[kind="primary"] {
        background-color: #0066cc !important;
        border: none !important;
    }
    
    .stButton button[kind="primary"]:hover {
        background-color: #0052a3 !important;
    }
    
    /* Slider styling */
    .stSlider div[data-baseweb="slider"] {
        background-color: #1a1a1a !important;
    }
    
    .stSlider div[role="slider"] {
        background-color: #0066cc !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        font-size: 16px !important;
        border-radius: 8px;
        border: 1px solid #333333;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: #333333 !important;
    }
    
    .streamlit-expanderContent {
        background-color: #0a0a0a !important;
        border: 1px solid #333333;
        border-top: none;
        border-radius: 0 0 8px 8px;
        padding: 1rem;
    }
    
    /* Code blocks */
    .stCode {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        border: 1px solid #333333;
        border-radius: 4px;
    }
    
    /* Metric cards */
    div[data-testid="metric-container"] {
        background-color: #1a1a1a;
        border: 1px solid #333333;
        border-radius: 8px;
        padding: 1rem;
    }
    
    div[data-testid="metric-container"] label {
        color: #cccccc !important;
        font-size: 14px !important;
    }
    
    div[data-testid="metric-container"] div {
        color: #ffffff !important;
        font-size: 20px !important;
        font-weight: 500;
    }
    
    /* Success/Error/Info messages */
    .stAlert {
        background-color: #1a1a1a !important;
        border: 1px solid #333333;
        border-radius: 8px;
        color: #ffffff !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0a0a0a !important;
        border-right: 1px solid #333333;
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: #ffffff !important;
    }
    
    /* Horizontal rule */
    hr {
        border-color: #333333 !important;
        margin: 2rem 0;
    }
    
    /* Progress bar */
    .stProgress div[role="progressbar"] {
        background-color: #0066cc !important;
    }
    
    /* Download button */
    .stDownloadButton button {
        background-color: #1a1a1a !important;
        border: 1px solid #333333 !important;
        width: 100%;
    }
    
    /* Occupation card styling */
    .occupation-card {
        background-color: #1a1a1a;
        border: 1px solid #333333;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .nco-code {
        background-color: #0a0a0a;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-family: monospace;
        border: 1px solid #333333;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    defaults = {
        "query": "",
        "top_k": 5,
        "results": None,
        "api_status": None,
        "last_query_time": None,
        "error_message": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

def check_api_health() -> tuple[bool, str]:
    """Check if the backend API is healthy"""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("rag_initialized"):
                return True, "Connected"
            else:
                return False, "Backend not fully initialized"
        return False, f"HTTP {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, f"Cannot connect to {HEALTH_ENDPOINT}"
    except requests.exceptions.Timeout:
        return False, "Connection timeout"
    except Exception as e:
        return False, str(e)

def perform_search(query: str, top_k: int) -> Optional[Dict[str, Any]]:
    """Perform search query against backend"""
    try:
        response = requests.post(
            SEARCH_ENDPOINT,
            json={"query": query, "top_k": top_k},
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.session_state.error_message = f"Server error: {response.status_code}"
            return None
            
    except requests.exceptions.ConnectionError:
        st.session_state.error_message = "Could not connect to backend. Is FastAPI running?"
    except requests.exceptions.Timeout:
        st.session_state.error_message = "Request timed out. The LLM might be taking too long."
    except Exception as e:
        st.session_state.error_message = f"Error: {str(e)}"
    
    return None

# Sidebar
with st.sidebar:
    st.title("NCO 2015 Advisor")
    
    st.markdown("---")
    
    # API Status
    st.subheader("System Status")
    api_ok, status_msg = check_api_health()
    if api_ok:
        st.success(f"Backend: {status_msg}")
    else:
        st.error(f"Backend: {status_msg}")
        st.caption("Check docker container logs if unavailable")
    
    st.markdown("---")
    
    # About section
    st.subheader("About")
    st.markdown("""
    This AI advisor uses the NCO 2015 (National Classification of Occupations) 
    to help you find career matches based on your interests and skills.
    
    How it works:
    1. Describe work you enjoy
    2. System retrieves matching occupations
    3. AI generates personalized guidance
    """)
    
    st.markdown("---")
    
    # Tips
    st.subheader("Tips")
    st.markdown("""
    • Be specific about your interests
    • Mention skills you have or want to develop
    • Include work environment preferences
    • Try different descriptions
    """)
    
    # Clear results button
    if st.button("Clear Results", use_container_width=True):
        st.session_state.results = None
        st.session_state.query = ""
        st.session_state.error_message = None
        st.rerun()

# Main content
st.title("NCO 2015 AI Career Advisor")
st.markdown("Find your ideal career path based on the official NCO 2015 classification")

# Query input
st.subheader("Describe Your Ideal Work")
query = st.text_area(
    "What kind of work are you looking for?",
    value=st.session_state.query,
    height=120,
    placeholder="Example: I enjoy working with my hands, solving mechanical problems, and designing new things. I'm detail-oriented and like both independent and team work.",
    key="query_input"
)

# Update session state when query changes
if query != st.session_state.query:
    st.session_state.query = query
    st.session_state.results = None

# Top-k slider
st.subheader("Search Settings")
col1, col2 = st.columns([3, 1])
with col1:
    top_k = st.slider(
        "Number of occupations to consider",
        min_value=3,
        max_value=10,
        value=st.session_state.top_k,
        step=1,
        help="More occupations provide broader context but may include less relevant matches"
    )
with col2:
    st.metric("Selected", top_k)

st.session_state.top_k = top_k

# Search button
search_button = st.button(
    "Find Matching Occupations",
    type="primary",
    use_container_width=True,
    disabled=not api_ok or not st.session_state.query.strip()
)

# Perform search
if search_button and st.session_state.query.strip():
    with st.spinner("Analyzing your preferences and searching occupations..."):
        # Add progress bar for better UX
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        
        results = perform_search(st.session_state.query, st.session_state.top_k)
        progress_bar.empty()
        
        if results:
            st.session_state.results = results
            st.session_state.error_message = None
            st.session_state.last_query_time = time.strftime("%Y-%m-%d %H:%M:%S")
        else:
            st.session_state.results = None

# Display error message if any
if st.session_state.error_message:
    st.error(st.session_state.error_message)

# Display results
if st.session_state.results:
    data = st.session_state.results
    
    # Success metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        query_display = data["query"][:30] + "..." if len(data["query"]) > 30 else data["query"]
        st.metric("Query", query_display)
    with col2:
        st.metric("Sources", len(data["sources"]))
    with col3:
        if st.session_state.last_query_time:
            st.metric("Searched at", st.session_state.last_query_time.split()[1])
    
    st.markdown("---")
    
    # AI Response
    st.subheader("AI Career Guidance")
    with st.container():
        st.markdown(data["generated_answer"])
    
    st.markdown("---")
    
    # Retrieved occupations
    def get_score(item):
        """Get score from the item, handling missing values"""
        score = item.get('score')
        if score is None:
            return -1  # Put items without scores at the end
        return score
    
    # Sort sources by score descending
    sorted_sources = sorted(data["sources"], key=get_score, reverse=True)
    
    # Export option
    if st.button("Export Results", use_container_width=True):
        # Simple export as text
        export_text = f"Query: {data['query']}\n\n"
        export_text += f"AI Response:\n{data['generated_answer']}\n\n"
        export_text += "Retrieved Occupations:\n"
        
        # Sort for export too
        for r in sorted_sources:
            score_text = ""
            if r.get('score') is not None:
                score = r['score']
                if -1 <= score <= 1:
                    score_text = f" (Match: {score:.1%})"
                else:
                    score_text = f" (Score: {score:.2f})"
            
            export_text += f"- {r['title']} (NCO: {r['nco_2015']}){score_text}\n"
        
        st.download_button(
            label="Download as TXT",
            data=export_text,
            file_name=f"career_advice_{time.strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

elif st.session_state.query and not st.session_state.error_message:
    st.info("Click 'Find Matching Occupations' to get career advice")