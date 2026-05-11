import streamlit as st
import os
import time
from inference import NewsClassifier

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.markdown("""
<link rel="stylesheet"
href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
""", unsafe_allow_html=True)
st.set_page_config(
    page_title="NovaNews • AI Classifier",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# STYLES
# --------------------------------------------------
st.markdown(
    """
    <style>
    /* Global Background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0a0a0f 0%, #1a0f2e 100%) !important;
    }
    .stApp {
        background: transparent;
        color: #f0f0f5;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #120d24 0%, #1a0f2e 100%) !important;
        border-right: 2px solid #6b4eff !important;
    }
    
    /* Main Header */
    h1 {
        background: linear-gradient(90deg, #a855f7, #22d3ee);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        letter-spacing: -1px;
    }
    
    /* Input Area */
    textarea {
        background-color: #1f1633 !important;
        color: #e0d9ff !important;
        border: 2px solid #6b4eff !important;
        border-radius: 16px !important;
        font-size: 1.05rem !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #7c3aed, #22d3ee) !important;
        color: #ffffff !important;
        border-radius: 50px !important;
        height: 3.2rem !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.4) !important;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(124, 58, 237, 0.6) !important;
    }
    
    /* Result Card */
    .result-card {
        background: linear-gradient(145deg, rgba(30, 20, 60, 0.9), rgba(15, 10, 35, 0.95));
        border: 1px solid #8b5cf6;
        border-radius: 20px;
        padding: 28px;
        margin-top: 20px;
        box-shadow: 0 10px 30px rgba(139, 92, 246, 0.25);
        backdrop-filter: blur(12px);
    }
    
    /* Sidebar Headers */
    .sidebar-header {
        color: #c4b5fd !important;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    
    /* Footer */
    .footer {
        position: fixed;
        bottom: 15px;
        left: 50%;
        transform: translateX(-50%);
        color: #64748b;
        font-size: 0.9rem;
        text-align: center;
        z-index: 100;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# MODEL LOAD
# --------------------------------------------------
@st.cache_resource
def load_classifier():
    return NewsClassifier()

classifier = load_classifier()

# --------------------------------------------------
# SIDEBAR 
# --------------------------------------------------
with st.sidebar:
    st.markdown(
    '# <i class="fa-solid fa-earth-americas"></i> NovaNews',
    unsafe_allow_html=True)
    st.markdown("**News Classifier AI**")
    st.markdown("---")
    
    #  Model 
    st.markdown(
    '<p class="sidebar-header"><i class="fa-solid fa-brain"></i> Model</p>',
    unsafe_allow_html=True)
    model_choice = st.selectbox(
        "Choose Architecture",
        ("Machine Learning (SVM)",),
        label_visibility="collapsed"
    )
    st.success("⚡ Fast & Lightweight")
    st.info(" We will add the Deep Learning model soon.")
    
    st.markdown("---")
    
    # System Information
    st.markdown(
    '<p class="sidebar-header"><i class="fa-solid fa-gear"></i> System Information</p>',
    unsafe_allow_html=True)
    st.markdown("""
    **NLP:** TF-IDF  
    **Model:** SVM  
    **Framework:** Streamlit
    """)
    
    st.markdown("---")
    
    st.markdown(
    '<p class="sidebar-header"><i class="fa-solid fa-bookmark"></i> Categories</p>',
    unsafe_allow_html=True)
    st.markdown("""
    - <i class="fa-solid fa-earth-americas"></i>  **World**  
    - <i class="fa-solid fa-futbol"></i>  **Sports**  
    - <i class="fa-solid fa-briefcase"></i>  **Business**  
    - <i class="fa-solid fa-microchip"></i>  **Sci/Tech**
    """, unsafe_allow_html=True)
    st.caption("**Note**: Currently the model only predicts these 4 categories")
    st.info(" More categories will be added soon when we expand to Deep Learning")
    
    st.markdown("---")
    st.markdown(
    '<p class="sidebar-header"><i class="fa-solid fa-lightbulb"></i> Quick Tips</p>',
    unsafe_allow_html=True)
    st.markdown("""
    • Paste full article or just headline  
    • SVM works best for both short and long text
    """)
    
    st.markdown("---")
    st.markdown("**Made by Azhar Mehmood**")
    st.markdown("[**GitHub Repo**](https://github.com/AzharMehmood4/news-classifier)")
    st.markdown("[**Connect on LinkedIn**](https://www.linkedin.com/in/azharmehmod)")

# --------------------------------------------------
# MAIN LAYOUT 
# --------------------------------------------------
col1, col2 = st.columns([1.1, 1])

with col1:
    st.markdown(
    '# <i class="fa-solid fa-earth-americas"></i> Classify News Instantly',
    unsafe_allow_html=True)
    st.markdown("#### Powered by AI • Real-time • Accurate")
    
    input_text = st.text_area(
        "",
        placeholder="Paste your news headline or full article here...",
        height=260,
        label_visibility="collapsed"
    )
    
    run_button = st.button(" Classify Now", use_container_width=True)

with col2:
    st.markdown("### How NovaNews Works")
    st.info("""
    1. Paste any news content  
    2. Choose model type ( Deep Learning will be added soon )
    3. Get accurate AI classification
    """)
    
    st.markdown("#### Example Headlines")
    examples = [
        "Apple unveils new MacBook with revolutionary M4 chip",
        "Pakistan cricket team wins T20 World Cup final",
        "Global markets crash amid new trade tensions",
        "Scientists discover potential cure for Alzheimer's"
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=ex):
            input_text = ex

# --------------------------------------------------
#  STREAMING
# --------------------------------------------------
def stream_text(text, placeholder):
    output = ""
    for char in text:
        output += char
        placeholder.markdown(
            f"""
            <div class="result-card">
                <h2 style="margin:0; color:#c4b5fd;">{output}<span style="color:#a855f7; animation: blink 0.8s infinite;">▋</span></h2>
            </div>
            """,
            unsafe_allow_html=True
        )
        time.sleep(0.018)

# --------------------------------------------------
# EXECUTION
# --------------------------------------------------
if run_button:
    if not input_text or not input_text.strip():
        st.error(" Please enter some text to classify.")
    else:
        with st.spinner("Analyzing with AI..."):
            result = classifier.predict(input_text, model_type='ml')
        
        st.markdown(
        '### <i class="fa-solid fa-bullseye"></i> Classification Result',
        unsafe_allow_html=True)
        placeholder = st.empty()
        stream_text(f"Based on detailed analysis, this article is categorized as:  {result}", placeholder )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown(
    """
    <div class="footer">
        NovaNews AI Classifier • using Streamlit &nbsp; | &nbsp; 
        <a href="https://github.com/AzharMehmood4/news-classifier" target="_blank">Azhar Mehmood</a>
    </div>
    """,
    unsafe_allow_html=True
)