import streamlit as st
import os
from inference import NewsClassifier

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="AI News Classifier",
    page_icon="📰",
    layout="wide"
)

# --------------------------------------------------
# GLOBAL STYLES (EMERALD DARK THEME)
# --------------------------------------------------
st.markdown(
    """
    <style>
    /* 1. Main Background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #020b08 0%, #051a11 100%) !important;
    }

    .stApp {
        background: transparent;
        color: #d1fae5;
    }

    /* 2. Sidebar Background */
    [data-testid="stSidebar"] {
        background-color: #010806 !important;
        border-right: 1px solid #064e3b !important;
    }

    /* 3. Headings & Custom Sidebar Emerald Headers */
    h1, h2, h3, .emerald-header {
        color: #6ee7b7 !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
        display: block;
        margin-bottom: 10px;
    }

    /* 4. Text Area */
    textarea {
        background-color: #04120c !important;
        color: #ecfdf5 !important;
        border-radius: 12px !important;
        border: 1px solid #065f46 !important;
    }

    /* 5. Modern Buttons (Main & Sidebar Widgets) */
    .stButton > button, div[data-baseweb="select"] > div {
        background: linear-gradient(90deg, #10b981, #059669) !important;
        color: #000000 !important;
        border-radius: 10px !important;
        border: none !important;
        font-weight: 700 !important;
    }

    .stButton > button:hover {
        background: linear-gradient(90deg, #34d399, #10b981) !important;
        box-shadow: 0px 4px 15px rgba(16, 185, 129, 0.4) !important;
    }

    /* 6. Styled Tip Box (Green background, white text) */
    div[data-testid="stNotification"] {
        background-color: #065f46 !important;
        color: #ffffff !important;
        border: 1px solid #10b981 !important;
        border-radius: 10px;
    }
    
    /* Ensure icon in tip box is also white */
    div[data-testid="stNotification"] svg {
        fill: #ffffff !important;
    }

    /* 7. Results Card */
    .card {
        background-color: rgba(6, 78, 59, 0.15);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid #065f46;
        backdrop-filter: blur(10px);
    }


    /* 8. Sticky Footer */
    .footer {
        position: fixed;
        left: 21rem; /* sidebar width */
        bottom: 12px;
        width: calc(100% - 21rem);
        color: #64748b;
        text-align: center;
        padding: 6px 0;
        font-size: 14px;
        z-index: 999;
        background: transparent;
}

    }

    /* Prevent content from hiding behind footer */
    .block-container {
        padding-bottom: 80px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# MODELS INITIALIZATION
# --------------------------------------------------
@st.cache_resource
def load_classifier():
    return NewsClassifier()

classifier = load_classifier()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.markdown('<span class="emerald-header">⚙️ Select Model</span>', unsafe_allow_html=True)
    
    model_choice = st.selectbox(
        "Select Model Architecture:",
        ("Machine Learning (SVM)", "Deep Learning (Neural Network)"),
        label_visibility="collapsed"
    )
    
    st.info("Tip: The ML model is faster for short headlines, while the DL model often handles nuanced context better.")
    
    st.markdown("---")
    st.markdown('<span class="emerald-header">🛠️ Tools & Technologies</span>', unsafe_allow_html=True)
    st.markdown("- **NLP:** TF-IDF / Tokenization\n- **ML:** Scikit-Learn (SVM)\n- **DL:** Keras/TensorFlow\n- **App:** Streamlit")
    

    st.markdown("---")
    st.markdown('<span class="emerald-header">👨‍💻 Project by:</span>', unsafe_allow_html=True)
    st.markdown("**Azhar Mehmood**")
    st.markdown("- [Check Github Repository](https://github.com/azharmehmood4/)\n- [Connect on Linkedin](https://www.linkedin.com/in/azharmehmod)")


# --------------------------------------------------
# MAIN CONTENT
# --------------------------------------------------
st.markdown("# 📰 AI News Classifier")
st.markdown("Instantly categorize news articles into **World, Sports, Business, or Sci/Tech** using hybrid modeling.")

input_text = st.text_area(
    "💡 Paste the news headline or content here:",
    placeholder="e.g., NASA's James Webb telescope captures stunning new images of distant galaxies...",
    height=180
)

run_button = st.button("🚀 Classify Article", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# EXECUTION
# --------------------------------------------------
if run_button:
    if not input_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("🤖 Processing text and running inference..."):
            m_type = 'ml' if "Machine Learning" in model_choice else 'dl'
            result = classifier.predict(input_text, model_type=m_type)
            
            st.markdown("### 🎯 Prediction Result")
            st.success(f"The article is classified as: **{result}**")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown(
    """
    <div class="footer">
        Developed by 
        <a href="https://github.com/Furqan09Ahmed/news-classification-system" target="_blank">
            Furqan Ahmed
        </a>
        | NLP Document Classification System
    </div>
    """,
    unsafe_allow_html=True
)