import warnings
warnings.filterwarnings("ignore", message=".*use_container_width.*")
warnings.filterwarnings("ignore", message=".*use_column_width.*")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

import streamlit as st

st.set_page_config(
    page_title="AI-102 Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0a2e 0%, #1a1a4e 50%, #0d1b2a 100%);
}
[data-testid="stSidebar"] * { color: #e0e8ff !important; }
.gradient-title {
    background: linear-gradient(135deg, #00d4ff 0%, #7b2ff7 50%, #ff6b6b 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.5rem;
    font-weight: 900;
    text-align: center;
    padding: 0.5rem 0;
}
.info-box {
    background: linear-gradient(135deg, #e8f4fd, #d1ecf1);
    border-left: 4px solid #17a2b8;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
}
.exam-tip {
    background: linear-gradient(135deg, #fff3cd, #ffeaa7);
    border-left: 4px solid #ffc107;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
}
.concept-box {
    background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
    border-left: 4px solid #28a745;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
}
.module-card {
    background: linear-gradient(135deg, #f8f9fa, #e9ecef);
    border: 1px solid #dee2e6;
    border-radius: 12px;
    padding: 1.2rem;
    margin: 0.5rem 0;
}
.warning-box {
    background: linear-gradient(135deg, #fce4ec, #f8bbd0);
    border-left: 4px solid #e91e63;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
defaults = {
    "completed_modules": set(),
    "quiz_score": 0,
    "quiz_completed": False,
    "quiz_questions": None,
    "current_question": 0,
    "answers_given": {},
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 AI-102 Explorer")
    st.markdown("*Azure AI Engineer Lab*")
    st.markdown("---")

    page = st.radio(
        "📚 Choose a Module",
        [
            "🏠 Home",
            "👁️ Azure AI Vision",
            "💬 Azure AI Language",
            "🎙️ Azure AI Speech",
            "📄 Document Intelligence",
            "🔍 Knowledge Mining",
            "✨ Generative AI",
            "🎯 Quiz — Test Yourself",
        ],
    )

    st.markdown("---")
    st.markdown("### 📈 Your Progress")

    module_labels = [
        ("👁️ AI Vision",        "AI Vision"),
        ("💬 AI Language",       "AI Language"),
        ("🎙️ AI Speech",         "AI Speech"),
        ("📄 Doc Intelligence",  "Doc Intelligence"),
        ("🔍 Knowledge Mining",  "Knowledge Mining"),
        ("✨ Generative AI",     "Generative AI"),
    ]

    for display, key in module_labels:
        icon = "✅" if key in st.session_state.completed_modules else "⬜"
        st.markdown(f"{icon} {display}")

    visited = len(st.session_state.completed_modules)
    st.progress(visited / 6)
    st.caption(f"{visited}/6 modules visited")

    if st.session_state.quiz_completed:
        st.markdown(f"🎯 Quiz: **{st.session_state.quiz_score}/25**")

    st.markdown("---")
    st.markdown("### 🔗 Useful Links")
    st.markdown("- [AI-102 Exam Page](https://learn.microsoft.com/certifications/exams/ai-102)")
    st.markdown("- [Azure AI Services Docs](https://learn.microsoft.com/azure/ai-services/)")
    st.markdown("- [Azure AI Vision](https://learn.microsoft.com/azure/ai-services/computer-vision/)")
    st.markdown("- [Azure AI Language](https://learn.microsoft.com/azure/ai-services/language-service/)")

# ── Page routing ───────────────────────────────────────────────────────────────
if page == "🏠 Home":
    from modules.home import show
elif page == "👁️ Azure AI Vision":
    from modules.ai_vision import show
    st.session_state.completed_modules.add("AI Vision")
elif page == "💬 Azure AI Language":
    from modules.ai_language import show
    st.session_state.completed_modules.add("AI Language")
elif page == "🎙️ Azure AI Speech":
    from modules.ai_speech import show
    st.session_state.completed_modules.add("AI Speech")
elif page == "📄 Document Intelligence":
    from modules.document_intelligence import show
    st.session_state.completed_modules.add("Doc Intelligence")
elif page == "🔍 Knowledge Mining":
    from modules.knowledge_mining import show
    st.session_state.completed_modules.add("Knowledge Mining")
elif page == "✨ Generative AI":
    from modules.generative_ai import show
    st.session_state.completed_modules.add("Generative AI")
elif page == "🎯 Quiz — Test Yourself":
    from modules.quiz import show

show()
