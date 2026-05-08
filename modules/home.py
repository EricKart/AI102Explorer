import streamlit as st
import plotly.graph_objects as go
import pandas as pd


def show():
    st.markdown('<p class="gradient-title">🧠 AI-102 Explorer</p>', unsafe_allow_html=True)
    st.markdown("### Your Free, Local Azure AI Engineer Lab — No Azure Account Needed!")

    st.markdown("""
    <div class="info-box">
    <strong>Welcome!</strong> This interactive app teaches you every hands-on concept you need to pass the
    <strong>Microsoft AI-102: Designing and Implementing a Microsoft Azure AI Solution</strong> exam —
    completely offline, using only Python libraries running on your own machine.<br><br>
    AI-102 is the <em>Associate-level</em> follow-up to AI-900. It requires coding knowledge and
    dives deep into <em>implementing</em> Azure AI services — not just knowing what they are.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # AI-900 vs AI-102 comparison
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🆚 AI-900 vs AI-102")
        comparison = pd.DataFrame({
            "Aspect":        ["Level",          "Audience",         "Coding required", "Azure depth",    "Exam questions", "Passing score"],
            "AI-900":        ["Fundamentals",   "Everyone",         "None",            "Conceptual",     "~60",            "700/1000"],
            "AI-102":        ["Associate",      "AI Engineers",     "Required",        "Implementation", "~60",            "700/1000"],
        })
        st.dataframe(comparison, hide_index=True, use_container_width=True)

    with col2:
        st.markdown("#### 📋 What You Need Before This Exam")
        st.markdown("""
        - ✅ Understanding of AI concepts (AI-900 level)
        - ✅ Python or C# programming experience
        - ✅ REST API familiarity (JSON requests/responses)
        - ✅ Basic cloud computing knowledge
        - ✅ Azure subscription familiarity (not needed here!)
        - ❌ No Azure account needed for this app
        - ❌ No API keys needed for this app
        """)

    st.markdown("---")

    # Radar chart — exam domain weights
    domains = [
        "Plan & Manage\nAI Solutions",
        "Computer\nVision",
        "NLP\nSolutions",
        "Document &\nKnowledge Mining",
        "Generative\nAI",
        "Speech\nSolutions",
    ]
    weights = [17, 17, 17, 17, 12, 10]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=weights + [weights[0]],
        theta=domains + [domains[0]],
        fill='toself',
        name='Exam Weight (%)',
        line_color='#7b2ff7',
        fillcolor='rgba(123,47,247,0.25)',
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 25])),
        showlegend=False,
        title="AI-102 Exam Domain Weights (%)",
        height=430,
        margin=dict(l=60, r=60, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("## 🗂️ Your Modules")

    modules = [
        ("👁️", "Azure AI Vision",        "Image analysis, object detection, OCR, custom vision classifiers",          "#2196F3"),
        ("💬", "Azure AI Language",       "NER, sentiment, key phrases, summarisation, language detection",           "#9C27B0"),
        ("🎙️", "Azure AI Speech",         "Speech-to-text, TTS, SSML, waveform analysis, phoneme concepts",           "#4CAF50"),
        ("📄", "Document Intelligence",   "Invoice/receipt parsing, table extraction, form field recognition",         "#FF9800"),
        ("🔍", "Knowledge Mining",        "Search indexing, BM25 ranking, vector similarity, faceted navigation",      "#F44336"),
        ("✨", "Generative AI",           "Prompt engineering, RAG simulation, Markov chains, token counting",         "#00BCD4"),
    ]

    cols = st.columns(3)
    for i, (icon, name, desc, color) in enumerate(modules):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="module-card" style="border-top: 4px solid {color};">
                <h4>{icon} {name}</h4>
                <p style="color:#555; font-size:0.9rem;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 🎓 Recommended Study Path")

    for step in [
        "1️⃣ **Home** — Understand how AI-102 differs from AI-900",
        "2️⃣ **Azure AI Vision** — Image analysis, OCR, and custom classifiers",
        "3️⃣ **Azure AI Language** — Text NLP services: NER, sentiment, summarisation",
        "4️⃣ **Azure AI Speech** — Voice services: STT, TTS, SSML",
        "5️⃣ **Document Intelligence** — Form and document data extraction",
        "6️⃣ **Knowledge Mining** — Azure AI Search, indexing, vector search",
        "7️⃣ **Generative AI** — Azure OpenAI, prompts, RAG architecture",
        "8️⃣ **Quiz** — 25 exam-style questions across all domains!",
    ]:
        st.markdown(step)

    st.markdown("---")
    st.markdown("""
    <div class="exam-tip">
    <strong>Exam Strategy:</strong> AI-102 scenario questions often ask you to
    <em>choose the right Azure service for a given business requirement</em>.
    As you go through each module, note the <strong>when to use which service</strong> comparisons.
    </div>
    """, unsafe_allow_html=True)

    with st.expander("💻 How to Run This App Locally"):
        st.code("""
# Clone and run (Windows - Command Prompt)
git clone https://github.com/EricKart/AI102Explorer.git
cd AI102Explorer
setup.bat

# Clone and run (macOS / Linux)
git clone https://github.com/EricKart/AI102Explorer.git
cd AI102Explorer
chmod +x setup.sh && ./setup.sh
""", language="bash")
