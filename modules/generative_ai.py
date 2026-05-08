import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import random
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ── Knowledge base for RAG ─────────────────────────────────────────────────────

RAG_DOCS = [
    ("Azure AI Search",        "Azure AI Search is a search-as-a-service that supports full-text, vector, and hybrid search. It integrates with Azure OpenAI for RAG scenarios."),
    ("Azure OpenAI GPT-4",     "GPT-4 is a multimodal large language model available via Azure OpenAI Service. It supports text and image inputs and excels at reasoning, code, and summarisation."),
    ("Responsible AI",         "Microsoft Responsible AI principles: Fairness, Reliability & Safety, Privacy & Security, Inclusiveness, Transparency, Accountability."),
    ("Azure Document Intel.",  "Azure Document Intelligence extracts structured data from documents using prebuilt models (invoice, receipt, ID) or custom trained models."),
    ("Azure AI Language",      "Azure AI Language provides NLP capabilities: sentiment analysis, entity recognition, key phrase extraction, summarisation, and translation."),
    ("Azure Custom Vision",    "Azure Custom Vision trains image classifiers and object detectors. Minimum 15 images per tag. Supports export to ONNX, CoreML, TensorFlow."),
    ("Azure AI Speech",        "Azure AI Speech includes TTS (neural voices, SSML), STT (real-time, batch), speech translation, and speaker recognition."),
    ("RAG Pattern",            "Retrieval Augmented Generation (RAG) grounds LLM responses in retrieved context to reduce hallucination. Steps: embed query, retrieve top-k docs, inject into prompt."),
]

BIGRAM_SEED = """
Azure AI services help developers build intelligent applications.
Machine learning enables computers to learn from data without being explicitly programmed.
Natural language processing allows computers to understand and generate human language.
Computer vision gives machines the ability to interpret visual information from images and video.
Azure OpenAI Service provides access to powerful language models including GPT-4 and GPT-3.5.
Responsible AI ensures that artificial intelligence is developed and deployed ethically and safely.
Azure AI Search integrates with cognitive services to enable intelligent search over large corpora.
Document intelligence extracts structured data from forms invoices receipts and other documents.
"""

PROMPT_TEMPLATES = {
    "Zero-Shot": (
        "You are a helpful AI assistant.\n\n"
        "User: {question}\n\nAssistant:"
    ),
    "Few-Shot": (
        "You are a knowledgeable AI assistant. Here are examples:\n\n"
        "Q: What is Azure AI Search?\n"
        "A: A cloud search service supporting full-text, vector, and hybrid search.\n\n"
        "Q: What is GPT-4?\n"
        "A: A large language model by OpenAI, available via Azure OpenAI Service.\n\n"
        "Q: {question}\nA:"
    ),
    "Chain-of-Thought": (
        "You are an AI assistant that thinks step by step.\n\n"
        "Q: {question}\n\n"
        "Let me think through this step by step:\n"
        "1. First, consider what the question is asking.\n"
        "2. Recall relevant facts.\n"
        "3. Synthesise a clear answer.\n\nA:"
    ),
    "System + User": (
        "[System]: You are a professional Azure exam tutor. "
        "Give concise, accurate answers based on Microsoft documentation.\n\n"
        "[User]: {question}\n\n[Assistant]:"
    ),
}


# ── Helper functions ───────────────────────────────────────────────────────────

def _build_ngrams(text, n=2):
    words = re.findall(r'\b[a-z]+\b', text.lower())
    model = defaultdict(list)
    for i in range(len(words) - n):
        key = tuple(words[i:i+n-1])
        model[key].append(words[i+n-1])
    return model, words


def _generate_text(model, seed_words, length=40, temperature=1.0):
    if not model:
        return "Not enough text to generate."
    words = list(seed_words[-2:]) if len(seed_words) >= 2 else list(seed_words)
    for _ in range(length):
        key = tuple(words[-len(list(model.keys())[0]):])
        candidates = model.get(key)
        if not candidates:
            key = random.choice(list(model.keys()))
            candidates = model[key]
        if temperature <= 0.3:
            next_word = max(set(candidates), key=candidates.count)
        elif temperature >= 1.8:
            next_word = random.choice(candidates)
        else:
            weights = np.array([candidates.count(c) for c in set(candidates)], dtype=float)
            weights = weights ** (1.0 / temperature)
            weights /= weights.sum()
            next_word = np.random.choice(list(set(candidates)), p=weights)
        words.append(next_word)
    return ' '.join(words)


def _rag_search(query, docs, top_k=3):
    texts = [title + " " + body for title, body in docs]
    vec = TfidfVectorizer(stop_words='english')
    mat = vec.fit_transform(texts + [query])
    q_vec = mat[-1]
    doc_vecs = mat[:-1]
    scores = cosine_similarity(q_vec, doc_vecs)[0]
    ranked = np.argsort(scores)[::-1][:top_k]
    return [(docs[i][0], docs[i][1], float(scores[i])) for i in ranked if scores[i] > 0]


def _approx_tokens(text):
    """Approximate BPE tokenisation: ~4 chars per token."""
    tokens = re.findall(r"[a-zA-Z']+|[0-9]+|[^a-zA-Z0-9\s]", text)
    bpe_count = 0
    for tok in tokens:
        bpe_count += max(1, len(tok) // 4)
    return len(tokens), bpe_count


# ── Main page ──────────────────────────────────────────────────────────────────

def show():
    st.markdown('<p class="gradient-title">🤖 Generative AI</p>', unsafe_allow_html=True)
    st.markdown("### Generative AI Concepts — Local Simulation (No API Key Required)")

    tab1, tab2, tab3, tab4 = st.tabs([
        "✍️ Prompt Engineering", "📚 RAG Simulation", "🎲 Text Generation", "🪙 Token Analysis",
    ])

    # ── Tab 1: Prompt Engineering ─────────────────────────────────────────────
    with tab1:
        st.markdown("""
        <div class="info-box">
        <strong>Prompt Engineering</strong><br>
        The craft of designing inputs to language models to elicit accurate, relevant, and
        well-structured outputs. Good prompt engineering can dramatically improve response quality
        without changing the model weights.
        </div>
        """, unsafe_allow_html=True)

        question = st.text_input(
            "Your question / task:",
            value="What is Azure AI Search and how does it support RAG?"
        )
        technique = st.selectbox("Prompting technique:", list(PROMPT_TEMPLATES.keys()))
        prompt = PROMPT_TEMPLATES[technique].format(question=question)

        st.markdown("**Generated Prompt:**")
        st.code(prompt, language="text")

        st.markdown("#### Technique Comparison")
        comp = pd.DataFrame([
            {"Technique":        "Zero-Shot",
             "When to Use":      "General tasks, clear questions",
             "Pros":             "Simple, fast",
             "Cons":             "Less consistent for complex tasks"},
            {"Technique":        "Few-Shot",
             "When to Use":      "Formatting, classification, extraction",
             "Pros":             "Guides output format",
             "Cons":             "Longer prompt, uses more tokens"},
            {"Technique":        "Chain-of-Thought",
             "When to Use":      "Reasoning, maths, multi-step problems",
             "Pros":             "Improves accuracy on hard tasks",
             "Cons":             "Verbose output, higher latency"},
            {"Technique":        "System + User",
             "When to Use":      "Production chat apps, persona control",
             "Pros":             "Persistent instructions, role separation",
             "Cons":             "System prompt consumes tokens"},
        ])
        st.dataframe(comp, hide_index=True, use_container_width=True)

        st.markdown("""
        <div class="exam-tip">
        <strong>Exam Tip:</strong> Azure OpenAI Chat Completion message roles:<br>
        &bull; <code>system</code> — persistent instructions / persona<br>
        &bull; <code>user</code> — human turn<br>
        &bull; <code>assistant</code> — model turn (previous responses in multi-turn)<br>
        &bull; <code>tool</code> — result from a function/tool call (for agentic scenarios)
        </div>
        """, unsafe_allow_html=True)

    # ── Tab 2: RAG Simulation ─────────────────────────────────────────────────
    with tab2:
        st.markdown("""
        <div class="info-box">
        <strong>Retrieval Augmented Generation (RAG)</strong><br>
        RAG grounds LLM responses in facts retrieved from a search index.
        This prevents hallucination by providing the model with relevant context at query time.
        </div>
        """, unsafe_allow_html=True)

        rag_q = st.text_input("RAG query:", value="How does Azure AI Search support RAG?")
        top_k = st.slider("Top-K documents to retrieve:", 1, 5, 3)

        retrieved = _rag_search(rag_q, RAG_DOCS, top_k=top_k)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Retrieved Context Documents**")
            for rank, (title, body, score) in enumerate(retrieved, 1):
                st.markdown(f"**{rank}. {title}** *(score: {score:.3f})*")
                st.caption(body[:120] + "…" if len(body) > 120 else body)
                st.progress(min(score * 2, 1.0))

        with col2:
            context = "\n".join(f"- {title}: {body}" for title, body, _ in retrieved)
            rag_prompt = (
                f"You are a helpful Azure AI assistant.\n\n"
                f"Use the following retrieved context to answer the question accurately:\n\n"
                f"{context}\n\n"
                f"Question: {rag_q}\n\nAnswer:"
            )
            st.markdown("**Assembled RAG Prompt:**")
            st.code(rag_prompt[:600] + ("\n…(truncated)" if len(rag_prompt)>600 else ""),
                    language="text")

        st.markdown("#### RAG vs No-RAG Comparison")
        comp_df = pd.DataFrame([
            {"Aspect": "Factual accuracy",  "Without RAG": "May hallucinate",  "With RAG (grounded)": "Uses verified sources"},
            {"Aspect": "Knowledge cutoff",  "Without RAG": "Limited to training", "With RAG (grounded)": "Can use fresh data"},
            {"Aspect": "Latency",           "Without RAG": "Lower",            "With RAG (grounded)": "Slightly higher (search + LLM)"},
            {"Aspect": "Cost",              "Without RAG": "LLM tokens only",  "With RAG (grounded)": "Search + LLM tokens"},
            {"Aspect": "Auditability",      "Without RAG": "Hard to trace",    "With RAG (grounded)": "Can cite source documents"},
        ])
        st.dataframe(comp_df, hide_index=True, use_container_width=True)

    # ── Tab 3: Text Generation ────────────────────────────────────────────────
    with tab3:
        st.markdown("""
        <div class="info-box">
        <strong>Text Generation — Markov Bigram Model</strong><br>
        A bigram language model predicts the next word based on the previous word.
        This simulates how temperature affects output randomness — a concept central
        to all LLM inference.
        </div>
        """, unsafe_allow_html=True)

        gen_corpus = st.text_area(
            "Training corpus (edit or extend):",
            value=BIGRAM_SEED.strip(),
            height=130,
        )
        seed_input = st.text_input(
            "Seed phrase (first 2 words to start generation):",
            value="azure ai"
        )
        gen_length = st.slider("Words to generate:", 15, 80, 35)

        temps = [0.2, 0.5, 1.0, 1.5, 2.0]
        model_ng, all_words = _build_ngrams(gen_corpus, n=2)
        seed_words = seed_input.lower().split()[:2] if seed_input.strip() else ["azure", "ai"]

        st.markdown("#### Generated Text at Different Temperatures")
        fig, axes = plt.subplots(1, 1, figsize=(10, 3))
        lengths = []
        for temp in temps:
            generated = _generate_text(model_ng, seed_words, length=gen_length, temperature=temp)
            unique_ratio = len(set(generated.split())) / max(len(generated.split()), 1)
            lengths.append(unique_ratio)

        axes.plot(temps, lengths, marker='o', color='#7b2ff7', linewidth=2.5)
        axes.fill_between(temps, lengths, alpha=0.2, color='#7b2ff7')
        axes.set_xlabel("Temperature")
        axes.set_ylabel("Unique Word Ratio")
        axes.set_title("Temperature vs Output Diversity")
        axes.set_xticks(temps)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        sel_temp = st.select_slider("Preview temperature:", temps, value=1.0)
        result = _generate_text(model_ng, seed_words, length=gen_length, temperature=sel_temp)
        st.markdown(f"**Generated** (temperature={sel_temp}):")
        st.info(result)

        st.markdown("""
        <div class="concept-box">
        <strong>Temperature in LLMs:</strong><br>
        &bull; <strong>Low (0.0–0.3)</strong> — deterministic, factual, repetitive. Good for code, extraction.<br>
        &bull; <strong>Medium (0.5–0.8)</strong> — balanced. Good for conversational assistants.<br>
        &bull; <strong>High (1.0–2.0)</strong> — creative, diverse, sometimes incoherent. Good for brainstorming.<br>
        In Azure OpenAI: <code>temperature</code> range is 0.0–2.0. Also see <code>top_p</code> (nucleus sampling).
        </div>
        """, unsafe_allow_html=True)

    # ── Tab 4: Token Analysis ─────────────────────────────────────────────────
    with tab4:
        st.markdown("""
        <div class="info-box">
        <strong>Token Analysis</strong><br>
        LLMs process text as <em>tokens</em> — subword units from BPE (Byte-Pair Encoding).
        One token ~ 4 characters or 0.75 words in English. Pricing and context limits are
        measured in tokens, not words.
        </div>
        """, unsafe_allow_html=True)

        token_text = st.text_area(
            "Text to analyse:",
            value="Azure OpenAI Service provides access to GPT-4 and GPT-3.5 models "
                  "through a REST API with enterprise security, compliance, and regional availability.",
            height=100,
        )
        raw_tokens, bpe_tokens = _approx_tokens(token_text)
        c1, c2, c3 = st.columns(3)
        c1.metric("Characters",     len(token_text))
        c2.metric("Word tokens",    raw_tokens)
        c3.metric("~BPE tokens",    bpe_tokens)

        st.markdown("#### Azure OpenAI Cost Estimation")
        pricing = pd.DataFrame([
            {"Model":           "GPT-4o",       "Input $/1K tok": 0.005,  "Output $/1K tok": 0.015,  "Context": "128K"},
            {"Model":           "GPT-4",         "Input $/1K tok": 0.03,   "Output $/1K tok": 0.06,   "Context": "128K"},
            {"Model":           "GPT-3.5 Turbo", "Input $/1K tok": 0.0005, "Output $/1K tok": 0.0015, "Context": "16K"},
            {"Model":           "text-embedding-3-small", "Input $/1K tok": 0.00002, "Output $/1K tok": 0.0, "Context": "8K"},
        ])
        st.dataframe(pricing, hide_index=True, use_container_width=True)
        st.caption("Illustrative pricing — check Azure portal for current rates")

        est_calls = st.number_input("Estimated daily API calls:", min_value=1, value=1000, step=100)
        sel_model = st.selectbox("Model:", pricing["Model"].tolist())
        row = pricing[pricing["Model"] == sel_model].iloc[0]
        daily_in  = (bpe_tokens * est_calls / 1000) * row["Input $/1K tok"]
        daily_out = (bpe_tokens * 2 * est_calls / 1000) * row["Output $/1K tok"]
        daily_total = daily_in + daily_out

        col1, col2, col3 = st.columns(3)
        col1.metric("Est. Daily Cost",   f"${daily_total:.2f}")
        col2.metric("Est. Monthly Cost", f"${daily_total*30:.0f}")
        col3.metric("Tokens per Call",   f"~{bpe_tokens} in / {bpe_tokens*2} out")

        st.markdown("""
        <div class="exam-tip">
        <strong>Exam Tip:</strong> Azure OpenAI key parameters:<br>
        &bull; <code>temperature</code> — randomness (0.0–2.0)<br>
        &bull; <code>max_tokens</code> — maximum output tokens<br>
        &bull; <code>top_p</code> — nucleus sampling threshold<br>
        &bull; <code>frequency_penalty</code> / <code>presence_penalty</code> — reduce repetition<br>
        &bull; <code>stop</code> — token sequence(s) to end generation<br>
        &bull; Content filters are applied by default; configure in Azure AI Foundry
        </div>
        """, unsafe_allow_html=True)
