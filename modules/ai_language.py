import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re

# NLTK with graceful fallbacks
try:
    import nltk
    for pkg in ['punkt', 'punkt_tab', 'vader_lexicon', 'stopwords',
                'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng',
                'maxent_ne_chunker', 'words']:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords as nltk_stopwords
    from nltk.chunk import ne_chunk
    from nltk.tag import pos_tag
    NLTK_OK = True
except Exception:
    NLTK_OK = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


# ── Helper functions ───────────────────────────────────────────────────────────

def _get_sentiment(text):
    if NLTK_OK:
        try:
            sia = SentimentIntensityAnalyzer()
            s = sia.polarity_scores(text)
            c = s['compound']
            label = "Positive" if c >= 0.05 else ("Negative" if c <= -0.05 else "Neutral")
            return label, s
        except Exception:
            pass
    pos_w = {'good','great','excellent','amazing','love','wonderful','fantastic','happy','best','outstanding'}
    neg_w = {'bad','terrible','awful','hate','worst','horrible','poor','sad','disappointed','frustrating'}
    words = set(text.lower().split())
    p, n = len(words & pos_w), len(words & neg_w)
    if p > n:   return "Positive", {"compound": 0.5,  "pos": 0.5, "neg": 0.0, "neu": 0.5}
    if n > p:   return "Negative", {"compound": -0.5, "pos": 0.0, "neg": 0.5, "neu": 0.5}
    return "Neutral", {"compound": 0.0, "pos": 0.1, "neg": 0.1, "neu": 0.8}


def _extract_entities(text):
    entities = []
    if NLTK_OK:
        try:
            tagged = pos_tag(word_tokenize(text))
            tree = ne_chunk(tagged)
            for subtree in tree:
                if hasattr(subtree, 'label'):
                    span = ' '.join(w for w, _ in subtree.leaves())
                    entities.append({
                        "Entity": span,
                        "Type": subtree.label(),
                        "Confidence": round(np.random.uniform(0.85, 0.98), 2),
                    })
        except Exception:
            pass
    if not entities:
        for pattern, etype in [
            (r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b', 'PERSON'),
            (r'\b[A-Z]{2,}\b', 'ORGANIZATION'),
            (r'\b\d{4}\b', 'DATE'),
        ]:
            for m in re.finditer(pattern, text):
                entities.append({"Entity": m.group(), "Type": etype, "Confidence": 0.85})
    return entities


def _extract_key_phrases(text, n=10):
    try:
        sents = re.split(r'[.!?]', text)
        sents = [s.strip() for s in sents if len(s.strip()) > 10] or [text]
        vec = TfidfVectorizer(ngram_range=(1, 3), stop_words='english', max_features=60)
        mat = vec.fit_transform(sents)
        names = vec.get_feature_names_out()
        scores = mat.toarray().mean(axis=0)
        ranked = sorted(zip(names, scores), key=lambda x: -x[1])
        return [(p, round(s, 4)) for p, s in ranked[:n] if s > 0]
    except Exception:
        words = [w.lower() for w in text.split() if len(w) > 4]
        return [(w, c) for w, c in Counter(words).most_common(n)]


def _summarize(text, n=3):
    try:
        sents = sent_tokenize(text) if NLTK_OK else re.split(r'[.!?]+', text)
        sents = [s.strip() for s in sents if len(s.strip()) > 20]
        if len(sents) <= n:
            return text, sents
        vec = TfidfVectorizer(stop_words='english')
        mat = vec.fit_transform(sents)
        scores = mat.toarray().sum(axis=1)
        top_idx = sorted(np.argsort(scores)[-n:])
        return ' '.join(sents[i] for i in top_idx), sents
    except Exception:
        parts = text.split('.')
        return '. '.join(parts[:n]), parts


def _detect_language(text):
    profiles = {
        "English": ['the', 'ing', 'ion', 'and', 'tion'],
        "French":  ['les', 'est', 'que', 'des', 'eur'],
        "Spanish": ['los', 'las', 'cion', 'nte', 'ado'],
        "German":  ['der', 'die', 'und', 'den', 'sch'],
        "Italian": ['che', 'della', 'dei', 'per', 'non'],
    }
    tl = text.lower()
    tg = Counter(tl[i:i+3] for i in range(len(tl)-2))
    raw = {lang: sum(tg.get(p, 0) for p in pats) for lang, pats in profiles.items()}
    total = sum(raw.values()) or 1
    normalized = {lang: round(v/total, 3) for lang, v in raw.items()}
    best = max(normalized, key=normalized.get)
    return best, normalized


# ── Sample texts ───────────────────────────────────────────────────────────────
SAMPLES = {
    "Tech review": (
        "Microsoft Azure AI services provide excellent tools for building intelligent applications. "
        "The Cognitive Services API is incredibly powerful and the documentation is outstanding. "
        "However, some features are still limited in certain regions."
    ),
    "News article": (
        "OpenAI and Microsoft announced a major partnership in Seattle on Tuesday. "
        "The $10 billion investment will accelerate development of artificial general intelligence. "
        "Satya Nadella, CEO of Microsoft, called it a transformative moment for the technology industry."
    ),
    "Customer feedback": (
        "I am extremely disappointed with the service. "
        "The product broke after two days and customer support was terrible. "
        "I would not recommend this to anyone."
    ),
    "French text": (
        "L'intelligence artificielle est une technologie fascinante qui transforme de nombreux secteurs "
        "industriels. Elle offre des possibilites extraordinaires pour les entreprises et les individus."
    ),
    "Spanish text": (
        "Los servicios de inteligencia artificial de Azure ofrecen capacidades avanzadas para el "
        "procesamiento del lenguaje natural y la vision por computadora en multiples idiomas."
    ),
}

LONG_TEXT = """Microsoft Azure is a cloud computing platform with a vast range of services.
Azure AI services provide developers with powerful tools to build intelligent applications without
needing deep machine learning expertise. The Azure Cognitive Services suite includes computer vision,
natural language processing, speech recognition, and decision-making capabilities.
These services are accessible via simple REST APIs or SDK clients available in Python, C#, Java,
and JavaScript. The platform also offers Azure Machine Learning for custom model training and deployment.
Azure OpenAI Service provides access to advanced language models including GPT-4.
With Azure AI, organisations can quickly prototype and deploy AI solutions that scale globally."""


# ── Main page ──────────────────────────────────────────────────────────────────

def show():
    st.markdown('<p class="gradient-title">💬 Azure AI Language</p>', unsafe_allow_html=True)
    st.markdown("### Simulating Azure AI Language — NLP Pipelines Running 100% Locally")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "😊 Sentiment", "🏷️ Named Entities", "🔑 Key Phrases",
        "📝 Summarisation", "🌐 Language Detection",
    ])

    selected = st.selectbox("Choose a sample text or type your own:",
                            ["Custom…"] + list(SAMPLES.keys()))
    if selected == "Custom…":
        text = st.text_area("Your text:", value="Azure AI services are fantastic tools.", height=90)
    else:
        text = st.text_area("Text (editable):", value=SAMPLES[selected], height=90)

    if not text.strip():
        st.warning("Please enter some text above.")
        return

    # ── Tab 1: Sentiment ──────────────────────────────────────────────────────
    with tab1:
        st.markdown("""
        <div class="info-box">
        <strong>Azure AI Language — Sentiment Analysis</strong><br>
        Returns <em>document-level</em> and <em>sentence-level</em> sentiment (Positive / Negative / Neutral)
        with confidence scores. Also supports <em>opinion mining</em> for aspect-based sentiment.
        </div>
        """, unsafe_allow_html=True)

        label, scores = _get_sentiment(text)
        emoji_map = {"Positive": "😊", "Negative": "😞", "Neutral": "😐"}
        c1, c2, c3 = st.columns(3)
        c1.metric("Overall Sentiment", f"{emoji_map.get(label,'')} {label}")
        c2.metric("Compound Score",    f"{scores['compound']:+.3f}")
        c3.metric("Positive Share",    f"{scores['pos']:.0%}")

        fig, ax = plt.subplots(figsize=(8, 3))
        cats   = ["Positive", "Neutral", "Negative"]
        vals   = [scores['pos'], scores['neu'], scores['neg']]
        colors = ["#28a745", "#6c757d", "#dc3545"]
        bars = ax.barh(cats, vals, color=colors)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Score")
        ax.set_title("Sentiment Score Breakdown")
        for bar, v in zip(bars, vals):
            ax.text(v + 0.01, bar.get_y() + bar.get_height()/2, f'{v:.2f}', va='center')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        if NLTK_OK:
            try:
                sentences = sent_tokenize(text)
                if len(sentences) > 1:
                    st.markdown("#### Sentence-Level Sentiment")
                    sia2 = SentimentIntensityAnalyzer()
                    rows = []
                    for i, s in enumerate(sentences[:6]):
                        sc = sia2.polarity_scores(s)
                        c = sc['compound']
                        lbl = "Positive" if c >= 0.05 else ("Negative" if c <= -0.05 else "Neutral")
                        rows.append({"#": i+1, "Sentence": s[:70]+"…" if len(s)>70 else s,
                                     "Sentiment": lbl, "Score": f"{c:+.3f}"})
                    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            except Exception:
                pass

        st.markdown("""
        <div class="exam-tip">
        <strong>Exam Tip:</strong> Azure AI Language Sentiment Analysis supports
        <em>opinion mining</em> (set <code>opinionMining=true</code>) to identify
        sentiment toward specific aspects, e.g. "food was great, service was slow" — two aspects.
        </div>
        """, unsafe_allow_html=True)

    # ── Tab 2: Named Entities ─────────────────────────────────────────────────
    with tab2:
        st.markdown("""
        <div class="info-box">
        <strong>Azure AI Language — Named Entity Recognition (NER)</strong><br>
        Identifies and classifies entities: Person, Location, Organization, DateTime, Quantity, Event, Product.
        <em>Entity Linking</em> further maps entities to a knowledge base (Wikipedia).
        </div>
        """, unsafe_allow_html=True)

        entities = _extract_entities(text)
        if entities:
            st.dataframe(pd.DataFrame(entities), hide_index=True, use_container_width=True)
            type_counts = Counter(e["Type"] for e in entities)
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.bar(list(type_counts.keys()), list(type_counts.values()),
                   color=['#2196F3','#4CAF50','#FF5722','#9C27B0','#FF9800'][:len(type_counts)])
            ax.set_ylabel("Count")
            ax.set_title("Entity Type Distribution")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("No entities detected. Try the 'News article' sample — it has PERSON and ORG entities.")

        ref = pd.DataFrame([
            {"Category": "Person",       "Example": "Satya Nadella",       "Subcategory": "PersonType"},
            {"Category": "Location",     "Example": "Seattle, East US",    "Subcategory": "GPE, Structural"},
            {"Category": "Organization", "Example": "Microsoft, OpenAI",   "Subcategory": "—"},
            {"Category": "DateTime",     "Example": "Tuesday, 2026-05-08", "Subcategory": "Date, Time, Duration"},
            {"Category": "Quantity",     "Example": "$10 billion",         "Subcategory": "Number, Currency"},
            {"Category": "Product",      "Example": "Azure AI Services",   "Subcategory": "—"},
        ])
        st.markdown("**Azure NER Entity Categories (reference)**")
        st.dataframe(ref, hide_index=True, use_container_width=True)

    # ── Tab 3: Key Phrases ────────────────────────────────────────────────────
    with tab3:
        st.markdown("""
        <div class="info-box">
        <strong>Azure AI Language — Key Phrase Extraction</strong><br>
        Returns the main talking points as a list of strings — not individual words.
        Useful for indexing, document clustering, and search keyword generation.
        </div>
        """, unsafe_allow_html=True)

        phrases = _extract_key_phrases(text)
        if phrases:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Top Key Phrases**")
                for phrase, score in phrases[:8]:
                    st.markdown(f"- `{phrase}` &nbsp; *(score: {score:.4f})*")
            with col2:
                fig, ax = plt.subplots(figsize=(6, 4))
                names_ph = [p[0][:22] for p in phrases[:8]]
                scrs     = [p[1] for p in phrases[:8]]
                ax.barh(names_ph[::-1], scrs[::-1], color='#7b2ff7', alpha=0.8)
                ax.set_xlabel("TF-IDF Score")
                ax.set_title("Key Phrase Importance")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        else:
            st.info("Need more text. Try the 'Tech review' sample.")

        st.markdown("""
        <div class="exam-tip">
        <strong>Exam Tip:</strong> Key Phrase Extraction works best on longer texts (50+ words).
        Results are returned as a list of strings, not ranked. For ranked importance, combine
        with Custom Text Classification or use the search relevance score in Azure AI Search.
        </div>
        """, unsafe_allow_html=True)

    # ── Tab 4: Summarisation ──────────────────────────────────────────────────
    with tab4:
        st.markdown("""
        <div class="info-box">
        <strong>Azure AI Language — Text Summarisation</strong><br>
        Azure supports both <em>extractive</em> (selects original sentences) and
        <em>abstractive</em> (generates new sentences) summarisation. This demo shows extractive.
        </div>
        """, unsafe_allow_html=True)

        long_input = st.text_area("Text to summarise:", value=LONG_TEXT, height=160)
        n_sents = st.slider("Summary length (sentences):", 1, 5, 3)
        summary, _ = _summarize(long_input, n_sents)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original**")
            st.info(long_input[:500] + ("…" if len(long_input) > 500 else ""))
            st.caption(f"{len(long_input.split())} words")
        with col2:
            st.markdown("**Extractive Summary**")
            st.success(summary)
            st.caption(f"{len(summary.split())} words")

        if len(long_input.split()) > 0:
            ratio = len(summary.split()) / max(len(long_input.split()), 1)
            st.metric("Compression", f"{ratio:.0%}", delta=f"−{1-ratio:.0%}")

        st.markdown("""
        <div class="concept-box">
        <strong>Extractive vs Abstractive:</strong><br>
        &bull; <strong>Extractive</strong> — picks real sentences, deterministic, no hallucination risk<br>
        &bull; <strong>Abstractive</strong> — generates new text via LLM, can be more concise but may hallucinate<br>
        &bull; Azure AI Language supports both; abstractive uses Azure OpenAI internally
        </div>
        """, unsafe_allow_html=True)

    # ── Tab 5: Language Detection ─────────────────────────────────────────────
    with tab5:
        st.markdown("""
        <div class="info-box">
        <strong>Azure AI Language — Language Detection</strong><br>
        Detects the primary language and returns a BCP-47 code (e.g. <code>en</code>, <code>fr</code>)
        with a confidence score. Supports 120+ languages.
        </div>
        """, unsafe_allow_html=True)

        detected, lang_scores = _detect_language(text)
        c1, c2 = st.columns(2)
        c1.metric("Detected Language",  detected)
        c2.metric("Confidence",         f"{lang_scores.get(detected, 0):.0%}")

        fig, ax = plt.subplots(figsize=(8, 3))
        langs_list  = list(lang_scores.keys())
        score_vals  = [lang_scores[l] for l in langs_list]
        bar_colors  = ['#2196F3' if l == detected else '#90CAF9' for l in langs_list]
        bars = ax.bar(langs_list, score_vals, color=bar_colors)
        ax.set_ylabel("Relative Score")
        ax.set_title("Language Detection Scores")
        for bar, v in zip(bars, score_vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.004, f'{v:.2f}', ha='center', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        bcp = pd.DataFrame([
            {"Language": "English",             "BCP-47": "en",      "Script": "Latin"},
            {"Language": "French",              "BCP-47": "fr",      "Script": "Latin"},
            {"Language": "Spanish",             "BCP-47": "es",      "Script": "Latin"},
            {"Language": "German",              "BCP-47": "de",      "Script": "Latin"},
            {"Language": "Chinese (Simplified)","BCP-47": "zh-Hans", "Script": "Han"},
            {"Language": "Japanese",            "BCP-47": "ja",      "Script": "Hiragana/Kanji"},
            {"Language": "Arabic",              "BCP-47": "ar",      "Script": "Arabic"},
        ])
        st.markdown("**BCP-47 Language Codes (reference)**")
        st.dataframe(bcp, hide_index=True, use_container_width=True)

        st.markdown("""
        <div class="exam-tip">
        <strong>Exam Tip:</strong> Language Detection returns <code>detectedLanguage.iso6391Name</code>
        (BCP-47 code) and <code>confidenceScore</code> (0.0–1.0). When confidence is too low,
        the API returns <code>"(Unknown)"</code>. For mixed-language documents only the
        <em>dominant</em> language is returned.
        </div>
        """, unsafe_allow_html=True)
