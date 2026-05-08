import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import re

try:
    from wordcloud import WordCloud
    WORDCLOUD_OK = True
except ImportError:
    WORDCLOUD_OK = False


# ── Sample corpus ──────────────────────────────────────────────────────────────

CORPUS = [
    {
        "id": 1, "title": "Azure AI Vision Overview",
        "category": "AI Services", "tags": ["vision","AI","azure"],
        "body": "Azure AI Vision provides computer vision capabilities including image analysis, OCR, "
                "object detection, and face recognition. It uses deep learning models hosted in Azure.",
    },
    {
        "id": 2, "title": "Getting Started with Azure OpenAI",
        "category": "Generative AI", "tags": ["openai","GPT","azure","LLM"],
        "body": "Azure OpenAI Service provides REST API access to OpenAI language models including "
                "GPT-4, GPT-3.5, Codex, and DALL-E. You can use it for text generation, summarisation, "
                "code completion, and embedding creation.",
    },
    {
        "id": 3, "title": "Azure AI Language — NLP Features",
        "category": "AI Services", "tags": ["NLP","language","azure"],
        "body": "Azure AI Language offers sentiment analysis, named entity recognition, key phrase "
                "extraction, summarisation, and language detection. Supports 120+ languages.",
    },
    {
        "id": 4, "title": "Knowledge Mining with Azure AI Search",
        "category": "Search", "tags": ["search","indexing","knowledge mining"],
        "body": "Azure AI Search enables full-text and vector search over large document corpora. "
                "It integrates with Azure AI services via skillsets to enrich documents during indexing.",
    },
    {
        "id": 5, "title": "Building RAG Applications",
        "category": "Generative AI", "tags": ["RAG","LLM","grounding","azure"],
        "body": "Retrieval Augmented Generation (RAG) combines a vector search index with a large "
                "language model. Azure AI Search retrieves relevant context; Azure OpenAI generates "
                "grounded answers. This reduces hallucinations significantly.",
    },
    {
        "id": 6, "title": "Azure Document Intelligence Prebuilt Models",
        "category": "AI Services", "tags": ["document","invoice","receipt","OCR"],
        "body": "Azure Document Intelligence prebuilt models extract structured data from invoices, "
                "receipts, ID documents, business cards, and more — no training required.",
    },
    {
        "id": 7, "title": "Azure AI Speech — TTS and STT",
        "category": "AI Services", "tags": ["speech","TTS","STT","azure"],
        "body": "Azure AI Speech offers text-to-speech with neural voices and SSML support, "
                "speech-to-text with real-time and batch transcription, and speech translation.",
    },
    {
        "id": 8, "title": "Responsible AI Principles",
        "category": "Ethics", "tags": ["responsible AI","fairness","transparency"],
        "body": "Microsoft's Responsible AI framework covers fairness, reliability, privacy, "
                "inclusiveness, transparency, and accountability. These principles guide AI product development.",
    },
    {
        "id": 9, "title": "Vector Embeddings and Semantic Search",
        "category": "Search", "tags": ["embeddings","vector search","semantic"],
        "body": "Vector embeddings map text to high-dimensional float vectors. Semantic search uses "
                "cosine similarity between query and document vectors to find conceptually similar content.",
    },
    {
        "id": 10, "title": "Custom Vision — Training Your Own Model",
        "category": "AI Services", "tags": ["vision","custom","training"],
        "body": "Azure Custom Vision lets you train image classifiers and object detectors with a "
                "small number of labelled images. Minimum 15 images per tag is recommended.",
    },
]


# ── Helper functions ───────────────────────────────────────────────────────────

def _build_tfidf(corpus):
    texts = [f"{d['title']} {d['body']}" for d in corpus]
    vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    mat = vec.fit_transform(texts)
    return vec, mat


def _search(query, vec, mat, corpus, top_n=5):
    q_vec = vec.transform([query])
    scores = cosine_similarity(q_vec, mat)[0]
    ranked = np.argsort(scores)[::-1][:top_n]
    results = []
    for idx in ranked:
        if scores[idx] > 0:
            results.append({
                "Rank": len(results) + 1,
                "Title": corpus[idx]["title"],
                "Category": corpus[idx]["category"],
                "Score": round(float(scores[idx]), 4),
            })
    return results


def _similarity_matrix(vec, mat, corpus):
    sim = cosine_similarity(mat)
    titles = [d["title"][:25] for d in corpus]
    return sim, titles


# ── Main page ──────────────────────────────────────────────────────────────────

def show():
    st.markdown('<p class="gradient-title">🔎 Knowledge Mining</p>', unsafe_allow_html=True)
    st.markdown("### Azure AI Search & Knowledge Mining — Local TF-IDF Simulation")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Full-Text Search", "🧮 Vector Search", "🗂️ Faceted Navigation", "☁️ Word Cloud",
    ])

    vec, mat = _build_tfidf(CORPUS)

    # ── Tab 1: Full-Text Search ───────────────────────────────────────────────
    with tab1:
        st.markdown("""
        <div class="info-box">
        <strong>Azure AI Search — Full-Text Search</strong><br>
        Uses <em>BM25</em> ranking (default) over an inverted index. Documents are indexed with
        analyzers (tokenisation, stemming) and scored based on term frequency and inverse document frequency.
        </div>
        """, unsafe_allow_html=True)

        query = st.text_input("Search query:", value="GPT language model Azure OpenAI")
        top_n = st.slider("Results:", 1, 10, 5)

        results = _search(query, vec, mat, CORPUS, top_n=top_n)
        if results:
            df = pd.DataFrame(results)
            st.dataframe(df, hide_index=True, use_container_width=True)

            fig, ax = plt.subplots(figsize=(9, 4))
            ax.barh([r["Title"][:35] for r in results][::-1],
                    [r["Score"] for r in results][::-1], color='#7b2ff7', alpha=0.85)
            ax.set_xlabel("Relevance Score (TF-IDF cosine)")
            ax.set_title(f'Search Results for: "{query}"')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("No results found. Try 'GPT' or 'vision' or 'search'.")

        st.markdown("#### Azure AI Search Index Structure")
        st.code("""
{
  "name": "knowledge-index",
  "fields": [
    { "name": "id",       "type": "Edm.String",     "key": true  },
    { "name": "title",    "type": "Edm.String",     "searchable": true, "analyzer": "en.microsoft" },
    { "name": "body",     "type": "Edm.String",     "searchable": true },
    { "name": "category", "type": "Edm.String",     "filterable": true, "facetable": true },
    { "name": "tags",     "type": "Collection(Edm.String)", "filterable": true, "facetable": true },
    { "name": "vector",   "type": "Collection(Edm.Single)", "dimensions": 1536,
                           "vectorSearchProfile": "hnsw-profile" }
  ]
}
        """, language="json")

    # ── Tab 2: Vector Search ──────────────────────────────────────────────────
    with tab2:
        st.markdown("""
        <div class="info-box">
        <strong>Azure AI Search — Vector Search</strong><br>
        Semantic / vector search computes <em>cosine similarity</em> between a query embedding
        and document embeddings. Azure AI Search supports HNSW (Hierarchical Navigable Small World)
        for approximate nearest-neighbour search at scale.
        </div>
        """, unsafe_allow_html=True)

        sim_matrix, titles = _similarity_matrix(vec, mat, CORPUS)

        st.markdown("#### Document Similarity Heatmap")
        fig, ax = plt.subplots(figsize=(11, 8))
        sns.heatmap(sim_matrix, xticklabels=titles, yticklabels=titles,
                    cmap='YlOrRd', ax=ax, annot=False, fmt='.2f',
                    linewidths=0.5, vmin=0, vmax=1)
        ax.set_title("Cosine Similarity Between All Documents", fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        v_query = st.selectbox(
            "Find documents similar to:",
            [d["title"] for d in CORPUS]
        )
        q_idx = next(i for i, d in enumerate(CORPUS) if d["title"] == v_query)
        sims = [(CORPUS[i]["title"], float(sim_matrix[q_idx, i]))
                for i in range(len(CORPUS)) if i != q_idx]
        sims.sort(key=lambda x: -x[1])

        st.markdown("**Most Similar Documents**")
        sim_df = pd.DataFrame([{"Title": t, "Similarity": f"{s:.3f}"} for t, s in sims[:5]])
        st.dataframe(sim_df, hide_index=True, use_container_width=True)

        st.markdown("""
        <div class="exam-tip">
        <strong>Exam Tip:</strong> Azure AI Search supports:<br>
        &bull; <strong>Pure Vector</strong> — semantic similarity via embeddings<br>
        &bull; <strong>Hybrid</strong> (recommended) — BM25 + vector combined with RRF (Reciprocal Rank Fusion)<br>
        &bull; <strong>Semantic Re-ranking</strong> (premium feature) — re-ranks results with a language model
        </div>
        """, unsafe_allow_html=True)

    # ── Tab 3: Faceted Navigation ─────────────────────────────────────────────
    with tab3:
        st.markdown("""
        <div class="info-box">
        <strong>Azure AI Search — Facets & Filters</strong><br>
        Faceted navigation lets users drill down results by category or tag.
        Define fields as <code>facetable: true</code> in the index schema.
        Facets are returned alongside search results with document counts.
        </div>
        """, unsafe_allow_html=True)

        categories = sorted(set(d["category"] for d in CORPUS))
        all_tags   = sorted(set(t for d in CORPUS for t in d["tags"]))

        sel_cats = st.multiselect("Filter by Category:", categories, default=categories)
        sel_tags = st.multiselect("Filter by Tag:", all_tags[:8])

        filtered = [
            d for d in CORPUS
            if d["category"] in sel_cats
            and (not sel_tags or any(t in d["tags"] for t in sel_tags))
        ]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Matching Documents", len(filtered))
            if filtered:
                fd = pd.DataFrame([{"Title": d["title"], "Category": d["category"]}
                                   for d in filtered])
                st.dataframe(fd, hide_index=True, use_container_width=True)

        with col2:
            cat_counts = Counter(d["category"] for d in filtered)
            if cat_counts:
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.barh(list(cat_counts.keys()), list(cat_counts.values()),
                        color='#2196F3', alpha=0.85)
                ax.set_xlabel("Document Count")
                ax.set_title("Facet Counts by Category")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        st.markdown("""
        <div class="concept-box">
        <strong>Azure AI Search Skillsets:</strong><br>
        Skillsets run during indexing to enrich documents:<br>
        &bull; <strong>EntityRecognitionSkill</strong> — extract entities<br>
        &bull; <strong>KeyPhraseExtractionSkill</strong> — index key phrases<br>
        &bull; <strong>OCRSkill</strong> — extract text from images<br>
        &bull; <strong>MergeSkill</strong> — combine fields<br>
        &bull; <strong>AzureOpenAIEmbeddingSkill</strong> — generate embeddings
        </div>
        """, unsafe_allow_html=True)

    # ── Tab 4: Word Cloud ─────────────────────────────────────────────────────
    with tab4:
        st.markdown("""
        <div class="info-box">
        <strong>Knowledge Store — Corpus Visualisation</strong><br>
        A <em>knowledge store</em> in Azure AI Search persists enriched data (entities, key phrases,
        images) to Azure Blob Storage or Azure Table Storage for downstream analytics.
        Word clouds are a quick visualisation of the most frequent terms in your corpus.
        </div>
        """, unsafe_allow_html=True)

        wc_cats = st.multiselect("Categories to include:", categories, default=categories)
        corpus_text = " ".join(
            d["body"] for d in CORPUS if d["category"] in wc_cats
        )

        if WORDCLOUD_OK and corpus_text.strip():
            wc = WordCloud(
                width=900, height=400,
                background_color='white',
                colormap='viridis',
                stopwords=set(['the','a','an','is','it','in','on','and','or','to','for',
                               'of','with','this','that','are','its','be','can','by']),
                max_words=80,
            ).generate(corpus_text)

            fig, ax = plt.subplots(figsize=(11, 5))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            ax.set_title("Corpus Word Cloud", fontsize=14)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        elif not WORDCLOUD_OK:
            st.warning("wordcloud package not installed. Install it with: `pip install wordcloud`")
            freq = Counter(re.sub(r'[^a-zA-Z ]', '', corpus_text.lower()).split())
            for stopword in ['the','a','an','is','it','in','on','and','or','to','for','of','with']:
                freq.pop(stopword, None)
            top = freq.most_common(20)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar([w for w, _ in top], [c for _, c in top], color='#7b2ff7', alpha=0.85)
            ax.set_xlabel("Term")
            ax.set_ylabel("Frequency")
            ax.set_title("Top 20 Terms (word cloud fallback)")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("#### Azure Knowledge Store")
        st.code("""
"knowledgeStore": {
  "storageConnectionString": "DefaultEndpointsProtocol=...",
  "projections": [
    {
      "tables": [
        { "tableName": "entities",   "source": "/document/entities/*"   },
        { "tableName": "keyPhrases", "source": "/document/keyPhrases/*" }
      ],
      "objects": [
        { "storageContainer": "enriched-docs", "source": "/document" }
      ]
    }
  ]
}
        """, language="json")

        st.markdown("""
        <div class="exam-tip">
        <strong>Exam Tip:</strong> Azure AI Search components:<br>
        &bull; <strong>Data Source</strong> — where content lives (Blob, SQL, Cosmos DB…)<br>
        &bull; <strong>Index</strong> — schema of searchable fields<br>
        &bull; <strong>Indexer</strong> — pipeline that pulls from source → enriches → populates index<br>
        &bull; <strong>Skillset</strong> — cognitive enrichment steps (OCR, NER, embedding…)<br>
        &bull; <strong>Knowledge Store</strong> — optional output for enriched data
        </div>
        """, unsafe_allow_html=True)
