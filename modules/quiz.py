import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import random
import pandas as pd
import matplotlib.pyplot as plt


# ── 25 Questions ───────────────────────────────────────────────────────────────
# Domains: Plan & Manage (5), Computer Vision (5), NLP (5), Doc Intel/Search (5), Generative AI (5)

ALL_QUESTIONS = [
    # ── Domain 1: Plan & Manage Azure AI Solutions (Q1–Q5) ──────────────────
    {
        "domain": "Plan & Manage",
        "question": "Your organisation wants to avoid embedding Azure AI service keys in application code. "
                    "Which authentication approach is recommended?",
        "options": [
            "A. Store keys in environment variables",
            "B. Use Azure Managed Identity to authenticate without credentials in code",
            "C. Hard-code keys as constants in a config file",
            "D. Pass keys as command-line arguments",
        ],
        "answer": "B",
        "explanation": (
            "Managed Identity removes the need to manage credentials. The application identity "
            "authenticates to Azure AD and obtains tokens automatically. This is the recommended "
            "approach and avoids secret leakage. Environment variables are better than hard-coding "
            "but still require manual rotation."
        ),
    },
    {
        "domain": "Plan & Manage",
        "question": "You are evaluating Azure AI Services pricing tiers for a proof-of-concept. "
                    "Which tier allows you to start with no upfront cost and low transaction limits?",
        "options": [
            "A. Standard S1",
            "B. Premium P1",
            "C. Free F0",
            "D. Developer D1",
        ],
        "answer": "C",
        "explanation": (
            "The Free (F0) tier provides a small number of free transactions per month — ideal for "
            "prototyping and POCs. The Standard (S1) tier is for production workloads. There is no "
            "P1 or D1 tier in Azure AI Services."
        ),
    },
    {
        "domain": "Plan & Manage",
        "question": "You need to store the API key for an Azure AI service securely and rotate it "
                    "without redeploying your application. What Azure service should you use?",
        "options": [
            "A. Azure App Configuration",
            "B. Azure Blob Storage",
            "C. Azure Key Vault",
            "D. Azure Service Bus",
        ],
        "answer": "C",
        "explanation": (
            "Azure Key Vault stores secrets, keys, and certificates securely. Applications can "
            "retrieve the key at runtime, and rotation is handled centrally without redeployment. "
            "App Configuration stores non-secret settings; Blob Storage is for data files."
        ),
    },
    {
        "domain": "Plan & Manage",
        "question": "You want to monitor the number of API calls, latency, and errors for an "
                    "Azure AI Services resource. Which Azure service provides these metrics?",
        "options": [
            "A. Azure Logic Apps",
            "B. Azure Monitor",
            "C. Azure DevOps",
            "D. Microsoft Sentinel",
        ],
        "answer": "B",
        "explanation": (
            "Azure Monitor collects metrics and logs from Azure resources including AI Services. "
            "You can set up alerts, dashboards, and diagnostic logs. Logic Apps is for workflow "
            "automation; Sentinel is a SIEM for security; DevOps is for CI/CD pipelines."
        ),
    },
    {
        "domain": "Plan & Manage",
        "question": "A company needs to deploy Azure AI Services in a region where data must not "
                    "leave a geographic boundary. What deployment option satisfies this requirement?",
        "options": [
            "A. Deploy in any region and enable geo-replication",
            "B. Use the global endpoint with regional routing",
            "C. Deploy the resource in a specific Azure region that meets the data residency requirement",
            "D. Use Azure Content Delivery Network",
        ],
        "answer": "C",
        "explanation": (
            "Azure AI Services are available in specific regions. Choosing a region within the "
            "required geographic boundary ensures data residency compliance. The global endpoint "
            "routes traffic worldwide, which would violate data boundary requirements."
        ),
    },

    # ── Domain 2: Computer Vision (Q6–Q10) ───────────────────────────────────
    {
        "domain": "Computer Vision",
        "question": "You need to extract text from a multi-page scanned PDF document. "
                    "Which Azure AI Vision feature should you use?",
        "options": [
            "A. Image Analysis API with 'tags' feature",
            "B. Custom Vision classification",
            "C. The Read API (Document Analysis)",
            "D. Face API",
        ],
        "answer": "C",
        "explanation": (
            "The Read API (also accessible via Azure Document Intelligence prebuilt-read) is the "
            "recommended OCR solution for multi-page documents. It handles PDFs and images, uses "
            "an async pattern, and returns structured bounding polygons with confidence per word."
        ),
    },
    {
        "domain": "Computer Vision",
        "question": "Azure AI Vision Image Analysis returns object bounding boxes. In what format "
                    "are the coordinates expressed?",
        "options": [
            "A. Absolute pixel coordinates (x, y, width, height)",
            "B. Normalised fractions of image dimensions (left, top, width, height between 0.0 and 1.0)",
            "C. Percentage coordinates from the centre of the image",
            "D. GPS latitude and longitude values",
        ],
        "answer": "B",
        "explanation": (
            "Azure AI Vision returns bounding boxes as normalised fractions of image width and height "
            "(values 0.0–1.0). Multiply by image dimensions to get pixel values. "
            "The JSON field is: boundingBox: { left, top, width, height }."
        ),
    },
    {
        "domain": "Computer Vision",
        "question": "You want to classify product images into categories that are not covered by the "
                    "standard Azure AI Vision tags. Which service allows you to train a custom image classifier?",
        "options": [
            "A. Azure AI Vision Image Analysis",
            "B. Azure Custom Vision",
            "C. Azure Face API",
            "D. Azure AI Document Intelligence",
        ],
        "answer": "B",
        "explanation": (
            "Azure Custom Vision lets you train image classifiers and object detectors with your own "
            "labelled images. Azure AI Vision Image Analysis uses prebuilt categories. "
            "Face API is for facial detection/recognition. Document Intelligence processes documents."
        ),
    },
    {
        "domain": "Computer Vision",
        "question": "What is the minimum recommended number of images per tag when training "
                    "an Azure Custom Vision classifier?",
        "options": [
            "A. 5 images",
            "B. 10 images",
            "C. 15 images",
            "D. 50 images",
        ],
        "answer": "C",
        "explanation": (
            "Microsoft recommends a minimum of 15 images per tag for Custom Vision training. "
            "More diverse images improve model accuracy. The model can train with fewer, "
            "but 15 is the official minimum recommendation for reasonable performance."
        ),
    },
    {
        "domain": "Computer Vision",
        "question": "An Azure AI Vision Image Analysis response includes which of the following top-level fields?",
        "options": [
            "A. tags, objects, caption, color, faces",
            "B. tokens, embeddings, entities, confidence",
            "C. pages, lines, words, paragraphs",
            "D. intent, entities, utterances, slots",
        ],
        "answer": "A",
        "explanation": (
            "The Image Analysis API v4.0 returns features including: tags, objects, caption, "
            "denseCaptions, color, faces, read (OCR), and smartCrops. Tags and objects are the "
            "most commonly tested. Pages/lines/words are Read API fields; tokens/embeddings are NLP."
        ),
    },

    # ── Domain 3: NLP (Q11–Q15) ───────────────────────────────────────────────
    {
        "domain": "NLP",
        "question": "You need to identify the main topics discussed in a collection of customer "
                    "support tickets to help route them. Which Azure AI Language feature is most appropriate?",
        "options": [
            "A. Sentiment Analysis",
            "B. Key Phrase Extraction",
            "C. Entity Linking",
            "D. Language Detection",
        ],
        "answer": "B",
        "explanation": (
            "Key Phrase Extraction returns the main talking points of a text as a list of strings. "
            "It is ideal for topic identification and routing. Sentiment determines positive/negative "
            "tone. Entity Linking maps mentions to Wikipedia. Language Detection identifies the language."
        ),
    },
    {
        "domain": "NLP",
        "question": "A VADER sentiment analyser returns a compound score of -0.02 for a review. "
                    "How should this be interpreted?",
        "options": [
            "A. Strongly Negative",
            "B. Mildly Negative",
            "C. Neutral",
            "D. Mildly Positive",
        ],
        "answer": "C",
        "explanation": (
            "VADER compound scores range from -1.0 to +1.0. A score between -0.05 and +0.05 is "
            "classified as Neutral. -0.02 falls in this neutral range. Positive is >= 0.05 and "
            "Negative is <= -0.05. Azure AI Language uses similar thresholds."
        ),
    },
    {
        "domain": "NLP",
        "question": "A text mentions 'Satya Nadella' and Azure AI Language identifies it as a "
                    "known entity. Which feature maps this mention to a knowledge base entry?",
        "options": [
            "A. Named Entity Recognition (NER)",
            "B. Entity Linking",
            "C. Key Phrase Extraction",
            "D. Custom Named Entity Recognition",
        ],
        "answer": "B",
        "explanation": (
            "Entity Linking identifies ambiguous entities (e.g. 'Nadella') and maps them to a "
            "knowledge base (Wikipedia) with a unique identifier and confidence score. "
            "NER only classifies entity type (PERSON, ORGANIZATION…) without linking to a source."
        ),
    },
    {
        "domain": "NLP",
        "question": "You are building a virtual assistant that determines what a user wants to do "
                    "(book a flight, check weather) and extracts parameters (destination, date). "
                    "Which Azure AI Language feature should you use?",
        "options": [
            "A. Sentiment Analysis with opinion mining",
            "B. Conversational Language Understanding (CLU)",
            "C. Custom Question Answering",
            "D. Text Analytics for health",
        ],
        "answer": "B",
        "explanation": (
            "Conversational Language Understanding (CLU) identifies the user's intent and extracts "
            "entities from natural language utterances. It is the successor to LUIS. "
            "Custom Question Answering is for FAQ-style knowledge bases. Sentiment is for tone."
        ),
    },
    {
        "domain": "NLP",
        "question": "You want to build a support chatbot that answers questions from an existing "
                    "FAQ document. Which Azure AI Language service is the best fit?",
        "options": [
            "A. Conversational Language Understanding (CLU)",
            "B. Azure OpenAI GPT-4",
            "C. Custom Question Answering",
            "D. Text Summarisation",
        ],
        "answer": "C",
        "explanation": (
            "Custom Question Answering (formerly QnA Maker) ingests FAQ documents, web pages, and "
            "structured Q&A pairs to build a question-and-answer knowledge base. "
            "CLU is for intent classification. GPT-4 could work but requires more setup and cost."
        ),
    },

    # ── Domain 4: Document Intelligence & Knowledge Mining (Q16–Q20) ─────────
    {
        "domain": "Doc Intelligence & Search",
        "question": "You need to automatically extract the vendor name, invoice date, and total "
                    "amount from thousands of supplier invoices. Which prebuilt model should you use?",
        "options": [
            "A. prebuilt-receipt",
            "B. prebuilt-invoice",
            "C. prebuilt-layout",
            "D. prebuilt-idDocument",
        ],
        "answer": "B",
        "explanation": (
            "The prebuilt-invoice model is trained on invoice documents and extracts fields including "
            "VendorName, InvoiceDate, InvoiceTotal, LineItems, and more. prebuilt-receipt is for "
            "store receipts. prebuilt-layout extracts structure but not semantic fields."
        ),
    },
    {
        "domain": "Doc Intelligence & Search",
        "question": "An Azure AI Search index contains which of the following components?",
        "options": [
            "A. Tables, views, stored procedures, and triggers",
            "B. Fields, analyzers, scoring profiles, and suggesters",
            "C. Topics, subscriptions, queues, and filters",
            "D. Datasets, pipelines, linked services, and triggers",
        ],
        "answer": "B",
        "explanation": (
            "An Azure AI Search index schema includes: fields (name, type, searchable/filterable/facetable), "
            "analyzers (language, custom tokenisation), scoring profiles (boosting rules), and "
            "suggesters (autocomplete). Tables/stored procedures are SQL; topics/queues are messaging."
        ),
    },
    {
        "domain": "Doc Intelligence & Search",
        "question": "You want to enrich documents during Azure AI Search indexing by extracting "
                    "entities and key phrases automatically. What Azure AI Search component enables this?",
        "options": [
            "A. Index",
            "B. Indexer",
            "C. Skillset",
            "D. Knowledge Store",
        ],
        "answer": "C",
        "explanation": (
            "A Skillset is a pipeline of cognitive enrichment steps (skills) attached to an indexer. "
            "Skills can call Azure AI Services to extract entities, key phrases, translate text, "
            "OCR images, or generate embeddings. The Knowledge Store persists enrichment output."
        ),
    },
    {
        "domain": "Doc Intelligence & Search",
        "question": "What is the key difference between Azure Document Intelligence and Azure AI Search?",
        "options": [
            "A. Document Intelligence trains custom classifiers; AI Search only does OCR",
            "B. Document Intelligence extracts structured fields from documents; AI Search indexes and queries content",
            "C. Azure AI Search is only for images; Document Intelligence handles text",
            "D. They are the same service with different names",
        ],
        "answer": "B",
        "explanation": (
            "Azure Document Intelligence (prebuilt or custom models) extracts structured data "
            "(fields, tables, key-value pairs) from documents. Azure AI Search indexes any content "
            "and enables full-text, vector, and faceted queries. They are often used together "
            "(Document Intelligence extracts → Search indexes the extracted data)."
        ),
    },
    {
        "domain": "Doc Intelligence & Search",
        "question": "You need to extract fields from driver's licences, passports, and national ID cards. "
                    "Which prebuilt Azure Document Intelligence model should you use?",
        "options": [
            "A. prebuilt-invoice",
            "B. prebuilt-businessCard",
            "C. prebuilt-idDocument",
            "D. prebuilt-layout",
        ],
        "answer": "C",
        "explanation": (
            "The prebuilt-idDocument model is specifically trained on identity documents including "
            "driver's licences and passports. It extracts fields such as FirstName, LastName, "
            "DateOfBirth, DocumentNumber, and CountryRegion with high confidence."
        ),
    },

    # ── Domain 5: Generative AI (Q21–Q25) ────────────────────────────────────
    {
        "domain": "Generative AI",
        "question": "Your organisation requires access to GPT-4 with enterprise data security, "
                    "compliance, and no data used for model training. Which service should you use?",
        "options": [
            "A. OpenAI API directly (openai.com)",
            "B. Azure OpenAI Service",
            "C. Azure AI Language",
            "D. Azure Machine Learning managed endpoints",
        ],
        "answer": "B",
        "explanation": (
            "Azure OpenAI Service provides GPT-4 (and other models) via Azure infrastructure with "
            "enterprise-grade security, private networking, data residency, and a guarantee that "
            "your data is not used to train Microsoft or OpenAI models. The public OpenAI API "
            "does not provide the same enterprise controls."
        ),
    },
    {
        "domain": "Generative AI",
        "question": "In a RAG architecture, what is the purpose of the retrieval step that "
                    "happens before the LLM generates a response?",
        "options": [
            "A. To translate the query into the model's language",
            "B. To fine-tune the model on user data",
            "C. To fetch relevant context from a search index to ground the LLM's answer",
            "D. To compress the user's query into fewer tokens",
        ],
        "answer": "C",
        "explanation": (
            "In RAG, the retrieval step searches an index (vector, keyword, or hybrid) for the "
            "most relevant documents and injects them into the prompt as context. This grounds "
            "the LLM response in factual, up-to-date information and reduces hallucination. "
            "Fine-tuning is a separate, expensive offline process."
        ),
    },
    {
        "domain": "Generative AI",
        "question": "Which Azure OpenAI parameter controls how random or deterministic "
                    "the generated text is?",
        "options": [
            "A. max_tokens",
            "B. frequency_penalty",
            "C. temperature",
            "D. stop",
        ],
        "answer": "C",
        "explanation": (
            "Temperature (0.0–2.0) scales the probability distribution over next tokens. "
            "Low temperature (e.g. 0.0–0.3) makes outputs more deterministic and focused. "
            "High temperature (e.g. 1.0+) increases randomness and creativity. "
            "max_tokens limits output length; stop ends generation at a token sequence."
        ),
    },
    {
        "domain": "Generative AI",
        "question": "You want the model to follow a specific output format. "
                    "You include two examples of input-output pairs in the prompt. "
                    "What prompting technique is this?",
        "options": [
            "A. Zero-shot prompting",
            "B. Chain-of-thought prompting",
            "C. Few-shot prompting",
            "D. Retrieval Augmented Generation",
        ],
        "answer": "C",
        "explanation": (
            "Few-shot prompting includes one or more (typically 2–5) example input-output pairs "
            "in the prompt to show the model the desired format and behaviour. "
            "Zero-shot has no examples. Chain-of-thought includes step-by-step reasoning. "
            "RAG retrieves external knowledge."
        ),
    },
    {
        "domain": "Generative AI",
        "question": "What is a token in the context of large language models like GPT-4?",
        "options": [
            "A. A full sentence in the input text",
            "B. A subword unit — approximately 3–4 characters or 0.75 words on average in English",
            "C. A unique word in the model's vocabulary",
            "D. An API authentication credential",
        ],
        "answer": "B",
        "explanation": (
            "LLMs tokenise text into subword units using algorithms like BPE (Byte-Pair Encoding). "
            "On average, 1 token ≈ 4 characters or 0.75 words in English. "
            "Common words are single tokens; rare words are split into multiple tokens. "
            "Pricing and context window limits are measured in tokens."
        ),
    },
]


# ── Quiz logic ─────────────────────────────────────────────────────────────────

def _init_quiz():
    questions = ALL_QUESTIONS.copy()
    random.shuffle(questions)
    st.session_state.quiz_questions    = questions
    st.session_state.quiz_score        = 0
    st.session_state.quiz_completed    = False
    st.session_state.current_question  = 0
    st.session_state.answers_given     = {}


def show():
    st.markdown('<p class="gradient-title">📝 AI-102 Practice Quiz</p>', unsafe_allow_html=True)

    if st.session_state.get("quiz_questions") is None:
        _init_quiz()

    questions = st.session_state.quiz_questions
    total = len(questions)

    # ── Completed view ────────────────────────────────────────────────────────
    if st.session_state.quiz_completed:
        score = st.session_state.quiz_score
        pct   = score / total

        st.markdown(f"## Final Score: {score} / {total} ({pct:.0%})")
        if pct >= 0.80:
            st.success("Excellent! You are well prepared for AI-102.")
        elif pct >= 0.60:
            st.warning("Good effort. Review the domains you missed and try again.")
        else:
            st.error("Keep studying. Focus on the domains with most errors.")

        # Score by domain
        domain_correct = {}
        domain_total   = {}
        for i, q in enumerate(questions):
            d = q["domain"]
            domain_total[d]   = domain_total.get(d, 0) + 1
            correct = st.session_state.answers_given.get(i, {}).get("correct", False)
            domain_correct[d] = domain_correct.get(d, 0) + (1 if correct else 0)

        rows = [{"Domain": d, "Correct": domain_correct.get(d,0),
                 "Total": domain_total[d],
                 "Score %": f"{domain_correct.get(d,0)/domain_total[d]:.0%}"}
                for d in domain_total]
        st.markdown("### Domain Breakdown")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        fig, ax = plt.subplots(figsize=(9, 4))
        domains = [r["Domain"] for r in rows]
        pcts    = [domain_correct.get(d,0) / domain_total[d] * 100 for d in domains]
        colors  = ['#4CAF50' if p >= 80 else '#FF9800' if p >= 60 else '#f44336' for p in pcts]
        ax.bar(domains, pcts, color=colors)
        ax.axhline(80, color='grey', linestyle='--', linewidth=1, label='80% target')
        ax.set_ylim(0, 110)
        ax.set_ylabel("Score %")
        ax.set_title("Performance by Domain")
        ax.legend()
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        if st.button("Retry Quiz (new question order)"):
            _init_quiz()
            st.rerun()
        return

    # ── Active question view ──────────────────────────────────────────────────
    idx = st.session_state.current_question
    q   = questions[idx]

    progress_pct = idx / total
    st.progress(progress_pct)
    st.caption(f"Question {idx + 1} of {total}  |  Domain: **{q['domain']}**")

    st.markdown(f"### {q['question']}")
    for option in q["options"]:
        st.markdown(f"- {option}")

    already_answered = idx in st.session_state.answers_given

    if not already_answered:
        choice = st.radio("Your answer:", ["A", "B", "C", "D"], key=f"q_{idx}", index=None)
        if st.button("Submit Answer", key=f"submit_{idx}"):
            if choice is None:
                st.warning("Please select an answer before submitting.")
            else:
                correct = choice == q["answer"]
                st.session_state.answers_given[idx] = {"choice": choice, "correct": correct}
                if correct:
                    st.session_state.quiz_score += 1
                st.rerun()
    else:
        data = st.session_state.answers_given[idx]
        if data["correct"]:
            st.success(f"Correct! You chose **{data['choice']}**.")
        else:
            st.error(
                f"Incorrect. You chose **{data['choice']}**, "
                f"but the correct answer is **{q['answer']}**."
            )
        st.markdown(f"**Explanation:** {q['explanation']}")

        if idx + 1 < total:
            if st.button("Next Question"):
                st.session_state.current_question += 1
                st.rerun()
        else:
            if st.button("View Results"):
                st.session_state.quiz_completed = True
                st.rerun()
