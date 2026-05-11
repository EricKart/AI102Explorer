import warnings
warnings.filterwarnings("ignore")

import time
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════════════════
# DATA — Round 1: Service Matcher
# ═══════════════════════════════════════════════════════════════════════════════

_PLACEHOLDER = "(select a service)"

SERVICE_OPTIONS = [
    "Azure AI Document Intelligence",
    "Azure AI Speech",
    "Azure AI Language",
    "Azure AI Content Safety",
    "Azure AI Vision / Custom Vision",
    "Azure AI Search",
    "Azure Key Vault",
    "Azure AI Translator",
]

# One hint per scenario — shown in a collapsible expander so students can try first.
SCENARIO_HINTS = [
    "Which service has a prebuilt-invoice model that extracts structured fields with zero custom training?",
    "This is about WHO is speaking, not WHAT they say. Look for a speaker verification / identification feature.",
    "The keywords are 'detect and redact PHI'. Which NLP service has a Text Analytics for Health variant?",
    "Which service was built specifically for detecting hate, violence, sexual content, and self-harm?",
    "Your defect categories aren't in any standard catalogue — you need to train on your own labelled images.",
    "Which service supports hybrid search — keyword + vector/semantic — with a semantic ranker at scale?",
    "The clue is 'centralise secret storage' and 'automatic rotation without redeployment'.",
    "Which service handles neural machine translation for 100+ languages in a single REST call?",
]

SCENARIOS = [
    {
        "scenario": (
            "A law firm wants to automatically extract vendor names, dates, and total amounts "
            "from thousands of supplier invoices without writing custom parsing code."
        ),
        "correct": "Azure AI Document Intelligence",
        "explanation": (
            "The <code>prebuilt-invoice</code> model in Azure AI Document Intelligence extracts structured "
            "fields including VendorName, InvoiceDate, InvoiceTotal, and line items with no custom parsing required."
        ),
    },
    {
        "scenario": (
            "A call centre wants to use a customer's voice as a biometric factor to verify their "
            "identity before granting account access."
        ),
        "correct": "Azure AI Speech",
        "explanation": (
            "Azure AI Speech includes Speaker Recognition with two modes: Speaker Verification "
            "(1:1 — confirm this is the claimed person) and Speaker Identification (1:N — identify "
            "who this person is from a group). This is a distinct capability from transcription."
        ),
    },
    {
        "scenario": (
            "A hospital needs to automatically detect and redact personally identifiable health "
            "information (PHI) — names, dates of birth, addresses — from clinical notes before "
            "sharing them for research."
        ),
        "correct": "Azure AI Language",
        "explanation": (
            "Azure AI Language's PII Extraction feature (and its Text Analytics for Health variant) "
            "identifies and can redact PHI entities from free text. This is the correct tool for "
            "structured entity detection and redaction."
        ),
    },
    {
        "scenario": (
            "An e-commerce platform allows user-generated reviews with optional image uploads. "
            "The platform needs to detect sexually explicit, violent, or hateful content before "
            "it goes live."
        ),
        "correct": "Azure AI Content Safety",
        "explanation": (
            "Azure AI Content Safety is specifically designed to detect harmful content "
            "(hate, violence, sexual content, self-harm) in both text and images, returning "
            "severity scores per category. It is distinct from general image analysis."
        ),
    },
    {
        "scenario": (
            "A manufacturer wants to detect surface defects on products moving along a conveyor "
            "belt using cameras. The defect categories are specific to their product line and "
            "are not covered by standard image analysis tags."
        ),
        "correct": "Azure AI Vision / Custom Vision",
        "explanation": (
            "Azure Custom Vision lets you train custom object detectors and classifiers on your "
            "own labelled images. For domain-specific categories like manufacturing defects, "
            "Custom Vision is the right choice over the prebuilt Image Analysis API."
        ),
    },
    {
        "scenario": (
            "A retailer has a 50,000-item product catalogue. They want customers to search it "
            "using natural language like 'lightweight waterproof jacket for hiking' and get "
            "semantically relevant results."
        ),
        "correct": "Azure AI Search",
        "explanation": (
            "Azure AI Search supports hybrid search combining keyword, vector (semantic), and "
            "faceted filtering. The semantic ranker understands natural language intent and "
            "re-ranks results — ideal for product catalogue search at scale."
        ),
    },
    {
        "scenario": (
            "A development team has hard-coded Azure AI service keys in their application config "
            "files. They need to centralise secret storage, enable automatic rotation, and avoid "
            "redeployment when keys change."
        ),
        "correct": "Azure Key Vault",
        "explanation": (
            "Azure Key Vault stores secrets, keys, and certificates securely. Applications "
            "retrieve secrets at runtime via the Key Vault SDK (ideally using Managed Identity). "
            "Key rotation is centralised and never requires code or deployment changes."
        ),
    },
    {
        "scenario": (
            "A global e-commerce platform needs to translate product titles and descriptions "
            "from English into 50 languages in real time as customers browse."
        ),
        "correct": "Azure AI Translator",
        "explanation": (
            "Azure AI Translator provides neural machine translation for 100+ languages. "
            "It supports real-time translation via REST and can target multiple languages "
            "in a single API call — ideal for dynamic, high-volume translation workloads."
        ),
    },
]

# ═══════════════════════════════════════════════════════════════════════════════
# DATA — Round 2: Spot the Bug
# ═══════════════════════════════════════════════════════════════════════════════

BUGS = [
    {
        "title": "Bug #1 — Speech SSML Prosody Rate",
        "language": "xml",
        "code": (
            '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">\n'
            '  <voice name="en-US-JennyNeural">\n'
            '    <prosody rate="very-fast" pitch="+5%">\n'
            '      Welcome to the Azure AI Speech service demonstration.\n'
            '    </prosody>\n'
            '  </voice>\n'
            '</speak>'
        ),
        "question": "This SSML snippet will cause the Speech service to return an error. What is the bug?",
        "options": [
            "A. The <speak> tag is missing the required xmlns:mstts namespace attribute",
            'B. "very-fast" is not a valid prosody rate — valid values are: x-slow, slow, medium, fast, x-fast',
            "C. en-US-JennyNeural is not a valid Azure neural voice name",
            'D. The version attribute should be "2.0" not "1.0"',
        ],
        "answer": "B",
        "explanation": (
            "The SSML `prosody` `rate` attribute only accepts predefined values: "
            "x-slow, slow, medium, fast, x-fast — or a relative percentage like '+20%'. "
            "'very-fast' is not in the valid set and will cause a synthesis error. "
            "JennyNeural is a valid voice; version 1.0 is correct; the xmlns attribute is present."
        ),
    },
    {
        "title": "Bug #2 — Azure AI Language API Endpoint",
        "language": "python",
        "code": (
            "import requests\n\n"
            'endpoint = "https://myaccount.cognitiveservices.azure.com"\n'
            'key = "YOUR_API_KEY"\n\n'
            "response = requests.post(\n"
            '    f"{endpoint}/text/analytics/v3.0/sentiment",\n'
            '    headers={"Ocp-Apim-Subscription-Key": key},\n'
            "    json={\n"
            '        "documents": [\n'
            '            {"id": "1", "text": "Azure AI Language is excellent!"}\n'
            "        ]\n"
            "    }\n"
            ")\n"
            "print(response.json())"
        ),
        "question": "This code will receive a deprecation error or unexpected response. What is the bug?",
        "options": [
            'A. The header should be "Authorization: Bearer {key}" not "Ocp-Apim-Subscription-Key"',
            "B. The path /text/analytics/v3.0/sentiment is the deprecated Text Analytics API — "
            "the current unified endpoint is /language/:analyze-text with a different request body",
            'C. Each document in the array must include a "language" field',
            "D. The base URL must end in .openai.azure.com not .cognitiveservices.azure.com",
        ],
        "answer": "B",
        "explanation": (
            "The Text Analytics v3.0 REST API is deprecated and has been replaced by the unified "
            "Azure AI Language service. The correct modern endpoint is "
            "POST /language/:analyze-text?api-version=2023-04-01 with a body that includes "
            "'kind': 'SentimentAnalysis' and an 'analysisInput' wrapper. "
            "The Ocp-Apim-Subscription-Key header is correct; the language field is optional; "
            "the .cognitiveservices.azure.com base URL is correct."
        ),
    },
    {
        "title": "Bug #3 — Azure AI Search Index Schema",
        "language": "json",
        "code": (
            "{\n"
            '  "name": "products-index",\n'
            '  "fields": [\n'
            "    {\n"
            '      "name": "id",\n'
            '      "type": "Edm.String",\n'
            '      "searchable": false,\n'
            '      "filterable": true,\n'
            '      "retrievable": true\n'
            "    },\n"
            "    {\n"
            '      "name": "title",\n'
            '      "type": "Edm.String",\n'
            '      "searchable": true,\n'
            '      "filterable": true\n'
            "    },\n"
            "    {\n"
            '      "name": "description",\n'
            '      "type": "Edm.String",\n'
            '      "searchable": true,\n'
            '      "retrievable": true\n'
            "    },\n"
            "    {\n"
            '      "name": "category",\n'
            '      "type": "Edm.String",\n'
            '      "filterable": true,\n'
            '      "facetable": true\n'
            "    }\n"
            "  ]\n"
            "}"
        ),
        "question": "This index definition will be rejected by the Azure AI Search REST API. What is the bug?",
        "options": [
            'A. The index is missing a required "suggesters" array',
            'B. The "id" field is missing "key": true — every index must have exactly one key field',
            'C. The "description" field must also have "filterable": true to enable full-text search',
            'D. The field type "Edm.String" is deprecated — use "Edm.Text" instead',
        ],
        "answer": "B",
        "explanation": (
            'Every Azure AI Search index requires exactly one field marked with "key": true. '
            "This field (type Edm.String) serves as the unique document identifier. "
            "Without a key field the index creation request fails with a 400 validation error. "
            "Suggesters are optional; filterable and searchable are independent; "
            "Edm.String is the correct and current type."
        ),
    },
    {
        "title": "Bug #4 — Custom Vision Prediction Client",
        "language": "python",
        "code": (
            "from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient\n"
            "from msrest.authentication import ApiKeyCredentials\n\n"
            'TRAINING_KEY = "abc123trainkey..."\n'
            'PREDICTION_ENDPOINT = "https://myaccount.cognitiveservices.azure.com/"\n\n'
            'credentials = ApiKeyCredentials(in_headers={"Training-key": TRAINING_KEY})\n'
            "predictor = CustomVisionPredictionClient(\n"
            "    endpoint=PREDICTION_ENDPOINT,\n"
            "    credentials=credentials\n"
            ")\n\n"
            'with open("test_image.jpg", "rb") as image_data:\n'
            "    results = predictor.classify_image(project_id, published_name, image_data)\n\n"
            "for pred in results.predictions:\n"
            '    print(f"{pred.tag_name}: {pred.probability:.2%}")'
        ),
        "question": "This code will fail with a 401 Unauthorized error. What is the bug?",
        "options": [
            "A. classify_image should be classify_image_with_no_store to avoid storing test images",
            "B. CustomVisionPredictionClient should be imported from the .training module",
            'C. The prediction client requires the "Prediction-key" header with the prediction '
            "resource key — not the training key",
            "D. project_id must be the project name string, not a GUID",
        ],
        "answer": "C",
        "explanation": (
            "Custom Vision has two separate Azure resources: a Training resource (for building models) "
            "and a Prediction resource (for running inference). The prediction client requires "
            "the Prediction resource key in the 'Prediction-key' header and must target the "
            "Prediction endpoint URL. Using the Training key causes 401 Unauthorized. "
            "classify_image is correct; the import is correct; project_id is a GUID."
        ),
    },
]

# ═══════════════════════════════════════════════════════════════════════════════
# DATA — Round 3: Architect the Solution
# ═══════════════════════════════════════════════════════════════════════════════

ARCHITECTURES = [
    {
        "title": "Scenario 1 — Healthcare Document Pipeline",
        "description": (
            "A hospital needs to: **(1)** extract structured data from handwritten patient intake forms, "
            "**(2)** redact all PHI before storing, **(3)** translate the cleaned records into Spanish "
            "for a partner clinic, and **(4)** ensure no API keys appear in application code. "
            "Which architecture satisfies all four requirements?"
        ),
        "options": [
            "A. Document Intelligence → AI Language (PII Extraction) → AI Translator "
            "| Managed Identity + Key Vault",
            "B. AI Vision (Read API) → AI Content Safety → AI Speech (TTS) "
            "| Environment variables",
            "C. Document Intelligence → AI Language (Sentiment Analysis) → Azure OpenAI "
            "| Service Principal credentials in a config file",
            "D. Azure Form Recognizer v2 → Text Analytics v3.0 → Translator v2 "
            "| Connection strings embedded in code",
        ],
        "answer": "A",
        "explanation": (
            "**A is correct:** Document Intelligence extracts structured fields from handwritten forms; "
            "AI Language PII Extraction detects and redacts PHI entities; AI Translator handles "
            "Spanish translation; Managed Identity removes credentials from code and Key Vault "
            "provides centralised, rotatable secret management.\n\n"
            "**B** uses the wrong services — Read API lacks semantic field extraction; "
            "Content Safety is for harm detection, not PHI redaction.\n\n"
            "**C** uses Sentiment Analysis (which scores tone, not redacts entities) and "
            "embeds credentials in a config file — a security violation.\n\n"
            "**D** uses *three deprecated services*: Form Recognizer v2 (now Document Intelligence), "
            "Text Analytics v3.0 (now Azure AI Language), and Translator v2 — "
            "and hard-codes connection strings in source code."
        ),
    },
    {
        "title": "Scenario 2 — Voice-Enabled Customer Service Bot",
        "description": (
            "A telecoms company wants a customer service bot that: **(1)** accepts voice input, "
            "**(2)** understands what the customer wants to do (pay a bill, upgrade a plan, report an outage) "
            "and extracts details like account numbers, "
            "**(3)** searches a knowledge base for relevant answers using semantic understanding, "
            "and **(4)** responds with a natural-sounding voice. What is the correct service pipeline?"
        ),
        "options": [
            "A. Speech-to-Text → Conversational Language Understanding (CLU) → "
            "Azure AI Search (semantic/vector) → Text-to-Speech",
            "B. Speech-to-Text → LUIS → Azure Cosmos DB → Text-to-Speech",
            "C. Azure Bot Service → Text Analytics (Sentiment) → Azure SQL → Azure Speech (TTS)",
            "D. Speech-to-Text → Azure OpenAI GPT-4 (direct chat) → Bing Search API → Text-to-Speech",
        ],
        "answer": "A",
        "explanation": (
            "**A is correct:** Speech-to-Text transcribes voice; CLU (the successor to LUIS) identifies "
            "intents and extracts entities; Azure AI Search with semantic ranking retrieves relevant "
            "knowledge base content; Text-to-Speech produces natural voice output.\n\n"
            "**B** uses **LUIS, which is deprecated** (replaced by CLU). Cosmos DB is a document "
            "database, not a semantic search service.\n\n"
            "**C** uses Sentiment Analysis (which identifies emotional tone, not intent) and "
            "SQL (not appropriate for semantic knowledge retrieval).\n\n"
            "**D** using GPT-4 direct for intent extraction and Bing Search bypasses purpose-built "
            "Azure AI services and introduces unnecessary complexity, latency, and cost."
        ),
    },
    {
        "title": "Scenario 3 — Content Moderation at Scale",
        "description": (
            "A social media platform needs to: **(1)** scan all uploaded images for violence and explicit content, "
            "**(2)** scan post captions for hate speech, **(3)** route flagged items to a human review queue, "
            "and **(4)** monitor API call latency and error rates with alerting. "
            "Which combination of Azure services is correct?"
        ),
        "options": [
            "A. AI Content Safety (image) → AI Content Safety (text) → "
            "Azure Service Bus + Logic Apps → Azure Monitor",
            "B. Azure Custom Vision → AI Language (Sentiment) → "
            "Azure Event Grid → Azure Application Insights",
            "C. AI Vision (Image Analysis) → AI Language (Key Phrase Extraction) → "
            "Azure Queue Storage → Azure Log Analytics",
            "D. AI Content Safety (image) → AI Language (NER) → "
            "Azure Event Hubs → Microsoft Sentinel",
        ],
        "answer": "A",
        "explanation": (
            "**A is correct:** Azure AI Content Safety provides dedicated image and text harm detection "
            "with severity scores for violence, sexual content, hate speech, and self-harm — "
            "exactly what's needed for both image and caption moderation. "
            "Service Bus provides reliable message queuing for the human review workflow; "
            "Logic Apps orchestrates routing. Azure Monitor provides dashboards and alerts "
            "for API latency and error rates.\n\n"
            "**B** uses Custom Vision (object classification, not harm detection) and "
            "Sentiment Analysis (tone scoring, not hate speech detection).\n\n"
            "**C** uses Key Phrase Extraction (not moderation) and Queue Storage lacks "
            "the workflow routing capabilities needed.\n\n"
            "**D** uses NER (entity recognition, not harm detection) and Sentinel is a SIEM "
            "for security threat management, not API performance monitoring."
        ),
    },
]

# ═══════════════════════════════════════════════════════════════════════════════
# DATA — Round 4: Speed Quiz Blitz (10 questions, 2 per domain)
# ═══════════════════════════════════════════════════════════════════════════════

SPEED_QUESTIONS = [
    # ── Plan & Manage ──────────────────────────────────────────────────────────
    {
        "domain": "Plan & Manage",
        "question": (
            "An Azure AI Services resource is returning HTTP 429 errors on every request. "
            "What does this indicate?"
        ),
        "options": [
            "A. The API key is invalid or has expired",
            "B. The JSON request body is malformed",
            "C. The rate limit or quota for this resource tier has been exceeded",
            "D. The service is currently unavailable in this Azure region",
        ],
        "answer": "C",
        "explanation": (
            "HTTP 429 is 'Too Many Requests'. Azure AI Services enforce per-minute and per-second "
            "transaction limits based on your pricing tier. Solutions include requesting a quota "
            "increase, implementing retry logic with exponential backoff, or upgrading the tier."
        ),
    },
    {
        "domain": "Plan & Manage",
        "question": (
            "Three separate microservices each need to call Azure AI Services with distinct "
            "identities and granular RBAC permissions. What is the recommended approach?"
        ),
        "options": [
            "A. Create one shared service principal and distribute credentials to all three services",
            "B. Assign a separate system-assigned Managed Identity to each microservice",
            "C. Store a single API key in Key Vault and share it across all three services",
            "D. Use a shared service account with username/password stored in App Configuration",
        ],
        "answer": "B",
        "explanation": (
            "System-assigned Managed Identities are tied to the resource lifecycle, require no "
            "credential management, and each can be granted distinct RBAC roles (least privilege). "
            "Sharing a single identity or key across services violates least-privilege and "
            "makes audit trails meaningless."
        ),
    },
    # ── Computer Vision ────────────────────────────────────────────────────────
    {
        "domain": "Computer Vision",
        "question": (
            "Azure AI Vision 4.0 (Florence model) consolidates which capabilities into a single API call?"
        ),
        "options": [
            "A. OCR/Read, dense captions, object detection, tagging, smart cropping, and background removal",
            "B. Face recognition, emotion detection, and speaker identification",
            "C. Document field extraction, table recognition, and signature detection",
            "D. Intent classification, entity extraction, and language detection",
        ],
        "answer": "A",
        "explanation": (
            "Azure AI Vision 4.0 (Florence-based) unified OCR (Read), dense captioning, object "
            "detection, tagging, smart cropping, background removal, and people detection into a "
            "single Image Analysis API. Face recognition is a separate service; document field "
            "extraction is Document Intelligence; intent and NLP belong to Azure AI Language."
        ),
    },
    {
        "domain": "Computer Vision",
        "question": (
            "A Custom Vision model returns a prediction probability of 0.35 for the tag 'defective'. "
            "What does this mean?"
        ),
        "options": [
            "A. The image has a 35% chance of crashing the model during inference",
            "B. The model is 35% confident the image belongs to the 'defective' class",
            "C. Only 35% of training images for this class were correctly labelled",
            "D. At least 35 more training images are required before this prediction is valid",
        ],
        "answer": "B",
        "explanation": (
            "Prediction probability scores are confidence values between 0.0 and 1.0. "
            "A score of 0.35 means the model assigns 35% confidence that the image matches "
            "the 'defective' tag. You typically set a probability threshold (e.g. 0.70) and "
            "only act on predictions above it."
        ),
    },
    # ── NLP ────────────────────────────────────────────────────────────────────
    {
        "domain": "NLP",
        "question": (
            "Azure AI Language returns a document-level sentiment of 'mixed'. What does this mean?"
        ),
        "options": [
            "A. The language detection step was inconclusive",
            "B. The document contains both positive and negative sentences",
            "C. The confidence scores for positive and negative sentiment are exactly equal",
            "D. The document is written in multiple languages",
        ],
        "answer": "B",
        "explanation": (
            "'Mixed' sentiment means the document-level analysis detected both positive and negative "
            "sentence-level sentiments within the same document. This is distinct from 'neutral' "
            "(neither strongly positive nor negative overall). Azure AI Language returns both "
            "document-level and sentence-level sentiment in a single response."
        ),
    },
    {
        "domain": "NLP",
        "question": (
            "Conversational Language Understanding (CLU) is the direct successor to which "
            "deprecated Azure AI service?"
        ),
        "options": [
            "A. Custom Question Answering (QnA Maker v1)",
            "B. Language Understanding Intelligent Service (LUIS)",
            "C. Text Analytics v3.0",
            "D. Azure Bot Service v3",
        ],
        "answer": "B",
        "explanation": (
            "LUIS (Language Understanding Intelligent Service) was deprecated and replaced by "
            "Conversational Language Understanding (CLU) within Azure AI Language. "
            "CLU supports the same intent and entity extraction workflows with a modernised API, "
            "improved accuracy, and integration with the unified Language service."
        ),
    },
    # ── Doc Intelligence & Search ──────────────────────────────────────────────
    {
        "domain": "Doc Intelligence & Search",
        "question": (
            "Adding vector search to an Azure AI Search index requires what additional field "
            "configuration compared to keyword-only search?"
        ),
        "options": [
            "A. A field with 'sortable': true and numeric type Edm.Double",
            "B. A field of type Collection(Edm.Single) with a 'dimensions' property matching "
            "the embedding model's output size",
            "C. A field with both 'facetable': true and 'filterable': true enabled simultaneously",
            "D. A field with 'analyzer': 'standard' to enable BPE tokenisation",
        ],
        "answer": "B",
        "explanation": (
            "Vector fields in Azure AI Search use type Collection(Edm.Single) and require a "
            "'dimensions' property (e.g. 1536 for text-embedding-ada-002) plus a "
            "'vectorSearchProfile' referencing an HNSW or KNN algorithm configuration. "
            "Standard text analyzers are for keyword search only and do not apply to vector fields."
        ),
    },
    {
        "domain": "Doc Intelligence & Search",
        "question": (
            "Which Azure AI Document Intelligence prebuilt model is designed specifically for "
            "W-2 wage statements and payroll documents?"
        ),
        "options": [
            "A. prebuilt-invoice",
            "B. prebuilt-receipt",
            "C. prebuilt-tax.us.w2",
            "D. prebuilt-idDocument",
        ],
        "answer": "C",
        "explanation": (
            "Azure Document Intelligence includes US tax-specific prebuilt models including "
            "prebuilt-tax.us.w2 (W-2 wage statements) and prebuilt-tax.us.1099 variants. "
            "prebuilt-invoice handles vendor invoices; prebuilt-receipt handles retail receipts; "
            "prebuilt-idDocument handles passports and driver's licences."
        ),
    },
    # ── Generative AI ──────────────────────────────────────────────────────────
    {
        "domain": "Generative AI",
        "question": (
            "In a RAG pipeline, what is the primary purpose of the retrieval step before "
            "the LLM generates a response?"
        ),
        "options": [
            "A. To compress the user's query into fewer tokens before sending to the model",
            "B. To fine-tune the model's weights on the most recent domain documents",
            "C. To fetch relevant context from a knowledge base and inject it into the prompt "
            "to ground the response",
            "D. To validate the query against a content blocklist before processing",
        ],
        "answer": "C",
        "explanation": (
            "In RAG, the retrieval step searches a vector or keyword index to find the most "
            "relevant documents and injects them as context into the LLM prompt. This grounds the "
            "model's response in factual, domain-specific, or up-to-date information and reduces "
            "hallucination — without requiring expensive model fine-tuning."
        ),
    },
    {
        "domain": "Generative AI",
        "question": (
            "Which Azure service adds rate limiting, content filtering, load balancing, and cost "
            "tracking across multiple Azure OpenAI deployments in a production environment?"
        ),
        "options": [
            "A. Azure API Management (APIM)",
            "B. Azure Application Gateway",
            "C. Azure Front Door",
            "D. Azure Service Bus",
        ],
        "answer": "A",
        "explanation": (
            "Azure API Management provides an AI Gateway layer for Azure OpenAI and other AI "
            "services, supporting token-based rate limiting, prompt/response content filtering, "
            "load balancing across multiple OpenAI endpoints, semantic caching, and usage analytics "
            "— all without modifying application code."
        ),
    },
]

# ═══════════════════════════════════════════════════════════════════════════════
# DATA — Round 5: Myth Busters (True / False)
# ═══════════════════════════════════════════════════════════════════════════════

MYTHS = [
    {
        "statement": (
            "LUIS (Language Understanding Intelligent Service) is Microsoft's current "
            "recommended service for building intent-based conversational AI."
        ),
        "answer": False,
        "explanation": (
            "LUIS is <strong>deprecated</strong>. It has been replaced by Conversational Language "
            "Understanding (CLU) within Azure AI Language. All new conversational AI projects "
            "should use CLU."
        ),
    },
    {
        "statement": "Azure AI Content Safety can analyse both text and images for harmful content.",
        "answer": True,
        "explanation": (
            "Azure AI Content Safety is multimodal — it returns severity scores for hate, violence, "
            "sexual content, and self-harm for both text and image inputs in a single unified API."
        ),
    },
    {
        "statement": (
            "Azure Document Intelligence can only process PDF files — "
            "it cannot analyse JPEG or PNG images."
        ),
        "answer": False,
        "explanation": (
            "Document Intelligence supports PDF, JPEG, PNG, BMP, TIFF, and HEIF. "
            "You can pass a file URL, base64-encoded content, or a raw file stream regardless of format."
        ),
    },
    {
        "statement": (
            "In a RAG pipeline, the language model's weights are updated each time "
            "it retrieves and processes new documents."
        ),
        "answer": False,
        "explanation": (
            "RAG does <strong>not</strong> update model weights. It retrieves relevant documents "
            "and injects them as context into the prompt at inference time. Weight updates only "
            "happen during fine-tuning, which is a separate and expensive training process."
        ),
    },
    {
        "statement": (
            "A single Azure AI Services multi-service resource provides one endpoint and one key "
            "to access Azure AI Vision, Language, Speech, and Translator."
        ),
        "answer": True,
        "explanation": (
            "An Azure AI Services multi-service resource gives you a unified key and endpoint that "
            "grants access to all supported services, simplifying configuration, secret management, "
            "and billing consolidation."
        ),
    },
    {
        "statement": (
            "Azure AI Vision and Azure Custom Vision are the same service — "
            "Custom Vision is simply an advanced tier of Azure AI Vision."
        ),
        "answer": False,
        "explanation": (
            "These are distinct services. Azure AI Vision uses Microsoft's prebuilt Florence models "
            "for general image analysis (tagging, OCR, captions, object detection). Custom Vision "
            "lets you train your own classifiers and object detectors on your own labelled images."
        ),
    },
    {
        "statement": (
            "Using Managed Identity to authenticate to Azure AI Services means no API keys "
            "or passwords need to appear in application code or configuration files."
        ),
        "answer": True,
        "explanation": (
            "Managed Identity authenticates the hosting resource (App Service, Function App, VM) "
            "directly to Azure AD. The token exchange is automatic — no keys, passwords, or "
            "connection strings appear in code."
        ),
    },
    {
        "statement": (
            "QnA Maker is Microsoft's current recommended service for building "
            "custom question-and-answer knowledge bases on Azure."
        ),
        "answer": False,
        "explanation": (
            "QnA Maker was <strong>retired</strong> and replaced by the Custom Question Answering "
            "feature within Azure AI Language. All new Q&amp;A projects should use Azure AI Language."
        ),
    },
]

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════

def _init():
    defaults = {
        "cc_initialized": True,
        # Round 1
        "cc_rm_submitted": False,
        "cc_rm_score": 0,
        "cc_rm_results": [],          # list of {"chosen", "is_correct"}
        # Round 2
        "cc_bug_submitted": False,
        "cc_bug_score": 0,
        "cc_bug_results": [],         # list of {"chosen", "is_correct"}
        # Round 3
        "cc_arch_submitted": False,
        "cc_arch_score": 0,
        "cc_arch_results": [],        # list of {"chosen", "is_correct"}
        # Round 4
        "cc_quiz_index": 0,
        "cc_quiz_answers": {},        # idx -> {"choice", "correct", "elapsed", "base", "bonus"}
        "cc_quiz_start_times": {},    # idx -> float
        "cc_quiz_submitted_flags": {},
        "cc_quiz_score": 0,
        # Round 5
        "cc_myth_submitted": False,
        "cc_myth_score": 0,
        "cc_myth_results": [],        # list of {"chosen", "is_correct"}
        # Final
        "cc_balloons_shown": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _total_score():
    return (
        st.session_state.cc_rm_score
        + st.session_state.cc_bug_score
        + st.session_state.cc_arch_score
        + st.session_state.cc_quiz_score
        + st.session_state.cc_myth_score
    )


def _badge(score):
    if score >= 561:
        return "🏆 Champion"
    elif score >= 431:
        return "🟢 Expert"
    elif score >= 251:
        return "🟡 Associate"
    else:
        return "🔵 Apprentice"


# ═══════════════════════════════════════════════════════════════════════════════
# ROUND 1 — Service Matcher
# ═══════════════════════════════════════════════════════════════════════════════

def _round1():
    st.markdown("### 🎯 Round 1: Service Matcher")
    st.markdown(
        "Match each business scenario to the correct Azure AI service. "
        "Each scenario has exactly one best-fit answer. **10 points per correct match.**"
    )
    st.markdown(
        '<div class="info-box">🏆 <strong>Maximum: 80 points</strong> &nbsp;|&nbsp; '
        "8 scenarios × 10 pts each</div>",
        unsafe_allow_html=True,
    )

    if st.session_state.cc_rm_submitted:
        score = st.session_state.cc_rm_score
        if score == 80:
            st.success(f"Perfect score! {score}/80 — Every scenario matched correctly. 🎯")
        elif score >= 50:
            st.info(f"Score: {score}/80 — Good work! Review the explanations below.")
        else:
            st.warning(f"Score: {score}/80 — Review each explanation carefully.")

        st.markdown("---")
        for i, (s, result) in enumerate(
            zip(SCENARIOS, st.session_state.cc_rm_results)
        ):
            icon = "✅" if result["is_correct"] else "❌"
            with st.container():
                st.markdown(f"**{icon} Scenario {i + 1}:** {s['scenario']}")
                if result["is_correct"]:
                    st.markdown(f"&nbsp;&nbsp;&nbsp;Your answer: **{result['chosen']}** ✓")
                else:
                    st.markdown(
                        f"&nbsp;&nbsp;&nbsp;Your answer: ~~{result['chosen']}~~  \n"
                        f"&nbsp;&nbsp;&nbsp;Correct answer: **{s['correct']}**"
                    )
                st.markdown(
                    f'<div class="concept-box">{s["explanation"]}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown("")
        return

    # ── Active form ────────────────────────────────────────────────────────────
    options_with_placeholder = [_PLACEHOLDER] + SERVICE_OPTIONS
    with st.form("cc_round1_form"):
        for i, s in enumerate(SCENARIOS):
            st.markdown(
                f'<div class="module-card"><strong>Scenario {i + 1}:</strong> {s["scenario"]}</div>',
                unsafe_allow_html=True,
            )
            st.selectbox(
                f"Service for Scenario {i + 1}",
                options=options_with_placeholder,
                key=f"cc_rm_sel_{i}",
                label_visibility="collapsed",
            )
            with st.expander("💡 Stuck? Get a hint"):
                st.caption(SCENARIO_HINTS[i])
            st.markdown("")

        submitted = st.form_submit_button(
            "Submit Round 1 →", type="primary", use_container_width=True
        )

    if submitted:
        results = []
        score = 0
        for i, s in enumerate(SCENARIOS):
            chosen = st.session_state.get(f"cc_rm_sel_{i}", _PLACEHOLDER)
            is_correct = chosen == s["correct"]
            if is_correct:
                score += 10
            results.append({"chosen": chosen, "is_correct": is_correct})
        st.session_state.cc_rm_results = results
        st.session_state.cc_rm_score = score
        st.session_state.cc_rm_submitted = True
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# ROUND 2 — Spot the Bug
# ═══════════════════════════════════════════════════════════════════════════════

def _round2():
    st.markdown("### 🐛 Round 2: Spot the Bug")
    st.markdown(
        "Each snippet contains exactly one bug. Read the code carefully and identify the error. "
        "**20 points per correct answer.**"
    )
    st.markdown(
        '<div class="info-box">🏆 <strong>Maximum: 80 points</strong> &nbsp;|&nbsp; '
        "4 bugs × 20 pts each</div>",
        unsafe_allow_html=True,
    )

    if st.session_state.cc_bug_submitted:
        score = st.session_state.cc_bug_score
        if score == 80:
            st.success(f"All 4 bugs found! {score}/80 — Excellent debugging skills. 🐛✅")
        elif score >= 40:
            st.info(f"Score: {score}/80 — Good catch! Review any you missed below.")
        else:
            st.warning(f"Score: {score}/80 — Tricky bugs! Study the explanations carefully.")

        st.markdown("---")
        for i, (bug, result) in enumerate(
            zip(BUGS, st.session_state.cc_bug_results)
        ):
            icon = "✅" if result["is_correct"] else "❌"
            st.markdown(f"#### {icon} {bug['title']}")
            st.code(bug["code"], language=bug["language"])
            if result["is_correct"]:
                st.markdown(f"Your answer: **{result['chosen']}** ✓")
            else:
                st.markdown(
                    f"Your answer: ~~{result['chosen']}~~  \n"
                    f"Correct answer: **{bug['answer']}. {bug['options'][ord(bug['answer']) - ord('A')][3:]}**"
                )
            st.markdown(
                f'<div class="concept-box">{bug["explanation"]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown("---")
        return

    # ── Active form ────────────────────────────────────────────────────────────
    with st.form("cc_round2_form"):
        for i, bug in enumerate(BUGS):
            st.markdown(f"#### {bug['title']}")
            st.code(bug["code"], language=bug["language"])
            st.markdown(f"**{bug['question']}**")
            st.radio(
                f"Answer for Bug {i + 1}",
                options=["A", "B", "C", "D"],
                format_func=lambda x, b=bug: next(
                    o for o in b["options"] if o.startswith(x)
                ),
                key=f"cc_bug_sel_{i}",
                index=None,
                label_visibility="collapsed",
            )
            st.markdown("---")

        submitted = st.form_submit_button(
            "Submit Round 2 →", type="primary", use_container_width=True
        )

    if submitted:
        results = []
        score = 0
        missing = []
        for i, bug in enumerate(BUGS):
            chosen = st.session_state.get(f"cc_bug_sel_{i}")
            if chosen is None:
                missing.append(i + 1)
                continue
            is_correct = chosen == bug["answer"]
            if is_correct:
                score += 20
            results.append({"chosen": chosen, "is_correct": is_correct})

        if missing:
            st.warning(f"Please answer Bug(s): {', '.join(str(m) for m in missing)} before submitting.")
        else:
            st.session_state.cc_bug_results = results
            st.session_state.cc_bug_score = score
            st.session_state.cc_bug_submitted = True
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# ROUND 3 — Architect the Solution
# ═══════════════════════════════════════════════════════════════════════════════

def _round3():
    st.markdown("### 🏗️ Round 3: Architect the Solution")
    st.markdown(
        "Each scenario has multiple technical constraints. Choose the architecture that satisfies "
        "**all** requirements. Watch for deprecated services! **30 points per correct answer.**"
    )
    st.markdown(
        '<div class="info-box">🏆 <strong>Maximum: 90 points</strong> &nbsp;|&nbsp; '
        "3 scenarios × 30 pts each</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="exam-tip">💡 <strong>Exam Tip:</strong> Watch for answer options that include '
        "LUIS (deprecated → CLU), Form Recognizer v2 (deprecated → Document Intelligence), "
        "or Text Analytics v3.0 (deprecated → Azure AI Language).</div>",
        unsafe_allow_html=True,
    )

    if st.session_state.cc_arch_submitted:
        score = st.session_state.cc_arch_score
        if score == 90:
            st.success(f"Perfect architecture! {score}/90 — All three scenarios solved correctly. 🏗️✅")
        elif score >= 60:
            st.info(f"Score: {score}/90 — Good instincts! Review the missed scenario below.")
        else:
            st.warning(f"Score: {score}/90 — Review the architecture explanations carefully.")

        st.markdown("---")
        for i, (arch, result) in enumerate(
            zip(ARCHITECTURES, st.session_state.cc_arch_results)
        ):
            icon = "✅" if result["is_correct"] else "❌"
            st.markdown(f"#### {icon} {arch['title']}")
            st.markdown(arch["description"])
            if result["is_correct"]:
                st.markdown(f"Your answer: **{result['chosen']}**. {arch['options'][ord(result['chosen']) - ord('A')][3:]} ✓")
            else:
                st.markdown(
                    f"Your answer: ~~{result['chosen']}~~ — "
                    f"Correct answer: **{arch['answer']}**. "
                    f"{arch['options'][ord(arch['answer']) - ord('A')][3:]}"
                )
            st.markdown(
                f'<div class="concept-box">{arch["explanation"]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown("---")
        return

    # ── Active form ────────────────────────────────────────────────────────────
    with st.form("cc_round3_form"):
        for i, arch in enumerate(ARCHITECTURES):
            st.markdown(
                f'<div class="module-card"><strong>{arch["title"]}</strong></div>',
                unsafe_allow_html=True,
            )
            st.markdown(arch["description"])
            st.radio(
                f"Answer for Scenario {i + 1}",
                options=["A", "B", "C", "D"],
                format_func=lambda x, a=arch: next(
                    o for o in a["options"] if o.startswith(x)
                ),
                key=f"cc_arch_sel_{i}",
                index=None,
                label_visibility="collapsed",
            )
            st.markdown("---")

        submitted = st.form_submit_button(
            "Submit Round 3 →", type="primary", use_container_width=True
        )

    if submitted:
        results = []
        score = 0
        missing = []
        for i, arch in enumerate(ARCHITECTURES):
            chosen = st.session_state.get(f"cc_arch_sel_{i}")
            if chosen is None:
                missing.append(i + 1)
                continue
            is_correct = chosen == arch["answer"]
            if is_correct:
                score += 30
            results.append({"chosen": chosen, "is_correct": is_correct})

        if missing:
            st.warning(f"Please answer Scenario(s): {', '.join(str(m) for m in missing)} before submitting.")
        else:
            st.session_state.cc_arch_results = results
            st.session_state.cc_arch_score = score
            st.session_state.cc_arch_submitted = True
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# ROUND 4 — Speed Quiz Blitz
# ═══════════════════════════════════════════════════════════════════════════════

def _round4():
    st.markdown("### ⚡ Round 4: Speed Quiz Blitz")
    st.markdown(
        "10 questions across all 5 AI-102 exam domains. Answer quickly to earn speed bonus points!"
    )
    st.markdown(
        '<div class="exam-tip">⏱️ <strong>Speed Bonus (correct answers only):</strong> '
        "Under 15 seconds = +10 pts &nbsp;|&nbsp; "
        "15–30 seconds = +5 pts &nbsp;|&nbsp; "
        "Over 30 seconds = +0 pts<br>"
        "Base score: 20 pts per correct answer. "
        "<strong>Maximum: 300 points</strong></div>",
        unsafe_allow_html=True,
    )

    idx = st.session_state.cc_quiz_index

    # ── Round complete view ────────────────────────────────────────────────────
    if idx >= len(SPEED_QUESTIONS):
        score = st.session_state.cc_quiz_score
        if score >= 250:
            st.success(f"Outstanding! Round 4 Score: **{score} / 300** ⚡")
        elif score >= 150:
            st.info(f"Round 4 Score: **{score} / 300** — Solid performance!")
        else:
            st.warning(f"Round 4 Score: **{score} / 300** — Speed and accuracy need work.")

        st.markdown("---")
        rows = []
        for qi, q in enumerate(SPEED_QUESTIONS):
            if qi in st.session_state.cc_quiz_answers:
                d = st.session_state.cc_quiz_answers[qi]
                rows.append({
                    "Q": f"Q{qi + 1}",
                    "Domain": q["domain"],
                    "Your Answer": d["choice"],
                    "Correct": q["answer"],
                    "Result": "✅" if d["correct"] else "❌",
                    "Time (s)": f"{d['elapsed']:.1f}",
                    "Points": d["base"] + d["bonus"],
                })
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, hide_index=True, use_container_width=True)
        return

    # ── Record start time on first render of this question ────────────────────
    if idx not in st.session_state.cc_quiz_start_times:
        st.session_state.cc_quiz_start_times[idx] = time.time()

    q = SPEED_QUESTIONS[idx]

    st.progress((idx) / len(SPEED_QUESTIONS))
    st.caption(f"Question {idx + 1} of {len(SPEED_QUESTIONS)}  |  Domain: **{q['domain']}**")

    st.markdown(f"### {q['question']}")

    already_submitted = idx in st.session_state.cc_quiz_submitted_flags

    if not already_submitted:
        elapsed_now = time.time() - st.session_state.cc_quiz_start_times[idx]
        if elapsed_now < 10:
            st.markdown(f"⚡ **{elapsed_now:.0f}s** — 🟢 +10 bonus zone! Submit fast!")
        elif elapsed_now < 15:
            st.markdown(f"⏱️ **{elapsed_now:.0f}s** — 🟡 Hurry! +10 bonus expires in {max(0, 15 - elapsed_now):.0f}s")
        elif elapsed_now < 30:
            st.markdown(f"⏱️ **{elapsed_now:.0f}s** — 🟠 +5 bonus zone (valid up to 30s)")
        else:
            st.markdown(f"⌛ **{elapsed_now:.0f}s** — 🔴 Speed bonus gone. Answer correctly for 20 base pts.")
        choice = st.radio(
            "Your answer:",
            options=["A", "B", "C", "D"],
            format_func=lambda x: next(o for o in q["options"] if o.startswith(x)),
            key=f"cc_sq_{idx}",
            index=None,
        )
        if st.button("Submit Answer", key=f"cc_sq_submit_{idx}", type="primary"):
            if choice is None:
                st.warning("Please select an answer before submitting.")
            else:
                elapsed = time.time() - st.session_state.cc_quiz_start_times[idx]
                correct = choice == q["answer"]
                base = 20 if correct else 0
                bonus = 0
                if correct:
                    bonus = 10 if elapsed < 15 else (5 if elapsed < 30 else 0)
                st.session_state.cc_quiz_answers[idx] = {
                    "choice": choice,
                    "correct": correct,
                    "elapsed": elapsed,
                    "base": base,
                    "bonus": bonus,
                }
                st.session_state.cc_quiz_score += base + bonus
                st.session_state.cc_quiz_submitted_flags[idx] = True
                st.rerun()
    else:
        data = st.session_state.cc_quiz_answers[idx]

        if data["correct"]:
            st.success(f"Correct! You chose **{data['choice']}**.")
        else:
            st.error(
                f"Incorrect. You chose **{data['choice']}** — "
                f"correct answer: **{q['answer']}**."
            )
        st.markdown(f"**Explanation:** {q['explanation']}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Time taken", f"{data['elapsed']:.1f}s")
        col2.metric("Speed bonus", f"+{data['bonus']} pts")
        col3.metric("Points earned", f"{data['base'] + data['bonus']}")

        st.markdown("")

        if idx + 1 < len(SPEED_QUESTIONS):
            if st.button("Next Question →", key=f"cc_sq_next_{idx}"):
                st.session_state.cc_quiz_index += 1
                st.rerun()
        else:
            if st.button("View Round 4 Results →", key="cc_sq_done", type="primary"):
                st.session_state.cc_quiz_index = len(SPEED_QUESTIONS)
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# ROUND 5 — Myth Busters (True / False)
# ═══════════════════════════════════════════════════════════════════════════════

def _round5():
    st.markdown("### 💥 Round 5: Myth Busters")
    st.markdown(
        "8 Azure AI statements — decide if each is **True** or **False**. "
        "These are based on real misconceptions that trip up exam candidates. **15 points per correct answer.**"
    )
    st.markdown(
        '<div class="info-box">🏆 <strong>Maximum: 120 points</strong> &nbsp;|&nbsp; '
        "8 statements × 15 pts each</div>",
        unsafe_allow_html=True,
    )

    if st.session_state.cc_myth_submitted:
        score = st.session_state.cc_myth_score
        if score == 120:
            st.success(f"All 8 myths busted! {score}/120 — Flawless Azure AI knowledge. 💥")
        elif score >= 75:
            st.info(f"Score: {score}/120 — Good instincts! Review the ones you missed below.")
        else:
            st.warning(f"Score: {score}/120 — Some tricky ones in there! Study the explanations.")

        st.markdown("---")
        for i, (myth, result) in enumerate(zip(MYTHS, st.session_state.cc_myth_results)):
            icon = "✅" if result["is_correct"] else "❌"
            correct_label = "TRUE" if myth["answer"] else "FALSE"
            chosen_label = "TRUE" if result["chosen"] else "FALSE"
            with st.container():
                st.markdown(f"**{icon} Statement {i + 1}:** _{myth['statement']}_")
                if result["is_correct"]:
                    st.markdown(f"&nbsp;&nbsp;&nbsp;Your answer: **{chosen_label}** ✓")
                else:
                    st.markdown(
                        f"&nbsp;&nbsp;&nbsp;Your answer: ~~{chosen_label}~~ — "
                        f"Correct answer: **{correct_label}**"
                    )
                st.markdown(
                    f'<div class="concept-box">{myth["explanation"]}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown("")
        return

    # ── Active form ────────────────────────────────────────────────────────────
    with st.form("cc_round5_form"):
        for i, myth in enumerate(MYTHS):
            st.markdown(
                f'<div class="module-card"><strong>Statement {i + 1}:</strong> {myth["statement"]}</div>',
                unsafe_allow_html=True,
            )
            st.radio(
                f"True or False for Statement {i + 1}",
                options=[True, False],
                format_func=lambda x: "✅  True" if x else "❌  False",
                key=f"cc_myth_sel_{i}",
                index=None,
                horizontal=True,
                label_visibility="collapsed",
            )
            st.markdown("")

        submitted = st.form_submit_button(
            "Submit Round 5 →", type="primary", use_container_width=True
        )

    if submitted:
        results = []
        score = 0
        missing = []
        for i, myth in enumerate(MYTHS):
            chosen = st.session_state.get(f"cc_myth_sel_{i}")
            if chosen is None:
                missing.append(i + 1)
                continue
            is_correct = chosen == myth["answer"]
            if is_correct:
                score += 15
            results.append({"chosen": chosen, "is_correct": is_correct})

        if missing:
            st.warning(
                f"Please answer Statement(s): {', '.join(str(m) for m in missing)} before submitting."
            )
        else:
            st.session_state.cc_myth_results = results
            st.session_state.cc_myth_score = score
            st.session_state.cc_myth_submitted = True
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL SCORE
# ═══════════════════════════════════════════════════════════════════════════════

def _final_score():
    st.markdown("### 🏆 Final Score")

    rounds_done = [
        st.session_state.cc_rm_submitted,
        st.session_state.cc_bug_submitted,
        st.session_state.cc_arch_submitted,
        st.session_state.cc_quiz_index >= len(SPEED_QUESTIONS),
        st.session_state.cc_myth_submitted,
    ]

    if not any(rounds_done):
        st.info("Complete at least one round to see your score here.")
        st.markdown(
            "Head to the round tabs on the left to start playing. "
            "You can do rounds in any order — your score accumulates as you go!"
        )
        return

    total = _total_score()
    badge = _badge(total)

    # ── Summary table ──────────────────────────────────────────────────────────
    rows = [
        {
            "Round": "🎯 Round 1: Service Matcher",
            "Score": st.session_state.cc_rm_score,
            "Max": 80,
            "Status": "✅ Submitted" if st.session_state.cc_rm_submitted else "⬜ Not yet submitted",
        },
        {
            "Round": "🐛 Round 2: Spot the Bug",
            "Score": st.session_state.cc_bug_score,
            "Max": 80,
            "Status": "✅ Submitted" if st.session_state.cc_bug_submitted else "⬜ Not yet submitted",
        },
        {
            "Round": "🏗️ Round 3: Architect",
            "Score": st.session_state.cc_arch_score,
            "Max": 90,
            "Status": "✅ Submitted" if st.session_state.cc_arch_submitted else "⬜ Not yet submitted",
        },
        {
            "Round": "⚡ Round 4: Speed Quiz",
            "Score": st.session_state.cc_quiz_score,
            "Max": 300,
            "Status": (
                "✅ Completed"
                if st.session_state.cc_quiz_index >= len(SPEED_QUESTIONS)
                else (
                    f"⏳ In progress ({st.session_state.cc_quiz_index}/{len(SPEED_QUESTIONS)})"
                    if st.session_state.cc_quiz_index > 0
                    else "⬜ Not yet started"
                )
            ),
        },
        {
            "Round": "💥 Round 5: Myth Busters",
            "Score": st.session_state.cc_myth_score,
            "Max": 120,
            "Status": "✅ Submitted" if st.session_state.cc_myth_submitted else "⬜ Not yet submitted",
        },
    ]
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, use_container_width=True)

    st.markdown("---")

    # ── Badge and total ────────────────────────────────────────────────────────
    col1, col2 = st.columns([1, 2])
    col1.metric("Total Score", f"{total} / 670")
    col2.markdown(f"## {badge}")

    if total >= 561:
        st.success("🏆 Champion — Exceptional! You've mastered every corner of Azure AI. 🎉")
    elif total >= 431:
        st.success("🟢 Expert — Outstanding! You clearly know the Azure AI Engineer landscape.")
    elif total >= 251:
        st.info("🟡 Associate — Solid foundation. Review the areas you missed and try again.")
    else:
        st.warning("🔵 Apprentice — Keep studying! Work through the Explorer modules and retry.")

    # ── Bar chart ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 3))
    labels = ["Round 1\nMatcher", "Round 2\nBug Hunt", "Round 3\nArchitect", "Round 4\nSpeed Quiz"]
    scores = [
        st.session_state.cc_rm_score,
        st.session_state.cc_bug_score,
        st.session_state.cc_arch_score,
        st.session_state.cc_quiz_score,
    ]
    maxes = [80, 80, 90, 300]
    pcts = [s / m for s, m in zip(scores, maxes)]
    colors = ["#4CAF50" if p >= 0.8 else "#FF9800" if p >= 0.5 else "#f44336" for p in pcts]
    bars = ax.bar(labels, pcts, color=colors)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("% of Max Score")
    ax.set_title("Performance by Round")
    ax.axhline(0.8, color="grey", linestyle="--", linewidth=1, label="80% target")
    ax.legend(fontsize=8)
    for bar, s, m in zip(bars, scores, maxes):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{s}/{m}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Sharable score string ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**Share your score with the class:**")
    share_str = (
        f"AI-102 Challenge Room | {badge} | Total: {total}/670 | "
        f"R1 (Matcher): {st.session_state.cc_rm_score}/80 | "
        f"R2 (Bugs): {st.session_state.cc_bug_score}/80 | "
        f"R3 (Architect): {st.session_state.cc_arch_score}/90 | "
        f"R4 (Speed): {st.session_state.cc_quiz_score}/300 | "
        f"R5 (Myths): {st.session_state.cc_myth_score}/120"
    )
    st.code(share_str, language=None)
    st.caption("Copy the line above and paste it in the class chat!")

    # ── Speed Quiz domain breakdown ────────────────────────────────────────────
    if st.session_state.cc_quiz_index >= len(SPEED_QUESTIONS) and st.session_state.cc_quiz_answers:
        domain_scores: dict = {}
        domain_max: dict = {}
        for qi, q in enumerate(SPEED_QUESTIONS):
            d = q["domain"]
            domain_scores.setdefault(d, 0)
            domain_max.setdefault(d, 0)
            domain_max[d] += 30
            if qi in st.session_state.cc_quiz_answers:
                ans = st.session_state.cc_quiz_answers[qi]
                domain_scores[d] += ans["base"] + ans["bonus"]
        pcts = {d: domain_scores[d] / domain_max[d] for d in domain_max if domain_max[d] > 0}
        if pcts:
            st.markdown("---")
            st.markdown("**📊 Speed Quiz — Domain Breakdown:**")
            cols = st.columns(len(pcts))
            for col, (domain, pct) in zip(cols, pcts.items()):
                emoji = "🟢" if pct >= 0.8 else "🟡" if pct >= 0.5 else "🔴"
                col.metric(f"{emoji} {domain}", f"{pct:.0%}", f"{domain_scores[domain]}/{domain_max[domain]} pts")
            weakest = min(pcts, key=pcts.get)
            if pcts[weakest] < 0.8:
                st.info(f"💡 **Focus area: {weakest}** ({pcts[weakest]:.0%}) — head to that module in the Explorer to review.")

    # ── Celebrations — fires once for Expert or Champion ──────────────────────
    if total >= 561 and not st.session_state.cc_balloons_shown:
        st.session_state.cc_balloons_shown = True
        st.balloons()
        st.snow()
    elif total >= 431 and not st.session_state.cc_balloons_shown:
        st.session_state.cc_balloons_shown = True
        st.balloons()

    # ── Reset ──────────────────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("Reset Challenge (start over)", type="secondary"):
        keys_to_clear = [k for k in st.session_state if k.startswith("cc_")]
        for k in keys_to_clear:
            del st.session_state[k]
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def show():
    _init()

    st.markdown(
        '<p class="gradient-title">🏆 AI Engineer Challenge Room</p>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Round 1", "80 pts", "Service Matcher")
    col2.metric("Round 2", "80 pts", "Spot the Bug")
    col3.metric("Round 3", "90 pts", "Architect")
    col4.metric("Round 4", "300 pts", "Speed Quiz")
    col5.metric("Round 5", "120 pts", "Myth Busters")

    st.markdown(
        '<div class="info-box">'
        "<strong>How to play:</strong> Complete all five rounds in any order. "
        "Each round tests a different skill — service knowledge, debugging, architecture, speed, and myth-busting. "
        "Your score accumulates automatically. Head to the 🏆 Final Score tab at any time to check your standing.<br><br>"
        "<strong>Badge thresholds:</strong> "
        "🔵 Apprentice (0–250) &nbsp;|&nbsp; "
        "🟡 Associate (251–430) &nbsp;|&nbsp; "
        "🟢 Expert (431–560) &nbsp;|&nbsp; "
        "🏆 Champion (561+)"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Round 1: Service Matcher",
        "🐛 Round 2: Spot the Bug",
        "🏗️ Round 3: Architect",
        "⚡ Round 4: Speed Quiz",
        "💥 Round 5: Myth Busters",
        "🏆 Final Score",
    ])

    with tab1:
        _round1()
    with tab2:
        _round2()
    with tab3:
        _round3()
    with tab4:
        _round4()
    with tab5:
        _round5()
    with tab6:
        _final_score()
