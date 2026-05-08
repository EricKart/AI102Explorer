# AI-102 Key Concepts Reference

A quick-reference guide for the **Azure AI Engineer Associate (AI-102)** exam.

---

## 1. Plan and Manage Azure AI Solutions

| Concept | Key Points |
|---|---|
| Authentication | Managed Identity (recommended), API key, Azure AD token |
| Pricing tiers | Free F0 (dev/test), Standard S0+ (production) |
| Key Vault | Store API keys as secrets; rotate without redeployment |
| Azure Monitor | Metrics, logs, alerts for AI Services resources |
| Data residency | Deploy resource in the required geographic region |
| Responsible AI | Fairness, Reliability, Privacy, Inclusiveness, Transparency, Accountability |

---

## 2. Azure AI Vision

### Image Analysis
- Features: `tags`, `objects`, `caption`, `denseCaptions`, `color`, `faces`, `read`, `smartCrops`
- Bounding boxes: **normalised fractions** (0.0–1.0) of image width/height
- Endpoint: `POST /computervision/imageanalysis:analyze?features=tags,objects`

### OCR / Read API
- **Read API** (async): best for multi-page PDFs and high-accuracy OCR
- **OCR API** (sync): images only, faster for small images
- Async pattern: POST → get `Operation-Location` header → GET to poll

### Custom Vision
- Modes: **Classification** (what?) vs **Object Detection** (where?)
- Minimum: **15 images per tag**
- Export formats: ONNX, CoreML, TensorFlow, TensorFlow Lite

---

## 3. Azure AI Language

| Feature | Use Case |
|---|---|
| Sentiment Analysis | Positive/Negative/Neutral per document and sentence |
| Opinion Mining | Aspect-based sentiment (food was great, service was slow) |
| NER | Classify entity type: Person, Location, Organization, DateTime… |
| Entity Linking | Map entity mentions → Wikipedia with disambiguation |
| Key Phrase Extraction | Main topics from text (list of strings) |
| Summarisation | Extractive (select sentences) or Abstractive (generate new text) |
| Language Detection | BCP-47 code + confidence; supports 120+ languages |
| CLU | Intent + entity extraction for conversational apps (successor to LUIS) |
| Custom QnA | FAQ-style knowledge base from documents or web pages |

---

## 4. Azure AI Speech

| Feature | Notes |
|---|---|
| Text-to-Speech (TTS) | Neural voices + SSML; `X-Microsoft-OutputFormat` header |
| Speech-to-Text (STT) | Real-time (`recognizeOnceAsync`) or batch transcription |
| Speech Translation | STT + translation in a single service call |
| Speaker Recognition | Verify (1:1) or Identify (1:N) a speaker from audio |
| SSML elements | `<prosody>` (rate, pitch), `<emphasis>`, `<break>`, `<phoneme>` |

---

## 5. Azure Document Intelligence

### Prebuilt Models
| Model | Extracts |
|---|---|
| `prebuilt-read` | Raw text + layout (OCR only) |
| `prebuilt-layout` | Text + tables + selection marks + figures |
| `prebuilt-invoice` | VendorName, InvoiceDate, InvoiceTotal, LineItems… |
| `prebuilt-receipt` | MerchantName, TransactionDate, Items, Total… |
| `prebuilt-idDocument` | FirstName, LastName, DateOfBirth, DocumentNumber… |
| `prebuilt-businessCard` | Name, Title, Email, Phone, Company… |

### Custom Models
- **Custom Template** — fixed layout (forms, structured docs)
- **Custom Neural** — varied layout (less structured documents)
- **Custom Classification** — route documents to N categories

---

## 6. Azure AI Search (Knowledge Mining)

### Components
| Component | Purpose |
|---|---|
| Data Source | Origin of content (Blob, SQL, Cosmos DB, SharePoint…) |
| Index | Schema — fields with `searchable`, `filterable`, `facetable` flags |
| Indexer | Pipeline: pull data → enrich → push to index |
| Skillset | Cognitive enrichment steps (OCR, NER, embedding generation…) |
| Knowledge Store | Persist enrichment output to Blob/Table storage |

### Search Types
| Type | Algorithm | Notes |
|---|---|---|
| Full-text | BM25 | Inverted index, analyzers, scoring profiles |
| Vector | HNSW (ANN) | Cosine/dot-product similarity on embeddings |
| Hybrid | BM25 + Vector + RRF | Recommended — combines keyword and semantic |
| Semantic re-ranking | Language model | Premium feature; re-ranks top results |

---

## 7. Generative AI (Azure OpenAI)

### Key Parameters
| Parameter | Range | Effect |
|---|---|---|
| `temperature` | 0.0–2.0 | Higher = more random output |
| `top_p` | 0.0–1.0 | Nucleus sampling; alternative to temperature |
| `max_tokens` | 1–context limit | Maximum output length |
| `frequency_penalty` | -2.0–2.0 | Reduce repetition of frequent tokens |
| `presence_penalty` | -2.0–2.0 | Encourage new topics |
| `stop` | string/list | End generation at this token |

### Message Roles
- `system` — persistent instructions / persona
- `user` — human input
- `assistant` — model response (previous turns)
- `tool` — function/tool call result (agentic flows)

### Prompting Techniques
| Technique | Description |
|---|---|
| Zero-shot | No examples — just instruction |
| Few-shot | 2–5 input-output examples in prompt |
| Chain-of-thought | Step-by-step reasoning in prompt |
| RAG | Inject retrieved context to ground response |

### Tokens
- 1 token ≈ 4 characters or 0.75 words (English)
- Pricing and context limits measured in tokens
- BPE (Byte-Pair Encoding) — subword tokenisation algorithm

---

## Quick Exam Tips

1. **Managed Identity** = no credentials in code — always recommend this
2. **Read API** = async OCR for multi-page PDFs; legacy OCR API = sync, images only
3. **Custom Vision min images** = **15 per tag**
4. **Bounding boxes** = normalised fractions (not pixels)
5. **CLU** = successor to LUIS; intent + entities for conversational apps
6. **Entity Linking** maps to a knowledge base; NER just classifies type
7. **Skillset** = enrichment pipeline attached to an indexer in AI Search
8. **prebuilt-invoice** ≠ **prebuilt-receipt** — different fields, different training
9. **Temperature** controls randomness; low = deterministic, high = creative
10. **RAG** reduces hallucination by grounding LLM responses in retrieved docs
