# AI-102 Explorer

An interactive, self-contained study tool for the **Microsoft Azure AI Engineer Associate (AI-102)** exam.  
No Azure subscription or API key required - everything runs **100% locally** using Python, Streamlit, and scikit-learn.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

---

## What's Inside

| Module | Topics Covered |
|---|---|
| **Home** | Exam overview, domain weights radar chart, study roadmap |
| **AI Vision** | Image Analysis, OCR Read API, Object Detection, Custom Vision |
| **AI Language** | Sentiment, NER, Key Phrases, Summarisation, Language Detection |
| **AI Speech** | TTS waveforms, SSML Builder, Speech Analysis, Spectrograms |
| **Document Intelligence** | Invoice, Receipt, Table Extraction, Document Classification |
| **Knowledge Mining** | Full-text Search, Vector Search, Facets, Word Cloud |
| **Generative AI** | Prompt Engineering, RAG Simulation, Text Generation, Token Analysis |
| **Practice Quiz** | 25 questions across all 5 domains with explanations and scoring |
| **Classroom Challenge** | 5-round gamified activity: Service Matcher, Spot the Bug, Architect, Speed Quiz Blitz, Azure AI Gauntlet |

---

## Quick Start

### Prerequisites

- **Python 3.9 or later** - [python.org/downloads](https://www.python.org/downloads/)
  - On Windows: tick **"Add Python to PATH"** during install
- **Git** - [git-scm.com](https://git-scm.com/)
- ~300 MB free disk space

---

### Windows

Open **Command Prompt** (cmd) and run:

```bat
git clone https://github.com/EricKart/AI102Explorer.git
cd AI102Explorer
setup.bat
```

> **PowerShell users:** If you see a script execution error, run this first, then re-run setup.bat:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
> ```

---

### macOS

Open **Terminal** and run:

```bash
git clone https://github.com/EricKart/AI102Explorer.git
cd AI102Explorer
chmod +x setup.sh
./setup.sh
```

> **macOS note:** If `python3` is not found, install it from [python.org](https://www.python.org/downloads/) or run `brew install python`.

---

### Linux (Ubuntu / Debian)

```bash
sudo apt update && sudo apt install -y python3 python3-venv python3-pip git

git clone https://github.com/EricKart/AI102Explorer.git
cd AI102Explorer
chmod +x setup.sh
./setup.sh
```

---

### Manual Setup (any OS)

Use this if the setup scripts do not work on your system:

```bash
# 1. Clone the repo
git clone https://github.com/EricKart/AI102Explorer.git
cd AI102Explorer

# 2. Create a virtual environment
python -m venv .venv          # Windows
# python3 -m venv .venv       # macOS / Linux

# 3. Activate the virtual environment
# Windows (Command Prompt):
.venv\Scripts\activate.bat
# Windows (PowerShell):
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 5. Download NLTK language data (one-time)
python -c "import nltk; [nltk.download(p, quiet=True) for p in ['punkt','punkt_tab','averaged_perceptron_tagger','averaged_perceptron_tagger_eng','vader_lexicon','stopwords','maxent_ne_chunker','words']]"

# 6. Launch the app
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---

## Project Structure

```
AI102Explorer/
- app.py                         Main Streamlit entry point
- requirements.txt               Python dependencies
- setup.bat                      Windows one-click setup
- setup.sh                       macOS/Linux one-click setup
- README.md                      This file
- CONCEPTS.md                    Exam concept reference sheet
- modules/
  - home.py                      Landing page and exam overview
  - ai_vision.py                 Computer Vision module
  - ai_language.py               NLP module
  - ai_speech.py                 Speech module
  - document_intelligence.py     Document Intelligence module
  - knowledge_mining.py          Knowledge Mining module
  - generative_ai.py             Generative AI module
  - quiz.py                      25-question practice quiz
  - classroom_challenge.py       5-round in-class gamified activity (670 pts)
```

---

## Exam Domain Coverage

| Domain | Weight | Modules |
|---|---|---|
| Plan and manage an Azure AI solution | 15-20% | Home, all modules |
| Implement image and video processing solutions | 15-20% | AI Vision |
| Implement natural language processing solutions | 15-20% | AI Language |
| Implement AI speech solutions | 10-15% | AI Speech |
| Implement knowledge mining and document intelligence | 15-20% | Document Intelligence, Knowledge Mining |
| Implement generative AI solutions | 10-15% | Generative AI |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` not found (Windows) | Re-install Python and tick **Add to PATH** |
| `python3` not found (macOS) | Run `brew install python` or download from python.org |
| `ModuleNotFoundError` | Activate the venv before running streamlit (step 3) |
| Script execution error (PowerShell) | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` |
| Port 8501 already in use | Run `streamlit run app.py --server.port 8502` |
| NLTK error at runtime | Re-run the NLTK download command from step 5 |

---

## Classroom Activity

The **AI Engineer Challenge Room** is a 5-round in-class activity designed for 45-60 minutes.
Rounds: 🎯 Service Matcher · 🐛 Spot the Bug · 🏗️ Architect · ⚡ Speed Quiz · ⚡ Azure AI Gauntlet  
Max score: **670 pts** — badges from Apprentice up to Champion.  
See [CLASSROOM_GUIDE.md](CLASSROOM_GUIDE.md) for full setup instructions, round descriptions, and scoring details.

---

## Also Check Out

- [AI-900 Explorer](https://github.com/EricKart/AI-900-Explorer) - foundational Azure AI study tool

---

## License

MIT License - free to use, share, and modify.
