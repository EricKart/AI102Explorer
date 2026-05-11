# AI Engineer Challenge Room — Classroom Guide

A 4-round gamified activity for **AI-102: Azure AI Engineer Associate** students.  
Designed for **30–45 minutes** in class. Runs **100% locally** — no Azure subscription or API keys required.

---

## For Students: Getting Started

### Step 1 — Install Python (if you haven't already)

Download and install **Python 3.9 or later** from [python.org/downloads](https://www.python.org/downloads/).

> **Windows:** During installation, tick **"Add Python to PATH"** before clicking Install.

---

### Step 2 — Clone the Repository

Open a terminal (Command Prompt, PowerShell, or macOS/Linux Terminal) and run:

```bash
git clone https://github.com/EricKart/AI102Explorer.git
cd AI102Explorer
```

> If you don't have Git: download the ZIP from [github.com/EricKart/AI102Explorer](https://github.com/EricKart/AI102Explorer) → click **Code → Download ZIP** → extract it.

---

### Step 3 — Run the Setup Script

**Windows (Command Prompt):**
```bat
setup.bat
```

> PowerShell users — run this first if you see a script error:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
> ```

**macOS / Linux:**
```bash
chmod +x setup.sh
./setup.sh
```

The script creates a virtual environment, installs all dependencies, and downloads NLTK language data automatically.

---

### Step 4 — Launch the App

**Windows:**
```bat
.venv\Scripts\activate.bat
streamlit run app.py
```

**PowerShell:**
```powershell
.venv\Scripts\Activate.ps1
streamlit run app.py
```

**macOS / Linux:**
```bash
source .venv/bin/activate
streamlit run app.py
```

Your browser will open automatically at **http://localhost:8501**.

> If the browser doesn't open, paste `http://localhost:8501` into your browser manually.

---

### Step 5 — Open the Challenge Room

In the left sidebar, scroll down and click **🏆 Classroom Challenge**.

---

## The 4 Rounds

### 🎯 Round 1: Service Matcher (80 points)
**Format:** 8 real-world business scenarios. Match each one to the correct Azure AI service.  
**Scoring:** 10 points per correct match.  
**Skills tested:** Knowing *which* Azure AI service solves which problem.

> Example: *"A call centre uses a customer's voice to verify their identity"* → **Azure AI Speech** (Speaker Recognition)

---

### 🐛 Round 2: Spot the Bug (80 points)
**Format:** 4 real code snippets (SSML, REST API calls, JSON schemas, Python SDK). Each has exactly one bug.  
**Scoring:** 20 points per correct answer.  
**Skills tested:** Reading API docs carefully, knowing valid parameter values, spotting deprecated endpoints.

> Example: A snippet using `rate="very-fast"` in SSML — which is not a valid prosody rate value.

---

### 🏗️ Round 3: Architect the Solution (90 points)
**Format:** 3 multi-constraint business scenarios. Choose the architecture that satisfies **all** requirements.  
**Scoring:** 30 points per correct answer.  
**Skills tested:** Composing multi-service pipelines, avoiding deprecated services, understanding security patterns.

> Watch out: wrong answers include **LUIS** (deprecated — use CLU), **Form Recognizer v2** (deprecated — use Document Intelligence), and **Text Analytics v3.0** (deprecated — use Azure AI Language).

---

### ⚡ Round 4: Speed Quiz Blitz (up to 300 points)
**Format:** 10 questions, one at a time, across all 5 AI-102 exam domains.  
**Scoring:** 20 base points for a correct answer, plus a speed bonus:

| Answer time | Speed bonus |
|---|---|
| Under 15 seconds | +10 pts |
| 15–30 seconds | +5 pts |
| Over 30 seconds | +0 pts |

**Maximum per question:** 30 pts (correct + fast). **Maximum round total:** 300 pts.

> The timer starts the moment the question appears — read fast!

---

## Scoring Summary

| Round | Max Points |
|---|---|
| 🎯 Service Matcher | 80 |
| 🐛 Spot the Bug | 80 |
| 🏗️ Architect | 90 |
| ⚡ Speed Quiz | 300 |
| **Total** | **550** |

### Badges

| Badge | Score Range |
|---|---|
| 🔵 Apprentice | 0–200 |
| 🟡 Associate | 201–350 |
| 🟢 Expert | 351–450 |
| 🏆 Champion | 451–550 |

---

## Sharing Your Score

When you finish all four rounds, go to the **🏆 Final Score** tab.  
Copy the score line shown there and paste it into the class chat — for example:

```
AI-102 Challenge Room | 🟢 Expert | Total: 392/550 | R1 (Matcher): 70/80 | R2 (Bugs): 60/80 | R3 (Architect): 60/90 | R4 (Speed): 202/300
```

---

## Rounds Can Be Done in Any Order

You don't have to do the rounds in sequence. Your score accumulates automatically.  
Check the **🏆 Final Score** tab at any point to see your current total.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` not found (Windows) | Re-install Python and tick **Add to PATH** |
| `python3` not found (macOS) | Run `brew install python` or download from python.org |
| Script execution error (PowerShell) | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` first |
| `ModuleNotFoundError` | Make sure you activated the virtual environment (Step 4) before running streamlit |
| Port 8501 already in use | Run `streamlit run app.py --server.port 8502` and go to `http://localhost:8502` |
| Browser doesn't open | Manually paste `http://localhost:8501` into your browser |
| NLTK error at runtime | Run this command with the venv active: `python -c "import nltk; [nltk.download(p, quiet=True) for p in ['punkt','punkt_tab','averaged_perceptron_tagger','averaged_perceptron_tagger_eng','vader_lexicon','stopwords','maxent_ne_chunker','words']]"` |
| Challenge scores reset on refresh | This is expected — Streamlit session state is local to your browser tab. Do not close or refresh the tab mid-challenge. |

---

## For Instructors

- **No server or shared infrastructure needed** — every student runs their own local copy
- **All rounds are independent** — you can demonstrate one round at a time on a projector while students follow along
- **Round 3 explanations** call out exactly which deprecated services appear in wrong answers — useful discussion anchor
- **Speed Quiz** creates natural competitive pressure; announcing the top speed bonus scores to the class works well as an icebreaker
- **Reset button** on the Final Score tab lets students retry the full challenge if time allows

---

## AI-102 Exam Domains Covered

| Domain | Rounds |
|---|---|
| Plan and manage Azure AI solutions | Round 1 (Key Vault scenario), Round 4 (Q1, Q2) |
| Computer Vision | Round 1 (Custom Vision scenario), Round 3 (architecture options), Round 4 (Q3, Q4) |
| Natural Language Processing | Round 1 (PII/NER scenario), Round 2 (Bug #2), Round 4 (Q5, Q6) |
| Document Intelligence & Knowledge Mining | Round 1 (Document Intelligence scenario), Round 2 (Bug #3), Round 4 (Q7, Q8) |
| Generative AI | Round 3 (architecture options), Round 4 (Q9, Q10) |

---

*Questions or issues? Open an issue at [github.com/EricKart/AI102Explorer](https://github.com/EricKart/AI102Explorer).*
