# AI Engineer Challenge Room — Classroom Guide

A 4-round gamified activity for **AI-102: Azure AI Engineer Associate** students.  
Designed for **30–45 minutes** in class. Runs **100% locally** — no Azure subscription or API keys required.

---

## Already Set Up? Quick Relaunch

If you cloned and set up the repo in a previous session, just pull the latest code and relaunch:

**Windows (Command Prompt):**
```bat
cd AI102Explorer
git pull origin main
.venv\Scripts\activate.bat
streamlit run app.py
```

**PowerShell:**
```powershell
cd AI102Explorer
git pull origin main
.venv\Scripts\Activate.ps1
streamlit run app.py
```

**macOS / Linux:**
```bash
cd AI102Explorer
git pull origin main
source .venv/bin/activate
streamlit run app.py
```

Then open **http://localhost:8501** in your browser and click **🏆 Classroom Challenge** in the sidebar.

> **If you downloaded a ZIP** (no Git): re-download the latest ZIP from [github.com/EricKart/AI102Explorer](https://github.com/EricKart/AI102Explorer), extract it into the same folder, and run `streamlit run app.py` from inside it.

---

## First Time? Full Setup Steps

### Step 1 — Install Git (if you haven't already)

| OS | How |
|---|---|
| **Windows** | Download from [git-scm.com/download/win](https://git-scm.com/download/win). During install, keep all defaults. Then open a **new** Command Prompt window. |
| **macOS** | Run `git --version` in Terminal. If not installed, macOS will prompt you to install Xcode Command Line Tools — click Install. |
| **Ubuntu / Debian** | `sudo apt update && sudo apt install git` |

---

### Step 2 — Install Python 3.9 or later

Download from [python.org/downloads](https://www.python.org/downloads/).

> **Windows:** Tick **"Add Python to PATH"** during installation before clicking Install.

> **macOS:** Python 3 is not pre-installed on recent macOS. Install via [python.org](https://www.python.org/downloads/) or `brew install python`.

> **Ubuntu / Debian:** You need three packages: `sudo apt install python3 python3-venv python3-pip`
> If you skip `python3-venv`, the setup script will fail with a venv error.

---

### Step 3 — Clone the Repository

> ⚠️ **Windows:** Do **not** clone into a folder synced by **OneDrive** or **SharePoint**. OneDrive locks files during sync and breaks virtual environment creation. Use a plain local path like `C:\Projects` or your Desktop.

```bash
git clone https://github.com/EricKart/AI102Explorer.git
cd AI102Explorer
```

**No Git?** Click **Code → Download ZIP** on [github.com/EricKart/AI102Explorer](https://github.com/EricKart/AI102Explorer), extract it, and open a terminal inside the extracted folder.

---

### Step 4 — Run the Setup Script

> The setup script creates the virtual environment, installs all dependencies, downloads NLTK language data, **and launches the app automatically at the end**. You do not need to run a separate launch command after this.

**Windows (Command Prompt):**
```bat
setup.bat
```

> **PowerShell note:** If you see *"running scripts is disabled on this system"*, run this first:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
> ```
> Then switch to **Command Prompt** and run `setup.bat` from there — it's simpler.

**macOS / Linux:**
```bash
chmod +x setup.sh
./setup.sh
```

Your browser will open automatically at **http://localhost:8501** when setup finishes. If it doesn't, paste that URL into your browser manually.

---

### Step 5 — Open the Challenge Room

In the left sidebar, scroll down and click **🏆 Classroom Challenge**.

---

## Relaunching in Future Sessions

The next time you want to use the app, **do not re-run the setup script** — just activate the virtual environment and run Streamlit:

**Windows (Command Prompt):**
```bat
cd AI102Explorer
.venv\Scripts\activate.bat
streamlit run app.py
```

**PowerShell:**
```powershell
cd AI102Explorer
.venv\Scripts\Activate.ps1
streamlit run app.py
```

**macOS / Linux:**
```bash
cd AI102Explorer
source .venv/bin/activate
streamlit run app.py
```

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
| 💥 Myth Busters | 120 |
| **Total** | **670** |

### Badges

| Badge | Score Range |
|---|---|
| 🔵 Apprentice | 0–250 |
| 🟡 Associate | 251–430 |
| 🟢 Expert | 431–560 |
| 🏆 Champion | 561–670 |

---

## Sharing Your Score

When you finish all five rounds, go to the **🏆 Final Score** tab.  
Copy the score line shown there and paste it into the class chat — for example:

```
AI-102 Challenge Room | 🟢 Expert | Total: 472/670 | R1 (Matcher): 70/80 | R2 (Bugs): 60/80 | R3 (Architect): 60/90 | R4 (Speed): 207/300 | R5 (Myths): 75/120
```

---

## Rounds Can Be Done in Any Order

You don't have to do the rounds in sequence. Your score accumulates automatically.  
Check the **🏆 Final Score** tab at any point to see your current total.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` not found (Windows) | Re-install Python and tick **Add to PATH**. Open a **new** Command Prompt after installing. |
| `python3` not found (macOS) | Download from python.org or run `brew install python` |
| `python3-venv` missing (Ubuntu/Debian) | Run `sudo apt install python3-venv python3-pip` then retry setup |
| *"running scripts is disabled"* (PowerShell) | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` then use Command Prompt for `setup.bat` |
| `ModuleNotFoundError: No module named 'streamlit'` | You ran streamlit without activating the venv. Activate first (see **Relaunching in Future Sessions** above). |
| `ModuleNotFoundError: No module named 'matplotlib'` or similar | Re-run `pip install -r requirements.txt` with the venv active |
| venv creation fails (Windows, OneDrive path) | Move the project folder out of OneDrive/SharePoint to a plain local path (e.g. `C:\Projects`) and retry |
| `git` not found after installing (Windows) | Close and reopen your Command Prompt so the PATH update takes effect |
| Port 8501 already in use | Run `streamlit run app.py --server.port 8502` and go to `http://localhost:8502` |
| Browser doesn't open automatically | Manually paste `http://localhost:8501` into your browser |
| NLTK error at runtime | With venv active, run: `python -c "import nltk; [nltk.download(p, quiet=True) for p in ['punkt','punkt_tab','averaged_perceptron_tagger','averaged_perceptron_tagger_eng','vader_lexicon','stopwords','maxent_ne_chunker','words']]"` |
| Challenge scores reset on refresh | Expected — Streamlit session state is local to your browser tab. Do **not** close or refresh the tab mid-challenge. |

---

## For Instructors

- **No server or shared infrastructure needed** — every student runs their own local copy
- **All rounds are independent** — you can demonstrate one round at a time on a projector while students follow along
- **Round 3 explanations** call out exactly which deprecated services appear in wrong answers — useful discussion anchor
- **Speed Quiz** creates natural competitive pressure; announcing the top speed bonus scores to the class works well as an icebreaker
- **Reset button** on the Final Score tab lets students retry the full challenge if time allows
- **Common Windows issue:** students who cloned into an OneDrive folder — have them move the folder to `C:\Projects` or Desktop and re-run `setup.bat`

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
