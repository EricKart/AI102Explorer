#!/usr/bin/env bash
set -e

echo "============================================================"
echo " AI-102 Explorer - Setup Script"
echo "============================================================"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 is not installed."
    echo "Install it with: sudo apt install python3 python3-venv python3-pip"
    exit 1
fi

echo "Python found. Creating virtual environment..."
python3 -m venv .venv

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip --quiet
pip install -r requirements.txt

echo "Downloading NLTK data..."
python3 -c "import nltk; [nltk.download(p, quiet=True) for p in ['punkt','punkt_tab','averaged_perceptron_tagger','averaged_perceptron_tagger_eng','vader_lexicon','stopwords','maxent_ne_chunker','words']]"

echo ""
echo "============================================================"
echo " Setup complete! Launching AI-102 Explorer..."
echo "============================================================"
streamlit run app.py
