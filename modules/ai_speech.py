import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import signal as sp_signal
import xml.etree.ElementTree as ET


# ── Audio synthesis helpers ───────────────────────────────────────────────────

def _text_to_signal(text, sr=22050, duration=2.0):
    """Return a synthetic waveform representing TTS output (not real TTS)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    words = text.split()
    n_words = max(len(words), 1)
    base_freq = 160 + 40 * min(n_words, 10) / 10
    wave = np.zeros_like(t)
    for i, w in enumerate(words[:12]):
        freq = base_freq + (hash(w) % 80)
        start = int(i * sr * duration / n_words)
        end   = int((i + 1) * sr * duration / n_words)
        end   = min(end, len(t))
        wave[start:end] += 0.4 * np.sin(2 * np.pi * freq * t[start:end])
    noise = np.random.normal(0, 0.02, len(wave))
    wave += noise
    wave = wave / (np.max(np.abs(wave)) + 1e-8)
    return wave, t, sr


def _build_ssml(voice, text, rate, pitch, emphasis):
    pitch_tag = {"low": "-2st", "medium": "+0st", "high": "+2st"}[pitch]
    root = ET.Element("speak", version="1.0", attrib={"xml:lang": "en-US",
        "xmlns": "https://www.w3.org/2001/10/synthesis"})
    voice_el = ET.SubElement(root, "voice", name=voice)
    prosody = ET.SubElement(voice_el, "prosody",
                            rate=rate, pitch=pitch_tag)
    if emphasis != "none":
        emph_el = ET.SubElement(prosody, "emphasis", level=emphasis)
        emph_el.text = text
    else:
        prosody.text = text
    return '<?xml version="1.0"?>\n' + ET.tostring(root, encoding="unicode")


def _phoneme_counts(text):
    vowels     = set("aeiouAEIOU")
    consonants = set("bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ")
    digits     = set("0123456789")
    v = sum(1 for c in text if c in vowels)
    co = sum(1 for c in text if c in consonants)
    d = sum(1 for c in text if c in digits)
    sp = text.count(' ')
    punc = sum(1 for c in text if c in '.,!?;:')
    return {"Vowels": v, "Consonants": co, "Digits": d, "Spaces": sp, "Punctuation": punc}


def _spectrogram(text, sr=4000):
    wave, _, _ = _text_to_signal(text, sr=sr, duration=1.5)
    f, t_sp, Sxx = sp_signal.spectrogram(wave, sr, nperseg=128, noverlap=64)
    return f, t_sp, Sxx


# ── Main page ──────────────────────────────────────────────────────────────────

def show():
    st.markdown('<p class="gradient-title">🎙️ Azure AI Speech</p>', unsafe_allow_html=True)
    st.markdown("### Exploring Azure AI Speech — Synthetic Waveforms (No Microphone Needed)")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔊 TTS Waveform", "🔧 SSML Builder", "📊 Speech Analysis", "🌊 Audio Features",
    ])

    # ── Tab 1: TTS Waveform ───────────────────────────────────────────────────
    with tab1:
        st.markdown("""
        <div class="info-box">
        <strong>Azure AI Speech — Text-to-Speech (TTS)</strong><br>
        Converts text to natural-sounding audio using <em>neural voices</em>.
        Azure Neural TTS uses a deep learning model to produce human-like prosody and intonation.
        This demo visualises what a TTS waveform looks like — without generating real audio.
        </div>
        """, unsafe_allow_html=True)

        tts_text = st.text_input("Text to synthesise:", value="Welcome to Azure AI Services!")
        voice_map = {
            "en-US-AriaNeural":   "US English — Aria (Female, conversational)",
            "en-US-GuyNeural":    "US English — Guy (Male, newscast)",
            "en-GB-SoniaNeural":  "UK English — Sonia (Female)",
            "fr-FR-DeniseNeural": "French — Denise (Female)",
            "es-ES-ElviraNeural": "Spanish — Elvira (Female)",
        }
        voice = st.selectbox("Voice:", list(voice_map.keys()),
                             format_func=lambda v: voice_map[v])

        if tts_text.strip():
            wave, t, sr = _text_to_signal(tts_text)
            fig, axes = plt.subplots(2, 1, figsize=(12, 5))

            axes[0].plot(t, wave, color='#2196F3', linewidth=0.6, alpha=0.85)
            axes[0].set_title(f"Simulated TTS Waveform — voice: {voice}")
            axes[0].set_xlabel("Time (seconds)")
            axes[0].set_ylabel("Amplitude")
            axes[0].axhline(0, color='grey', linewidth=0.5, linestyle='--')

            rms_frame = 512
            rms = [np.sqrt(np.mean(wave[i:i+rms_frame]**2))
                   for i in range(0, len(wave)-rms_frame, rms_frame//2)]
            t_rms = np.linspace(0, t[-1], len(rms))
            axes[1].fill_between(t_rms, rms, alpha=0.6, color='#FF5722')
            axes[1].set_title("RMS Energy (volume envelope)")
            axes[1].set_xlabel("Time (seconds)")
            axes[1].set_ylabel("RMS")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            c1, c2, c3 = st.columns(3)
            c1.metric("Sample Rate",  f"{sr:,} Hz")
            c2.metric("Duration",     f"{t[-1]:.1f} s")
            c3.metric("Word Count",   len(tts_text.split()))

        st.markdown("""
        <div class="exam-tip">
        <strong>Exam Tip:</strong> Azure Neural TTS endpoint:<br>
        <code>POST /cognitiveservices/v1</code> — body can be plain text or SSML.<br>
        Response is audio binary (WAV, MP3, OGG). Set <code>X-Microsoft-OutputFormat</code> header.
        </div>
        """, unsafe_allow_html=True)

    # ── Tab 2: SSML Builder ───────────────────────────────────────────────────
    with tab2:
        st.markdown("""
        <div class="info-box">
        <strong>Speech Synthesis Markup Language (SSML)</strong><br>
        SSML gives fine-grained control over pronunciation, rate, pitch, pauses,
        and emphasis. Azure TTS supports a large subset of W3C SSML 1.0.
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            s_text  = st.text_input("Text:", value="Azure AI makes development easy.")
            s_voice = st.selectbox("Voice:", [
                "en-US-AriaNeural", "en-US-GuyNeural",
                "en-GB-SoniaNeural", "fr-FR-DeniseNeural",
            ])
            s_rate  = st.select_slider("Rate:", ["x-slow","slow","medium","fast","x-fast"], value="medium")
            s_pitch = st.selectbox("Pitch:", ["low","medium","high"], index=1)
            s_emph  = st.selectbox("Emphasis:", ["none","reduced","moderate","strong"], index=2)

        with col2:
            ssml = _build_ssml(s_voice, s_text, s_rate, s_pitch, s_emph)
            st.markdown("**Generated SSML:**")
            st.code(ssml, language="xml")

        st.markdown("#### SSML Reference")
        ref = pd.DataFrame([
            {"Element":    "<prosody>", "Key Attributes": "rate, pitch, volume",
             "Example": 'rate="fast" pitch="+2st"'},
            {"Element":    "<emphasis>","Key Attributes": "level",
             "Example": 'level="strong"'},
            {"Element":    "<break>",   "Key Attributes": "time, strength",
             "Example": 'time="500ms"'},
            {"Element":    "<phoneme>", "Key Attributes": "alphabet, ph",
             "Example": 'alphabet="ipa" ph="ˈæzʊər"'},
            {"Element":    "<say-as>",  "Key Attributes": "interpret-as",
             "Example": 'interpret-as="date"'},
            {"Element":    "<mstts:express-as>","Key Attributes": "style",
             "Example": 'style="cheerful"'},
        ])
        st.dataframe(ref, hide_index=True, use_container_width=True)

        st.markdown("""
        <div class="concept-box">
        <strong>Neural Voice Styles:</strong>
        Some neural voices support style parameters:<br>
        <code>cheerful</code>, <code>sad</code>, <code>angry</code>, <code>newscast</code>,
        <code>customerservice</code>, <code>assistant</code><br>
        Check <em>Voice Gallery</em> at <code>speech.microsoft.com/portal</code> for supported styles.
        </div>
        """, unsafe_allow_html=True)

    # ── Tab 3: Speech Analysis ────────────────────────────────────────────────
    with tab3:
        st.markdown("""
        <div class="info-box">
        <strong>Azure AI Speech — Speech-to-Text Concepts</strong><br>
        Azure STT converts spoken audio to text with word-level timestamps, punctuation,
        speaker diarisation (who said what), and profanity filtering.
        This demo analyses text to show phonemic composition — a proxy for acoustic complexity.
        </div>
        """, unsafe_allow_html=True)

        analyze_text = st.text_area(
            "Text to analyse:",
            value="The quick brown fox jumped over the lazy dog near the river.",
            height=80,
        )
        counts = _phoneme_counts(analyze_text)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].bar(list(counts.keys()), list(counts.values()),
                    color=['#4CAF50','#2196F3','#FF5722','#9E9E9E','#FF9800'])
        axes[0].set_title("Character Composition")
        axes[0].set_ylabel("Count")

        syllable_approx = sum(
            max(1, sum(1 for c in w.lower() if c in "aeiou"))
            for w in analyze_text.split()
        )
        word_count = len(analyze_text.split())
        axes[1].bar(["Words","Syllables","Chars"],
                    [word_count, syllable_approx, len(analyze_text.replace(" ",""))],
                    color=['#3F51B5','#009688','#FF5722'])
        axes[1].set_title("Text Metrics")
        axes[1].set_ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        c1, c2, c3 = st.columns(3)
        c1.metric("Words",              word_count)
        c2.metric("Approx. Syllables",  syllable_approx)
        c3.metric("Consonant:Vowel",    f"{counts['Consonants']}:{counts['Vowels']}")

        st.markdown("#### Azure STT Features Reference")
        feat = pd.DataFrame([
            {"Feature": "Continuous recognition", "API parameter": "continuous=true",
             "Notes": "Transcribe long audio streams"},
            {"Feature": "Word timestamps",        "API parameter": "wordLevelTimestamps=true",
             "Notes": "Start/end ms per word"},
            {"Feature": "Speaker diarisation",    "API parameter": "diarizationEnabled=true",
             "Notes": "Distinguish multiple speakers"},
            {"Feature": "Punctuation",            "API parameter": "punctuation=DictatedAndAutomatic",
             "Notes": "Auto-punctuate transcript"},
            {"Feature": "Profanity filter",       "API parameter": "profanity=Masked",
             "Notes": "Raw, Masked, or Removed"},
        ])
        st.dataframe(feat, hide_index=True, use_container_width=True)

    # ── Tab 4: Audio Features ─────────────────────────────────────────────────
    with tab4:
        st.markdown("""
        <div class="info-box">
        <strong>Audio Features — Spectrogram Concepts</strong><br>
        A spectrogram shows how frequency content changes over time — the foundation of how
        neural STT models interpret speech. The x-axis is time, y-axis is frequency,
        and colour represents energy (power spectral density).
        </div>
        """, unsafe_allow_html=True)

        spec_text = st.text_input(
            "Text (affects synthetic signal complexity):",
            value="Artificial intelligence transforms how we build software."
        )

        f, t_sp, Sxx = _spectrogram(spec_text)
        Sxx_db = 10 * np.log10(Sxx + 1e-12)

        fig, ax = plt.subplots(figsize=(11, 4))
        pcm = ax.pcolormesh(t_sp, f[:40], Sxx_db[:40, :], cmap='inferno', shading='gouraud')
        plt.colorbar(pcm, ax=ax, label="Power (dB)")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title("Simulated Spectrogram")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("#### MFCC Concept")
        st.markdown("""
        **Mel-Frequency Cepstral Coefficients (MFCCs)** are the feature vectors used by
        almost all modern speech recognition systems:
        1. Apply a window to short audio frame (~25 ms)
        2. Compute FFT → power spectrum
        3. Apply mel filter bank (log-spaced, matching human hearing)
        4. Take log of filter bank energies
        5. Apply DCT → MFCCs (typically 13–40 coefficients per frame)

        Azure STT handles all of this internally — you just send raw WAV/MP3/OGG audio.
        """)

        mfcc_sim = np.random.randn(13, 50)
        for i in range(13):
            for j in range(50):
                mfcc_sim[i, j] = np.sin(np.pi * i / 13) * np.cos(np.pi * j / 50) * (hash(spec_text + str(i)) % 10 / 10.0)

        fig2, ax2 = plt.subplots(figsize=(11, 3))
        pcm2 = ax2.pcolormesh(mfcc_sim, cmap='RdBu_r', shading='auto')
        plt.colorbar(pcm2, ax=ax2)
        ax2.set_xlabel("Frame index")
        ax2.set_ylabel("MFCC Coefficient")
        ax2.set_title("Simulated MFCC Coefficients (13 x 50 frames)")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        st.markdown("""
        <div class="exam-tip">
        <strong>Exam Tip:</strong> Azure AI Speech main capabilities:<br>
        &bull; <strong>STT</strong> — Audio → text (recognizeOnceAsync or continuous)<br>
        &bull; <strong>TTS</strong> — Text → audio, via plain text or SSML<br>
        &bull; <strong>Speech Translation</strong> — STT + translation in one service<br>
        &bull; <strong>Speaker Recognition</strong> — Identify or verify a speaker by voice
        </div>
        """, unsafe_allow_html=True)
