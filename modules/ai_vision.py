import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import io


# ── Helper functions ───────────────────────────────────────────────────────────

def simulate_tags(img_array):
    """Return (tag, confidence) pairs derived from colour statistics."""
    r = img_array[:, :, 0].mean()
    g = img_array[:, :, 1].mean()
    b = img_array[:, :, 2].mean()
    brightness = (r + g + b) / 3

    tags = []
    if brightness > 180:
        tags += [("bright", 0.95), ("light", 0.88), ("overexposed", 0.72)]
    elif brightness < 80:
        tags += [("dark", 0.93), ("night", 0.80), ("low-light", 0.74)]

    dominant = ["red", "green", "blue"][int(np.argmax([r, g, b]))]
    if dominant == "green":
        tags += [("nature", 0.91), ("outdoor", 0.86), ("vegetation", 0.78)]
    elif dominant == "blue":
        tags += [("sky", 0.89), ("water", 0.77), ("cool", 0.72)]
    else:
        tags += [("warm", 0.85), ("indoor", 0.75), ("sunset", 0.68)]

    gray = img_array.mean(axis=2)
    edge = (np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean()) / 2
    if edge > 8:
        tags += [("detailed", 0.84), ("complex", 0.76)]
    else:
        tags += [("smooth", 0.80), ("minimal", 0.73)]

    return sorted(tags, key=lambda x: -x[1])[:8]


def make_synthetic_scene():
    """Return a (H, W, 3) uint8 array with a simple outdoor scene."""
    arr = np.ones((400, 600, 3), dtype=np.uint8) * 220
    arr[:160, :]       = [135, 200, 245]   # sky
    arr[300:, :]       = [101, 143,  58]   # grass
    arr[80:300, 90:250]  = [180, 160, 140] # building
    arr[110:165, 120:175] = [100, 150, 200] # window 1
    arr[110:165, 190:245] = [100, 150, 200] # window 2
    arr[275:330, 350:520] = [200,  60,  60] # car
    arr[210:305, 458:492] = [101,  67,  33] # tree trunk
    arr[145:225, 425:525] = [ 34, 120,  34] # tree top
    return arr


def make_doc_image():
    """Return a PIL Image of a synthetic invoice document."""
    img = Image.new("RGB", (460, 340), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 460, 40], fill=(30, 90, 180))
    draw.text((10, 10), "INVOICE", fill=(255, 255, 255))
    lines = [
        (30,  60, "Invoice #:    INV-20260508"),
        (30,  85, "Date:         08 May 2026"),
        (30, 110, "Bill To:      Contoso Ltd."),
        (30, 135, "Service:      AI Workshop x 2"),
        (30, 160, "Unit Price:   $125.00"),
        (30, 185, "Subtotal:     $250.00"),
        (30, 210, "Tax (10%):    $25.00"),
        (30, 240, "TOTAL DUE:    $275.00"),
        (30, 280, "Due Date:     01 June 2026"),
        (30, 310, "Thank you for your business!"),
    ]
    for x, y, text in lines:
        draw.rectangle([x - 2, y - 2, 420, y + 16], outline="#BBBBBB")
        draw.text((x, y), text, fill=(30, 30, 30))
    return img


# ── Main page ──────────────────────────────────────────────────────────────────

def show():
    st.markdown('<p class="gradient-title">👁️ Azure AI Vision</p>', unsafe_allow_html=True)
    st.markdown("### Simulating Azure AI Vision — 100% Local with Python + PIL + scikit-learn")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Image Analysis", "📝 OCR (Read API)", "📦 Object Detection", "🏷️ Custom Vision",
    ])

    # ─── Tab 1: Image Analysis ────────────────────────────────────────────────
    with tab1:
        st.markdown("""
        <div class="info-box">
        <strong>Azure AI Vision — Image Analysis</strong><br>
        The Image Analysis API accepts an image and returns <em>tags</em>, <em>objects</em>,
        <em>descriptions</em>, <em>colour scheme</em>, and <em>metadata</em> in a single JSON
        response. Here we replicate the same logic locally using PIL and NumPy.
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Upload an image (JPG/PNG) or scroll down to use the synthetic sample",
            type=["jpg", "jpeg", "png", "bmp"],
        )

        if uploaded:
            img = Image.open(uploaded).convert("RGB")
        else:
            arr_sample = np.zeros((300, 400, 3), dtype=np.uint8)
            arr_sample[:150, :200] = [34, 139, 34]
            arr_sample[:150, 200:] = [135, 206, 235]
            arr_sample[150:, :]    = [210, 180, 140]
            img = Image.fromarray(arr_sample)
            st.info("No image uploaded — using a synthetic colour-block sample.")

        img_array = np.array(img)
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("**Original Image**")
            st.image(img, use_column_width=True)

        with col2:
            st.markdown("**Simulated Tags (confidence)**")
            tags = simulate_tags(img_array)
            for tag, conf in tags:
                st.markdown(f"`{tag}` — {conf:.0%}")
                st.progress(conf)

        st.markdown("#### Colour Channel Histograms")
        fig, axes = plt.subplots(1, 3, figsize=(12, 3))
        for i, (name, color) in enumerate(zip(
            ["Red Channel", "Green Channel", "Blue Channel"],
            ["#e74c3c", "#27ae60", "#2980b9"],
        )):
            hist, bins = np.histogram(img_array[:, :, i], bins=32, range=(0, 256))
            axes[i].fill_between(bins[:-1], hist, alpha=0.75, color=color)
            axes[i].set_title(name)
            axes[i].set_xlabel("Pixel Intensity (0-255)")
            axes[i].set_ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        r, g, b = img_array[:,:,0].mean(), img_array[:,:,1].mean(), img_array[:,:,2].mean()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Width (px)",   img.size[0])
        c2.metric("Height (px)",  img.size[1])
        c3.metric("Brightness",   f"{(r+g+b)/3:.0f}/255")
        c4.metric("Dominant",     ["Red","Green","Blue"][int(np.argmax([r,g,b]))])

        st.markdown("""
        <div class="exam-tip">
        <strong>Exam Tip:</strong> The Azure AI Vision Image Analysis v4.0 API endpoint is<br>
        <code>POST /computervision/imageanalysis:analyze?features=tags,objects,caption,color</code><br>
        The response includes <code>tags[].confidence</code> (0.0–1.0) and
        <code>objects[].boundingBox</code> (normalised fractions).
        </div>
        """, unsafe_allow_html=True)

    # ─── Tab 2: OCR ───────────────────────────────────────────────────────────
    with tab2:
        st.markdown("""
        <div class="info-box">
        <strong>Azure AI Vision — Read API (OCR)</strong><br>
        The Read API is the recommended OCR solution. It handles printed and handwritten text,
        supports multi-page PDFs, and returns bounding polygons plus confidence for every word.
        It uses an async pattern: POST to start, then GET to poll for results.
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Synthetic Document Image**")
            doc_img = make_doc_image()
            st.image(doc_img, use_column_width=True)

        with col2:
            st.markdown("**Simulated Read API Response**")
            ocr_data = [
                {"Line": 1, "Extracted Text":    "Invoice #:    INV-20260508", "Confidence": 0.97},
                {"Line": 2, "Extracted Text":    "Date:         08 May 2026",  "Confidence": 0.95},
                {"Line": 3, "Extracted Text":    "Bill To:      Contoso Ltd.", "Confidence": 0.93},
                {"Line": 4, "Extracted Text":    "Service:      AI Workshop",  "Confidence": 0.96},
                {"Line": 5, "Extracted Text":    "Unit Price:   $125.00",      "Confidence": 0.99},
                {"Line": 6, "Extracted Text":    "Subtotal:     $250.00",      "Confidence": 0.98},
                {"Line": 7, "Extracted Text":    "Tax (10%):    $25.00",       "Confidence": 0.97},
                {"Line": 8, "Extracted Text":    "TOTAL DUE:    $275.00",      "Confidence": 0.99},
                {"Line": 9, "Extracted Text":    "Due Date:     01 June 2026", "Confidence": 0.94},
                {"Line":10, "Extracted Text":    "Thank you for your business!","Confidence": 0.91},
            ]
            df = pd.DataFrame(ocr_data)
            st.dataframe(df, hide_index=True, use_container_width=True)
            avg = sum(r["Confidence"] for r in ocr_data) / len(ocr_data)
            st.metric("Average Confidence", f"{avg:.1%}")

        st.markdown("#### Read API Flow (Async Pattern)")
        st.code("""
# Step 1 — submit the image
POST /vision/v3.2/read/analyze
Headers: Ocp-Apim-Subscription-Key: <key>
Body: binary image or { "url": "https://..." }
Response: 202 Accepted
          Operation-Location: https://.../read/analyzeResults/{operationId}

# Step 2 — poll until status = "succeeded"
GET {Operation-Location}
Response: { "status": "succeeded",
            "analyzeResult": {
              "readResults": [{
                "lines": [{ "text": "Invoice #", "words": [...] }]
              }] } }
        """, language="text")

        st.markdown("""
        <div class="concept-box">
        <strong>Read vs OCR API:</strong><br>
        &bull; <strong>Read (v3.2+)</strong> — async, handles multi-page PDFs, best accuracy, recommended<br>
        &bull; <strong>OCR (v3.0)</strong> — synchronous, images only, lower latency for small images<br>
        &bull; For AI-102: always use the <em>Read API</em> for new solutions
        </div>
        """, unsafe_allow_html=True)

    # ─── Tab 3: Object Detection ──────────────────────────────────────────────
    with tab3:
        st.markdown("""
        <div class="info-box">
        <strong>Azure AI Vision — Object Detection</strong><br>
        Returns detected objects with a class label, confidence score, and a bounding box
        expressed as normalised fractions <code>{ left, top, width, height }</code> of the image dimensions.
        </div>
        """, unsafe_allow_html=True)

        scene = make_synthetic_scene()

        detections = [
            {"Object": "Building",  "Confidence": 0.93, "box": (90, 80, 160, 220)},
            {"Object": "Car",       "Confidence": 0.89, "box": (350, 275, 170, 55)},
            {"Object": "Tree",      "Confidence": 0.87, "box": (425, 145, 100, 160)},
            {"Object": "Sky",       "Confidence": 0.97, "box": (0, 0, 600, 160)},
        ]
        colors_map = {"Building": "#FF5722", "Car": "#2196F3", "Tree": "#4CAF50", "Sky": "#9C27B0"}

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(scene)
        for det in detections:
            x, y, w, h = det["box"]
            color = colors_map[det["Object"]]
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            ax.text(x + 4, y - 6, f"{det['Object']} {det['Confidence']:.0%}",
                    color='white', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.85))
        ax.set_title("Simulated Object Detection with Bounding Boxes", fontsize=13)
        ax.axis('off')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        det_df = pd.DataFrame([
            {
                "Object":     d["Object"],
                "Confidence": f"{d['Confidence']:.0%}",
                "Box (x,y,w,h)": str(d["box"]),
                "Normalised left": f"{d['box'][0]/600:.3f}",
                "Normalised top":  f"{d['box'][1]/400:.3f}",
            }
            for d in detections
        ])
        st.dataframe(det_df, hide_index=True, use_container_width=True)

        st.markdown("""
        <div class="exam-tip">
        <strong>Exam Tip:</strong> Object Detection bounding boxes use <em>normalised fractions</em>
        (0.0–1.0), not pixel coordinates. Multiply by image width/height to get actual pixels.
        The JSON response field is <code>boundingBox: { left, top, width, height }</code>.
        </div>
        """, unsafe_allow_html=True)

    # ─── Tab 4: Custom Vision ─────────────────────────────────────────────────
    with tab4:
        st.markdown("""
        <div class="info-box">
        <strong>Azure Custom Vision</strong><br>
        Train your own image classifier or object detector with a small number of labelled images.
        Azure Custom Vision wraps a CNN training pipeline with a simple portal and REST API.
        Here we replicate the training + evaluation pipeline using colour features and scikit-learn.
        </div>
        """, unsafe_allow_html=True)

        np.random.seed(42)
        n = 160
        r0 = np.column_stack([np.random.normal(205, 18, n), np.random.normal(75, 18, n),  np.random.normal(75, 18, n)])
        r1 = np.column_stack([np.random.normal(75, 18, n),  np.random.normal(205, 18, n), np.random.normal(75, 18, n)])
        r2 = np.column_stack([np.random.normal(75, 18, n),  np.random.normal(75, 18, n),  np.random.normal(205, 18, n)])
        X = np.vstack([r0, r1, r2]).clip(0, 255)
        y = np.array([0]*n + [1]*n + [2]*n)
        class_names = ["Warm/Red", "Nature/Green", "Sky/Blue"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        n_trees = st.slider("Number of trees (model complexity):", 10, 200, 50, step=10)
        clf = RandomForestClassifier(n_estimators=n_trees, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = (y_pred == y_test).mean()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Custom Vision Accuracy", f"{acc:.1%}")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names, ax=ax)
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
            rows = [{"Class": c, "Precision": f"{report[c]['precision']:.2f}",
                     "Recall": f"{report[c]['recall']:.2f}", "F1": f"{report[c]['f1-score']:.2f}"}
                    for c in class_names]
            st.markdown("**Per-Class Performance**")
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            st.markdown("""
            <div class="concept-box">
            <strong>Custom Vision Workflow:</strong><br>
            1. Create a Custom Vision project (Classification or Detection)<br>
            2. Upload and tag images (min 15 per tag)<br>
            3. Train — Azure manages CNN fine-tuning<br>
            4. Evaluate precision/recall per tag<br>
            5. Publish iteration to a prediction endpoint<br>
            6. Call <code>POST /classify/iterations/{name}/image</code>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="exam-tip">
        <strong>Exam Tip:</strong> Know the difference between <em>Classification</em>
        (multi-class or multi-label — what is in the image?) and <em>Object Detection</em>
        (where is the object? returns bounding boxes). Minimum 15 images per tag for training.
        Use <code>customvision.ai</code> portal or the Azure AI Vision SDK.
        </div>
        """, unsafe_allow_html=True)
