import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


# ── Sample documents ───────────────────────────────────────────────────────────

INVOICE_TEXT = """
INVOICE
Invoice Number: INV-20260508
Invoice Date: May 8, 2026
Due Date: June 1, 2026
Bill To: Contoso Ltd., 123 Main Street, Seattle WA 98101

Line Items:
  1. Azure AI Workshop (2 days) .... $250.00
  2. Travel & Expenses ............. $85.00
  3. Materials & Handouts .......... $40.00

Subtotal: $375.00
Tax (8.5%): $31.88
Total Due: $406.88

Payment Method: Bank Transfer
Bank: Woodgrove Bank  Account: 12-345678  Sort: 04-00-75
"""

RECEIPT_TEXT = """
NORTHWIND GROCERY
123 Commerce Street, Redmond WA
Tel: (425) 555-0198

Date: 2026-05-08   Time: 14:33
Cashier: Alice

  Organic Milk 2L          $3.49
  Sourdough Bread 800g     $4.29
  Free Range Eggs x12      $5.99
  Butter 250g              $2.89
  Coffee Beans 250g        $8.79
  Orange Juice 1L          $3.49

Subtotal:                 $28.94
Tax (9.0%):                $2.60
TOTAL:                    $31.54

Card: **** 4892   Auth: 829140
Thank you for shopping with us!
"""

TABLE_TEXT = """
| Product         | Category    | Units Sold | Unit Price | Revenue    |
|-----------------|-------------|-----------|-----------|------------|
| Widget Alpha    | Electronics |       142 |    $29.99  |  $4,258.58 |
| Gadget Beta     | Electronics |        87 |    $59.99  |  $5,219.13 |
| Book Gamma      | Education   |       310 |     $9.99  |  $3,096.90 |
| Course Delta    | Education   |        55 |   $199.00  | $10,945.00 |
| Tool Epsilon    | Tools       |       203 |    $14.99  |  $3,042.97 |
"""

CLASSIFY_DOCS = {
    "invoice": [
        "Invoice for consulting services. Total due $500. Payment due 30 days.",
        "Please remit payment for the following invoice. Invoice number 4521.",
        "Bill for professional services rendered. Subtotal $750. Tax $60. Total $810.",
        "Invoice from Fabrikam Corp for AI training. Due date January 31.",
        "Payment requested for completed development work. Amount $1,200 net 30.",
    ],
    "receipt": [
        "Thank you for your purchase at Northwind Store. Total $45.99.",
        "Receipt for coffee and pastry. Change $4.01. Have a great day!",
        "Store receipt. 3 items purchased. Total $23.75. Card ending 1234.",
        "Grocery purchase receipt. Items: bread, milk, eggs. Total $18.40.",
        "Restaurant receipt. Table 5. Food $62. Tip $10. Total $72.",
    ],
    "contract": [
        "This agreement is entered into between the parties on the date written below.",
        "Service Level Agreement for cloud infrastructure. Term 12 months.",
        "Non-disclosure agreement between Contoso and Fabrikam. Effective immediately.",
        "Master Service Agreement governing all work orders issued hereunder.",
        "Employment contract outlining duties, compensation, and termination clauses.",
    ],
    "report": [
        "Q1 Financial Report. Revenue increased 12% year-over-year to $4.2M.",
        "Annual performance review of AI systems deployed in production.",
        "Quarterly business review: customer satisfaction score 4.3/5.",
        "Technical report on model accuracy improvements following retraining.",
        "Executive summary: market analysis and competitive landscape overview.",
    ],
}


# ── Extraction helpers ─────────────────────────────────────────────────────────

def _extract_invoice(text):
    fields = {}
    for key, patterns in [
        ("Invoice Number",  [r'Invoice (?:Number|#|No)[:\s#]+([A-Z0-9\-]+)']),
        ("Invoice Date",    [r'Invoice Date[:\s]+(.+)']),
        ("Due Date",        [r'Due Date[:\s]+(.+)']),
        ("Bill To",         [r'Bill To[:\s]+(.+)']),
        ("Subtotal",        [r'Subtotal[:\s]+(\$[\d,\.]+)']),
        ("Tax",             [r'Tax[^:]*[:\s]+(\$[\d,\.]+)']),
        ("Total Due",       [r'Total Due[:\s]+(\$[\d,\.]+)']),
    ]:
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                fields[key] = (m.group(1).strip(), round(np.random.uniform(0.92, 0.99), 2))
                break
        else:
            fields[key] = ("Not found", 0.0)
    return fields


def _extract_receipt(text):
    fields = {}
    patterns = {
        "Merchant":   r'^([A-Z][A-Z &]+)\n',
        "Date":       r'Date[:\s]+(\d{4}-\d{2}-\d{2})',
        "Total":      r'TOTAL[:\s]+(\$[\d\.]+)',
        "Tax":        r'Tax[^:]*[:\s]+(\$[\d\.]+)',
        "Subtotal":   r'Subtotal[:\s]+(\$[\d\.]+)',
        "Card Last4": r'\*{4}\s*(\d{4})',
    }
    for key, pat in patterns.items():
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        fields[key] = (m.group(1).strip() if m else "Not found",
                       round(np.random.uniform(0.88, 0.99), 2) if m else 0.0)
    return fields


def _parse_table(text):
    lines = [l for l in text.strip().split('\n') if '|' in l and '---' not in l]
    if len(lines) < 2:
        return pd.DataFrame()
    header = [h.strip() for h in lines[0].strip('|').split('|')]
    rows = []
    for line in lines[1:]:
        cells = [c.strip() for c in line.strip('|').split('|')]
        if len(cells) == len(header):
            rows.append(cells)
    return pd.DataFrame(rows, columns=header)


# ── Main page ──────────────────────────────────────────────────────────────────

def show():
    st.markdown('<p class="gradient-title">📄 Azure Document Intelligence</p>', unsafe_allow_html=True)
    st.markdown("### Simulating Azure Document Intelligence — Local Regex + ML Extraction")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🧾 Invoice Model", "🛒 Receipt Model", "📊 Table Extraction", "🗂️ Document Classification",
    ])

    # ── Tab 1: Invoice ────────────────────────────────────────────────────────
    with tab1:
        st.markdown("""
        <div class="info-box">
        <strong>Azure Document Intelligence — Invoice Model</strong><br>
        A prebuilt model trained on millions of invoices. Extracts structured fields
        (VendorName, InvoiceDate, TotalAmount, LineItems…) with confidence scores — no custom training needed.
        </div>
        """, unsafe_allow_html=True)

        inv_text = st.text_area("Invoice text (editable):", value=INVOICE_TEXT, height=250)
        if st.button("Extract Invoice Fields"):
            fields = _extract_invoice(inv_text)
            rows = [{"Field": k, "Extracted Value": v[0], "Confidence": f"{v[1]:.0%}"}
                    for k, v in fields.items()]
            df = pd.DataFrame(rows)
            st.dataframe(df, hide_index=True, use_container_width=True)

            line_items = re.findall(r'\d+\.\s(.+?)\s+(\$[\d,\.]+)', inv_text)
            if line_items:
                st.markdown("**Extracted Line Items**")
                li_df = pd.DataFrame(line_items, columns=["Description", "Amount"])
                st.dataframe(li_df, hide_index=True, use_container_width=True)

        st.markdown("""
        <div class="concept-box">
        <strong>Prebuilt Invoice Fields (Azure):</strong><br>
        VendorName, VendorAddress, CustomerName, CustomerAddress, InvoiceId,
        InvoiceDate, DueDate, SubTotal, TotalTax, InvoiceTotal, AmountDue,
        Items[].Description, Items[].Quantity, Items[].UnitPrice, Items[].Amount
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="exam-tip">
        <strong>Exam Tip:</strong> Azure Document Intelligence prebuilt models:<br>
        <code>prebuilt-invoice</code>, <code>prebuilt-receipt</code>,
        <code>prebuilt-idDocument</code>, <code>prebuilt-businessCard</code>,
        <code>prebuilt-layout</code> (for table + text structure),
        <code>prebuilt-read</code> (OCR only).<br>
        Custom models: <em>Custom Template</em> (fixed layout) vs <em>Custom Neural</em> (varied layout).
        </div>
        """, unsafe_allow_html=True)

    # ── Tab 2: Receipt ────────────────────────────────────────────────────────
    with tab2:
        st.markdown("""
        <div class="info-box">
        <strong>Azure Document Intelligence — Receipt Model</strong><br>
        Extracts MerchantName, MerchantAddress, TransactionDate, TransactionTime,
        Items (Name, Quantity, Price), Subtotal, Tax, Tip, and Total.
        </div>
        """, unsafe_allow_html=True)

        rec_text = st.text_area("Receipt text (editable):", value=RECEIPT_TEXT, height=250)
        if st.button("Extract Receipt Fields"):
            fields = _extract_receipt(rec_text)
            rows = [{"Field": k, "Extracted Value": v[0], "Confidence": f"{v[1]:.0%}"}
                    for k, v in fields.items()]
            df = pd.DataFrame(rows)
            st.dataframe(df, hide_index=True, use_container_width=True)

            items = re.findall(r'([A-Za-z][A-Za-z0-9 ]+?)\s{2,}(\$[\d\.]+)', rec_text)
            if items:
                st.markdown("**Extracted Line Items**")
                items_df = pd.DataFrame(items, columns=["Item", "Price"])
                st.dataframe(items_df, hide_index=True, use_container_width=True)

        receipt_types = pd.DataFrame([
            {"Receipt Type": "Retail",       "Example": "Grocery, clothing store"},
            {"Receipt Type": "Meal",         "Example": "Restaurant, cafe, food delivery"},
            {"Receipt Type": "Gas",          "Example": "Fuel station receipts"},
            {"Receipt Type": "Hotel",        "Example": "Hotel folio / accommodation"},
            {"Receipt Type": "CreditCard",   "Example": "Credit card slip"},
        ])
        st.markdown("**Supported Receipt Types**")
        st.dataframe(receipt_types, hide_index=True, use_container_width=True)

    # ── Tab 3: Table Extraction ───────────────────────────────────────────────
    with tab3:
        st.markdown("""
        <div class="info-box">
        <strong>Azure Document Intelligence — Table Extraction</strong><br>
        The <code>prebuilt-layout</code> model extracts tables with cell content,
        row/column spans, and bounding polygons. Returns a structured JSON with
        <code>tables[].cells[].rowIndex</code>, <code>.columnIndex</code>, <code>.content</code>.
        </div>
        """, unsafe_allow_html=True)

        table_text = st.text_area("Markdown table (editable):", value=TABLE_TEXT, height=200)
        df = _parse_table(table_text)
        if not df.empty:
            st.markdown("**Extracted Table**")
            st.dataframe(df, hide_index=True, use_container_width=True)
            st.metric("Rows extracted", len(df))
            st.metric("Columns detected", len(df.columns))

            conf_data = np.random.uniform(0.88, 0.99, (len(df), len(df.columns)))
            conf_df = pd.DataFrame(conf_data, columns=df.columns).round(2)
            st.markdown("**Confidence per Cell (simulated)**")
            st.dataframe(conf_df, hide_index=True, use_container_width=True)
        else:
            st.warning("Could not parse table. Ensure it uses `|` delimiters.")

        st.markdown("""
        <div class="exam-tip">
        <strong>Exam Tip:</strong> <code>prebuilt-layout</code> returns:<br>
        &bull; <code>paragraphs[]</code> — text blocks with role (title, sectionHeading, footnote…)<br>
        &bull; <code>tables[]</code> — structured cells with row/col indices<br>
        &bull; <code>figures[]</code> — detected images/charts<br>
        &bull; Works on scanned PDFs, image files (JPEG, PNG, TIFF), and Office files
        </div>
        """, unsafe_allow_html=True)

    # ── Tab 4: Document Classification ───────────────────────────────────────
    with tab4:
        st.markdown("""
        <div class="info-box">
        <strong>Azure Document Intelligence — Custom Classification</strong><br>
        A <em>Custom Classification</em> model learns to route documents to one of N categories.
        Here we train a TF-IDF + Naive Bayes classifier on four document types locally.
        </div>
        """, unsafe_allow_html=True)

        all_texts, all_labels = [], []
        for label, docs in CLASSIFY_DOCS.items():
            all_texts += docs
            all_labels += [label] * len(docs)

        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words='english')),
            ("clf",   MultinomialNB()),
        ])
        X_tr, X_te, y_tr, y_te = train_test_split(
            all_texts, all_labels, test_size=0.25, random_state=42
        )
        pipe.fit(X_tr, y_tr)
        acc = pipe.score(X_te, y_te)
        st.metric("Classifier Accuracy (held-out)", f"{acc:.0%}")

        user_doc = st.text_area(
            "Paste a document snippet to classify:",
            value="Please find attached the invoice for services rendered. Total amount due is $850.",
            height=100,
        )
        if user_doc.strip():
            pred = pipe.predict([user_doc])[0]
            proba = pipe.predict_proba([user_doc])[0]
            classes = pipe.classes_
            col1, col2 = st.columns(2)
            col1.metric("Predicted Type", pred.upper())
            col1.metric("Confidence",     f"{max(proba):.0%}")

            with col2:
                fig, ax = plt.subplots(figsize=(5, 3))
                colors = ['#2196F3' if c == pred else '#90CAF9' for c in classes]
                ax.barh(list(classes), proba, color=colors)
                ax.set_xlim(0, 1)
                ax.set_xlabel("Probability")
                ax.set_title("Classification Probabilities")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        st.markdown("""
        <div class="concept-box">
        <strong>Azure Custom Classification Workflow:</strong><br>
        1. Label documents in Azure Document Intelligence Studio<br>
        2. Train — min 5 samples per class recommended<br>
        3. Evaluate precision/recall per class<br>
        4. Deploy classifier → call <code>analyzeDocument</code> with model ID<br>
        5. Response: <code>documents[].docType</code> + <code>confidence</code>
        </div>
        """, unsafe_allow_html=True)
