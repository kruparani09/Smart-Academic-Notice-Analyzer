from pathlib import Path
import sys
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.pipeline import (
    load_classifier, classify_notice, extract_entities, extract_dates,
    summarize_text, extract_pdf_text, evaluate_classifier, CATEGORIES
)

st.set_page_config(page_title="Smart Academic Notice Analyzer", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Training the notice classifier...")
def get_classifier():
    return load_classifier(ROOT / "data" / "notices.csv")


@st.cache_data(show_spinner=False)
def get_metrics():
    return evaluate_classifier(ROOT / "data" / "notices.csv")


st.title("📚 Smart Academic Notice Analyzer")
st.caption(
    "An NLP mini-project for classifying academic notices, extracting important information, "
    "and generating concise summaries."
)
with st.sidebar:
    st.header("About this project")
    st.write("**NLP techniques**")
    st.markdown(
        "- TF-IDF + Logistic Regression\n"
        "- Named Entity Recognition (spaCy)\n"
        "- Date extraction\n"
        "- Pretrained summarization with an extractive fallback"
    )
    st.divider()
    st.write("**Privacy:** Uploaded text is processed by this app. If enabled, the pretrained summarizer downloads and runs on the app host.")
    st.info("The first analysis may take longer while the classifier and language model initialize.")

tab_analyze, tab_metrics, tab_about = st.tabs(
    ["🔎 Analyze a notice", "📊 Model evaluation", "ℹ️ Methodology"]
)

with tab_analyze:
    st.subheader("Provide an academic notice")
    mode = st.radio("Input method", ["Paste text", "Upload file"], horizontal=True)
    text = ""
    uploaded_name = "Pasted notice"

    if mode == "Paste text":
        text = st.text_area(
            "Notice text",
            height=220,
            placeholder="Paste the full academic notice here..."
        )
    else:
        uploaded = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
        if uploaded is not None:
            uploaded_name = uploaded.name
            if uploaded.name.lower().endswith(".pdf"):
                try:
                    text = extract_pdf_text(uploaded.getvalue())
                except Exception as exc:
                    st.error(f"Could not read this PDF: {exc}")
            else:
                text = uploaded.getvalue().decode("utf-8", errors="replace")
            if not text.strip():
                st.warning(
                    "No selectable text was found. This may be a scanned PDF; OCR is not included."
                )
            else:
                st.success(f"Text extracted from {uploaded_name}")

    st.caption("Tip: confirm all predicted categories and extracted deadlines against the original notice.")
    summarize = st.checkbox(
        "Generate a summary using the pretrained model (may be slow on first use)",
        value=False
    )
    analyze = st.button("Analyze notice", type="primary", use_container_width=True)

    if analyze:
        if not text.strip():
            st.warning("Please paste notice text or upload a readable file first.")
        elif len(text.strip()) < 25:
            st.warning("The notice is very short. Add more text for more meaningful results.")
        else:
            with st.spinner("Analyzing the notice..."):
                classifier = get_classifier()
                category, confidence = classify_notice(classifier, text)
                entities = extract_entities(text)
                dates = extract_dates(text)
                if summarize:
                    summary, summary_note = summarize_text(text)
                else:
                    summary = ""
                    summary_note = "Summary skipped by user."

            st.divider()
            st.subheader("Analysis results")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted category", category)
            with col2:
                st.metric("Classifier confidence", f"{confidence:.1%}")
            st.caption(
                "Confidence is the model's estimated probability, not a guarantee that the prediction is correct."
            )
            if confidence < 0.45:
                st.warning("The classifier is uncertain. Please verify the predicted category.")

            if summary:
                st.markdown("### 📝 Summary")
                st.write(summary)
                st.caption(summary_note)
            st.markdown("### 📅 Dates and possible deadlines")
            if dates:
                dates_df = pd.DataFrame(dates)
                st.dataframe(dates_df, use_container_width=True, hide_index=True)
                st.caption(
                    "Dates are extracted from the text. Verify their meaning and whether they are actual deadlines."
                )
            else:
                st.write("No dates were detected.")

            st.markdown("### 🏷️ Named entities")
            if entities:
                ent_df = pd.DataFrame(entities)
                st.dataframe(ent_df, use_container_width=True, hide_index=True)
            else:
                st.write("No named entities were detected, or the spaCy language model is unavailable.")

            st.markdown("### 📄 Extracted source text")
            with st.expander("View original text"):
                st.text(text)

            date_report = dates_df.to_csv(index=False) if dates else "No dates detected"
            entity_report = ent_df.to_csv(index=False) if entities else "No entities detected"
            report = [
                "SMART ACADEMIC NOTICE ANALYZER",
                f"Input: {uploaded_name}",
                f"Predicted category: {category}",
                f"Classifier confidence: {confidence:.1%}",
                "",
                "SUMMARY",
                summary or "Not generated",
                "",
                "DATES",
                date_report,
                "",
                "NAMED ENTITIES",
                entity_report,
                "",
                "ORIGINAL TEXT",
                text
            ]
            st.download_button(
                "⬇️ Download analysis report (.txt)",
                data="\n".join(report),
                file_name="notice_analysis_report.txt",
                mime="text/plain",
                use_container_width=True
            )

with tab_metrics:
    st.subheader("Classifier evaluation")
    st.write(
        "Metrics are calculated using a held-out test split from the bundled sample dataset. "
        "Because this starter dataset is small and illustrative, results are not evidence of performance "
        "on all real-world college notices."
    )
    try:
        metrics, report, cm_df = get_metrics()
        a, b, c = st.columns(3)
        a.metric("Test accuracy", f"{metrics['accuracy']:.1%}")
        b.metric("Macro precision", f"{metrics['macro_precision']:.1%}")
        c.metric("Macro F1", f"{metrics['macro_f1']:.1%}")
        st.markdown("#### Per-category report")
        st.dataframe(report, use_container_width=True, hide_index=True)
        st.markdown("#### Confusion matrix")
        st.dataframe(cm_df, use_container_width=True)
    except Exception as exc:
        st.error(f"Could not evaluate classifier: {exc}")

with tab_about:
    st.subheader("Project methodology")
    st.markdown(
        """
1. **Input:** A user supplies an academic notice as text or a PDF/TXT file.
2. **Text extraction:** PyMuPDF extracts selectable text from PDFs.
3. **Classification:** TF-IDF turns text into numeric features; Logistic Regression predicts a notice category.
4. **Entity extraction:** spaCy identifies entities such as people, organizations, and locations.
5. **Date extraction:** dateparser detects date expressions and shows nearby context.
6. **Summarization:** A pretrained Hugging Face model can be enabled. If it cannot load, the app uses an extractive fallback.
7. **Output:** Results are displayed and can be downloaded as a text report.

**Limitations:** The bundled training data is a small illustrative dataset. Classification quality should be re-evaluated using a larger, representative, manually labeled dataset. Named entity recognition may miss uncommon names or locations. Scanned PDF OCR is not included.
        """
    )
    st.markdown("#### Supported categories")
    st.write(", ".join(CATEGORIES))
    st.markdown("#### Dataset")
    st.write(
        "The bundled data/notices.csv file contains example notices for demonstration and development. "
        "Extend it with appropriately sourced and consistently labeled notices before making strong claims about model performance."
    )
