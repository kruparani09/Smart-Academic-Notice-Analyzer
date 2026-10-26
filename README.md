# Smart Academic Notice Analyzer

An individual NLP mini-project using Python and Streamlit. It classifies academic notices, extracts dates and named entities, summarizes text, and exports an analysis report.

## Features
- Paste a notice or upload a selectable-text PDF/TXT.
- Six categories: Examination, Event, Holiday, Assignment/Submission, Placement, General Announcement.
- TF-IDF + Logistic Regression classification.
- spaCy named entity recognition.
- Date expression extraction with nearby context.
- Hugging Face summarization with an extractive fallback.
- Classifier evaluation dashboard and downloadable TXT report.

## Run locally
Recommended: Python 3.11 (64-bit).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Visit https://share.streamlit.io/ and sign in with GitHub.
2. Click **Create app**.
3. Select repository `kruparani09/Smart-Academic-Notice-Analyzer`, branch `main`, and main file path `app.py`.
4. In **Advanced settings**, choose **Python 3.11** (the spaCy model dependency is pinned for spaCy 3.7.x).
5. Click **Deploy**. Initial setup can take several minutes because Python packages may need to download. The large summarization model downloads only when a user enables summarization.

## Important deployment note
The summarizer `sshleifer/distilbart-cnn-6-6` downloads model weights on first use, so the first summary can be slow and use considerable memory. If deployment fails because the instance is memory-limited, the summarizer can be disabled or replaced with the built-in extractive fallback.

## Dataset and limitations
`data/notices.csv` contains illustrative example notices for a demonstration; it is not a benchmark dataset. Evaluation scores from this small dataset are not evidence of real-world performance. Add a larger, representative, consistently labeled dataset before making accuracy claims.

- Scanned PDFs need OCR, which is not included.
- Date extraction finds date-like expressions; verify whether they are deadlines or event dates.
- Named entity recognition may miss uncommon names or locations.
- Summaries may omit important instructions, so always verify dates and deadlines.

## Project structure
```text
app.py
requirements.txt
README.md
data/notices.csv
src/__init__.py
src/pipeline.py
```
