from pathlib import Path
import re
import pandas as pd
import fitz
from dateparser.search import search_dates
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, classification_report, confusion_matrix

CATEGORIES = [
    "Examination", "Event", "Holiday", "Assignment/Submission",
    "Placement", "General Announcement"
]
SUMMARIZER_MODEL = "sshleifer/distilbart-cnn-6-6"


def load_dataset(csv_path):
    df = pd.read_csv(csv_path).dropna(subset=["text", "category"])
    df["text"] = df["text"].astype(str)
    df["category"] = df["category"].astype(str)
    return df


def make_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 2),
            max_features=12000, sublinear_tf=True
        )),
        ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced"))
    ])


def load_classifier(csv_path):
    df = load_dataset(csv_path)
    if df["category"].nunique() < 2 or len(df) < 12:
        raise ValueError("Dataset must contain at least two categories and 12 notices.")
    model = make_model()
    model.fit(df["text"], df["category"])
    return model


def classify_notice(model, text):
    probs = model.predict_proba([text])[0]
    index = probs.argmax()
    return str(model.classes_[index]), float(probs[index])


def evaluate_classifier(csv_path):
    df = load_dataset(csv_path)
    counts = df["category"].value_counts()
    if len(counts) < 2 or counts.min() < 2:
        raise ValueError("Each category needs at least two examples for evaluation.")
    train_x, test_x, train_y, test_y = train_test_split(
        df["text"], df["category"], test_size=0.25, random_state=42, stratify=df["category"]
    )
    model = make_model()
    model.fit(train_x, train_y)
    pred = model.predict(test_x)
    metrics = {
        "accuracy": accuracy_score(test_y, pred),
        "macro_precision": precision_score(test_y, pred, average="macro", zero_division=0),
        "macro_f1": f1_score(test_y, pred, average="macro", zero_division=0)
    }
    labels = sorted(df["category"].unique())
    report_dict = classification_report(test_y, pred, labels=labels, output_dict=True, zero_division=0)
    report = pd.DataFrame([
        {"Category": label, "Precision": report_dict[label]["precision"],
         "Recall": report_dict[label]["recall"], "F1-score": report_dict[label]["f1-score"],
         "Support": int(report_dict[label]["support"])}
        for label in labels
    ])
    cm = confusion_matrix(test_y, pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.index.name = "Actual / Predicted"
    return metrics, report, cm_df


def extract_pdf_text(file_bytes):
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        pages = [page.get_text("text") for page in doc]
    return "\n".join(pages).strip()


def get_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        return None


def extract_entities(text):
    nlp = get_spacy_model()
    if nlp is None:
        return []
    doc = nlp(text[:100000])
    seen = set()
    entities = []
    for ent in doc.ents:
        item = (ent.text.strip(), ent.label_)
        if item[0] and item not in seen:
            seen.add(item)
            entities.append({"Entity": item[0], "Type": item[1]})
    return entities


def extract_dates(text):
    found = search_dates(
        text,
        settings={"PREFER_DATES_FROM": "future", "RETURN_AS_TIMEZONE_AWARE": False}
    ) if text.strip() else None
    results = []
    seen = set()
    if found:
        for phrase, parsed in found:
            phrase_clean = phrase.strip()
            key = (phrase_clean.lower(), parsed.date().isoformat())
            if not phrase_clean or key in seen:
                continue
            seen.add(key)
            pos = text.lower().find(phrase_clean.lower())
            left = max(0, pos - 65)
            right = min(len(text), pos + len(phrase_clean) + 65)
            context = re.sub(r"\s+", " ", text[left:right]).strip()
            results.append({
                "Extracted text": phrase_clean,
                "Parsed date": parsed.strftime("%Y-%m-%d"),
                "Context": ("…" if left > 0 else "") + context + ("…" if right < len(text) else "")
            })
    return results


def extractive_fallback(text, max_sentences=3):
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    if not sentences:
        return text.strip()[:600]
    if len(sentences) <= max_sentences:
        return " ".join(sentences)
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    stop = set("the and for are with this that from have has will must should into your their they them was were been being can could would may might about after before".split())
    freq = {}
    for word in words:
        if word not in stop:
            freq[word] = freq.get(word, 0) + 1
    scored = []
    for i, sentence in enumerate(sentences):
        tokens = [w for w in re.findall(r"\b[a-zA-Z]{3,}\b", sentence.lower()) if w not in stop]
        score = sum(freq.get(w, 0) for w in tokens) / max(len(tokens), 1)
        scored.append((score, i, sentence))
    selected = sorted(sorted(scored, reverse=True)[:max_sentences], key=lambda x: x[1])
    return " ".join(item[2] for item in selected)


def summarize_text(text):
    # The first call downloads model files; if anything fails, use a transparent local fallback.
    try:
        summarizer = _get_summarizer()
        if len(text.split()) < 45:
            return extractive_fallback(text, max_sentences=2), "Short notice: using extractive summary."
        words = text.split()
        chunks = []
        step = 750
        for start in range(0, len(words), step):
            chunk = " ".join(words[start:start + 850])
            if chunk:
                chunks.append(chunk)
        summaries = []
        for chunk in chunks[:6]:
            out = summarizer(chunk, max_length=100, min_length=20, do_sample=False, truncation=True)
            summaries.append(out[0]["summary_text"].strip())
        result = " ".join(summaries).strip()
        return result or extractive_fallback(text), f"Generated with {SUMMARIZER_MODEL}."
    except Exception as exc:
        return extractive_fallback(text), f"Pretrained summarizer unavailable; used extractive fallback ({type(exc).__name__})."


_SUMMARIZER = None


def _get_summarizer():
    global _SUMMARIZER
    if _SUMMARIZER is None:
        from transformers import pipeline
        _SUMMARIZER = pipeline(
            "summarization", model=SUMMARIZER_MODEL,
            tokenizer=SUMMARIZER_MODEL, device=-1
        )
    return _SUMMARIZER
