import logging
import os
import re
import webbrowser
from pathlib import Path
from threading import Timer

import difflib
import nltk
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, flash, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent

# Download required NLTK data once at startup
for _resource, _category in [('stopwords', 'corpora'), ('punkt', 'tokenizers'), ('punkt_tab', 'tokenizers')]:
    try:
        nltk.data.find(f'{_category}/{_resource}')
    except LookupError:
        nltk.download(_resource, quiet=True)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))

# ── NLP helpers (module-level, not nested inside routes) ─────────────────────
_stop_words = set(stopwords.words('english'))


def preprocess_text(text: str) -> str:
    words = word_tokenize(str(text).lower())
    return ' '.join(w for w in words if w.isalpha() and w not in _stop_words)


def correct_symptoms(symptoms: list, vocabulary: list) -> list:
    corrected = []
    for sym in symptoms:
        match = difflib.get_close_matches(sym, vocabulary, n=1, cutoff=0.5)
        corrected.append(match[0] if match else sym)
    return corrected


# ── Model cache — trained once at startup, reused for every request ───────────
_cache: dict = {}


def _load_models() -> dict:
    log.info("Loading CSVs and training models...")

    # ── Disease prediction: KNN + TF-IDF ──────────────────────────────────
    data = pd.read_csv(BASE_DIR / 'Symptom2Disease.csv')
    data.drop(columns=[c for c in data.columns if 'Unnamed' in c], inplace=True)

    preprocessed = data['text'].apply(preprocess_text)
    tfidf_disease = TfidfVectorizer(max_features=2000)
    X = tfidf_disease.fit_transform(preprocessed).toarray()

    X_train, X_test, y_train, y_test = train_test_split(
        X, data['label'], test_size=0.2, random_state=42
    )
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    accuracy = round(accuracy_score(y_test, preds) * 100, 1)
    report_html = (
        pd.DataFrame(classification_report(y_test, preds, output_dict=True))
        .transpose()
        .round(3)
        .to_html(classes='report-table', border=0)
    )
    log.info(f"KNN trained. Accuracy: {accuracy}%")

    # ── Formulation recommendation: Naive Bayes + TF-IDF ──────────────────
    df_form = pd.read_csv(BASE_DIR / 'Formulation-Indications.csv')
    indications = df_form['Main Indications'].astype(str).apply(
        lambda x: x.replace(',', ' ').lower()
    )
    tfidf_form = TfidfVectorizer()
    X_form = tfidf_form.fit_transform(indications)
    nb = MultinomialNB()
    nb.fit(X_form, df_form['Name of Medicine'])

    # Vocabulary for spell correction (built from all indication tokens)
    vocab = list({w for ind in indications for w in ind.split()})
    log.info("Naive Bayes formulation model trained.")

    # ── Formulation class lookup (A → "Asava Arista", etc.) ───────────────
    class_map: dict = {}
    class_csv = BASE_DIR / 'FormulationClass.csv'
    if class_csv.exists():
        df_class = pd.read_csv(class_csv, header=0)
        desc_col = df_class.columns[1]
        class_map = {
            str(k).strip(): str(v).strip()
            for k, v in zip(df_class['Class'], df_class[desc_col])
        }

    # ── Optional ayurvedic symptom lookup table ────────────────────────────
    symptoms_df = None
    sym_csv = BASE_DIR / 'ayurvedic_symptoms_desc_updated.csv'
    if sym_csv.exists():
        symptoms_df = pd.read_csv(sym_csv)
        symptoms_df['Symptom'] = symptoms_df['Symptom'].str.lower().str.strip()
        log.info("Ayurvedic symptom lookup loaded.")
    else:
        log.warning(
            "ayurvedic_symptoms_desc_updated.csv not found — "
            "using simplified formulation matching via disease name."
        )

    return {
        'tfidf_disease': tfidf_disease,
        'knn': knn,
        'accuracy': accuracy,
        'report_html': report_html,
        'df_form': df_form,
        'tfidf_form': tfidf_form,
        'nb': nb,
        'vocab': vocab,
        'class_map': class_map,
        'symptoms_df': symptoms_df,
    }


def get_models() -> dict:
    if not _cache:
        _cache.update(_load_models())
    return _cache


# ── Prediction pipeline ───────────────────────────────────────────────────────
def predict(user_input: str) -> dict:
    m = get_models()

    # Step 1 — predict disease
    preprocessed = preprocess_text(user_input)
    if not preprocessed.strip():
        raise ValueError("Could not extract meaningful symptoms. Please describe them in more detail.")

    disease_vec = m['tfidf_disease'].transform([preprocessed])
    predicted_disease = m['knn'].predict(disease_vec)[0]

    # Step 2 — build symptom query for formulation lookup
    if m['symptoms_df'] is not None:
        # Full pipeline: map user words → ayurvedic symptom DB → get Symptom keywords
        words = [w for w in re.split(r'[,\s]+', user_input.lower()) if w.strip()]
        sym_df = m['symptoms_df'].copy()
        sym_df['score'] = sym_df['English_Symptoms'].apply(
            lambda x: sum(w in str(x).lower() for w in words)
        )
        matched = sym_df[sym_df['score'] > 0].sort_values('score', ascending=False).head(10)
        query_keywords = ' '.join(matched['Symptom'].tolist()) or predicted_disease.lower()
    else:
        # Simplified: disease name drives the formulation lookup
        query_keywords = predicted_disease.lower()

    # Step 3 — spell-correct keywords, predict top 3 formulations
    keyword_list = query_keywords.split()
    corrected = correct_symptoms(keyword_list, m['vocab'])

    formulation_names: list = []
    seen: set = set()
    for kw in corrected:
        try:
            kw_vec = m['tfidf_form'].transform([kw])
            pred = m['nb'].predict(kw_vec)[0]
            if pred not in seen:
                seen.add(pred)
                formulation_names.append(pred)
                if len(formulation_names) >= 3:
                    break
        except Exception:
            continue

    # Fallback: use disease name as query directly
    if not formulation_names:
        try:
            disease_q = m['tfidf_form'].transform([predicted_disease.lower()])
            formulation_names = [m['nb'].predict(disease_q)[0]]
        except Exception:
            pass

    # Step 4 — fetch full medicine details
    df_form = m['df_form']
    class_map = m['class_map']
    medicines = []
    for name in formulation_names:
        rows = df_form[df_form['Name of Medicine'] == name]
        if not rows.empty:
            r = rows.iloc[0]
            code = str(r.get('Class', ''))
            medicines.append({
                'name': str(r.get('Name of Medicine', '')),
                'indications': str(r.get('Main Indications', '')),
                'dose': str(r.get('Dose', 'N/A')),
                'precaution': str(r.get('Precaution/ Contraindication', 'N/A')),
                'reference': str(r.get('Reference text', '')),
                'preferred_use': str(r.get('Preferred use (OPD/ IPD)', 'N/A')),
                'class_code': code,
                'class_name': class_map.get(code, code),
            })

    return {
        'disease': predicted_disease,
        'medicines': medicines,
        'accuracy': m['accuracy'],
        'report_html': m['report_html'],
    }


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    symptom_input = ''

    if request.method == 'POST':
        symptom_input = request.form.get('symptomInput', '').strip()
        if not symptom_input:
            flash("Please describe your symptoms.")
        elif len(symptom_input) < 5:
            flash("Please provide more detail about your symptoms.")
        else:
            try:
                result = predict(symptom_input)
            except ValueError as e:
                flash(str(e))
            except Exception as e:
                log.error(f"Prediction error: {e}", exc_info=True)
                flash("An error occurred during analysis. Please try again.")

    return render_template('index.html', result=result, symptom_input=symptom_input)


if __name__ == '__main__':
    get_models()  # pre-warm before first request
    Timer(1, webbrowser.open, args=('http://127.0.0.1:5000',)).start()
    app.run(debug=os.environ.get('FLASK_DEBUG', '0') == '1')
