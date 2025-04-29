### --- Full Final Streamlit UID Matcher Script --- ###

# --- Install and Import Required Libraries ---
try:
    import streamlit as st
    import pandas as pd
    import requests
    import re
    import logging
    import json
    from uuid import uuid4
    from sqlalchemy import create_engine, text
    from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
    from sklearn.metrics.pairwise import cosine_similarity
    from sentence_transformers import SentenceTransformer, util
    from rapidfuzz import fuzz
except ImportError as e:
    missing_package = str(e).split("No module named ")[-1].replace("'", "")
    raise ImportError(f"""
Missing required package: {missing_package}
Please install all required libraries first by running:
pip install pandas openpyxl rapidfuzz python-Levenshtein SQLAlchemy scikit-learn sentence-transformers streamlit requests
""")

# --- Streamlit Setup and Logger ---
st.set_page_config(page_title="UID Matcher Combined", layout="wide")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants ---
TFIDF_HIGH_CONFIDENCE = 0.60
TFIDF_LOW_CONFIDENCE = 0.50
SEMANTIC_THRESHOLD = 0.60
FUZZY_THRESHOLD = 0.95
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 1000

DEFAULT_SYNONYM_MAP = {
    "please select": "what is",
    "sector you are from": "your sector",
    "identity type": "id type",
    "what type of": "type of",
    "are you": "do you",
}

# --- Utility Functions ---
@st.cache_resource
def load_sentence_transformer():
    logger.info(f"Loading SentenceTransformer model: {MODEL_NAME}")
    return SentenceTransformer(MODEL_NAME)

def enhanced_normalize(text, synonym_map=DEFAULT_SYNONYM_MAP):
    text = str(text).lower()
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'[^a-z0-9 ]', '', text)
    for phrase, replacement in synonym_map.items():
        text = text.replace(phrase, replacement)
    return ' '.join(w for w in text.split() if w not in ENGLISH_STOP_WORDS)

@st.cache_data
def get_tfidf_vectors(df_reference):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    vectors = vectorizer.fit_transform(df_reference["norm_text"])
    return vectorizer, vectors

@st.cache_resource
def get_snowflake_engine():
    try:
        sf = st.secrets["snowflake"]
        logger.info(f"Connecting to Snowflake: user={sf.user}, account={sf.account}")
        engine = create_engine(
            f"snowflake://{sf.user}:{sf.password}@{sf.account}/{sf.database}/{sf.schema}?warehouse={sf.warehouse}&role={sf.role}"
        )
        with engine.connect() as conn:
            conn.execute(text("SELECT CURRENT_VERSION()"))
        return engine
    except Exception as e:
        logger.error(f"Snowflake connection failed: {e}")
        if "250001" in str(e):
            st.warning("Snowflake connection failed: User account is locked.")
        raise

def run_snowflake_reference_query(limit=10000, offset=0):
    query = """
        SELECT HEADING_0, MAX(UID) AS UID
        FROM AMI_DBT.DBT_SURVEY_MONKEY.SURVEY_DETAILS_RESPONSES_COMBINED_LIVE
        WHERE UID IS NOT NULL
        GROUP BY HEADING_0
        LIMIT :limit OFFSET :offset
    """
    with get_snowflake_engine().connect() as conn:
        return pd.read_sql(text(query), conn, params={"limit": limit, "offset": offset})

def run_snowflake_target_query():
    query = """
        SELECT DISTINCT HEADING_0
        FROM AMI_DBT.DBT_SURVEY_MONKEY.SURVEY_DETAILS_RESPONSES_COMBINED_LIVE
        WHERE UID IS NULL AND NOT LOWER(HEADING_0) LIKE 'our privacy policy%'
    """
    with get_snowflake_engine().connect() as conn:
        return pd.read_sql(text(query), conn)

def compute_tfidf_matches(df_reference, df_target, synonym_map=DEFAULT_SYNONYM_MAP):
    df_reference = df_reference[df_reference["heading_0"].notna()].reset_index(drop=True)
    df_target = df_target[df_target["heading_0"].notna()].reset_index(drop=True)

    df_reference["norm_text"] = df_reference["heading_0"].apply(enhanced_normalize)
    df_target["norm_text"] = df_target["heading_0"].apply(enhanced_normalize)

    vectorizer, ref_vectors = get_tfidf_vectors(df_reference)
    target_vectors = vectorizer.transform(df_target["norm_text"])
    similarity_matrix = cosine_similarity(target_vectors, ref_vectors)

    matched_uids, matched_qs, scores, confs = [], [], [], []
    for sim_row in similarity_matrix:
        best_idx = sim_row.argmax()
        best_score = sim_row[best_idx]
        if best_score >= TFIDF_HIGH_CONFIDENCE:
            conf = "✅ High"
        elif best_score >= TFIDF_LOW_CONFIDENCE:
            conf = "⚠️ Low"
        else:
            conf = "❌ No match"
            best_idx = None
        matched_uids.append(df_reference.iloc[best_idx]["UID"] if best_idx is not None else None)
        matched_qs.append(df_reference.iloc[best_idx]["heading_0"] if best_idx is not None else None)
        scores.append(round(best_score, 4))
        confs.append(conf)

    df_target["Suggested_UID"] = matched_uids
    df_target["Matched_Question"] = matched_qs
    df_target["Similarity"] = scores
    df_target["Match_Confidence"] = confs
    return df_target

def compute_semantic_matches(df_reference, df_target):
    model = load_sentence_transformer()
    emb_target = model.encode(df_target["heading_0"].tolist(), convert_to_tensor=True)
    emb_ref = model.encode(df_reference["heading_0"].tolist(), convert_to_tensor=True)
    cosine_scores = util.cos_sim(emb_target, emb_ref)

    sem_matches, sem_scores = [], []
    for i in range(len(df_target)):
        best_idx = cosine_scores[i].argmax().item()
        score = cosine_scores[i][best_idx].item()
        sem_matches.append(df_reference.iloc[best_idx]["UID"] if score >= SEMANTIC_THRESHOLD else None)
        sem_scores.append(round(score, 4) if score >= SEMANTIC_THRESHOLD else None)

    df_target["Semantic_UID"] = sem_matches
    df_target["Semantic_Similarity"] = sem_scores
    return df_target

def get_surveys(token):
    url = "https://api.surveymonkey.com/v3/surveys"
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json().get("data", [])

def get_survey_details(survey_id, token):
    url = f"https://api.surveymonkey.com/v3/surveys/{survey_id}/details"
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

# --- Streamlit App Logic ---
option = st.radio("Select Data Source", ["SurveyMonkey", "Snowflake"], horizontal=True)

if option == "Snowflake":
    if st.button("🔁 Run UID Matching on Snowflake Data"):
        try:
            with st.spinner("Fetching Snowflake Data..."):
                df_reference = run_snowflake_reference_query()
                df_target = run_snowflake_target_query()

            df_target = compute_tfidf_matches(df_reference, df_target)
            df_target = compute_semantic_matches(df_reference, df_target)

            st.success("UID Matching Completed!")
            st.dataframe(df_target)

            st.download_button(
                "Download Results as CSV",
                df_target.to_csv(index=False).encode('utf-8'),
                "uid_matching_results.csv",
                "text/csv"
            )

        except Exception as e:
            st.error(f"Error: {e}")

elif option == "SurveyMonkey":
    st.title("SurveyMonkey Integration")

    token = st.secrets.get("surveymonkey", {}).get("token", None)
    if not token:
        st.error("SurveyMonkey token missing in secrets.")
        st.stop()

    with st.spinner("Fetching available surveys..."):
        surveys = get_surveys(token)

    survey_dict = {s['title']: s['id'] for s in surveys}

    selected_title = st.selectbox("Choose Survey", list(survey_dict.keys()))
    selected_id = survey_dict[selected_title]

    if st.button("Fetch Survey Details"):
        with st.spinner("Fetching Survey Details..."):
            survey_json = get_survey_details(selected_id, token)
            questions = []
            for page in survey_json.get("pages", []):
                for question in page.get("questions", []):
                    questions.append(question.get("headings", [{}])[0].get("heading", ""))

            df_questions = pd.DataFrame({"Question": questions})
            st.dataframe(df_questions)

            st.download_button(
                "Download Questions as CSV",
                df_questions.to_csv(index=False).encode('utf-8'),
                "survey_questions.csv",
                "text/csv"
            )
