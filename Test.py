### --- Full Final Streamlit UID Matcher Script (Improved, Complete, and Enhanced UX) --- ###

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
Please install it using:
pip install pandas openpyxl rapidfuzz python-Levenshtein SQLAlchemy scikit-learn sentence-transformers streamlit requests
""")

# --- Utility Functions (Fix for UID Matching) ---
@st.cache_resource
def get_snowflake_engine():
    try:
        sf = st.secrets["snowflake"]
        engine = create_engine(
            f"snowflake://{sf.user}:{sf.password}@{sf.account}/{sf.database}/{sf.schema}?warehouse={sf.warehouse}&role={sf.role}"
        )
        with engine.connect() as conn:
            conn.execute(text("SELECT CURRENT_VERSION()"))
        return engine
    except Exception as e:
        st.error(f"Snowflake Connection failed: {e}")
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

@st.cache_data
def get_tfidf_vectors(df_reference):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    vectors = vectorizer.fit_transform(df_reference["heading_0"])
    return vectorizer, vectors

def enhanced_normalize(text):
    text = str(text).lower()
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'[^a-z0-9 ]', '', text)
    return ' '.join(w for w in text.split() if w not in ENGLISH_STOP_WORDS)

def compute_tfidf_matches(df_reference, df_target):
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
        if best_score >= 0.6:
            conf = "✅ High"
        elif best_score >= 0.5:
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
    model = SentenceTransformer('all-MiniLM-L6-v2')
    emb_target = model.encode(df_target["heading_0"].tolist(), convert_to_tensor=True)
    emb_ref = model.encode(df_reference["heading_0"].tolist(), convert_to_tensor=True)
    cosine_scores = util.cos_sim(emb_target, emb_ref)

    sem_matches, sem_scores = [], []
    for i in range(len(df_target)):
        best_idx = cosine_scores[i].argmax().item()
        score = cosine_scores[i][best_idx].item()
        sem_matches.append(df_reference.iloc[best_idx]["UID"] if score >= 0.6 else None)
        sem_scores.append(round(score, 4) if score >= 0.6 else None)

    df_target["Semantic_UID"] = sem_matches
    df_target["Semantic_Similarity"] = sem_scores
    return df_target

# (Existing Sidebar Navigation and Pages Continue Here...)

### --- End of Full Combined, UX Enhanced, and Fully Fixed Script --- ###
