# Combined Streamlit App: SurveyMonkey + Snowflake UID Matching with Choices Support

import streamlit as st
import pandas as pd
import requests
import re
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="UID Matcher Combined", layout="wide")

# --- UID Matching Logic ---
synonym_map = {
    "please select": "what is",
    "sector you are from": "your sector",
    "identity type": "id type",
    "what type of": "type of",
    "are you": "do you",
}

def apply_synonyms(text):
    for phrase, replacement in synonym_map.items():
        text = text.replace(phrase, replacement)
    return text

def enhanced_normalize(text):
    text = str(text).lower()
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'[^a-z0-9 ]', '', text)
    text = apply_synonyms(text)
    words = text.split()
    return ' '.join([w for w in words if w not in ENGLISH_STOP_WORDS])

def run_uid_match(df_mapped, df_unmapped):
    df_mapped = df_mapped[df_mapped["heading_0"].notna()].reset_index(drop=True)
    df_unmapped = df_unmapped[df_unmapped["heading_0"].notna()].reset_index(drop=True)
    df_mapped["norm_text"] = df_mapped["heading_0"].apply(enhanced_normalize)
    df_unmapped["norm_text"] = df_unmapped["heading_0"].apply(enhanced_normalize)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    vectorizer.fit(df_mapped["norm_text"].tolist() + df_unmapped["norm_text"].tolist())
    similarity_matrix = cosine_similarity(vectorizer.transform(df_unmapped["norm_text"]), vectorizer.transform(df_mapped["norm_text"]))

    matched_uids, matched_qs, scores, confs = [], [], [], []
    for i, sim_row in enumerate(similarity_matrix):
        best_idx = sim_row.argmax()
        best_score = sim_row[best_idx]
        if best_score >= 0.60:
            uid = df_mapped.iloc[best_idx]["uid"]
            q = df_mapped.iloc[best_idx]["heading_0"]
            conf = "✅ High"
        elif best_score >= 0.50:
            uid = df_mapped.iloc[best_idx]["uid"]
            q = df_mapped.iloc[best_idx]["heading_0"]
            conf = "⚠️ Low"
        else:
            uid, q, conf = None, None, "❌ No match"
        matched_uids.append(uid)
        matched_qs.append(q)
        scores.append(round(best_score, 4))
        confs.append(conf)

    df_unmapped["Suggested_UID"] = matched_uids
    df_unmapped["Matched_Question"] = matched_qs
    df_unmapped["Similarity"] = scores
    df_unmapped["Match_Confidence"] = confs

    st.info("Running semantic fallback...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb_u = model.encode(df_unmapped["heading_0"].tolist(), convert_to_tensor=True)
    emb_m = model.encode(df_mapped["heading_0"].tolist(), convert_to_tensor=True)
    cosine_scores = util.cos_sim(emb_u, emb_m)

    sem_matches, sem_scores = [], []
    for i in range(len(df_unmapped)):
        best_idx = cosine_scores[i].argmax().item()
        score = cosine_scores[i][best_idx].item()
        if score >= 0.60:
            sem_matches.append(df_mapped.iloc[best_idx]["uid"])
            sem_scores.append(round(score, 4))
        else:
            sem_matches.append(None)
            sem_scores.append(None)

    df_unmapped["Semantic_UID"] = sem_matches
    df_unmapped["Semantic_Similarity"] = sem_scores
    df_unmapped["Final_UID"] = df_unmapped["Suggested_UID"].combine_first(df_unmapped["Semantic_UID"])
    df_unmapped["Final_Question"] = df_unmapped["Matched_Question"]
    df_unmapped["Final_Match_Type"] = df_unmapped.apply(lambda row: row["Match_Confidence"] if pd.notnull(row["Suggested_UID"]) else ("🧠 Semantic" if pd.notnull(row["Semantic_UID"]) else "❌ No match"), axis=1)

    uid_conflicts = df_unmapped.groupby("Final_UID")["heading_0"].nunique()
    duplicate_uids = uid_conflicts[uid_conflicts > 1].index
    df_unmapped["UID_Conflict"] = df_unmapped["Final_UID"].apply(lambda x: "⚠️ Conflict" if x in duplicate_uids else "")

    return df_unmapped

# --- Snowflake Setup ---
@st.cache_resource
def get_snowflake_engine():
    sf = st.secrets["snowflake"]
    return create_engine(f"snowflake://{sf.user}:{sf.password}@{sf.account}/{sf.database}/{sf.schema}?warehouse={sf.warehouse}&role={sf.role}")

def run_snowflake_query():
    engine = get_snowflake_engine()
    df = pd.read_sql("""
        SELECT HEADING_0, MAX(UID) AS UID
        FROM AMI_DBT.DBT_SURVEY_MONKEY.SURVEY_DETAILS_RESPONSES_COMBINED_LIVE
        WHERE UID IS NOT NULL
        GROUP BY HEADING_0
    """, engine)
    return df

# --- SurveyMonkey API ---
def get_surveys(token):
    url = "https://api.surveymonkey.com/v3/surveys"
    headers = {"Authorization": f"Bearer {token}"}
    return requests.get(url, headers=headers).json()["data"]

def get_survey_details(survey_id, token):
    url = f"https://api.surveymonkey.com/v3/surveys/{survey_id}/details"
    headers = {"Authorization": f"Bearer {token}"}
    return requests.get(url, headers=headers).json()

def extract_questions(survey_json):
    questions = []
    for page in survey_json.get("pages", []):
        for question in page.get("questions", []):
            q_text = question.get("headings", [{}])[0].get("heading", "")
            questions.append(q_text)
            for choice in question.get("answers", {}).get("choices", []):
                choice_text = choice.get("text", "")
                if choice_text:
                    questions.append(f"{q_text} - {choice_text}")
    return questions

# --- App UI ---
st.title("🧠 UID Matcher: Snowflake + SurveyMonkey")

option = st.radio("Choose Source", ["SurveyMonkey", "Snowflake"], horizontal=True)

if option == "SurveyMonkey":
    token = st.secrets["surveymonkey"]["token"]
    if token:
        try:
            surveys = get_surveys(token)
            choices = {s['title']: s['id'] for s in surveys}
            selected = st.selectbox("Choose survey", list(choices.keys()))
            if selected:
                survey_json = get_survey_details(choices[selected], token)
                questions = extract_questions(survey_json)
                df_unmapped = pd.DataFrame({"heading_0": questions})
                df_mapped = pd.DataFrame(columns=["heading_0", "uid"])  # dummy fallback if Snowflake not available
                df_final = run_uid_match(df_mapped, df_unmapped)
                st.dataframe(df_final)
                st.download_button("📥 Download UID Matches", df_final.to_csv(index=False), "uid_matches.csv")
        except Exception as e:
            st.error(f"Error: {e}")

if option == "Snowflake":
    if st.button("🔁 Run Matching on Snowflake data"):
        df_mapped = run_snowflake_query()
        df_unmapped = pd.read_sql("""
            SELECT DISTINCT HEADING_0 FROM AMI_DBT.DBT_SURVEY_MONKEY.SURVEY_DETAILS_RESPONSES_COMBINED_LIVE
            WHERE UID IS NULL AND NOT LOWER(HEADING_0) LIKE 'our privacy policy%'
        """, get_snowflake_engine())
        df_final = run_uid_match(df_mapped, df_unmapped)
        st.dataframe(df_final)
        st.download_button("📥 Download UID Matches", df_final.to_csv(index=False), "uid_matches.csv")
