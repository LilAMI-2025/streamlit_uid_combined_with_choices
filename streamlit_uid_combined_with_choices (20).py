import streamlit as st
import pandas as pd
import requests
import re
import logging
import json
import time
import os
from uuid import uuid4
from sqlalchemy import create_engine, text
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Setup
st.set_page_config(page_title="UID Matcher Optimized", layout="wide")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
TFIDF_HIGH_CONFIDENCE = 0.60
TFIDF_LOW_CONFIDENCE = 0.50
SEMANTIC_THRESHOLD = 0.60
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 1000
CACHE_FILE = "survey_cache.json"
REQUEST_DELAY = 0.5
SURVEYS_PER_PAGE = 10

# Synonym Mapping
DEFAULT_SYNONYM_MAP = {
    "please select": "what is",
    "sector you are from": "your sector",
    "identity type": "id type",
    "what type of": "type of",
    "are you": "do you",
}

# Cached Resources
@st.cache_resource
def load_sentence_transformer():
    logger.info(f"Loading SentenceTransformer model: {MODEL_NAME}")
    return SentenceTransformer(MODEL_NAME)

@st.cache_resource
def get_snowflake_engine():
    try:
        sf = st.secrets["snowflake"]
        engine = create_engine(
            f"snowflake://{sf.user}:{sf.password}@{sf.account}/{sf.database}/{sf.schema}"
            f"?warehouse={sf.warehouse}&role={sf.role}"
        )
        with engine.connect() as conn:
            conn.execute(text("SELECT CURRENT_VERSION()"))
        return engine
    except Exception as e:
        logger.error(f"Snowflake engine creation failed: {e}")
        if "250001" in str(e):
            st.warning("Snowflake connection failed: User account locked. UID matching disabled.")
        raise

@st.cache_data
def get_tfidf_vectors(df_reference):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    vectors = vectorizer.fit_transform(df_reference["norm_text"])
    return vectorizer, vectors

# Cache Management
def load_cached_survey_data():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                cache = json.load(f)
            cache_time = cache.get("timestamp", 0)
            if time.time() - cache_time < 24 * 3600:
                return (
                    pd.DataFrame(cache.get("all_questions", [])),
                    cache.get("dedup_questions", []),
                    cache.get("dedup_choices", [])
                )
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
    return None, [], []

def save_cached_survey_data(all_questions, dedup_questions, dedup_choices):
    cache = {
        "timestamp": time.time(),
        "all_questions": all_questions.to_dict(orient="records"),
        "dedup_questions": dedup_questions,
        "dedup_choices": dedup_choices
    }
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f)
    except Exception as e:
        logger.error(f"Failed to save cache: {e}")

# Normalization
def enhanced_normalize(text, synonym_map=DEFAULT_SYNONYM_MAP):
    text = str(text).lower()
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'[^a-z0-9 ]', '', text)
    for phrase, replacement in synonym_map.items():
        text = text.replace(phrase, replacement)
    return ' '.join(w for w in text.split() if w not in ENGLISH_STOP_WORDS)

# Calculate Matched Questions Percentage
def calculate_matched_percentage(df_final):
    if df_final is None or df_final.empty:
        return 0.0
    df_main = df_final[df_final["is_choice"] == False].copy()
    privacy_filter = ~df_main["heading_0"].str.contains("Our Privacy Policy", case=False, na=False)
    html_pattern = r"<div.*text-align:\s*center.*<span.*font-size:\s*12pt.*<em>If you have any questions, please contact your AMI Learner Success Manager.*</em>.*</span>.*</div>"
    html_filter = ~df_main["heading_0"].str.contains(html_pattern, case=False, na=False, regex=True)
    eligible_questions = df_main[privacy_filter & html_filter]
    if eligible_questions.empty:
        return 0.0
    matched_questions = eligible_questions[eligible_questions["Final_UID"].notna()]
    return round((len(matched_questions) / len(eligible_questions)) * 100, 2)

# Snowflake Queries
def run_snowflake_reference_query(limit=10000, offset=0):
    query = """
        SELECT HEADING_0, MAX(UID) AS UID
        FROM AMI_DBT.DBT_SURVEY_MONKEY.SURVEY_DETAILS_RESPONSES_COMBINED_LIVE
        WHERE HEADING_0 IS NOT NULL AND UID IS NOT NULL
        GROUP BY HEADING_0
        LIMIT :limit OFFSET :offset
    """
    try:
        with get_snowflake_engine().connect() as conn:
            result = pd.read_sql(text(query), conn, params={"limit": limit, "offset": offset})
        return result
    except Exception as e:
        logger.error(f"Snowflake reference query failed: {e}")
        if "250001" in str(e):
            st.warning("Snowflake connection failed: User account locked. UID matching disabled.")
        return pd.DataFrame(columns=["HEADING_0", "UID"])

# SurveyMonkey API
@st.cache_data
def get_surveys_cached(token, page=1, per_page=SURVEYS_PER_PAGE):
    url = f"https://api.surveymonkey.com/v3/surveys?page={page}&per_page={per_page}"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json().get("data", []), response.json().get("page", 1)
    except requests.RequestException as e:
        logger.error(f"Failed to fetch surveys: {e}")
        raise

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(requests.HTTPError)
)
def get_survey_details_with_retry(survey_id, token):
    url = f"https://api.surveymonkey.com/v3/surveys/{survey_id}/details"
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers)
    if response.status_code == 429:
        raise requests.HTTPError("429 Too Many Requests")
    response.raise_for_status()
    return response.json()

def create_survey(token, survey_template):
    url = "https://api.surveymonkey.com/v3/surveys"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    try:
        response = requests.post(url, headers=headers, json={
            "title": survey_template["title"],
            "nickname": survey_template["nickname"],
            "language": survey_template.get("language", "en")
        })
        response.raise_for_status()
        return response.json().get("id")
    except requests.RequestException as e:
        logger.error(f"Failed to create survey: {e}")
        raise

def create_page(token, survey_id, page_template):
    url = f"https://api.surveymonkey.com/v3/surveys/{survey_id}/pages"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    try:
        response = requests.post(url, headers=headers, json={
            "title": page_template.get("title", ""),
            "description": page_template.get("description", "")
        })
        response.raise_for_status()
        return response.json().get("id")
    except requests.RequestException as e:
        logger.error(f"Failed to create page: {e}")
        raise

def create_question(token, survey_id, page_id, question_template):
    url = f"https://api.surveymonkey.com/v3/surveys/{survey_id}/pages/{page_id}/questions"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    try:
        payload = {
            "family": question_template["family"],
            "subtype": question_template["subtype"],
            "headings": [{"heading": question_template["heading"]}],
            "position": question_template["position"],
            "required": question_template.get("is_required", False)
        }
        if "choices" in question_template:
            payload["answers"] = {"choices": question_template["choices"]}
        if question_template["family"] == "matrix":
            payload["answers"] = {
                "rows": question_template.get("rows", []),
                "choices": question_template.get("choices", [])
            }
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json().get("id")
    except Exception as e:
        logger.error(f"Failed to create question: {e}")
        raise

def extract_questions(survey_json):
    questions = []
    global_position = 0
    for page in survey_json.get("pages", []):
        for question in page.get("questions", []):
            q_text = question.get("headings", [{}])[0].get("heading", "")
            q_id = question.get("id", None)
            family = question.get("family", None)
            subtype = question.get("subtype", None)
            if family == "single_choice":
                schema_type = "Single Choice"
            elif family == "multiple_choice":
                schema_type = "Multiple Choice"
            elif family == "open_ended":
                schema_type = "Open-Ended"
            elif family == "matrix":
                schema_type = "Matrix"
            else:
                choices = question.get("answers", {}).get("choices", [])
                schema_type = "Multiple Choice" if choices else "Open-Ended"
                if choices and ("select one" in q_text.lower() or len(choices) <= 2):
                    schema_type = "Single Choice"
            
            if q_text:
                global_position += 1
                questions.append({
                    "heading_0": q_text,  # SurveyMonkey question text
                    "position": global_position,
                    "is_choice": False,
                    "parent_question": None,
                    "question_uid": q_id,
                    "schema_type": schema_type,
                    "mandatory": False,
                    "mandatory_editable": True,
                    "survey_id": survey_json.get("id", ""),
                    "survey_title": survey_json.get("title", "")
                })
                choices = question.get("answers", {}).get("choices", [])
                for choice in choices:
                    choice_text = choice.get("text", "")
                    if choice_text:
                        questions.append({
                            "heading_0": f"{q_text} - {choice_text}",  # SurveyMonkey question-choice text
                            "position": global_position,
                            "is_choice": True,
                            "parent_question": q_text,
                            "question_uid": q_id,
                            "schema_type": schema_type,
                            "mandatory": False,
                            "mandatory_editable": False,
                            "survey_id": survey_json.get("id", ""),
                            "survey_title": survey_json.get("title", "")
                        })
    return questions

# UID Matching
def compute_tfidf_matches(df_reference, df_target, synonym_map=DEFAULT_SYNONYM_MAP):
    df_reference = df_reference[df_reference["HEADING_0"].notna()].reset_index(drop=True)
    df_target = df_target[df_target["heading_0"].notna()].reset_index(drop=True)
    df_reference["norm_text"] = df_reference["HEADING_0"].apply(enhanced_normalize)
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
        matched_qs.append(df_reference.iloc[best_idx]["HEADING_0"] if best_idx is not None else None)
        scores.append(round(best_score, 4))
        confs.append(conf)

    df_target["Suggested_UID"] = matched_uids
    df_target["Matched_Question"] = matched_qs
    df_target["Similarity"] = scores
    df_target["Match_Confidence"] = confs
    return df_target

def compute_semantic_matches(df_reference, df_target):
    try:
        model = load_sentence_transformer()
        emb_target = model.encode(df_target["heading_0"].tolist(), convert_to_tensor=True)
        emb_ref = model.encode(df_reference["HEADING_0"].tolist(), convert_to_tensor=True)
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
    except Exception as e:
        logger.error(f"Semantic matching failed: {e}")
        return df_target

def assign_match_type(row):
    if pd.notnull(row["Suggested_UID"]):
        return row["Match_Confidence"]
    return "🧠 Semantic" if pd.notnull(row["Semantic_UID"]) else "❌ No match"

def finalize_matches(df_target, df_reference):
    df_target["Final_UID"] = df_target["Suggested_UID"].combine_first(df_target["Semantic_UID"])
    df_target["configured_final_UID"] = df_target["Final_UID"]
    df_target["Final_Question"] = df_target["Matched_Question"]
    df_target["Final_Match_Type"] = df_target.apply(assign_match_type, axis=1)
    df_target["Change_UID"] = df_target["Final_UID"].apply(
        lambda x: f"{x} - {df_reference[df_reference['UID'] == x]['HEADING_0'].iloc[0]}" if pd.notnull(x) and x in df_reference["UID"].values else None
    )
    
    df_target["Final_UID"] = df_target.apply(
        lambda row: df_target[df_target["heading_0"] == row["parent_question"]]["Final_UID"].iloc[0]
        if row["is_choice"] and pd.notnull(row["parent_question"]) else row["Final_UID"],
        axis=1
    )
    df_target["configured_final_UID"] = df_target["Final_UID"]
    df_target["Change_UID"] = df_target["Final_UID"].apply(
        lambda x: f"{x} - {df_reference[df_reference['UID'] == x]['HEADING_0'].iloc[0]}" if pd.notnull(x) and x in df_reference["UID"].values else None
    )
    
    return df_target

def detect_uid_conflicts(df_target):
    uid_conflicts = df_target.groupby("Final_UID")["heading_0"].nunique()
    duplicate_uids = uid_conflicts[uid_conflicts > 1].index
    df_target["UID_Conflict"] = df_target["Final_UID"].apply(
        lambda x: "⚠️ Conflict" if pd.notnull(x) and x in duplicate_uids else ""
    )
    return df_target

def run_uid_match(df_reference, df_target, synonym_map=DEFAULT_SYNONYM_MAP, batch_size=BATCH_SIZE):
    if df_reference.empty or df_target.empty:
        logger.warning("Empty input dataframes.")
        return df_target.assign(Final_UID=None)
    
    df_results = []
    for start in range(0, len(df_target), batch_size):
        batch_target = df_target.iloc[start:start + batch_size].copy()
        with st.spinner(f"Processing batch {start//batch_size + 1}..."):
            batch_target = compute_tfidf_matches(df_reference, batch_target, synonym_map)
            batch_target = compute_semantic_matches(df_reference, batch_target)
            batch_target = finalize_matches(batch_target, df_reference)
            batch_target = detect_uid_conflicts(batch_target)
        df_results.append(batch_target)
    
    return pd.concat(df_results, ignore_index=True) if df_results else df_target.assign(Final_UID=None)

# App UI
st.title("🧠 UID Matcher: SurveyMonkey Optimization")

# Secrets Validation
if "snowflake" not in st.secrets or "surveymonkey" not in st.secrets:
    st.error("Missing secrets configuration for Snowflake or SurveyMonkey.")
    st.stop()

# Initialize Session State
if "df_target" not in st.session_state:
    st.session_state.df_target = None
if "df_final" not in st.session_state:
    st.session_state.df_final = None
if "uid_changes" not in st.session_state:
    st.session_state.uid_changes = {}
if "custom_questions" not in st.session_state:
    st.session_state.custom_questions = pd.DataFrame(columns=["Customized Question", "Original Question", "Final_UID"])
if "question_bank" not in st.session_state:
    st.session_state.question_bank = None
if "survey_template" not in st.session_state:
    st.session_state.survey_template = None
if "preview_df" not in st.session_state:
    st.session_state.preview_df = None
if "all_questions" not in st.session_state:
    st.session_state.all_questions = None
if "dedup_questions" not in st.session_state:
    st.session_state.dedup_questions = []
if "dedup_choices" not in st.session_state:
    st.session_state.dedup_choices = []
if "pending_survey" not in st.session_state:
    st.session_state.pending_survey = None
if "fetched_survey_ids" not in st.session_state:
    st.session_state.fetched_survey_ids = []
if "edited_df" not in st.session_state:
    st.session_state.edited_df = pd.DataFrame(columns=["heading_0", "schema_type", "is_choice", "mandatory"])
if "survey_page" not in st.session_state:
    st.session_state.survey_page = 1

# Load SurveyMonkey Data
try:
    token = st.secrets["surveymonkey"]["token"]
    surveys, current_page = get_surveys_cached(token, page=st.session_state.survey_page)
    if not surveys:
        st.error("No surveys found.")
        st.stop()

    # Load cached survey data
    if st.session_state.all_questions is None:
        cached_questions, cached_dedup_questions, cached_dedup_choices = load_cached_survey_data()
        if cached_questions is not None:
            st.session_state.all_questions = cached_questions
            st.session_state.dedup_questions = cached_dedup_questions
            st.session_state.dedup_choices = cached_dedup_choices
            st.session_state.fetched_survey_ids = cached_questions["survey_id"].unique().tolist()

    # Load question bank
    if st.session_state.question_bank is None:
        try:
            st.session_state.question_bank = run_snowflake_reference_query()
        except Exception:
            st.warning("Failed to load question bank. Standardization checks disabled.")
            st.session_state.question_bank = pd.DataFrame(columns=["HEADING_0", "UID"])

except Exception as e:
    st.error(f"SurveyMonkey initialization failed: {e}")
    st.stop()

# Tabs
tab1, tab2, tab3 = st.tabs([
    "Survey Selection & Question Bank",
    "UID Matching & Configuration",
    "Survey Creation"
])

# Tab 1: Survey Selection and Question Bank
with tab1:
    st.subheader("Select Surveys")
    survey_options = [f"{s['id']} - {s['title']}" for s in surveys]
    selected_surveys = st.multiselect("Surveys", survey_options)
    selected_survey_ids = [s.split(" - ")[0] for s in selected_surveys]

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh Survey Data"):
            st.session_state.all_questions = None
            st.session_state.dedup_questions = []
            st.session_state.dedup_choices = []
            st.session_state.fetched_survey_ids = []
            st.session_state.survey_page = 1
            if os.path.exists(CACHE_FILE):
                os.remove(CACHE_FILE)
            st.experimental_rerun()
    with col2:
        if st.button("📥 Load More Surveys"):
            st.session_state.survey_page += 1
            new_surveys, _ = get_surveys_cached(token, page=st.session_state.survey_page)
            surveys.extend(new_surveys)
            st.experimental_rerun()

    if selected_survey_ids:
        combined_questions = []
        new_survey_ids = [sid for sid in selected_survey_ids if sid not in st.session_state.fetched_survey_ids]
        for survey_id in new_survey_ids:
            with st.spinner(f"Fetching survey {survey_id}..."):
                survey_json = get_survey_details_with_retry(survey_id, token)
                questions = extract_questions(survey_json)
                combined_questions.extend(questions)
                st.session_state.fetched_survey_ids.append(survey_id)
                time.sleep(REQUEST_DELAY)
        
        if combined_questions:
            new_questions = pd.DataFrame(combined_questions)
            st.session_state.all_questions = pd.concat(
                [st.session_state.all_questions, new_questions] if st.session_state.all_questions is not None else [new_questions],
                ignore_index=True
            )
            st.session_state.dedup_questions = sorted(st.session_state.all_questions[
                st.session_state.all_questions["is_choice"] == False
            ]["heading_0"].unique().tolist())
            st.session_state.dedup_choices = sorted(st.session_state.all_questions[
                st.session_state.all_questions["is_choice"] == True
            ]["heading_0"].apply(lambda x: x.split(" - ", 1)[1] if " - " in x else x).unique().tolist())
            save_cached_survey_data(
                st.session_state.all_questions,
                st.session_state.dedup_questions,
                st.session_state.dedup_choices
            )

        st.session_state.df_target = st.session_state.all_questions[
            st.session_state.all_questions["survey_id"].isin(selected_survey_ids)
        ].copy()
        
        if st.session_state.df_target.empty:
            st.error("No questions found.")
        else:
            st.write("### Selected Questions (SurveyMonkey)")
            show_main_only = st.checkbox("Show main questions only")
            display_df = st.session_state.df_target[st.session_state.df_target["is_choice"] == False] if show_main_only else st.session_state.df_target
            st.dataframe(display_df[["heading_0", "schema_type", "is_choice", "survey_title"]])

    st.write("### Standardized Question Bank (Snowflake)")
    if st.button("View Question Bank"):
        search_query = st.text_input("Search Questions or UIDs")
        filtered_bank = st.session_state.question_bank[
            st.session_state.question_bank["HEADING_0"].str.contains(search_query, case=False, na=False) |
            st.session_state.question_bank["UID"].str.contains(search_query, case=False, na=False)
        ] if search_query else st.session_state.question_bank
        st.dataframe(filtered_bank.rename(columns={"HEADING_0": "Question/Choice", "UID": "UID"}))
    if st.button("Add to Question Bank"):
        st.markdown("[Submit New Question](https://docs.google.com/forms/d/1LoY_La59UJ4ZsuxckM8Wl52kVeLI7a1t1MF8zIQxGUs)")

# Tab 2: UID Matching and Configuration
with tab2:
    if st.session_state.df_target is not None and not st.session_state.df_target.empty:
        try:
            with st.spinner("Matching UIDs..."):
                if st.session_state.question_bank is not None and not st.session_state.question_bank.empty:
                    st.session_state.df_final = run_uid_match(st.session_state.question_bank, st.session_state.df_target)
                else:
                    st.session_state.df_final = st.session_state.df_target.copy()
                    st.session_state.df_final["Final_UID"] = None
        except Exception as e:
            logger.error(f"UID matching failed: {e}")
            st.warning("UID matching failed. Continuing without UIDs.")
            st.session_state.df_final = st.session_state.df_target.copy()
            st.session_state.df_final["Final_UID"] = None

        matched_percentage = calculate_matched_percentage(st.session_state.df_final)
        st.metric("Matched Questions", f"{matched_percentage}%")

        st.write("### UID Matching (SurveyMonkey Questions)")
        col1, col2 = st.columns(2)
        with col1:
            show_main_only = st.checkbox("Show main questions only")
        with col2:
            match_filter = st.multiselect(
                "Match Status",
                ["✅ High", "⚠️ Low", "🧠 Semantic", "❌ No match"],
                default=["✅ High", "⚠️ Low", "🧠 Semantic"]
            )
        similarity_threshold = st.slider("Minimum Similarity Score", 0.0, 1.0, 0.5)
        schema_filter = st.multiselect(
            "Question Type",
            ["Single Choice", "Multiple Choice", "Open-Ended", "Matrix"],
            default=["Single Choice", "Multiple Choice", "Open-Ended", "Matrix"]
        )

        search_query = st.text_input("Search Questions/Choices")
        result_df = st.session_state.df_final.copy()
        if search_query:
            result_df = result_df[result_df["heading_0"].str.contains(search_query, case=False, na=False)]
        if match_filter:
            result_df = result_df[result_df["Final_Match_Type"].isin(match_filter)]
        if show_main_only:
            result_df = result_df[result_df["is_choice"] == False]
        if schema_filter:
            result_df = result_df[result_df["schema_type"].isin(schema_filter)]
        if similarity_threshold > 0.5:
            result_df = result_df[
                result_df["Similarity"].ge(similarity_threshold) |
                result_df["Semantic_Similarity"].ge(similarity_threshold) |
                result_df["Final_UID"].isna()
            ]

        if st.button("Re-run UID Matching"):
            with st.spinner("Re-running UID matching..."):
                updated_df = run_uid_match(st.session_state.question_bank, result_df)
                st.session_state.df_final.update(updated_df)
                result_df = st.session_state.df_final.copy()

        uid_options = [None] + [f"{row['UID']} - {row['HEADING_0']}" for _, row in st.session_state.question_bank.iterrows()]
        edited_df_tab2 = st.data_editor(
            result_df[["heading_0", "schema_type", "is_choice", "Final_UID", "Change_UID", "mandatory"]],
            column_config={
                "heading_0": st.column_config.TextColumn("Question/Choice (SurveyMonkey)"),
                "schema_type": st.column_config.TextColumn("Question Type"),
                "is_choice": st.column_config.CheckboxColumn("Is Choice"),
                "Final_UID": st.column_config.TextColumn("Final UID"),
                "Change_UID": st.column_config.SelectboxColumn(
                    "Change UID (Snowflake)",
                    options=uid_options,
                    default=None
                ),
                "mandatory": st.column_config.CheckboxColumn("Required")
            },
            disabled=["heading_0", "schema_type", "is_choice", "Final_UID"],
            hide_index=True
        )

        for idx, row in edited_df_tab2.iterrows():
            if pd.notnull(row["Change_UID"]):
                new_uid = row["Change_UID"].split(" - ")[0]
                st.session_state.df_final.at[idx, "Final_UID"] = new_uid
                st.session_state.df_final.at[idx, "configured_final_UID"] = new_uid
                st.session_state.uid_changes[idx] = new_uid
            st.session_state.df_final.at[idx, "mandatory"] = row["mandatory"]

        st.write("### Customize Questions (SurveyMonkey)")
        customize_df = pd.DataFrame({
            "Pre-existing Question": [None],
            "Customized Question": [""]
        })
        question_options = [None] + st.session_state.df_target[
            st.session_state.df_target["is_choice"] == False
        ]["heading_0"].tolist()
        customize_edited_df = st.data_editor(
            customize_df,
            column_config={
                "Pre-existing Question": st.column_config.SelectboxColumn(
                    "Pre-existing Question (SurveyMonkey)",
                    options=question_options,
                    default=None
                ),
                "Customized Question": st.column_config.TextColumn(
                    "Customized Question",
                    default=""
                )
            },
            hide_index=True,
            num_rows="dynamic"
        )

        for _, row in customize_edited_df.iterrows():
            if row["Pre-existing Question"] and row["Customized Question"]:
                original_question = row["Pre-existing Question"]
                custom_question = row["Customized Question"]
                uid = st.session_state.df_final[
                    st.session_state.df_final["heading_0"] == original_question
                ]["Final_UID"].iloc[0] if not st.session_state.df_final[
                    st.session_state.df_final["heading_0"] == original_question
                ].empty else None
                new_row = pd.DataFrame({
                    "Customized Question": [custom_question],
                    "Original Question": [original_question],
                    "Final_UID": [uid]
                })
                st.session_state.custom_questions = pd.concat(
                    [st.session_state.custom_questions, new_row], ignore_index=True
                )

        if not st.session_state.custom_questions.empty:
            st.dataframe(st.session_state.custom_questions)

        st.write("### Submit New Questions/UIDs")
        st.markdown("[Submit New Question](https://docs.google.com/forms/d/1LoY_La59UJ4ZsuxckM8Wl52kVeLI7a1t1MF8zIQxGUs)")
        st.markdown("[Submit New UID](https://docs.google.com/forms/d/1lkhfm1-t5-zwLxfbVEUiHewveLpGXv5yEVRlQx5XjxA)")

        st.write("### Export")
        export_df = st.session_state.df_final[[
            "survey_id", "survey_title", "heading_0", "configured_final_UID", "schema_type", "is_choice", "mandatory"
        ]].copy()
        export_df = export_df.rename(columns={"configured_final_UID": "uid"})
        st.download_button(
            "📥 Download CSV",
            export_df.to_csv(index=False),
            f"survey_with_uids_{uuid4()}.csv",
            "text/csv"
        )

        if st.button("🚀 Upload to Snowflake"):
            if export_df[export_df["is_choice"] == False]["uid"].isna().any():
                st.error("All main questions must have a UID before upload.")
            else:
                try:
                    with get_snowflake_engine().connect() as conn:
                        export_df.to_sql(
                            'SURVEY_DETAILS_RESPONSES_COMBINED_LIVE',
                            conn,
                            schema='DBT_SURVEY_MONKEY',
                            if_exists='append',
                            index=False
                        )
                    st.success("Uploaded to Snowflake!")
                except Exception as e:
                    st.error(f"Snowflake upload failed: {e}")

# Tab 3: Survey Creation
with tab3:
    st.subheader("Create New Survey")
    with st.container():
        col_left, col_form, col_right = st.columns([1, 2, 1])
        with col_form:
            with st.form("survey_creation_form"):
                st.write("### Survey Details")
                survey_title = st.text_input("Survey Title", value="New Survey")
                survey_nickname = st.text_input("Survey Nickname", value=survey_title)
                survey_language = st.selectbox("Language", ["en", "es", "fr", "de"], index=0)

                st.write("### Questions (SurveyMonkey)")
                question_options = [""] + st.session_state.dedup_questions
                choice_options = [""] + st.session_state.dedup_choices
                edited_df = st.data_editor(
                    st.session_state.edited_df,
                    column_config={
                        "heading_0": st.column_config.SelectboxColumn(
                            "Question/Choice (SurveyMonkey)",
                            options=question_options if not st.session_state.edited_df.empty and not st.session_state.edited_df["is_choice"].iloc[-1] else choice_options,
                            default=""
                        ),
                        "schema_type": st.column_config.SelectboxColumn(
                            "Question Type",
                            options=["Single Choice", "Multiple Choice", "Open-Ended", "Matrix"],
                            default="Open-Ended"
                        ),
                        "is_choice": st.column_config.CheckboxColumn("Is Choice"),
                        "mandatory": st.column_config.CheckboxColumn("Required")
                    },
                    hide_index=True,
                    num_rows="dynamic"
                )
                st.session_state.edited_df = edited_df

                if st.button("➕ Add Question/Choice"):
                    new_row = pd.DataFrame({
                        "heading_0": [""],
                        "schema_type": ["Open-Ended"],
                        "is_choice": [False],
                        "mandatory": [False]
                    })
                    st.session_state.edited_df = pd.concat([st.session_state.edited_df, new_row], ignore_index=True)
                    st.experimental_rerun()

                if st.button("Validate Against Question Bank"):
                    non_standard = edited_df[
                        ~edited_df["heading_0"].str.split(" - ", n=1, expand=True)[0].isin(st.session_state.question_bank["HEADING_0"])
                    ]
                    if not non_standard.empty:
                        st.error("Non-standard questions detected. Submit them to the question bank:")
                        st.dataframe(non_standard[["heading_0"]])
                        st.markdown("[Submit New Question](https://docs.google.com/forms/d/1LoY_La59UJ4ZsuxckM8Wl52kVeLI7a1t1MF8zIQxGUs)")

                preview, create = st.columns(2)
                with preview:
                    preview_btn = st.form_submit_button("👁️ Preview")
                with create:
                    create_btn = st.form_submit_button("Create Survey")

                if preview_btn or create_btn:
                    if not survey_title or edited_df.empty or edited_df["heading_0"].eq("").any():
                        st.error("Survey title and valid questions required.")
                    else:
                        questions = []
                        preview_rows = []
                        position = 1
                        for idx, row in edited_df.iterrows():
                            question_template = {
                                "heading": row["heading_0"].split(" - ")[0] if row["is_choice"] else row["heading_0"],
                                "position": position,
                                "is_required": row["mandatory"]
                            }
                            if row["schema_type"] == "Single Choice":
                                question_template["family"] = "single_choice"
                                question_template["subtype"] = "vertical"
                            elif row["schema_type"] == "Multiple Choice":
                                question_template["family"] = "multiple_choice"
                                question_template["subtype"] = "vertical"
                            elif row["schema_type"] == "Open-Ended":
                                question_template["family"] = "open_ended"
                                question_template["subtype"] = "essay"
                            elif row["schema_type"] == "Matrix":
                                question_template["family"] = "matrix"
                                question_template["subtype"] = "rating"
                            
                            if row["is_choice"]:
                                parent_question = row["heading_0"].split(" - ")[0]
                                parent_idx = edited_df[edited_df["heading_0"] == parent_question].index
                                if parent_idx.empty:
                                    continue
                                parent_idx = parent_idx[0]
                                if "choices" not in questions[parent_idx]:
                                    questions[parent_idx]["choices"] = []
                                questions[parent_idx]["choices"].append({
                                    "text": row["heading_0"].split(" - ")[1],
                                    "position": len(questions[parent_idx]["choices"]) + 1
                                })
                            else:
                                questions.append(question_template)
                                position += 1
                            
                            preview_rows.append({
                                "position": question_template["position"],
                                "title": survey_title,
                                "nickname": survey_nickname,
                                "heading_0": row["heading_0"],
                                "schema_type": row["schema_type"],
                                "is_choice": row["is_choice"],
                                "mandatory": row["mandatory"]
                            })

                        survey_template = {
                            "title": survey_title,
                            "nickname": survey_nickname,
                            "language": survey_language,
                            "pages": [{
                                "title": "Page 1",
                                "description": "",
                                "questions": questions
                            }],
                            "settings": {
                                "progress_bar": False,
                                "hide_asterisks": True,
                                "one_question_at_a_time": False
                            }
                        }

                        preview_df = pd.DataFrame(preview_rows)
                        if st.session_state.question_bank is not None and not st.session_state.question_bank.empty:
                            uid_target = preview_df[preview_df["is_choice"] == False][["heading_0"]].copy()
                            if not uid_target.empty:
                                uid_matched = run_uid_match(st.session_state.question_bank, uid_target)
                                preview_df = preview_df.merge(
                                    uid_matched[["heading_0", "Final_UID"]],
                                    on="heading_0",
                                    how="left"
                                )
                        else:
                            preview_df["Final_UID"] = None

                        st.session_state.preview_df = preview_df
                        st.session_state.survey_template = survey_template

                        if preview_btn:
                            st.success("Preview generated!")
                            st.subheader("Survey Preview (SurveyMonkey)")
                            st.json(survey_template)
                            def highlight_duplicates(df):
                                styles = pd.DataFrame('', index=df.index, columns=df.columns)
                                main_questions = df[df["is_choice"] == False]["heading_0"]
                                duplicates = main_questions[main_questions.duplicated(keep=False)]
                                if not duplicates.empty:
                                    mask = (df["is_choice"] == False) & (df["heading_0"].isin(duplicates))
                                    styles.loc[mask, "heading_0"] = 'background-color: red'
                                return styles
                            st.dataframe(preview_df.style.apply(highlight_duplicates, axis=None))

                        if create_btn:
                            try:
                                with st.spinner("Creating survey..."):
                                    survey_id = create_survey(token, survey_template)
                                    for page_template in survey_template["pages"]:
                                        page_id = create_page(token, survey_id, page_template)
                                        for question_template in page_template["questions"]:
                                            create_question(token, survey_id, page_id, question_template)
                                st.success(f"Survey created! Survey ID: {survey_id}")
                                st.session_state.pending_survey = {
                                    "survey_id": survey_id,
                                    "survey_title": survey_title,
                                    "df": preview_df
                                }
                                st.info("Configure questions in Tab 2 before uploading to Snowflake.")
                                if st.button("Configure Now"):
                                    st.session_state.df_final = preview_df.copy()
                                    st.session_state.df_final["survey_id"] = survey_id
                                    st.session_state.df_final["survey_title"] = survey_title
                                    st.experimental_rerun()
                            except Exception as e:
                                st.error(f"Failed to create survey: {e}")

    if st.session_state.pending_survey:
        st.subheader("Pending Survey Configuration")
        st.write(f"Survey ID: {st.session_state.pending_survey['survey_id']}")
        st.write(f"Title: {st.session_state.pending_survey['survey_title']}")
        if st.button("Configure Pending Survey"):
            st.session_state.df_final = st.session_state.pending_survey["df"].copy()
            st.session_state.df_final["survey_id"] = st.session_state.pending_survey["survey_id"]
            st.session_state.df_final["survey_title"] = st.session_state.pending_survey["survey_title"]
            st.experimental_rerun()
