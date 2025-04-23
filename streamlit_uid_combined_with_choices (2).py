import streamlit as st
import pandas as pd
import requests
import re
import json
import logging
from uuid import uuid4
from sqlalchemy import create_engine, text
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

# Setup
st.set_page_config(page_title="UID Matcher Combined", layout="wide")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
TFIDF_HIGH_CONFIDENCE = 0.60  # Threshold for high-confidence syntactic matches
TFIDF_LOW_CONFIDENCE = 0.50   # Threshold for low-confidence syntactic matches
SEMANTIC_THRESHOLD = 0.60     # Threshold for semantic matches
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 1000             # Batch size for processing large datasets

# Synonym Mapping (used internally, not exposed in UI)
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
    try:
        return SentenceTransformer(MODEL_NAME)
    except Exception as e:
        logger.error(f"Failed to load SentenceTransformer: {e}")
        raise

@st.cache_resource
def get_snowflake_engine():
    try:
        sf = st.secrets["snowflake"]
        return create_engine(
            f"snowflake://{sf.user}:{sf.password}@{sf.account}/{sf.database}/{sf.schema}"
            f"?warehouse={sf.warehouse}&role={sf.role}"
        )
    except Exception as e:
        logger.error(f"Failed to create Snowflake engine: {e}")
        raise

@st.cache_data
def get_tfidf_vectors(df_reference):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    vectors = vectorizer.fit_transform(df_reference["norm_text"])
    return vectorizer, vectors

# Normalization
def enhanced_normalize(text, synonym_map=DEFAULT_SYNONYM_MAP):
    text = str(text).lower()
    text = re.sub(r'\(.*?\)', '', text)  # Remove parenthetical content
    text = re.sub(r'[^a-z0-9 ]', '', text)  # Keep alphanumeric and spaces
    for phrase, replacement in synonym_map.items():
        text = text.replace(phrase, replacement)
    return ' '.join(w for w in text.split() if w not in ENGLISH_STOP_WORDS)

# Snowflake Queries
def run_snowflake_reference_query(limit=10000, offset=0):
    query = """
        SELECT HEADING_0, MAX(UID) AS UID
        FROM AMI_DBT.DBT_SURVEY_MONKEY.SURVEY_DETAILS_RESPONSES_COMBINED_LIVE
        WHERE UID IS NOT NULL
        GROUP BY HEADING_0
        LIMIT :limit OFFSET :offset
    """
    try:
        with get_snowflake_engine().connect() as conn:
            result = pd.read_sql(text(query), conn, params={"limit": limit, "offset": offset})
        return result
    except Exception as e:
        logger.error(f"Snowflake reference query failed: {e}")
        raise

def run_snowflake_target_query():
    query = """
        SELECT DISTINCT HEADING_0
        FROM AMI_DBT.DBT_SURVEY_MONKEY.SURVEY_DETAILS_RESPONSES_COMBINED_LIVE
        WHERE UID IS NULL AND NOT LOWER(HEADING_0) LIKE 'our privacy policy%'
    """
    try:
        with get_snowflake_engine().connect() as conn:
            result = pd.read_sql(text(query), conn)
        return result
    except Exception as e:
        logger.error(f"Snowflake target query failed: {e}")
        raise

# SurveyMonkey API
def test_surveymonkey_token(token):
    url = "https://api.surveymonkey.com/v3/surveys"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        logger.info("Token is valid! Retrieved surveys:")
        for survey in data.get("data", []):
            logger.info(f"- {survey['title']} (ID: {survey['id']})")
        return True
    except requests.HTTPError as e:
        logger.error(f"HTTP Error: {e.response.status_code} - {e.response.text}")
        return False
    except requests.RequestException as e:
        logger.error(f"Request failed: {e}")
        return False

def get_surveys(token):
    url = "https://api.surveymonkey.com/v3/surveys"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json().get("data", [])
    except requests.RequestException as e:
        logger.error(f"Failed to fetch surveys: {e}")
        raise

def get_survey_details(survey_id, token):
    url = f"https://api.surveymonkey.com/v3/surveys/{survey_id}/details"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch survey details for ID {survey_id}: {e}")
        raise

def extract_questions(survey_json):
    questions = []
    global_position = 0  # Running counter for main questions
    for page in survey_json.get("pages", []):
        for question in page.get("questions", []):
            q_text = question.get("headings", [{}])[0].get("heading", "")
            api_position = question.get("position", None)
            q_id = question.get("id", None)
            family = question.get("family", None)
            subtype = question.get("subtype", None)
            # Determine schema type
            if family == "single_choice":
                schema_type = "Single Choice"
            elif family == "multiple_choice":
                schema_type = "Multiple Choice"
            elif family == "open_ended":
                schema_type = "Open-Ended"
            elif family == "matrix":
                schema_type = "Matrix"
            else:
                # Infer based on choices
                choices = question.get("answers", {}).get("choices", [])
                schema_type = "Multiple Choice" if choices else "Open-Ended"
                if choices and ("select one" in q_text.lower() or len(choices) <= 2):
                    schema_type = "Single Choice"
            
            if q_text:
                global_position += 1  # Increment for main questions only
                # Add the question
                questions.append({
                    "heading_0": q_text,
                    "position": global_position,
                    "is_choice": False,
                    "parent_question": None,
                    "question_id": q_id,
                    "schema_type": schema_type,
                    "mandatory": False,
                    "mandatory_editable": True  # Main questions can edit mandatory
                })
                # Add choices for applicable question types
                choices = question.get("answers", {}).get("choices", [])
                for choice in choices:
                    choice_text = choice.get("text", "")
                    if choice_text:
                        questions.append({
                            "heading_0": f"{q_text} - {choice_text}",
                            "position": global_position,
                            "is_choice": True,
                            "parent_question": q_text,
                            "question_id": q_id,
                            "schema_type": schema_type,
                            "mandatory": False,
                            "mandatory_editable": False  # Choices cannot edit mandatory
                        })
    return questions

# UID Matching
def compute_tfidf_matches(df_reference, df_target, synonym_map=DEFAULT_SYNONYM_MAP):
    df_reference = df_reference[df_reference["heading_0"].notna()].reset_index(drop=True)
    df_target = df_target[df_reference["heading_0"].notna()].reset_index(drop=True)
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
        matched_uids.append(df_reference.iloc[best_idx]["uid"] if best_idx is not None else None)
        matched_qs.append(df_reference.iloc[best_idx]["heading_0"] if best_idx is not None else None)
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
        emb_ref = model.encode(df_reference["heading_0"].tolist(), convert_to_tensor=True)
        cosine_scores = util.cos_sim(emb_target, emb_ref)

        sem_matches, sem_scores = [], []
        for i in range(len(df_target)):
            best_idx = cosine_scores[i].argmax().item()
            score = cosine_scores[i][best_idx].item()
            sem_matches.append(df_reference.iloc[best_idx]["uid"] if score >= SEMANTIC_THRESHOLD else None)
            sem_scores.append(round(score, 4) if score >= SEMANTIC_THRESHOLD else None)

        df_target["Semantic_UID"] = sem_matches
        df_target["Semantic_Similarity"] = sem_scores
        return df_target
    except Exception as e:
        logger.error(f"Semantic matching failed: {e}")
        st.error(f"Semantic matching failed: {e}")
        return df_target

def assign_match_type(row):
    if pd.notnull(row["Suggested_UID"]):
        return row["Match_Confidence"]
    return "🧠 Semantic" if pd.notnull(row["Semantic_UID"]) else "❌ No match"

def finalize_matches(df_target):
    df_target["Final_UID"] = df_target["Suggested_UID"].combine_first(df_target["Semantic_UID"])
    df_target["Final_Question"] = df_target["Matched_Question"]
    df_target["Final_Match_Type"] = df_target.apply(assign_match_type, axis=1)
    
    # Propagate UID to choices
    df_target["Final_UID"] = df_target.apply(
        lambda row: df_target[df_target["heading_0"] == row["parent_question"]]["Final_UID"].iloc[0]
        if row["is_choice"] and pd.notnull(row["parent_question"]) else row["Final_UID"],
        axis=1
    )
    return df_target

def detect_uid_conflicts(df_target):
    uid_conflicts = df_target.groupby("Final_UID")["heading_0"].nunique()
    duplicate_uids = uid_conflicts[uid_conflicts > 1].index
    df_target["UID_Conflict"] = df_target["Final_UID"].apply(
        lambda x: "⚠️ Conflict" if x in duplicate_uids else ""
    )
    return df_target

def run_uid_match(df_reference, df_target, synonym_map=DEFAULT_SYNONYM_MAP, batch_size=BATCH_SIZE):
    if df_reference.empty or df_target.empty:
        logger.warning("Empty input dataframes provided.")
        st.error("Input data is empty.")
        return pd.DataFrame()

    if len(df_target) > 10000:
        st.warning("Large dataset detected. Processing may take time.")

    logger.info(f"Processing {len(df_target)} target questions against {len(df_reference)} reference questions.")
    df_results = []
    for start in range(0, len(df_target), batch_size):
        batch_target = df_target.iloc[start:start + batch_size].copy()
        with st.spinner(f"Processing batch {start//batch_size + 1}..."):
            batch_target = compute_tfidf_matches(df_reference, batch_target, synonym_map)
            batch_target = compute_semantic_matches(df_reference, batch_target)
            batch_target = finalize_matches(batch_target)
            batch_target = detect_uid_conflicts(batch_target)
        df_results.append(batch_target)
    
    if not df_results:
        logger.warning("No results from batch processing.")
        return pd.DataFrame()
    return pd.concat(df_results, ignore_index=True)

# Sidebar
st.sidebar.title("SurveyMonkey Token")
token = st.secrets.get("surveymonkey", {}).get("token", None)
if st.sidebar.button("Test SurveyMonkey Token"):
    with st.sidebar:
        with st.spinner("Testing token..."):
            if token and test_surveymonkey_token(token):
                st.success("Token is valid! Surveys are accessible.")
            else:
                st.error("Token test failed. Check logs or token configuration.")

# App UI
st.title("🧠 UID Matcher: Snowflake + SurveyMonkey")

# Secrets Validation
if "snowflake" not in st.secrets or "surveymonkey" not in st.secrets:
    st.error("Missing secrets configuration for Snowflake or SurveyMonkey.")
    st.stop()

# Data Source Selection
option = st.radio("Choose Data Source", ["SurveyMonkey", "Snowflake"], horizontal=True)

if option == "SurveyMonkey":
    try:
        if not token:
            st.error("SurveyMonkey token is missing in secrets configuration.")
            st.stop()
        with st.spinner("Fetching surveys..."):
            surveys = get_surveys(token)
        if not surveys:
            st.error("No surveys found or invalid API response.")
        else:
            choices = {s["title"]: s["id"] for s in surveys}
            selected_survey = st.selectbox("Choose Survey", list(choices.keys()))
            df_target = None
            if selected_survey:
                with st.spinner("Fetching survey details..."):
                    survey_json = get_survey_details(choices[selected_survey], token)
                    questions = extract_questions(survey_json)
                    df_target = pd.DataFrame(questions)
                if df_target.empty:
                    st.error("No questions found in the selected survey.")
                else:
                    st.write("Survey questions and choices retrieved successfully. Edit mandatory status and click below to match UIDs.")
                    # Filter for main questions
                    show_main_only = st.checkbox("Show only main questions", value=False)
                    display_df = df_target[df_target["is_choice"] == False] if show_main_only else df_target
                    
                    # Editable table for mandatory status
                    st.subheader("Survey Questions and Choices")
                    edited_df = st.data_editor(
                        display_df,
                        column_config={
                            "mandatory": st.column_config.CheckboxColumn(
                                "Mandatory",
                                help="Mark question as mandatory",
                                default=False
                            ),
                            "mandatory_editable": st.column_config.CheckboxColumn(
                                "Editable",
                                help="Indicates if mandatory can be edited",
                                disabled=True
                            ),
                            "heading_0": st.column_config.TextColumn("Question/Choice"),
                            "position": st.column_config.NumberColumn("Position"),
                            "is_choice": st.column_config.CheckboxColumn("Is Choice"),
                            "parent_question": st.column_config.TextColumn("Parent Question"),
                            "schema_type": st.column_config.TextColumn("Schema Type"),
                            "question_id": st.column_config.TextColumn("Question ID")
                        },
                        disabled=["heading_0", "position", "is_choice", "parent_question", "schema_type", "question_id", "mandatory_editable"] + (["mandatory"] if not display_df["mandatory_editable"].any() else []),
                        hide_index=True
                    )
                    
                    # Update df_target with edited mandatory status for editable rows
                    if not edited_df.empty:
                        editable_rows = df_target[df_target["mandatory_editable"]]
                        if not editable_rows.empty:
                            df_target.loc[df_target["mandatory_editable"], "mandatory"] = edited_df[edited_df["mandatory_editable"]]["mandatory"]
                    
                    if st.button("Run UID Matching"):
                        with st.spinner("Running UID matching..."):
                            df_reference = run_snowflake_reference_query()
                            df_final = run_uid_match(df_reference, df_target)
                            
                            # Apply filter to results
                            display_final = df_final[df_final["is_choice"] == False] if show_main_only else df_final
                            
                            # Filter Results
                            confidence_filter = st.multiselect(
                                "Filter by Match Type",
                                ["✅ High", "⚠️ Low", "🧠 Semantic", "❌ No match"],
                                default=["✅ High", "⚠️ Low", "🧠 Semantic"]
                            )
                            filtered_df = display_final[display_final["Final_Match_Type"].isin(confidence_filter)]
                            
                            st.subheader("UID Matching Results")
                            st.dataframe(filtered_df)
                            st.download_button(
                                "📥 Download UID Matches",
                                filtered_df.to_csv(index=False),
                                f"uid_matches_{uuid4()}.csv"
                            )
                    
                    # Add New Question Option
                    st.subheader("Add New Question")
                    add_question_method = st.radio(
                        "Choose method to add new question",
                        ["Google Form", "Snowflake Query"],
                        index=0
                    )
                    if add_question_method == "Google Form":
                        st.write(
                            "Submit a new question request via Google Form. "
                            "Create a form with fields: Question Text, Question Type, Choices (optional), Program, Mandatory."
                        )
                        st.markdown(
                            "[Placeholder Google Form](https://forms.gle/your_form_link) "
                            "(Replace with your actual form link after creation)"
                        )
                        st.write(
                            "To create a form: Go to [Google Forms](https://forms.google.com), "
                            "add fields, and share the link. Update this app with the link later."
                        )
                    else:  # Snowflake Query
                        st.write("Use the following SQL query to add a new question to the question bank:")
                        new_question_sql = """
INSERT INTO AMI_DBT.DBT_SURVEY_MONKEY.QUESTION_BANK
(HEADING_0, UID, POSITION, SCHEMA_TYPE, MANDATORY, CREATED_AT)
VALUES
(:question_text, :uid, :position, :schema_type, :mandatory, CURRENT_TIMESTAMP);
"""
                        st.code(new_question_sql, language="sql")
                        st.write(
                            "Parameters:\n"
                            "- question_text: The question text (e.g., 'What is your favorite color?')\n"
                            "- uid: A unique identifier (e.g., generate via UUID)\n"
                            "- position: The desired position (e.g., 1, 2, 3)\n"
                            "- schema_type: The question type (e.g., 'Single Choice', 'Multiple Choice', 'Open-Ended')\n"
                            "- mandatory: Boolean (TRUE/FALSE)"
                        )
                        st.write(
                            "Execute this query in Snowflake after replacing parameters. "
                            "Ensure the QUESTION_BANK table exists with appropriate columns."
                        )
    except Exception as e:
        logger.error(f"SurveyMonkey processing failed: {e}")
        st.error(f"Error: {e}")

if option == "Snowflake":
    if st.button("🔁 Run Matching on Snowflake Data"):
        with st.spinner("Fetching Snowflake data..."):
            df_reference = run_snowflake_reference_query()
            df_target = run_snowflake_target_query()
        
        if df_reference.empty or df_target.empty:
            st.error("No data retrieved from Snowflake.")
        else:
            df_final = run_uid_match(df_reference, df_target)
            
            # Filter Results
            confidence_filter = st.multiselect(
                "Filter by Match Type",
                ["✅ High", "⚠️ Low", "🧠 Semantic", "❌ No match"],
                default=["✅ High", "⚠️ Low", "🧠 Semantic"]
            )
            filtered_df = df_final[df_final["Final_Match_Type"].isin(confidence_filter)]
            
            st.dataframe(filtered_df)
            st.download_button(
                "📥 Download UID Matches",
                filtered_df.to_csv(index=False),
                f"uid_matches_{uuid4()}.csv"
            )
