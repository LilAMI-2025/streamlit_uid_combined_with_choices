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
import numpy as np

# Setup
st.set_page_config(page_title="UID Matcher", layout="wide")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
TFIDF_HIGH_CONFIDENCE = 0.60
TFIDF_LOW_CONFIDENCE = 0.50
SEMANTIC_THRESHOLD = 0.60
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 1000
CLUSTER_UID = "9999"  # For Heading questions

# Synonym Mapping
DEFAULT_SYNONYM_MAP = {
    "please select": "what is",
    "sector you are from": "your sector",
    "identity type": "id type",
    "what type of": "type of",
    "are you": "do you",
}

# Heading Patterns
HEADING_PATTERNS = [
    r"as we prepare to implement our programme in your company",
    r"now, we'd like to find out a little bit about your company's learning initiatives",
    r"this section contains the heart of what we would like you to tell us",
    r"now, we want to delve a bit deeper to examine how the winning behaviours",
    r"as a last step, we ask that you rank order",
    r"thank you for taking the time to reflect on how aligned",
    r"please provide the following details",
    r"business details",
    r"learning needs",
    r"confidentiality assurance",
    r"contact information",
    r"institutional profile",
    r"section \w+:",
    r"introduction",
    r"welcome to the business development service provider",
    r"confidentiality",
    r"thank you for your participation",
    r"msme survey tool",
    r"your organisation's current learning initiatives",
    r"rate: pinpoint",
    r"frequency: tell us",
    r"rank: prioritise",
    r"ecosystem support organizations interview guide",
]

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
        logger.info(f"Attempting Snowflake connection: user={sf.user}, account={sf.account}")
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
            st.warning(
                "Snowflake connection failed: User account is locked. "
                "UID matching is disabled. Please resolve the lockout."
            )
        elif "sqlalchemy.dialects:snowflake" in str(e):
            st.error(
                "Snowflake SQLAlchemy dialect not found. Ensure `snowflake-sqlalchemy` is installed. "
                "Run: `pip install snowflake-sqlalchemy` and restart the app."
            )
        raise

@st.cache_data
def get_tfidf_vectors(df_reference):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    vectors = vectorizer.fit_transform(df_reference["norm_text"])
    return vectorizer, vectors

# Normalization
def enhanced_normalize(text, synonym_map=DEFAULT_SYNONYM_MAP):
    text = str(text).lower()
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'[^a-z0-9 ]', '', text)
    for phrase, replacement in synonym_map.items():
        text = text.replace(phrase, replacement)
    return ' '.join(w for w in text.split() if w not in ENGLISH_STOP_WORDS)

# Categorize Questions
def categorize_question(text):
    text_lower = str(text).lower()
    for pattern in HEADING_PATTERNS:
        if re.search(pattern, text_lower):
            return "Heading"
    return "Main Question"

# Calculate Matched Questions Percentage
def calculate_matched_percentage(df_final):
    if df_final is None or df_final.empty:
        return 0.0
    df_main = df_final[(df_final["category"] == "Main Question")].copy()
    if df_main.empty:
        return 0.0
    matched_questions = df_main[df_main["Final_UID"].notna()]
    percentage = (len(matched_questions) / len(df_main)) * 100
    return round(percentage, 2)

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
                category = categorize_question(q_text)
                questions.append({
                    "heading_0": q_text,
                    "position": global_position,
                    "is_choice": False,
                    "category": category,
                    "parent_question": None,
                    "question_uid": q_id,
                    "schema_type": schema_type,
                    "mandatory": False,
                    "mandatory_editable": True,
                    "survey_id": survey_json.get("id", ""),
                    "survey_title": survey_json.get("title", "")
                })
                if category == "Heading":
                    questions[-1]["Final_UID"] = CLUSTER_UID
                    questions[-1]["configured_final_UID"] = CLUSTER_UID
                
                choices = question.get("answers", {}).get("choices", [])
                for choice in choices:
                    choice_text = choice.get("text", "")
                    if choice_text:
                        questions.append({
                            "heading_0": f"{q_text} - {choice_text}",
                            "position": global_position,
                            "is_choice": True,
                            "category": "Choice",
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
    df_reference = df_reference[df_reference["heading_0"].notna()].reset_index(drop=True)
    df_target = df_target[df_target["heading_0"].notna() & (df_target["category"] != "Heading")].reset_index(drop=True)
    if df_target.empty:
        logger.info("No non-Heading questions to match.")
        return df_target.assign(
            Suggested_UID=None,
            Matched_Question=None,
            Similarity=None,
            Match_Confidence="❌ No match"
        )

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
        scores.append(round(best_score, 4) if best_score >= TFIDF_LOW_CONFIDENCE else None)
        confs.append(conf)

    df_target["Suggested_UID"] = matched_uids
    df_target["Matched_Question"] = matched_qs
    df_target["Similarity"] = scores
    df_target["Match_Confidence"] = confs
    return df_target

def compute_semantic_matches(df_reference, df_target):
    try:
        if df_target.empty or df_target["category"].eq("Heading").all():
            logger.info("No non-Heading questions for semantic matching.")
            return df_target.assign(Semantic_UID=None, Semantic_Similarity=None)

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
        return df_target.assign(Semantic_UID=None, Semantic_Similarity=None)

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
        lambda x: f"{x} - {df_reference[df_reference['uid'] == x]['heading_0'].iloc[0]}" if pd.notnull(x) and x in df_reference["uid"].values else None
    )

    # Propagate parent UID to choices
    df_target["Final_UID"] = df_target.apply(
        lambda row: df_target[df_target["heading_0"] == row["parent_question"]]["Final_UID"].iloc[0]
        if row["is_choice"] and pd.notnull(row["parent_question"]) and not df_target[df_target["heading_0"] == row["parent_question"]].empty
        else row["Final_UID"],
        axis=1
    )
    df_target["configured_final_UID"] = df_target["Final_UID"]
    df_target["Change_UID"] = df_target["Final_UID"].apply(
        lambda x: f"{x} - {df_reference[df_reference['uid'] == x]['heading_0'].iloc[0]}" if pd.notnull(x) and x in df_reference["uid"].values else None
    )

    # Add survey_id_title
    if "survey_id" in df_target.columns and "survey_title" in df_target.columns:
        df_target["survey_id_title"] = df_target.apply(
            lambda x: f"{x['survey_id']} - {x['survey_title']}" if pd.notnull(x['survey_id']) and pd.notnull(x['survey_title']) else "",
            axis=1
        )

    return df_target

def detect_uid_conflicts(df_target):
    uid_conflicts = df_target.groupby("Final_UID")["heading_0"].nunique()
    duplicate_uids = uid_conflicts[uid_conflicts > 1].index
    df_target["UID_Conflict"] = df_target["Final_UID"].apply(
        lambda x: "⚠️ Conflict" if pd.notnull(x) and x in duplicate_uids and x != CLUSTER_UID else ""
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
    non_heading_target = df_target[df_target["category"] != "Heading"].copy()
    if not non_heading_target.empty:
        for start in range(0, len(non_heading_target), batch_size):
            batch_target = non_heading_target.iloc[start:start + batch_size].copy()
            with st.spinner(f"Processing batch {start//batch_size + 1}..."):
                batch_target = compute_tfidf_matches(df_reference, batch_target, synonym_map)
                batch_target = compute_semantic_matches(df_reference, batch_target)
                batch_target = finalize_matches(batch_target, df_reference)
                batch_target = detect_uid_conflicts(batch_target)
            df_results.append(batch_target)

    # Include Heading questions
    heading_df = df_target[df_target["category"] == "Heading"].copy()
    heading_df["Suggested_UID"] = None
    heading_df["Matched_Question"] = None
    heading_df["Similarity"] = None
    heading_df["Match_Confidence"] = "❌ No match"
    heading_df["Semantic_UID"] = None
    heading_df["Semantic_Similarity"] = None
    heading_df["Final_Match_Type"] = "❌ No match"
    heading_df["Change_UID"] = heading_df["Final_UID"].apply(
        lambda x: f"{x} - Cluster Heading" if pd.notnull(x) else None
    )
    heading_df["UID_Conflict"] = ""

    df_results.append(heading_df)
    
    if not df_results:
        logger.warning("No results from batch processing.")
        return df_target
    return pd.concat(df_results, ignore_index=True).sort_values(by=["position", "is_choice"])

# App UI
st.title("🧠 UID Matcher: Snowflake + SurveyMonkey")

# Secrets Validation
if "snowflake" not in st.secrets or "surveymonkey" not in st.secrets:
    st.error("Missing secrets configuration for Snowflake or SurveyMonkey. Please check .streamlit/secrets.toml.")
    st.stop()

# Initialize session state
if "df_target" not in st.session_state:
    st.session_state.df_target = None
if "df_final" not in st.session_state:
    st.session_state.df_final = None
if "uid_changes" not in st.session_state:
    st.session_state.uid_changes = {}
if "df_reference" not in st.session_state:
    st.session_state.df_reference = None

# Sidebar for Interactions
with st.sidebar:
    st.header("Control Panel")
    
    # Data Source Selection
    option = st.radio("Select Data Source", ["SurveyMonkey", "Snowflake"], horizontal=False)
    
    if option == "SurveyMonkey":
        try:
            token = st.secrets.get("surveymonkey", {}).get("token", None)
            if not token:
                st.error("SurveyMonkey token missing in secrets configuration.")
                st.stop()
            with st.spinner("Fetching surveys..."):
                surveys = get_surveys(token)
            if not surveys:
                st.error("No surveys found or invalid API response.")
            else:
                choices = {s["title"]: s["id"] for s in surveys}
                survey_id_title_choices = [f"{s['id']} - {s['title']}" for s in surveys]
                
                st.subheader("Select Surveys")
                selected_survey = st.selectbox("Survey Title", [""] + list(choices.keys()), index=0)
                selected_survey_ids = st.multiselect(
                    "Survey ID/Title",
                    survey_id_title_choices,
                    default=[],
                    help="Select one or more surveys by ID and title"
                )
                
                # Process selected surveys
                selected_survey_ids_from_title = []
                if selected_survey:
                    selected_survey_ids_from_title.append(choices[selected_survey])
                
                all_selected_survey_ids = list(set(selected_survey_ids_from_title + [
                    s.split(" - ")[0] for s in selected_survey_ids
                ]))
                
                # Filters and Options
                if all_selected_survey_ids:
                    st.subheader("Filters")
                    show_main_only = st.checkbox("Show only main questions", value=False)
                    match_filter = st.selectbox(
                        "Filter by Match Status",
                        ["All", "Matched", "Not Matched"],
                        index=0
                    )
                    
                    st.subheader("Search Questions")
                    question_options = [""] + st.session_state.df_target[st.session_state.df_target["is_choice"] == False]["heading_0"].tolist() if st.session_state.df_target is not None else [""]
                    search_query = st.text_input("Type to filter questions", "")
                    filtered_questions = [q for q in question_options if not search_query or search_query.lower() in q.lower()]
                    selected_question = st.selectbox("Select a question", filtered_questions, index=0)
                
                # Google Form Links
                st.subheader("Add Content")
                st.markdown("[Submit New Question](https://docs.google.com/forms/d/1LoY_La59UJ4ZsuxckM8Wl52kVeLI7a1t1MF8zIQxGUs)")
                st.markdown("[Submit New UID](https://docs.google.com/forms/d/1lkhfm1-t5-zwLxfbVEUiHewveLpGXv5yEVRlQx5XjxA)")

    elif option == "Snowflake":
        st.subheader("Snowflake Actions")
        if st.button("🔁 Run Matching on Snowflake Data"):
            try:
                with st.spinner("Fetching Snowflake data..."):
                    df_reference = run_snowflake_reference_query()
                    df_target = run_snowflake_target_query()
                
                if df_reference.empty or df_target.empty:
                    st.error("No data retrieved from Snowflake.")
                else:
                    df_final = run_uid_match(df_reference, df_target)
                    
                    st.subheader("Filter by Match Type")
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
            except Exception as e:
                logger.error(f"Snowflake processing failed: {e}")
                if "250001" in str(e):
                    st.error("Snowflake connection failed: User account is locked. Contact your Snowflake admin.")
                elif "sqlalchemy.dialects:snowflake" in str(e):
                    st.error(
                        "Snowflake SQLAlchemy dialect not found. Run: `pip install snowflake-sqlalchemy` and restart the app."
                    )
                else:
                    st.error(f"Error: {e}")

# Main Content
if option == "SurveyMonkey" and all_selected_survey_ids:
    try:
        combined_questions = []
        for survey_id in all_selected_survey_ids:
            with st.spinner(f"Fetching survey questions for ID {survey_id}..."):
                survey_json = get_survey_details(survey_id, token)
                questions = extract_questions(survey_json)
                combined_questions.extend(questions)
        
        st.session_state.df_target = pd.DataFrame(combined_questions)
        
        if st.session_state.df_target.empty:
            st.error("No questions found in the selected survey(s).")
        else:
            try:
                with st.spinner("Matching questions to UIDs..."):
                    st.session_state.df_reference = run_snowflake_reference_query()
                    st.session_state.df_final = run_uid_match(st.session_state.df_reference, st.session_state.df_target)
                    st.session_state.uid_changes = {}
            except Exception as e:
                logger.error(f"UID matching failed: {e}")
                if "250001" in str(e):
                    st.warning(
                        "Snowflake connection failed: User account is locked. "
                        "UID matching is disabled, but you can view and edit questions."
                    )
                elif "sqlalchemy.dialects:snowflake" in str(e):
                    st.error(
                        "Snowflake SQLAlchemy dialect not found. Run: `pip install snowflake-sqlalchemy` and restart the app."
                    )
                else:
                    st.error(f"UID matching failed: {e}")
                st.session_state.df_final = st.session_state.df_target.copy()
                st.session_state.df_final["survey_id_title"] = st.session_state.df_final.apply(
                    lambda x: f"{x['survey_id']} - {x['survey_title']}" if pd.notnull(x['survey_id']) and pd.notnull(x['survey_title']) else "",
                    axis=1
                )
        
        # Display Results
        if st.session_state.df_final is not None:
            matched_percentage = calculate_matched_percentage(st.session_state.df_final)
            st.metric("Matched Questions", f"{matched_percentage}%")
            
            result_df = st.session_state.df_final.copy()
            if selected_question:
                result_df = result_df[result_df["heading_0"] == selected_question]
            if match_filter == "Matched":
                result_df = result_df[result_df["Final_UID"].notna()]
            elif match_filter == "Not Matched":
                result_df = result_df[result_df["Final_UID"].isna()]
            if show_main_only:
                result_df = result_df[result_df["category"] == "Main Question"]
            
            uid_options = [None]
            if st.session_state.df_reference is not None:
                uid_options += [f"{row['uid']} - {row['heading_0']}" for _, row in st.session_state.df_reference.iterrows()]
            
            display_columns = ["survey_id_title", "heading_0", "category", "position", "is_choice", "Final_UID", "schema_type", "Change_UID"]
            display_df = result_df[display_columns].copy()
            display_df = display_df.rename(columns={"heading_0": "Question/Choice", "Final_UID": "Final UID"})
            
            edited_df = st.data_editor(
                display_df,
                column_config={
                    "survey_id_title": st.column_config.TextColumn("Survey ID/Title"),
                    "Question/Choice": st.column_config.TextColumn("Question/Choice"),
                    "category": st.column_config.TextColumn("Category"),
                    "position": st.column_config.NumberColumn("Position"),
                    "is_choice": st.column_config.CheckboxColumn("Is Choice"),
                    "Final UID": st.column_config.TextColumn("Final UID"),
                    "schema_type": st.column_config.TextColumn("Schema Type"),
                    "Change_UID": st.column_config.SelectboxColumn(
                        "Change UID",
                        help="Select a UID from Snowflake",
                        options=uid_options,
                        default=None
                    )
                },
                disabled=["survey_id_title", "Question/Choice", "category", "position", "is_choice", "Final UID", "schema_type"],
                hide_index=True
            )
            
            for idx, row in edited_df.iterrows():
                current_change_uid = st.session_state.df_final.at[idx, "Change_UID"] if "Change_UID" in st.session_state.df_final.columns else None
                if pd.notnull(row["Change_UID"]) and row["Change_UID"] != current_change_uid and row["category"] != "Heading":
                    new_uid = row["Change_UID"].split(" - ")[0] if row["Change_UID"] and " - " in row["Change_UID"] else None
                    st.session_state.df_final.at[idx, "Final_UID"] = new_uid
                    st.session_state.df_final.at[idx, "configured_final_UID"] = new_uid
                    st.session_state.df_final.at[idx, "Change_UID"] = row["Change_UID"]
                    st.session_state.uid_changes[idx] = new_uid
            
            # Export Options
            st.subheader("Export to Snowflake")
            export_columns = [
                "survey_id", "survey_title", "heading_0", "configured_final_UID", "position",
                "is_choice", "category", "parent_question", "schema_type", "mandatory"
            ]
            export_df = st.session_state.df_final[export_columns].copy()
            export_df = export_df.rename(columns={"configured_final_UID": "uid"})
            
            st.download_button(
                "📥 Download as CSV",
                export_df.to_csv(index=False),
                f"survey_with_uids_{uuid4()}.csv",
                "text/csv"
            )
            
            if st.button("🚀 Upload to Snowflake"):
                try:
                    with st.spinner("Uploading to Snowflake..."):
                        with get_snowflake_engine().connect() as conn:
                            export_df.to_sql(
                                'SURVEY_DETAILS_RESPONSES_COMBINED_LIVE',
                                conn,
                                schema='DBT_SURVEY_MONKEY',
                                if_exists='append',
                                index=False
                            )
                        st.success("Successfully uploaded to Snowflake!")
                except Exception as e:
                    logger.error(f"Snowflake upload failed: {e}")
                    if "250001" in str(e):
                        st.error("Snowflake upload failed: User account is locked. Contact your Snowflake admin.")
                    elif "sqlalchemy.dialects:snowflake" in str(e):
                        st.error(
                            "Snowflake SQLAlchemy dialect not found. Run: `pip install snowflake-sqlalchemy` and restart the app."
                        )
                    else:
                        st.error(f"Snowflake upload failed: {e}")

    except Exception as e:
        logger.error(f"SurveyMonkey processing failed: {e}")
        if "429" in str(e):
            st.error("SurveyMonkey API rate limit exceeded. Please wait and try again later.")
        else:
            st.error(f"Error: {e}")
else:
    st.write("Select a data source and survey to begin.")
