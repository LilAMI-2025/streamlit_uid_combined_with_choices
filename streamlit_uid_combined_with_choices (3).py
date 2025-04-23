import streamlit as st
import pandas as pd
import requests
import re
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
TFIDF_HIGH_CONFIDENCE = 0.60
TFIDF_LOW_CONFIDENCE = 0.50
SEMANTIC_THRESHOLD = 0.60
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 1000

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
                "UID matching is disabled, but you can edit questions, search, and use Google Forms. "
                "Visit: https://community.snowflake.com/s/error-your-user-login-has-been-locked"
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

# Calculate Matched Questions Percentage
def calculate_matched_percentage(df_final):
    if df_final is None or df_final.empty:
        return 0.0
    
    # Filter for main questions
    df_main = df_final[df_final["is_choice"] == False].copy()
    
    # Exclusion criteria
    privacy_filter = ~df_main["heading_0"].str.contains("Our Privacy Policy", case=False, na=False)
    html_pattern = r"<div.*text-align:\s*center.*<span.*font-size:\s*12pt.*<em>If you have any questions, please contact your AMI Learner Success Manager.*</em>.*</span>.*</div>"
    html_filter = ~df_main["heading_0"].str.contains(html_pattern, case=False, na=False, regex=True)
    
    # Apply filters
    eligible_questions = df_main[privacy_filter & html_filter]
    
    if eligible_questions.empty:
        return 0.0
    
    # Calculate percentage
    matched_questions = eligible_questions[eligible_questions["Final_UID"].notna()]
    percentage = (len(matched_questions) / len(eligible_questions)) * 100
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
        if "250001" in str(e):
            st.warning(
                "Cannot fetch Snowflake data: User account is locked. "
                "UID matching is disabled. Please resolve the lockout and retry."
            )
        elif "invalid identifier" in str(e).lower():
            st.warning(
                "Snowflake query failed due to invalid column. "
                "UID matching is disabled, but you can edit questions, search, and use Google Forms. "
                "Contact your Snowflake admin to verify table schema."
            )
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
        if "250001" in str(e):
            st.warning(
                "Cannot fetch Snowflake data: User account is locked. "
                "Please resolve the lockout and retry."
            )
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
    # Extract SurveyMonkey questions and choices as question/choice
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
                    "mandatory_editable": True
                })
                choices = question.get("answers", {}).get("choices", [])
                for choice in choices:
                    choice_text = choice.get("text", "")
                    if choice_text:
                        questions.append({
                            "heading_0": f"{q_text} - {choice_text}",  # SurveyMonkey choice text
                            "position": global_position,
                            "is_choice": True,
                            "parent_question": q_text,
                            "question_uid": q_id,
                            "schema_type": schema_type,
                            "mandatory": False,
                            "mandatory_editable": False
                        })
    return questions

# UID Matching
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

def finalize_matches(df_target, df_reference):
    df_target["Final_UID"] = df_target["Suggested_UID"].combine_first(df_target["Semantic_UID"])
    df_target["configured_final_UID"] = df_target["Final_UID"]
    df_target["Final_Question"] = df_target["Matched_Question"]
    df_target["Final_Match_Type"] = df_target.apply(assign_match_type, axis=1)
    df_target["Change_UID"] = df_target["Final_UID"].apply(
        lambda x: f"{x} - {df_reference[df_reference['uid'] == x]['heading_0'].iloc[0]}" if pd.notnull(x) and x in df_reference["uid"].values else None
    )
    
    df_target["Final_UID"] = df_target.apply(
        lambda row: df_target[df_target["heading_0"] == row["parent_question"]]["Final_UID"].iloc[0]
        if row["is_choice"] and pd.notnull(row["parent_question"]) else row["Final_UID"],
        axis=1
    )
    df_target["configured_final_UID"] = df_target["Final_UID"]
    df_target["Change_UID"] = df_target["Final_UID"].apply(
        lambda x: f"{x} - {df_reference[df_reference['uid'] == x]['heading_0'].iloc[0]}" if pd.notnull(x) and x in df_reference["uid"].values else None
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
            batch_target = finalize_matches(batch_target, df_reference)
            batch_target = detect_uid_conflicts(batch_target)
        df_results.append(batch_target)
    
    if not df_results:
        logger.warning("No results from batch processing.")
        return pd.DataFrame()
    return pd.concat(df_results, ignore_index=True)

# App UI
st.title("🧠 UID Matcher: Snowflake + SurveyMonkey")

# Secrets Validation
if "snowflake" not in st.secrets or "surveymonkey" not in st.secrets:
    st.error("Missing secrets configuration for Snowflake or SurveyMonkey.")
    st.stop()

# Data Source Selection
option = st.radio("Choose Data Source", ["SurveyMonkey", "Snowflake"], horizontal=True)

# Initialize session state
if "df_target" not in st.session_state:
    st.session_state.df_target = None
if "df_final" not in st.session_state:
    st.session_state.df_final = None
if "uid_changes" not in st.session_state:
    st.session_state.uid_changes = {}
if "custom_questions" not in st.session_state:
    st.session_state.custom_questions = pd.DataFrame(columns=["Customized Question", "Original Question", "Final_UID"])
if "df_reference" not in st.session_state:
    st.session_state.df_reference = None

if option == "SurveyMonkey":
    try:
        token = st.secrets.get("surveymonkey", {}).get("token", None)
        if not token:
            st.error("SurveyMonkey token is missing in secrets configuration.")
            st.stop()
        with st.spinner("Fetching surveys..."):
            surveys = get_surveys(token)
        if not surveys:
            st.error("No surveys found or invalid API response.")
        else:
            choices = {s["title"]: s["id"] for s in surveys}
            selected_survey = st.selectbox("Choose Survey", [""] + list(choices.keys()), index=0)
            if selected_survey:
                with st.spinner("Fetching survey questions..."):
                    survey_json = get_survey_details(choices[selected_survey], token)
                    questions = extract_questions(survey_json)
                    st.session_state.df_target = pd.DataFrame(questions)
                
                if st.session_state.df_target.empty:
                    st.error("No questions found in the selected survey.")
                else:
                    # Run UID matching
                    try:
                        with st.spinner("Matching questions to UIDs..."):
                            st.session_state.df_reference = run_snowflake_reference_query()
                            st.session_state.df_final = run_uid_match(st.session_state.df_reference, st.session_state.df_target)
                            st.session_state.uid_changes = {}
                    except Exception as e:
                        logger.error(f"UID matching failed: {e}")
                        if "250001" in str(e) or "invalid identifier" in str(e).lower():
                            st.warning(
                                "Snowflake connection failed: Account may be locked or table schema is incorrect. "
                                "UID matching is disabled, but you can edit questions, search, and use Google Forms. "
                                "Contact your Snowflake admin to resolve lockout or verify table schema."
                            )
                            st.session_state.df_reference = None
                            st.session_state.df_final = st.session_state.df_target.copy()
                            st.session_state.df_final["Final_UID"] = None
                            st.session_state.df_final["configured_final_UID"] = None
                            st.session_state.df_final["Change_UID"] = None
                            st.session_state.uid_changes = {}
                        else:
                            st.error(f"UID matching failed: {e}")
                            raise
                    
                    # Three Tabs
                    tab1, tab2, tab3 = st.tabs(["Survey Questions and Choices", "UID Matching and Configuration", "Configured Survey"])

                    # Tab 1: Survey Questions and Choices
                    with tab1:
                        st.write("Edit mandatory status for questions/choices. Go to next tab for UID matching.")
                        show_main_only = st.checkbox("Show only main questions", value=False)
                        display_df = st.session_state.df_target[st.session_state.df_target["is_choice"] == False] if show_main_only else st.session_state.df_target
                        
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
                                    help="Can mandatory status be edited?",
                                    disabled=True
                                ),
                                "heading_0": st.column_config.TextColumn("Question/Choice"),
                                "position": st.column_config.NumberColumn("Position"),
                                "is_choice": st.column_config.CheckboxColumn("Is Choice"),
                                "parent_question": st.column_config.TextColumn("Parent Question"),
                                "schema_type": st.column_config.TextColumn("Schema Type"),
                                "question_uid": st.column_config.TextColumn("Question UID")
                            },
                            disabled=["heading_0", "position", "is_choice", "parent_question", "schema_type", "question_uid", "mandatory_editable"] + (["mandatory"] if not display_df["mandatory_editable"].any() else []),
                            hide_index=True
                        )
                        
                        if not edited_df.empty:
                            editable_rows = st.session_state.df_target[st.session_state.df_target["mandatory_editable"]]
                            if not editable_rows.empty:
                                st.session_state.df_target.loc[st.session_state.df_target["mandatory_editable"], "mandatory"] = edited_df[edited_df["mandatory_editable"]]["mandatory"]

                    # Tab 2: UID Matching and Configuration
                    with tab2:
                        if st.session_state.df_final is not None:
                            # Display matched percentage
                            matched_percentage = calculate_matched_percentage(st.session_state.df_final)
                            st.metric("Matched Questions", f"{matched_percentage}%")
                            
                            st.subheader("UID Matching for Questions/Choices")
                            if st.session_state.df_final["Final_UID"].isna().all():
                                st.info("UID matching disabled due to Snowflake issues. Assign UIDs manually or fix the connection.")
                            
                            show_main_only = st.checkbox("Show only main questions", value=False, key="tab2_main_only")
                            match_filter = st.selectbox(
                                "Filter by Match Status",
                                ["All", "Matched", "Not Matched"],
                                index=0
                            )
                            
                            # Search Questions
                            st.subheader("Search Questions/Choices")
                            question_options = [""] + st.session_state.df_target[st.session_state.df_target["is_choice"] == False]["heading_0"].tolist()
                            search_query = st.text_input("Type to filter questions/choices", "")
                            filtered_questions = [q for q in question_options if not search_query or search_query.lower() in q.lower()]
                            selected_question = st.selectbox("Select a question/choice", filtered_questions, index=0)
                            
                            result_df = st.session_state.df_final.copy()
                            if selected_question:
                                result_df = result_df[result_df["heading_0"] == selected_question]
                            if match_filter == "Matched":
                                result_df = result_df[result_df["Final_UID"].notna()]
                            elif match_filter == "Not Matched":
                                result_df = result_df[result_df["Final_UID"].isna()]
                            result_df = result_df[result_df["is_choice"] == False] if show_main_only else result_df
                            
                            # UID dropdown
                            uid_options = [None]
                            if st.session_state.df_reference is not None:
                                uid_options += [f"{row['uid']} - {row['heading_0']}" for _, row in st.session_state.df_reference.iterrows()]
                            else:
                                st.warning("UID options unavailable due to Snowflake issues. Fix connection to load UIDs.")
                            
                            result_columns = ["heading_0", "position", "is_choice", "Final_UID", "question_uid", "schema_type", "Change_UID"]
                            display_df = result_df[result_columns].copy()
                            display_df = display_df.rename(columns={"heading_0": "Question/Choice", "Final_UID": "final_UID"})
                            
                            edited_df = st.data_editor(
                                display_df,
                                column_config={
                                    "Question/Choice": st.column_config.TextColumn("Question/Choice"),
                                    "position": st.column_config.NumberColumn("Position"),
                                    "is_choice": st.column_config.CheckboxColumn("Is Choice"),
                                    "final_UID": st.column_config.TextColumn("Final UID"),
                                    "question_uid": st.column_config.TextColumn("Question UID"),
                                    "schema_type": st.column_config.TextColumn("Schema Type"),
                                    "Change_UID": st.column_config.SelectboxColumn(
                                        "Change UID",
                                        help="Select a UID from Snowflake",
                                        options=uid_options,
                                        default=None
                                    )
                                },
                                disabled=["Question/Choice", "position", "is_choice", "final_UID", "question_uid", "schema_type"],
                                hide_index=True
                            )
                            
                            # Update UIDs
                            for idx, row in edited_df.iterrows():
                                current_change_uid = st.session_state.df_final.at[idx, "Change_UID"] if "Change_UID" in st.session_state.df_final.columns else None
                                if pd.notnull(row["Change_UID"]) and row["Change_UID"] != current_change_uid:
                                    new_uid = row["Change_UID"].split(" - ")[0] if row["Change_UID"] and " - " in row["Change_UID"] else None
                                    st.session_state.df_final.at[idx, "Final_UID"] = new_uid
                                    st.session_state.df_final.at[idx, "configured_final_UID"] = new_uid
                                    st.session_state.df_final.at[idx, "Change_UID"] = row["Change_UID"]
                                    st.session_state.uid_changes[idx] = new_uid
                            
                            # New Questions
                            st.subheader("Create New Questions")
                            st.write("Submit new questions via Google Form. Fields: Question Text, Type, Choices, Program, Mandatory.")
                            st.markdown("[Submit New Question](https://docs.google.com/forms/d/1LoY_La59UJ4ZsuxckM8Wl52kVeLI7a1t1MF8zIQxGUs)")
                            
                            # New UID
                            st.subheader("Create New UID")
                            st.write("Submit new UIDs via Google Form. Fields: Question Text, Proposed UID, Program, Type, Mandatory.")
                            st.markdown("[Submit New UID](https://docs.google.com/forms/d/1lkhfm1-t5-zwLxfbVEUiHewveLpGXv5yEVRlQx5XjxA)")
                            
                            # Customize Questions
                            st.subheader("Customize Questions/Choices")
                            customize_df = pd.DataFrame({
                                "Pre-existing Question": [None],
                                "Customized Question": [""]
                            })
                            question_options = [None]
                            if st.session_state.df_target is not None:
                                question_options += st.session_state.df_target[st.session_state.df_target["is_choice"] == False]["heading_0"].tolist()
                            
                            customize_edited_df = st.data_editor(
                                customize_df,
                                column_config={
                                    "Pre-existing Question": st.column_config.SelectboxColumn(
                                        "Pre-existing Question",
                                        help="Select a question from the current SurveyMonkey survey",
                                        options=question_options,
                                        default=None
                                    ),
                                    "Customized Question": st.column_config.TextColumn(
                                        "Customized Question",
                                        help="Enter customized question text",
                                        default=""
                                    )
                                },
                                hide_index=True,
                                num_rows="dynamic"
                            )
                            
                            # Save customized questions
                            for _, row in customize_edited_df.iterrows():
                                if row["Pre-existing Question"] and row["Customized Question"]:
                                    original_question = row["Pre-existing Question"]
                                    custom_question = row["Customized Question"]
                                    uid = None
                                    if st.session_state.df_final is not None:
                                        uid_row = st.session_state.df_final[st.session_state.df_final["heading_0"] == original_question]
                                        uid = uid_row["Final_UID"].iloc[0] if not uid_row.empty else None
                                    if custom_question:
                                        new_row = pd.DataFrame({
                                            "Customized Question": [custom_question],
                                            "Original Question": [original_question],
                                            "Final_UID": [uid]
                                        })
                                        st.session_state.custom_questions = pd.concat([st.session_state.custom_questions, new_row], ignore_index=True)
                            
                            # Show customized questions
                            if not st.session_state.custom_questions.empty:
                                st.subheader("Customized Questions/Choices")
                                st.dataframe(st.session_state.custom_questions)

                    # Tab 3: Configured Survey
                    with tab3:
                        if st.session_state.df_final is not None:
                            # Display matched percentage
                            matched_percentage = calculate_matched_percentage(st.session_state.df_final)
                            st.metric("Matched Questions", f"{matched_percentage}%")
                            
                            st.subheader("Configured Survey")
                            config_columns = [
                                "heading_0", "position", "is_choice", "parent_question", 
                                "schema_type", "mandatory", "mandatory_editable", "configured_final_UID"
                            ]
                            config_df = st.session_state.df_final[config_columns].copy()
                            config_df = config_df[config_df["is_choice"] == False] if show_main_only else config_df
                            config_df = config_df.rename(columns={"heading_0": "Question/Choice"})
                            st.dataframe(config_df)
                        else:
                            st.write("Select a survey to view the configured survey.")

    except Exception as e:
        logger.error(f"SurveyMonkey processing failed: {e}")
        st.error(f"Error: {e}")

if option == "Snowflake":
    if st.button("🔁 Run Matching on Snowflake Data"):
        try:
            with st.spinner("Fetching Snowflake data..."):
                df_reference = run_snowflake_reference_query()
                df_target = run_snowflake_target_query()
        
            if df_reference.empty or df_target.empty:
                st.error("No data retrieved from Snowflake.")
            else:
                df_final = run_uid_match(df_reference, df_target)
                
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
                st.error(
                    "Snowflake connection failed: User account is locked. "
                    "Contact your Snowflake admin or wait 15–30 minutes."
                )
            else:
                st.error(f"Error: {e}")
