### --- Full Final Streamlit UID Matcher Script (Improved and Complete) --- ###

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
    import snowflake.sqlalchemy  # Explicit import for Snowflake dialect
except ImportError as e:
    missing_package = str(e).split("No module named ")[-1].replace("'", "")
    raise ImportError(f"""
Missing required package: {missing_package}
Please install it using:
pip install pandas openpyxl rapidfuzz python-Levenshtein SQLAlchemy scikit-learn sentence-transformers streamlit requests snowflake-sqlalchemy snowflake-connector-python
""")

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Streamlit Configuration ---
st.set_page_config(page_title="UID Matcher App", layout="wide")

# --- Constants ---
TFIDF_HIGH_CONFIDENCE = 0.60
TFIDF_LOW_CONFIDENCE = 0.50
SEMANTIC_THRESHOLD = 0.60
FUZZY_THRESHOLD = 0.95
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 1000

# --- Synonym Mapping ---
DEFAULT_SYNONYM_MAP = {
    "please select": "what is",
    "sector you are from": "your sector",
    "identity type": "id type",
    "what type of": "type of",
    "are you": "do you",
}

# --- Excluded Questions List ---
questions_to_exclude = [
    "As we prepare to implement our programme in your company, we would like to define what learning interventions are needed to help you achieve your strategic objectives. Clearly establishing these parameters now will help to inform and support the embedding of a culture of learning and personal development from your leaders all the way through to your non-management staff.",
    "Please provide the following details:",
    "Now, we'd like to find out a little bit about your company's learning initiatives and how well aligned they are to your strategic objectives. A learning initiative is any formal or informal way in which your people are encouraged to learn. It may include online or face-to-face courses, coaching, projects, etc.",
    "This section contains the heart of what we would like you to tell us. The following twenty Winning Behaviours represent what managers and staff do in any successful and growing organisation. Most or all of them would be important for your people too, but we are interested in identifying those few that stand out as supporting your unique strategic opportunities and challenges - those that would make a significant and obvious difference to your organisation’s performance. So we are asking you to rate their importance.\n\nYou will notice that we have not included specific technical or functional skills, like budgeting. This is, firstly, because it would make far too long a list for a busy leader like you to review; and secondly, because we assume that these functional skills are \"threshold behaviours” that every member has to have to be in their jobs, and so are likely to be equally present in your competitors. We would rather focus on the Winning Behaviours that will distinguish your company from others. Of course, if there are specific technical or functional skills that do or could provide you with a competitive advantage, you should certainly include those in your company's learning strategy.",
    "Now, we want to delve a bit deeper to examine how the winning behaviours you have prioritised at the company-wide level might look different if you just focused on those employees who manage people. We will go through the same process of considering the importance, then frequency, and finally ranking of the Winning Behaviours for our learning initiative focus.",
    "Now, we want to delve a bit deeper to examine how the winning behaviours you have prioritised at the company-wide and manager levels might look different if you just focused on those employees who do not manage any people. We will go through the same process of considering the importance, then frequency, and finally ranking of the Winning Behaviours for our learning initiative focus.",
    "As a last step, we ask that you rank order the short list of those you have indicated are both important and less frequent for non-managers. In that way, you will tell us the priority behaviours you would like us to encourage your non-managers to demonstrate. You can either drag and drop the Winning behaviours into your chosen rank position, or you can adjust their numbering manually.",
    "Thank you for taking the time to reflect on how aligned your learning initiatives are with your key strategic priorities. Once you submit, we will review your responses (alongside those of your CEO, people lead, senior managers) in order to advise on owner",
    "BUSINESS DETAILS",
    "LEARNING NEEDS",
    "Confidentiality Assurance",
    "The information provided in this assessment will be treated with the utmost confidentiality and will be used solely for the purpose of developing and improving our access to finance programs. Your institution's identity and specific details will not be disclosed or shared with third parties without prior consent.",
    "Contact Information",
    "Institutional Profile",
    "Section II: Financial Products and Services",
    "Section III: Credit Assessment and Risk Management",
    "Introduction",
    "Welcome to the Business Development Service Provider (BDSP) Diagnostic Tool, a crucial component in our mission to map and enhance the BDS landscape in Rwanda. This survey is designed to capture vital information about your organization's profile, services, target market, business model, and impact measurement practices. Your participation is essential in identifying gaps, challenges, and opportunities within the BDS ecosystem, and will directly inform the development of standards for high-quality, MSME-centric BDS provision.",
    "Your participation in this 15-20 minute survey is invaluable in shaping the future of BDS in Rwanda. Your insights will contribute to improving support for MSMEs, enhancing collaboration among BDS providers, and influencing policy decisions to strengthen the sector. By participating, you'll have the opportunity to reflect on your organization's strengths and areas for growth, gain insights into industry trends, and have a voice in shaping BDS standards and policies. We appreciate your honest and thoughtful responses in this collective effort to foster sustainable enterprise development, growth, and resilience in Rwanda.",
    "Confidentiality",
    "1. Contact Information",
    "2. Organizational Profile and Reach",
    "3 Service Offering and Delivery",
    "4 Target Market and Specialization",
    "Understanding your target market helps us identify any underserved segments and opportunities for expanding BDS reach.",
    "5 Business Model and Sustainability",
    "This section assesses your organization's financial health and sustainability, helping us identify areas where BDS providers might need support.",
    "Understanding how you measure impact and the challenges you face helps us develop better support systems and identify areas for improvement in the BDS ecosystem.",
    "7 Ecosystem Collaboration and Support",
    "This section explores how BDS providers interact within the larger ecosystem and what support would be most beneficial.",
    "8. Future Outlook",
    "Understanding your future plans and perspectives helps us anticipate trends and prepare for the evolving needs of the BDS sector.",
    "Thank You for Your Participation",
    "Thank you for dedicating your time and effort to complete this diagnostic tool. Your valuable insights are crucial in our mission to map the landscape of BDS provision in Rwanda, identify opportunities for improvement, and drive growth in the sector. By sharing your experiences and perspectives, you've contributed to a comprehensive understanding of the challenges and opportunities facing BDS providers and MSMEs in Rwanda, which will inform targeted interventions and enhance collaboration within the ecosystem.",
    "Your participation is a significant step towards creating a more robust, responsive, and effective BDS ecosystem that can drive sustainable MSME growth and contribute to Rwanda's economic development. We look forward to sharing the outcomes of this study and exploring potential opportunities for collaboration in the future. Together, we're shaping a stronger BDS sector that will support Rwanda's vision of becoming a knowledge-based, middle-income economy. Thank you again for your commitment to excellence in business development services.",
    "MSME Survey Tool: Understanding BDS Provision in Rwanda",
    "Section 1: Contact Information",
    "Section 2: Business Challenges and BDS Engagement",
    "Section 3: BDS Quality and Needs Assessment",
    "Section 4: Future Engagement",
    "Conclusion",
    "Your organisation's current learning initiatives",
    "Please describe in very practical behaviours what your people need to do differently to achieve your strategic goals. In other words, what would you notice that they would be doing differently if the learning was entirely successful?",
    "RATE: Pinpoint your organisation's key Winning Behaviours",
    "FREQUENCY: Tell us about how often the Winning Behaviours are displayed in your organisation",
    "As a last step, we ask that you rank order the short list of those you have indicated are both important and less frequent. In that way, you will tell us the priority behaviours you would like us to encourage your people to demonstrate. You can either drag and drop the Winning behaviours into your chosen rank position, or you can adjust their numbering manually.",
    "Prioritise the Winning Behaviours to focus on",
    "First, we need to know a bit about your role",
    "RATE: Pinpoint key Winning Behaviours for managers",
    "FREQUENCY: Tell us how often the Winning Behaviours are displayed by managers",
    "RANK: Prioritise the Winning Behaviours to focus on for managers",
    "RATE: Pinpoint key Winning Behaviours for non-managers",
    "FREQUENCY: Tell us how often the Winning Behaviours are displayed by non-managers",
    "RANK: Prioritise the Winning Behaviours to focus on for non-managers",
    "Ecosystem Support Organizations Interview Guide",
    "Introduction (5 minutes)",
    "Introduce yourself and the purpose of the interview",
    "Assure confidentiality and ask for permission to record the interview",
    "Explain that the interview will take about 1 hour"
]

# --- Cached Resources ---
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
        import snowflake.sqlalchemy
        import sqlalchemy
        logger.info(f"SQLAlchemy version: {sqlalchemy.__version__}")
        logger.info(f"Snowflake-SQLAlchemy version: {snowflake.sqlalchemy.__version__}")
    except ImportError:
        st.error("Snowflake SQLAlchemy dialect not found. Install it using: pip install snowflake-sqlalchemy")
        raise ImportError("Missing snowflake-sqlalchemy package")
    
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
            st.error("Snowflake connection failed: User account is locked. Contact your Snowflake admin.")
        elif "Can't load plugin: sqlalchemy.dialects:snowflake" in str(e):
            st.error("Snowflake SQLAlchemy dialect failed to load. Ensure snowflake-sqlalchemy is installed.")
        raise

@st.cache_data
def get_tfidf_vectors(df_reference):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    vectors = vectorizer.fit_transform(df_reference["norm_text"])
    return vectorizer, vectors

# --- Snowflake Queries ---
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
            st.error("Snowflake query failed: User account is locked. Contact your Snowflake admin.")
        elif "invalid identifier" in str(e).lower():
            st.error("Snowflake query failed: Invalid table or column. Contact your Snowflake admin.")
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
            st.error("Snowflake query failed: User account is locked. Contact your Snowflake admin.")
        raise

# --- Text Normalization ---
def enhanced_normalize(text, synonym_map=DEFAULT_SYNONYM_MAP):
    text = str(text).lower()
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'[^a-z0-9 ]', '', text)
    for phrase, replacement in synonym_map.items():
        text = text.replace(phrase, replacement)
    return ' '.join(w for w in text.split() if w not in ENGLISH_STOP_WORDS)

# --- UID Matching Functions ---
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

def is_excluded_question(question, questions_to_exclude, model):
    logger.info(f"Checking exclusion for: {question}")
    if question in questions_to_exclude:
        logger.info(f"Exact match found for: {question}")
        return True
    
    question_emb = model.encode([question], convert_to_tensor=True)
    exclude_emb = model.encode(questions_to_exclude, convert_to_tensor=True)
    cosine_scores = util.cos_sim(question_emb, exclude_emb)[0]
    max_score = cosine_scores.max().item()
    
    fuzzy_scores = [fuzz.ratio(question.lower(), excl.lower()) / 100.0 for excl in questions_to_exclude]
    max_fuzzy_score = max(fuzzy_scores) if fuzzy_scores else 0.0
    
    logger.info(f"Question: {question}, Max Semantic Score: {max_score}, Max Fuzzy Score: {max_fuzzy_score}")
    return max_score >= FUZZY_THRESHOLD or max_fuzzy_score >= FUZZY_THRESHOLD

def assign_match_type(row):
    if pd.notnull(row["Suggested_UID"]):
        return row["Match_Confidence"]
    return "🧠 Semantic" if pd.notnull(row["Semantic_UID"]) else "❌ No match"

def detect_uid_conflicts(df_target):
    uid_conflicts = df_target.groupby("Final_UID")["heading_0"].nunique()
    duplicate_uids = uid_conflicts[uid_conflicts > 1].index
    df_target["UID_Conflict"] = df_target["Final_UID"].apply(
        lambda x: "⚠️ Conflict" if pd.notnull(x) and x in duplicate_uids else ""
    )
    return df_target

def finalize_matches(df_target, df_reference):
    model = load_sentence_transformer()
    df_target["Final_UID"] = df_target["Suggested_UID"].combine_first(df_target["Semantic_UID"])
    df_target["Final_Question"] = df_target["Matched_Question"]
    df_target["Final_Match_Type"] = df_target.apply(assign_match_type, axis=1)
    
    df_target["Final_UID"] = df_target.apply(
        lambda row: "9999" if is_excluded_question(row["heading_0"], questions_to_exclude, model) else row["Final_UID"],
        axis=1
    )
    df_target["Change_UID"] = df_target["Final_UID"].apply(
        lambda x: f"9999 - Excluded Question" if x == "9999" else (
            f"{x} - {df_reference[df_reference['uid'] == x]['heading_0'].iloc[0]}" if pd.notnull(x) and x in df_reference["uid"].values else None
        )
    )
    return df_target

def run_uid_match(df_reference, df_target, synonym_map=DEFAULT_SYNONYM_MAP, batch_size=BATCH_SIZE):
    if df_reference.empty or df_target.empty:
        logger.warning("Empty input dataframes provided.")
        st.error("Input data is empty.")
        return pd.DataFrame()

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

# --- Streamlit Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "UID Matching", "SurveyMonkey Fetch", "Create New Survey"])

# --- Home Page ---
if page == "Home":
    st.title("Welcome to UID Matcher App")
    st.write("Use the sidebar to navigate between functionalities.")

# --- UID Matching Workflow ---
elif page == "UID Matching":
    st.title("UID Matching")
    option = st.radio("Select Data Source", ["SurveyMonkey", "Snowflake"], horizontal=True)

    if option == "Snowflake":
        if st.button("🔁 Run UID Matching on Snowflake Data"):
            try:
                with st.spinner("Fetching Snowflake Data..."):
                    df_reference = run_snowflake_reference_query()
                    df_target = run_snowflake_target_query()
                
                with st.spinner("Running UID Matching..."):
                    df_final = run_uid_match(df_reference, df_target)
                
                if df_final.empty:
                    st.error("No matching results to display.")
                else:
                    st.success("UID Matching Completed!")
                    
                    # Interactive Filtering
                    match_filter = st.selectbox(
                        "Filter by Match Status",
                        ["All", "Matched", "Not Matched"],
                        index=0
                    )
                    confidence_filter = st.multiselect(
                        "Filter by Match Type",
                        ["✅ High", "⚠️ Low", "🧠 Semantic", "❌ No match"],
                        default=["✅ High", "⚠️ Low", "🧠 Semantic"]
                    )
                    
                    result_df = df_final.copy()
                    if match_filter == "Matched":
                        result_df = result_df[result_df["Final_UID"].notna()]
                    elif match_filter == "Not Matched":
                        result_df = result_df[result_df["Final_UID"].isna()]
                    if confidence_filter:
                        result_df = result_df[result_df["Final_Match_Type"].isin(confidence_filter)]
                    
                    # Interactive UID Editing
                    st.subheader("Edit UID Assignments")
                    uid_options = [None] + [f"{row['uid']} - {row['heading_0']}" for _, row in df_reference.iterrows()]
                    edited_df = st.data_editor(
                        result_df[["heading_0", "Final_UID", "Final_Match_Type", "Change_UID", "UID_Conflict"]],
                        column_config={
                            "heading_0": st.column_config.TextColumn("Question"),
                            "Final_UID": st.column_config.TextColumn("Final UID"),
                            "Final_Match_Type": st.column_config.TextColumn("Match Type"),
                            "Change_UID": st.column_config.SelectboxColumn(
                                "Change UID",
                                help="Select a UID from Snowflake",
                                options=uid_options,
                                default=None
                            ),
                            "UID_Conflict": st.column_config.TextColumn("Conflict")
                        },
                        disabled=["heading_0", "Final_UID", "Final_Match_Type", "UID_Conflict"],
                        hide_index=True
                    )
                    
                    for idx, row in edited_df.iterrows():
                        if pd.notnull(row["Change_UID"]) and row["Change_UID"] != result_df.at[idx, "Change_UID"]:
                            new_uid = row["Change_UID"].split(" - ")[0] if row["Change_UID"] and " - " in row["Change_UID"] else None
                            df_final.at[idx, "Final_UID"] = new_uid
                            df_final.at[idx, "Change_UID"] = row["Change_UID"]
                    
                    st.dataframe(result_df)
                    st.download_button(
                        "Download Results as CSV",
                        result_df.to_csv(index=False).encode('utf-8'),
                        "uid_matching_results.csv",
                        "text/csv"
                    )
                    
                    # Google Forms Links
                    st.subheader("Submit New Questions or UIDs")
                    st.markdown("[Submit New Question](https://docs.google.com/forms/d/1LoY_La59UJ4ZsuxckM8Wl52kVeLI7a1t1MF8zIQxGUs)")
                    st.markdown("[Submit New UID](https://docs.google.com/forms/d/1lkhfm1-t5-zwLxfbVEUiHewveLpGXv5yEVRlQx5XjxA)")

            except Exception as e:
                logger.error(f"Snowflake processing failed: {e}")
                if "250001" in str(e):
                    st.error("Snowflake processing failed: User account is locked. Contact your Snowflake admin to resolve.")
                elif "Can't load plugin: sqlalchemy.dialects:snowflake" in str(e):
                    st.error("Snowflake processing failed: Snowflake SQLAlchemy dialect not found. Install snowflake-sqlalchemy.")
                else:
                    st.error(f"Snowflake processing failed: {e}")

# --- SurveyMonkey Fetch Workflow ---
elif page == "SurveyMonkey Fetch":
    st.title("SurveyMonkey Fetch + UID Matching")

    token = st.secrets.get("surveymonkey", {}).get("token", None)
    if not token:
        st.error("SurveyMonkey token missing in secrets.")
        st.stop()

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

    with st.spinner("Fetching available surveys..."):
        surveys = get_surveys(token)

    survey_dict = {s['title']: s['id'] for s in surveys}
    selected_title = st.selectbox("Choose Survey", list(survey_dict.keys()))
    selected_id = survey_dict[selected_title]

    if st.button("Fetch Survey and Run UID Matching"):
        try:
            with st.spinner("Fetching Survey Details..."):
                survey_json = get_survey_details(selected_id, token)
                questions_list = []
                for page in survey_json.get("pages", []):
                    for question in page.get("questions", []):
                        q_text = question.get("headings", [{}])[0].get("heading", "")
                        q_id = question.get("id", "")
                        q_family = question.get("family", "")
                        q_subtype = question.get("subtype", "")
                        q_required = question.get("required", False)
                        q_position = question.get("position", "")
                        choices = question.get("answers", {}).get("choices", [])
                        schema_type = (
                            "Single Choice" if q_family == "single_choice"
                            else "Multiple Choice" if q_family == "multiple_choice"
                            else "Open-Ended" if q_family == "open_ended"
                            else "Matrix" if q_family == "matrix"
                            else "Other"
                        )
                        
                        questions_list.append({
                            "survey_id": survey_json.get("id", ""),
                            "survey_title": survey_json.get("title", ""),
                            "question_id": q_id,
                            "heading_0": q_text,
                            "family": q_family,
                            "subtype": q_subtype,
                            "required": q_required,
                            "mandatory_editable": True,
                            "position": q_position,
                            "choices": ", ".join([choice.get("text", "") for choice in choices]),
                            "schema_type": schema_type,
                            "is_choice": False
                        })
                        
                        for choice in choices:
                            choice_text = choice.get("text", "")
                            if choice_text:
                                questions_list.append({
                                    "survey_id": survey_json.get("id", ""),
                                    "survey_title": survey_json.get("title", ""),
                                    "question_id": q_id,
                                    "heading_0": f"{q_text} - {choice_text}",
                                    "family": q_family,
                                    "subtype": q_subtype,
                                    "required": False,
                                    "mandatory_editable": False,
                                    "position": q_position,
                                    "choices": choice_text,
                                    "schema_type": schema_type,
                                    "is_choice": True,
                                    "parent_question": q_text
                                })

                df_target = pd.DataFrame(questions_list)
                df_target["survey_id_title"] = df_target.apply(
                    lambda x: f"{x['survey_id']} - {x['survey_title']}", axis=1
                )

            with st.spinner("Running UID Matching..."):
                df_reference = run_snowflake_reference_query()
                df_final = run_uid_match(df_reference, df_target)

            if df_final.empty:
                st.error("No matching results to display.")
            else:
                st.success("UID Matching Completed on SurveyMonkey Fetched Data!")
                
                # Interactive Filtering
                match_filter = st.selectbox(
                    "Filter by Match Status",
                    ["All", "Matched", "Not Matched"],
                    index=0
                )
                confidence_filter = st.multiselect(
                    "Filter by Match Type",
                    ["✅ High", "⚠️ Low", "🧠 Semantic", "❌ No match"],
                    default=["✅ High", "⚠️ Low", "🧠 Semantic"]
                )
                
                result_df = df_final.copy()
                if match_filter == "Matched":
                    result_df = result_df[result_df["Final_UID"].notna()]
                elif match_filter == "Not Matched":
                    result_df = result_df[result_df["Final_UID"].isna()]
                if confidence_filter:
                    result_df = result_df[result_df["Final_Match_Type"].isin(confidence_filter)]
                
                # Interactive UID Editing
                st.subheader("Edit UID Assignments")
                uid_options = [None] + [f"{row['uid']} - {row['heading_0']}" for _, row in df_reference.iterrows()]
                edited_df = st.data_editor(
                    result_df[["survey_id_title", "heading_0", "schema_type", "is_choice", "Final_UID", "Final_Match_Type", "Change_UID", "UID_Conflict"]],
                    column_config={
                        "survey_id_title": st.column_config.TextColumn("Survey ID/Title"),
                        "heading_0": st.column_config.TextColumn("Question/Choice"),
                        "schema_type": st.column_config.TextColumn("Schema Type"),
                        "is_choice": st.column_config.CheckboxColumn("Is Choice"),
                        "Final_UID": st.column_config.TextColumn("Final UID"),
                        "Final_Match_Type": st.column_config.TextColumn("Match Type"),
                        "Change_UID": st.column_config.SelectboxColumn(
                            "Change UID",
                            help="Select a UID from Snowflake",
                            options=uid_options,
                            default=None
                        ),
                        "UID_Conflict": st.column_config.TextColumn("Conflict")
                    },
                    disabled=["survey_id_title", "heading_0", "schema_type", "is_choice", "Final_UID", "Final_Match_Type", "UID_Conflict"],
                    hide_index=True
                )
                
                for idx, row in edited_df.iterrows():
                    if pd.notnull(row["Change_UID"]) and row["Change_UID"] != result_df.at[idx, "Change_UID"]:
                        new_uid = row["Change_UID"].split(" - ")[0] if row["Change_UID"] and " - " in row["Change_UID"] else None
                        df_final.at[idx, "Final_UID"] = new_uid
                        df_final.at[idx, "Change_UID"] = row["Change_UID"]
                
                st.dataframe(result_df)
                st.download_button(
                    "Download UID Matched Survey as CSV",
                    result_df.to_csv(index=False).encode('utf-8'),
                    "survey_uid_matched.csv",
                    "text/csv"
                )
                
                # Google Forms Links
                st.subheader("Submit New Questions or UIDs")
                st.markdown("[Submit New Question](https://docs.google.com/forms/d/1LoY_La59UJ4ZsuxckM8Wl52kVeLI7a1t1MF8zIQxGUs)")
                st.markdown("[Submit New UID](https://docs.google.com/forms/d/1lkhfm1-t5-zwLxfbVEUiHewveLpGXv5yEVRlQx5XjxA)")

        except Exception as e:
            logger.error(f"SurveyMonkey processing failed: {e}")
            st.error(f"Failed: {e}")

# --- Create New Survey Workflow ---
elif page == "Create New Survey":
    st.title("Create New Survey")

    token = st.secrets.get("surveymonkey", {}).get("token", None)
    if not token:
        st.error("SurveyMonkey token missing in secrets.")
        st.stop()

    def create_survey(token, survey_template):
        url = "https://api.surveymonkey.com/v3/surveys"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        response = requests.post(url, headers=headers, json={
            "title": survey_template["title"],
            "nickname": survey_template.get("nickname", survey_template["title"]),
            "language": survey_template.get("language", "en")
        })
        response.raise_for_status()
        return response.json().get("id")

    def create_page(token, survey_id, page_template):
        url = f"https://api.surveymonkey.com/v3/surveys/{survey_id}/pages"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        response = requests.post(url, headers=headers, json={
            "title": page_template.get("title", ""),
            "description": page_template.get("description", "")
        })
        response.raise_for_status()
        return response.json().get("id")

    def create_question(token, survey_id, page_id, question_template):
        url = f"https://api.surveymonkey.com/v3/surveys/{survey_id}/pages/{page_id}/questions"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        payload = {
            "family": question_template["family"],
            "subtype": question_template["subtype"],
            "headings": [{"heading": question_template["heading"]}],
            "position": question_template["position"],
            "required": question_template.get("is_required", False)
        }
        if "choices" in question_template:
            payload["answers"] = {"choices": question_template["choices"]}
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json().get("id")

    with st.form("new_survey_form"):
        survey_title = st.text_input("Survey Title", value="New Survey")
        survey_language = st.selectbox("Language", ["en", "es", "fr", "de"], index=0)
        num_questions = st.number_input("Number of Questions", min_value=1, max_value=10, value=1)
        
        questions = []
        for i in range(num_questions):
            st.write(f"#### Question {i+1}")
            question_text = st.text_input(f"Question Text {i+1}", key=f"q_text_{i}")
            question_type = st.selectbox(
                f"Question Type {i+1}",
                ["Single Choice", "Multiple Choice", "Open-Ended"],
                key=f"q_type_{i}"
            )
            is_required = st.checkbox(f"Required {i+1}", key=f"q_required_{i}")
            
            question_template = {
                "heading": question_text,
                "position": i + 1,
                "is_required": is_required
            }
            
            if question_type == "Single Choice":
                question_template["family"] = "single_choice"
                question_template["subtype"] = "vertical"
                num_choices = st.number_input(
                    f"Number of Choices for Question {i+1}",
                    min_value=1,
                    max_value=10,
                    value=2,
                    key=f"num_choices_{i}"
                )
                choices = []
                for j in range(num_choices):
                    choice_text = st.text_input(
                        f"Choice {j+1} for Question {i+1}",
                        key=f"choice_{i}_{j}"
                    )
                    if choice_text:
                        choices.append({"text": choice_text, "position": j + 1})
                if choices:
                    question_template["choices"] = choices
            elif question_type == "Multiple Choice":
                question_template["family"] = "multiple_choice"
                question_template["subtype"] = "vertical"
                num_choices = st.number_input(
                    f"Number of Choices for Question {i+1}",
                    min_value=1,
                    max_value=10,
                    value=2,
                    key=f"num_choices_{i}"
                )
                choices = []
                for j in range(num_choices):
                    choice_text = st.text_input(
                        f"Choice {j+1} for Question {i+1}",
                        key=f"choice_{i}_{j}"
                    )
                    if choice_text:
                        choices.append({"text": choice_text, "position": j + 1})
                if choices:
                    question_template["choices"] = choices
            elif question_type == "Open-Ended":
                question_template["family"] = "open_ended"
                question_template["subtype"] = "essay"
            
            if question_text:
                questions.append(question_template)
        
        submit = st.form_submit_button("Create Survey")
        if submit:
            if not survey_title or not questions:
                st.error("Survey title and at least one question are required.")
            else:
                survey_template = {
                    "title": survey_title,
                    "language": survey_language,
                    "pages": [{
                        "title": "Page 1",
                        "description": "",
                        "questions": questions
                    }]
                }
                try:
                    with st.spinner("Creating survey in SurveyMonkey..."):
                        survey_id = create_survey(token, survey_template)
                        for page_template in survey_template["pages"]:
                            page_id = create_page(token, survey_id, page_template)
                            for question_template in page_template["questions"]:
                                create_question(token, survey_id, page_id, question_template)
                        st.success(f"Survey created successfully! Survey ID: {survey_id}")
                except Exception as e:
                    logger.error(f"Survey creation failed: {e}")
                    st.error(f"Failed to create survey: {e}")

### --- End of Full Combined and Final Script --- ###
