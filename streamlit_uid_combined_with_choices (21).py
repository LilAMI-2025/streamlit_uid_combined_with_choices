
# Install libraries (run once)
!pip install pandas openpyxl fuzzywuzzy python-Levenshtein SQLAlchemy


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
from fuzzywuzzy import fuzz

# Setup
st.set_page_config(page_title="UID Matcher Combined", layout="wide")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
TFIDF_HIGH_CONFIDENCE = 0.60
TFIDF_LOW_CONFIDENCE = 0.50
SEMANTIC_THRESHOLD = 0.60
FUZZY_THRESHOLD = 0.95  # For near-exact matches in questions_to_exclude
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
        logger.info("calculate_matched_percentage: df_final is None or empty")
        return 0.0
    
    df_main = df_final[df_final["is_choice"] == False].copy()
    logger.info(f"calculate_matched_percentage: Total main questions: {len(df_main)}")
    
    privacy_filter = ~df_main["heading_0"].str.contains("Our Privacy Policy", case=False, na=False)
    html_pattern = r"<div.*text-align:\s*center.*<span.*font-size:\s*12pt.*<em>If you have any questions, please contact your AMI Learner Success Manager.*</em>.*</span>.*</div>"
    html_filter = ~df_main["heading_0"].str.contains(html_pattern, case=False, na=False, regex=True)
    
    eligible_questions = df_main[privacy_filter & html_filter]
    logger.info(f"calculate_matched_percentage: Eligible questions after exclusions: {len(eligible_questions)}")
    
    if eligible_questions.empty:
        logger.info("calculate_matched_percentage: No eligible questions after exclusions")
        return 0.0
    
    matched_questions = eligible_questions[eligible_questions["Final_UID"].notna()]
    logger.info(f"calculate_matched_percentage: Matched questions: {len(matched_questions)}")
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

def create_survey(token, survey_template):
    url = "https://api.surveymonkey.com/v3/surveys"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    try:
        response = requests.post(url, headers=headers, json={
            "title": survey_template["title"],
            "nickname": survey_template.get("nickname", survey_template["title"]),
            "language": survey_template.get("language", "en")
        })
        response.raise_for_status()
        survey_id = response.json().get("id")
        return survey_id
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
        page_id = response.json().get("id")
        return page_id
    except requests.RequestException as e:
        logger.error(f"Failed to create page for survey {survey_id}: {e}")
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
        logger.error(f"Failed to create question for page {page_id}: {e}")
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
                    "heading_0": q_text,
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
                            "heading_0": f"{q_text} - {choice_text}",
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

def is_excluded_question(question, questions_to_exclude, model):
    # Exact match check
    if question in questions_to_exclude:
        return True
    
    # Fuzzy match using SentenceTransformer
    question_emb = model.encode([question], convert_to_tensor=True)
    exclude_emb = model.encode(questions_to_exclude, convert_to_tensor=True)
    cosine_scores = util.cos_sim(question_emb, exclude_emb)[0]
    max_score = cosine_scores.max().item()
    
    # Additional fuzzy match using fuzzywuzzy for string similarity
    fuzzy_scores = [fuzz.ratio(question.lower(), excl.lower()) / 100.0 for excl in questions_to_exclude]
    max_fuzzy_score = max(fuzzy_scores) if fuzzy_scores else 0.0
    
    # Consider it a match if either semantic or fuzzy score is above threshold
    return max_score >= FUZZY_THRESHOLD or max_fuzzy_score >= FUZZY_THRESHOLD

def finalize_matches(df_target, df_reference):
    # List of questions to assign UID 9999
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

    # Load SentenceTransformer for fuzzy matching
    model = load_sentence_transformer()

    # Combine TF-IDF and semantic UIDs
    df_target["Final_UID"] = df_target["Suggested_UID"].combine_first(df_target["Semantic_UID"])
    df_target["configured_final_UID"] = df_target["Final_UID"]
    df_target["Final_Question"] = df_target["Matched_Question"]
    df_target["Final_Match_Type"] = df_target.apply(assign_match_type, axis=1)
    df_target["Change_UID"] = df_target["Final_UID"].apply(
        lambda x: f"{x} - {df_reference[df_reference['uid'] == x]['heading_0'].iloc[0]}" if pd.notnull(x) and x in df_reference["uid"].values else None
    )

    # Override Final_UID for questions in questions_to_exclude (exact or fuzzy match)
    df_target["Final_UID"] = df_target.apply(
        lambda row: "9999" if is_excluded_question(row["heading_0"], questions_to_exclude, model) else row["Final_UID"],
        axis=1
    )
    df_target["configured_final_UID"] = df_target["Final_UID"]
    df_target["Change_UID"] = df_target["Final_UID"].apply(
        lambda x: f"9999 - Excluded Question" if x == "9999" else (
            f"{x} - {df_reference[df_reference['uid'] == x]['heading_0'].iloc[0]}" if pd.notnull(x) and x in df_reference["uid"].values else None
        )
    )

    # Assign parent question UID for choices
    df_target["Final_UID"] = df_target.apply(
        lambda row: df_target[df_target["heading_0"] == row["parent_question"]]["Final_UID"].iloc[0]
        if row["is_choice"] and pd.notnull(row["parent_question"]) else row["Final_UID"],
        axis=1
    )
    df_target["configured_final_UID"] = df_target["Final_UID"]
    df_target["Change_UID"] = df_target["Final_UID"].apply(
        lambda x: f"9999 - Excluded Question" if x == "9999" else (
            f"{x} - {df_reference[df_reference['uid'] == x]['heading_0'].iloc[0]}" if pd.notnull(x) and x in df_reference["uid"].values else None
        )
    )

    # Add survey_id_title if columns exist
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
if "survey_template" not in st.session_state:
    st.session_state.survey_template = None
if "all_questions" not in st.session_state:
    st.session_state.all_questions = None
if "dedup_questions" not in st.session_state:
    st.session_state.dedup_questions = []
if "dedup_choices" not in st.session_state:
    st.session_state.dedup_choices = []
if "mode" not in st.session_state:
    st.session_state.mode = None

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
                st.stop()
            
            # Fetch all questions for deduplicated lists
            if st.session_state.all_questions is None:
                all_questions = []
                for survey in surveys:
                    survey_json = get_survey_details(survey["id"], token)
                    questions = extract_questions(survey_json)
                    all_questions.extend(questions)
                
                st.session_state.all_questions = pd.DataFrame(all_questions)
                st.session_state.dedup_questions = sorted(st.session_state.all_questions[
                    st.session_state.all_questions["is_choice"] == False
                ]["heading_0"].unique().tolist())
                st.session_state.dedup_choices = sorted(st.session_state.all_questions[
                    st.session_state.all_questions["is_choice"] == True
                ]["heading_0"].apply(lambda x: x.split(" - ", 1)[1] if " - " in x else x).unique().tolist())
        
        # Prepare choices for both filters
        choices = {s["title"]: s["id"] for s in surveys}
        survey_id_title_choices = [f"{s['id']} - {s['title']}" for s in surveys]
        
        # Create two columns for filters
        col1, col2 = st.columns(2)
        
        with col1:
            selected_survey = st.selectbox("Choose Survey", [""] + list(choices.keys()), index=0)
        with col2:
            selected_survey_ids = st.multiselect(
                "SurveyID/Title",
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
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "Survey Questions and Choices",
            "UID Matching and Configuration",
            "Configured Survey",
            "Create Survey"
        ])

        # Tab 1: Survey Questions and Choices
        with tab1:
            if all_selected_survey_ids:
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
                            st.session_state.df_final["survey_id_title"] = st.session_state.df_final.apply(
                                lambda x: f"{x['survey_id']} - {x['survey_title']}" if pd.notnull(x['survey_id']) and pd.notnull(x['survey_title']) else "",
                                axis=1
                            )
                            st.session_state.uid_changes = {}
                        else:
                            st.error(f"UID matching failed: {e}")
                            raise
                    
                    st.write("Edit mandatory status for questions/choices. Go to next tab for UID matching.")
                    show_main_only = st.checkbox("Show only main questions", value=False)
                    display_df = st.session_state.df_target[st.session_state.df_target["is_choice"] == False] if show_main_only else st.session_state.df_target
                    
                    display_df = display_df.copy()
                    display_df["survey_id_title"] = display_df.apply(
                        lambda x: f"{x['survey_id']} - {x['survey_title']}" if pd.notnull(x['survey_id']) and pd.notnull(x['survey_title']) else "",
                        axis=1
                    )
                    
                    edited_df = st.data_editor(
                        display_df,
                        column_config={
                            "survey_id_title": st.column_config.TextColumn("Survey ID/Title"),
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
                            "question_uid": st.column_config.TextColumn("Question UID"),
                            "survey_id": st.column_config.TextColumn("Survey ID"),
                            "survey_title": st.column_config.TextColumn("Survey Title")
                        },
                        disabled=["survey_id_title", "heading_0", "position", "is_choice", "parent_question", "schema_type", "question_uid", "survey_id", "survey_title", "mandatory_editable"] + (["mandatory"] if not display_df["mandatory_editable"].any() else []),
                        hide_index=True
                    )
                    
                    if not edited_df.empty:
                        editable_rows = st.session_state.df_target[st.session_state.df_target["mandatory_editable"]]
                        if not editable_rows.empty:
                            st.session_state.df_target.loc[st.session_state.df_target["mandatory_editable"], "mandatory"] = edited_df[edited_df["mandatory_editable"]]["mandatory"]
            else:
                st.write("Select a survey to view questions.")

        # Tab 2: UID Matching and Configuration
        with tab2:
            if st.session_state.df_final is not None:
                matched_percentage = calculate_matched_percentage(st.session_state.df_final)
                st.metric("Matched Questions", f"{matched_percentage}%")
                
                if matched_percentage == 0.0:
                    eligible_count = len(st.session_state.df_final[
                        (st.session_state.df_final["is_choice"] == False) &
                        (~st.session_state.df_final["heading_0"].str.contains("Our Privacy Policy", case=False, na=False)) &
                        (~st.session_state.df_final["heading_0"].str.contains(
                            r"<div.*text-align:\s*center.*<span.*font-size:\s*12pt.*<em>If you have any questions, please contact your AMI Learner Success Manager.*</em>.*</span>.*</div>",
                            case=False, na=False, regex=True))
                    ])
                    if eligible_count == 0:
                        st.warning("No eligible questions found (all may be excluded due to 'Our Privacy Policy' or specific HTML format).")
                
                st.subheader("UID Matching for Questions/Choices")
                if st.session_state.df_final["Final_UID"].isna().all():
                    st.info("UID matching disabled due to Snowflake issues. Assign UIDs manually or fix the connection.")
                
                show_main_only = st.checkbox("Show only main questions", value=False, key="tab2_main_only")
                match_filter = st.selectbox(
                    "Filter by Match Status",
                    ["All", "Matched", "Not Matched"],
                    index=0
                )
                
                st.subheader("Search Questions/Choices")
                question_options = [""] + st.session_state.df_target[st.session_state.df_target["is_choice"] == False]["heading_0"].tolist()
                search_query = st.text_input("Type to filter questions/choices", "")
                filtered_questions = [q for q in question_options if not search_query or search_query.lower() in q.lower()]
                selected_question = st.selectbox("Select a question/choice", filtered_questions, index=0)
                
                result_df = st.session_state.df_final.copy()
                if "survey_id" in result_df.columns and "survey_title" in result_df.columns:
                    result_df["survey_id_title"] = result_df.apply(
                        lambda x: f"{x['survey_id']} - {x['survey_title']}" if pd.notnull(x['survey_id']) and pd.notnull(x['survey_title']) else "",
                        axis=1
                    )
                else:
                    result_df["survey_id_title"] = ""
                
                if selected_question:
                    result_df = result_df[result_df["heading_0"] == selected_question]
                if match_filter == "Matched":
                    result_df = result_df[result_df["Final_UID"].notna()]
                elif match_filter == "Not Matched":
                    result_df = result_df[result_df["Final_UID"].isna()]
                result_df = result_df[result_df["is_choice"] == False] if show_main_only else result_df
                
                uid_options = [None]
                if st.session_state.df_reference is not None:
                    uid_options += [f"{row['uid']} - {row['heading_0']}" for _, row in st.session_state.df_reference.iterrows()]
                else:
                    st.warning("UID options unavailable due to Snowflake issues. Fix connection to load UIDs.")
                
                result_columns = ["survey_id_title", "heading_0", "position", "is_choice", "Final_UID", "question_uid", "schema_type", "Change_UID"]
                result_columns = [col for col in result_columns if col in result_df.columns]
                display_df = result_df[result_columns].copy()
                display_df = display_df.rename(columns={"heading_0": "Question/Choice", "Final_UID": "final_UID"})
                
                edited_df = st.data_editor(
                    display_df,
                    column_config={
                        "survey_id_title": st.column_config.TextColumn("Survey ID/Title"),
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
                    disabled=["survey_id_title", "Question/Choice", "position", "is_choice", "final_UID", "question_uid", "schema_type"],
                    hide_index=True
                )
                
                for idx, row in edited_df.iterrows():
                    current_change_uid = st.session_state.df_final.at[idx, "Change_UID"] if "Change_UID" in st.session_state.df_final.columns else None
                    if pd.notnull(row["Change_UID"]) and row["Change_UID"] != current_change_uid:
                        new_uid = row["Change_UID"].split(" - ")[0] if row["Change_UID"] and " - " in row["Change_UID"] else None
                        st.session_state.df_final.at[idx, "Final_UID"] = new_uid
                        st.session_state.df_final.at[idx, "configured_final_UID"] = new_uid
                        st.session_state.df_final.at[idx, "Change_UID"] = row["Change_UID"]
                        st.session_state.uid_changes[idx] = new_uid
                
                st.subheader("Create New Questions")
                st.write("Submit new questions via Google Form. Fields: Question Text, Type, Choices, Program, Mandatory.")
                st.markdown("[Submit New Question](https://docs.google.com/forms/d/1LoY_La59UJ4ZsuxckM8Wl52kVeLI7a1t1MF8zIQxGUs)")
                
                st.subheader("Create New UID")
                st.write("Submit new UIDs via Google Form. Fields: Question Text, Proposed UID, Program, Type, Mandatory.")
                st.markdown("[Submit New UID](https://docs.google.com/forms/d/1lkhfm1-t5-zwLxfbVEUiHewveLpGXv5yEVRlQx5XjxA)")
                
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
                
                if not st.session_state.custom_questions.empty:
                    st.subheader("Customized Questions/Choices")
                    st.dataframe(st.session_state.custom_questions)

        # Tab 3: Configured Survey
        with tab3:
            if st.session_state.df_final is not None:
                matched_percentage = calculate_matched_percentage(st.session_state.df_final)
                st.metric("Matched Questions", f"{matched_percentage}%")
                
                st.subheader("Configured Survey")
                config_columns = [
                    "heading_0", "position", "is_choice", "parent_question", 
                    "schema_type", "mandatory", "mandatory_editable", "configured_final_UID"
                ]
                if "survey_id" in st.session_state.df_final.columns and "survey_title" in st.session_state.df_final.columns:
                    config_columns.insert(0, "survey_id_title")
                    st.session_state.df_final["survey_id_title"] = st.session_state.df_final.apply(
                        lambda x: f"{x['survey_id']} - {x['survey_title']}" if pd.notnull(x['survey_id']) and pd.notnull(x['survey_title']) else "",
                        axis=1
                    )
                
                config_df = st.session_state.df_final[config_columns].copy()
                config_df = config_df[config_df["is_choice"] == False] if show_main_only else config_df
                config_df = config_df.rename(columns={"heading_0": "Question/Choice"})
                st.dataframe(config_df)
                
                st.subheader("Export to Snowflake")
                export_columns = [
                    "survey_id", "survey_title", "heading_0", "configured_final_UID", "position",
                    "is_choice", "parent_question", "question_uid", "schema_type", "mandatory",
                    "mandatory_editable"
                ]
                export_columns = [col for col in export_columns if col in st.session_state.df_final.columns]
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
                        else:
                            st.error(f"Snowflake upload failed: {e}")
            else:
                st.write("Select a survey to view the configured survey.")

        # Tab 4: Create Survey
        with tab4:
            st.subheader("Create New Survey")
            
            # Buttons for creation modes
            col1, col2 = st.columns(2)
            with col1:
                from_pre_existing = st.button("📋 From Pre-existing Survey")
            with col2:
                create_new_template = st.button("✨ Create New Template")
            
            # From Pre-existing Survey
            if from_pre_existing or st.session_state.mode == "pre_existing":
                st.session_state.mode = "pre_existing"
                st.write("### Create Survey from Pre-existing Surveys")
                
                with st.form("pre_existing_survey_form"):
                    survey_options = [f"{s['id']} - {s['title']}" for s in surveys]
                    selected_surveys = st.multiselect(
                        "Select Surveys",
                        survey_options,
                        help="Choose one or more surveys to include questions from"
                    )
                    
                    selected_survey_ids = [s.split(" - ")[0] for s in selected_surveys]
                    selected_questions = st.session_state.all_questions[
                        st.session_state.all_questions["survey_id"].isin(selected_survey_ids)
                    ].copy() if selected_survey_ids else pd.DataFrame()
                    
                    if not selected_questions.empty:
                        selected_questions["survey_id_title"] = selected_questions.apply(
                            lambda x: f"{x['survey_id']} - {x['survey_title']}", axis=1
                        )
                        edit_df = selected_questions[["survey_id_title", "heading_0", "schema_type", "is_choice"]].copy()
                        edit_df["required"] = False
                        
                        st.write("#### Selected Questions")
                        edited_df = st.data_editor(
                            edit_df,
                            column_config={
                                "survey_id_title": st.column_config.TextColumn("Survey ID/Title"),
                                "heading_0": st.column_config.TextColumn("Question/Choice"),
                                "schema_type": st.column_config.SelectboxColumn(
                                    "Question Type",
                                    options=["Single Choice", "Multiple Choice", "Open-Ended", "Matrix"],
                                    default="Open-Ended"
                                ),
                                "is_choice": st.column_config.CheckboxColumn("Is Choice", disabled=True),
                                "required": st.column_config.CheckboxColumn("Required")
                            },
                            hide_index=True,
                            num_rows="dynamic"
                        )
                        
                        if st.button("➕ Add Question"):
                            new_row = pd.DataFrame({
                                "survey_id_title": [""],
                                "heading_0": [""],
                                "schema_type": ["Open-Ended"],
                                "is_choice": [False],
                                "required": [False]
                            })
                            edited_df = pd.concat([edited_df, new_row], ignore_index=True)
                            st.session_state.edited_df = edited_df
                        
                        if "edited_df" in st.session_state:
                            edited_df = st.session_state.edited_df
                        
                        st.write("#### Add Question from Existing")
                        question_search = st.text_input("Search Questions", key="question_search")
                        filtered_questions = [q for q in st.session_state.dedup_questions if not question_search or question_search.lower() in q.lower()]
                        new_question = st.selectbox(
                            "Select Question",
                            [""] + filtered_questions,
                            key="new_question"
                        )
                        if new_question:
                            new_row = pd.DataFrame({
                                "survey_id_title": [""],
                                "heading_0": [new_question],
                                "schema_type": ["Open-Ended"],
                                "is_choice": [False],
                                "required": [False]
                            })
                            edited_df = pd.concat([edited_df, new_row], ignore_index=True)
                            st.session_state.edited_df = edited_df
                            st.experimental_rerun()
                        
                        for idx, row in edited_df.iterrows():
                            if row["schema_type"] in ["Single Choice", "Multiple Choice"] and not row["is_choice"]:
                                st.write(f"##### Choices for: {row['heading_0']}")
                                choice_search = st.text_input("Search Choices", key=f"choice_search_{idx}")
                                filtered_choices = [c for c in st.session_state.dedup_choices if not choice_search or choice_search.lower() in c.lower()]
                                new_choice = st.selectbox(
                                    "Add Choice",
                                    [""] + filtered_choices,
                                    key=f"new_choice_{idx}"
                                )
                                if new_choice:
                                    new_row = pd.DataFrame({
                                        "survey_id_title": [row["survey_id_title"]],
                                        "heading_0": [f"{row['heading_0']} - {new_choice}"],
                                        "schema_type": [row["schema_type"]],
                                        "is_choice": [True],
                                        "required": [False]
                                    })
                                    edited_df = pd.concat([edited_df, new_row], ignore_index=True)
                                    st.session_state.edited_df = edited_df
                                    st.experimental_rerun()
                    
                    survey_title = st.text_input("Survey Title", value="New Survey from Pre-existing")
                    survey_language = st.selectbox("Language", ["en", "es", "fr", "de"], index=0)
                    
                    submit = st.form_submit_button("Create Survey")
                    if submit:
                        if not survey_title or edited_df.empty:
                            st.error("Survey title and at least one question are required.")
                        else:
                            questions = []
                            position = 1
                            for idx, row in edited_df.iterrows():
                                question_template = {
                                    "heading": row["heading_0"].split(" - ")[0] if row["is_choice"] else row["heading_0"],
                                    "position": position,
                                    "is_required": row["required"]
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
                                    parent_idx = edited_df[edited_df["heading_0"] == parent_question].index[0]
                                    if "choices" not in questions[parent_idx]:
                                        questions[parent_idx]["choices"] = []
                                    questions[parent_idx]["choices"].append({
                                        "text": row["heading_0"].split(" - ")[1],
                                        "position": len(questions[parent_idx]["choices"]) + 1
                                    })
                                else:
                                    questions.append(question_template)
                                    position += 1
                            
                            survey_template = {
                                "title": survey_title,
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
                                },
                                "theme": {
                                    "font": "Arial",
                                    "background_color": "#FFFFFF",
                                    "question_color": "#000000",
                                    "answer_color": "#000000"
                                }
                            }
                            
                            try:
                                with st.spinner("Creating survey in SurveyMonkey..."):
                                    survey_id = create_survey(token, survey_template)
                                    for page_template in survey_template["pages"]:
                                        page_id = create_page(token, survey_id, page_template)
                                        for question_template in page_template["questions"]:
                                            create_question(token, survey_id, page_id, question_template)
                                    st.success(f"Survey created successfully! Survey ID: {survey_id}")
                                    st.session_state.survey_template = survey_template
                            except Exception as e:
                                st.error(f"Failed to create survey: {e}")
                
                if st.session_state.survey_template:
                    st.subheader("Preview Survey Template")
                    st.json(st.session_state.survey_template)
            
            # Create New Template
            if create_new_template or st.session_state.mode == "new_template":
                st.session_state.mode = "new_template"
                st.write("### Create New Survey Template")
                
                with st.form("new_template_form"):
                    survey_title = st.text_input("Survey Title", value="New Survey")
                    survey_language = st.selectbox("Language", ["en", "es", "fr", "de"], index=0)
                    
                    num_pages = st.number_input("Number of Pages", min_value=1, max_value=10, value=1)
                    pages = []
                    for i in range(num_pages):
                        st.write(f"#### Page {i+1}")
                        page_title = st.text_input(f"Page {i+1} Title", value=f"Page {i+1}", key=f"page_title_{i}")
                        page_description = st.text_area(f"Page {i+1} Description", value="", key=f"page_desc_{i}")
                        
                        num_questions = st.number_input(
                            f"Number of Questions for Page {i+1}",
                            min_value=1,
                            max_value=10,
                            value=1,
                            key=f"num_questions_{i}"
                        )
                        questions = []
                        for j in range(num_questions):
                            st.write(f"##### Question {j+1}")
                            question_search = st.text_input("Search Questions", key=f"q_search_{i}_{j}")
                            filtered_questions = [q for q in st.session_state.dedup_questions if not question_search or question_search.lower() in q.lower()]
                            question_text = st.selectbox(
                                "Question Text",
                                [""] + filtered_questions,
                                key=f"q_text_{i}_{j}",
                                help="Select or type a question"
                            )
                            question_type = st.selectbox(
                                "Question Type",
                                ["Single Choice", "Multiple Choice", "Open-Ended", "Matrix"],
                                key=f"q_type_{i}_{j}"
                            )
                            is_required = st.checkbox("Required", key=f"q_required_{i}_{j}")
                            
                            question_template = {
                                "heading": question_text,
                                "position": j + 1,
                                "is_required": is_required
                            }
                            
                            if question_type == "Single Choice":
                                question_template["family"] = "single_choice"
                                question_template["subtype"] = "vertical"
                                num_choices = st.number_input(
                                    "Number of Choices",
                                    min_value=1,
                                    max_value=10,
                                    value=2,
                                    key=f"num_choices_{i}_{j}"
                                )
                                choices = []
                                for k in range(num_choices):
                                    choice_search = st.text_input("Search Choices", key=f"choice_search_{i}_{j}_{k}")
                                    filtered_choices = [c for c in st.session_state.dedup_choices if not choice_search or choice_search.lower() in c.lower()]
                                    choice_text = st.selectbox(
                                        f"Choice {k+1}",
                                        [""] + filtered_choices,
                                        key=f"choice_{i}_{j}_{k}"
                                    )
                                    if choice_text:
                                        choices.append({"text": choice_text, "position": k + 1})
                                if choices:
                                    question_template["choices"] = choices
                            elif question_type == "Multiple Choice":
                                question_template["family"] = "multiple_choice"
                                question_template["subtype"] = "vertical"
                                num_choices = st.number_input(
                                    "Number of Choices",
                                    min_value=1,
                                    max_value=10,
                                    value=2,
                                    key=f"num_choices_{i}_{j}"
                                )
                                choices = []
                                for k in range(num_choices):
                                    choice_search = st.text_input("Search Choices", key=f"choice_search_{i}_{j}_{k}")
                                    filtered_choices = [c for c in st.session_state.dedup_choices if not choice_search or choice_search.lower() in c.lower()]
                                    choice_text = st.selectbox(
                                        f"Choice {k+1}",
                                        [""] + filtered_choices,
                                        key=f"choice_{i}_{j}_{k}"
                                    )
                                    if choice_text:
                                        choices.append({"text": choice_text, "position": k + 1})
                                if choices:
                                    question_template["choices"] = choices
                            elif question_type == "Open-Ended":
                                question_template["family"] = "open_ended"
                                question_template["subtype"] = "essay"
                            elif question_type == "Matrix":
                                question_template["family"] = "matrix"
                                question_template["subtype"] = "rating"
                                num_rows = st.number_input(
                                    "Number of Rows",
                                    min_value=1,
                                    max_value=10,
                                    value=2,
                                    key=f"num_rows_{i}_{j}"
                                )
                                rows = []
                                for k in range(num_rows):
                                    row_search = st.text_input("Search Rows", key=f"row_search_{i}_{j}_{k}")
                                    filtered_rows = [q for q in st.session_state.dedup_questions if not row_search or row_search.lower() in q.lower()]
                                    row_text = st.selectbox(
                                        f"Row {k+1}",
                                        [""] + filtered_rows,
                                        key=f"row_{i}_{j}_{k}"
                                    )
                                    if row_text:
                                        rows.append({"text": row_text, "position": k + 1})
                                num_choices = st.number_input(
                                    "Number of Rating Choices",
                                    min_value=1,
                                    max_value=10,
                                    value=5,
                                    key=f"num_choices_{i}_{j}"
                                )
                                choices = []
                                for k in range(num_choices):
                                    choice_search = st.text_input("Search Choices", key=f"rating_search_{i}_{j}_{k}")
                                    filtered_choices = [c for c in st.session_state.dedup_choices if not choice_search or choice_search.lower() in c.lower()]
                                    choice_text = st.selectbox(
                                        f"Rating Choice {k+1}",
                                        [""] + filtered_choices,
                                        key=f"rating_{i}_{j}_{k}"
                                    )
                                    if choice_text:
                                        choices.append({"text": choice_text, "position": k + 1})
                                if rows and choices:
                                    question_template["rows"] = rows
                                    question_template["choices"] = choices
                            
                            if question_text:
                                questions.append(question_template)
                        
                        if questions:
                            pages.append({
                                "title": page_title,
                                "description": page_description,
                                "questions": questions
                            })
                    
                    st.write("#### Survey Settings")
                    show_progress_bar = st.checkbox("Show Progress Bar", value=False)
                    hide_asterisks = st.checkbox("Hide Asterisks for Required Questions", value=True)
                    one_question_at_a_time = st.checkbox("Show One Question at a Time", value=False)
                    
                    survey_template = {
                        "title": survey_title,
                        "language": survey_language,
                        "pages": pages,
                        "settings": {
                            "progress_bar": show_progress_bar,
                            "hide_asterisks": hide_asterisks,
                            "one_question_at_a_time": one_question_at_a_time
                        },
                        "theme": {
                            "font": "Arial",
                            "background_color": "#FFFFFF",
                            "question_color": "#000000",
                            "answer_color": "#000000"
                        }
                    }
                    
                    submit = st.form_submit_button("Create Survey")
                    if submit:
                        if not survey_title or not pages:
                            st.error("Survey title and at least one page with questions are required.")
                        else:
                            st.session_state.survey_template = survey_template
                            try:
                                with st.spinner("Creating survey in SurveyMonkey..."):
                                    survey_id = create_survey(token, survey_template)
                                    for page_template in survey_template["pages"]:
                                        page_id = create_page(token, survey_id, page_template)
                                        for question_template in page_template["questions"]:
                                            create_question(token, survey_id, page_id, question_template)
                                    st.success(f"Survey created successfully! Survey ID: {survey_id}")
                            except Exception as e:
                                st.error(f"Failed to create survey: {e}")
                    
                    if st.session_state.survey_template:
                        st.subheader("Preview Survey Template")
                        st.json(st.session_state.survey_template)

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
