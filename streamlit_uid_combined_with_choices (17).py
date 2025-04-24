import streamlit as st
import pandas as pd
import requests
import logging
import json
from uuid import uuid4

# Setup
st.set_page_config(page_title="Survey Creator", layout="wide")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SurveyMonkey API Functions
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
                            "survey_id": survey_json.get("id", ""),
                            "survey_title": survey_json.get("title", "")
                        })
    return questions

# App UI
st.title("🧠 Survey Creator: SurveyMonkey")

# Secrets Validation
if "surveymonkey" not in st.secrets:
    st.error("Missing SurveyMonkey secrets configuration.")
    st.stop()

# Initialize session state
if "all_questions" not in st.session_state:
    st.session_state.all_questions = None
if "dedup_questions" not in st.session_state:
    st.session_state.dedup_questions = []
if "dedup_choices" not in st.session_state:
    st.session_state.dedup_choices = []
if "survey_template" not in st.session_state:
    st.session_state.survey_template = None

# Fetch all surveys and questions
try:
    token = st.secrets.get("surveymonkey", {}).get("token", None)
    if not token:
        st.error("SurveyMonkey token is missing in secrets configuration.")
        st.stop()
    with st.spinner("Fetching all surveys and questions..."):
        surveys = get_surveys(token)
        if not surveys:
            st.error("No surveys found or invalid API response.")
            st.stop()
        
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

except Exception as e:
    logger.error(f"Failed to fetch surveys/questions: {e}")
    st.error(f"Error: {e}")
    st.stop()

# Buttons for creation modes
col1, col2 = st.columns(2)
with col1:
    from_pre_existing = st.button("📋 From Pre-existing Survey")
with col2:
    create_new_template = st.button("✨ Create New Template")

# From Pre-existing Survey
if from_pre_existing or st.session_state.get("mode") == "pre_existing":
    st.session_state.mode = "pre_existing"
    st.subheader("Create Survey from Pre-existing Surveys")
    
    with st.form("pre_existing_survey_form"):
        # Survey filter
        survey_options = [f"{s['id']} - {s['title']}" for s in surveys]
        selected_surveys = st.multiselect(
            "Select Surveys",
            survey_options,
            help="Choose one or more surveys to include questions from"
        )
        
        # Fetch questions for selected surveys
        selected_survey_ids = [s.split(" - ")[0] for s in selected_surveys]
        selected_questions = st.session_state.all_questions[
            st.session_state.all_questions["survey_id"].isin(selected_survey_ids)
        ].copy() if selected_survey_ids else pd.DataFrame()
        
        if not selected_questions.empty:
            selected_questions["survey_id_title"] = selected_questions.apply(
                lambda x: f"{x['survey_id']} - {x['survey_title']}", axis=1
            )
            # Prepare editable DataFrame
            edit_df = selected_questions[["survey_id_title", "heading_0", "schema_type", "is_choice"]].copy()
            edit_df["required"] = False
            
            st.write("### Selected Questions")
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
            
            # Add new question
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
            
            # Update edited_df from session state if modified
            if "edited_df" in st.session_state:
                edited_df = st.session_state.edited_df
            
            # Dropdown for adding questions
            st.write("### Add Question from Existing")
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
            
            # Add choices for choice-based questions
            for idx, row in edited_df.iterrows():
                if row["schema_type"] in ["Single Choice", "Multiple Choice"] and not row["is_choice"]:
                    st.write(f"#### Choices for: {row['heading_0']}")
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
        
        # Survey details
        survey_title = st.text_input("Survey Title", value="New Survey from Pre-existing")
        survey_language = st.selectbox("Language", ["en", "es", "fr", "de"], index=0)
        
        # Submit
        submit = st.form_submit_button("Create Survey")
        if submit:
            if not survey_title or edited_df.empty:
                st.error("Survey title and at least one question are required.")
            else:
                # Build survey template
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
if create_new_template or st.session_state.get("mode") == "new_template":
    st.session_state.mode = "new_template"
    st.subheader("Create New Survey Template")
    
    with st.form("new_template_form"):
        survey_title = st.text_input("Survey Title", value="New Survey")
        survey_language = st.selectbox("Language", ["en", "es", "fr", "de"], index=0)
        
        # Pages
        num_pages = st.number_input("Number of Pages", min_value=1, max_value=10, value=1)
        pages = []
        for i in range(num_pages):
            st.write(f"### Page {i+1}")
            page_title = st.text_input(f"Page {i+1} Title", value=f"Page {i+1}", key=f"page_title_{i}")
            page_description = st.text_area(f"Page {i+1} Description", value="", key=f"page_desc_{i}")
            
            # Questions per page
            num_questions = st.number_input(
                f"Number of Questions for Page {i+1}",
                min_value=1,
                max_value=10,
                value=1,
                key=f"num_questions_{i}"
            )
            questions = []
            for j in range(num_questions):
                st.write(f"#### Question {j+1}")
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
        
        # Survey settings
        st.write("### Survey Settings")
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
