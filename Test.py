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
except ImportError as e:
    missing_package = str(e).split("No module named ")[-1].replace("'", "")
    raise ImportError(f"""
Missing required package: {missing_package}
Please install it using:
pip install pandas openpyxl rapidfuzz python-Levenshtein SQLAlchemy scikit-learn sentence-transformers streamlit requests
""")

# --- Streamlit Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "UID Matching", "SurveyMonkey Fetch", "Create New Survey"])

# --- Home Page ---
if page == "Home":
    st.title("Welcome to UID Matcher App")
    st.write("Use the sidebar to navigate between functionalities.")

# --- UID Matching Workflow ---
elif page == "UID Matching":
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

# --- SurveyMonkey Fetch Workflow ---
elif page == "SurveyMonkey Fetch":
    st.title("SurveyMonkey Integration")

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

# --- SurveyMonkey Fetch Workflow (Updated) ---
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
                        choices = ", ".join([choice.get("text", "") for choice in question.get("answers", {}).get("choices", [])])

                        questions_list.append({
                            "question_id": q_id,
                            "heading_0": q_text,
                            "family": q_family,
                            "subtype": q_subtype,
                            "required": q_required,
                            "position": q_position,
                            "choices": choices
                        })

                df_target = pd.DataFrame(questions_list)

            # --- Now Run UID Matching ---
            with st.spinner("Running UID Matching..."):
                df_reference = run_snowflake_reference_query()

                df_target = compute_tfidf_matches(df_reference, df_target)
                df_target = compute_semantic_matches(df_reference, df_target)

            st.success("UID Matching Completed on SurveyMonkey Fetched Data!")
            st.dataframe(df_target)

            st.download_button(
                "Download UID Matched Survey as CSV",
                df_target.to_csv(index=False).encode('utf-8'),
                "survey_uid_matched.csv",
                "text/csv"
            )

        except Exception as e:
            st.error(f"Failed: {e}")


### --- End of Full Combined and Final Script --- ###
