#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# app.py

import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI 
from langchain_experimental.agents import create_pandas_dataframe_agent

# === Load OpenAI API Key ===
load_dotenv("/Users/swaraj/Downloads/QAC387 Project/ai-data-analysis-assistant/env")
api_key = os.getenv("OPENAI_API_KEY")

# === Streamlit UI ===
st.set_page_config(page_title="VO2 Optimization Assistant", layout="centered")
st.title("🏃 VO2 Optimization Assistant")

# === Tabs ===
tab1, tab2 = st.tabs(["📋 Fitness Card Generator", "💬 Ask My VO2 Data"])

# === Load survey data once ===
@st.cache_data
def load_reference_data():
    base_path = "data"
    demographics = pd.read_csv(os.path.join(base_path, "Demographics.csv"))
    pss = pd.read_csv(os.path.join(base_path, "Perceived Stress Scale (PSS) (Responses) - Form Responses 1.csv"))
    ifis = pd.read_csv(os.path.join(base_path, "International Fitness Scale (IFIS) (Responses) - Form Responses 1.csv"))
    diet = pd.read_csv(os.path.join(base_path, "How healthy is your diet_ Questionnaire (Responses) - Form Responses 1.csv"))
    return demographics, pss, ifis, diet

demographics, pss, ifis, diet = load_reference_data()

# === TAB 1: Fitness Card ===
with tab1:
    st.subheader("📥 Upload VO2 Report and Enter Participant ID")
    vo2_file = st.file_uploader("Upload VO2 .csv", type=["csv"], key="upload1")
    participant_id = st.text_input("Participant ID (e.g., P01)", key="pid1")

    if vo2_file and participant_id:
        try:
            vo2_df = pd.read_csv(vo2_file)
            avg_vo2 = vo2_df["VO2[mL/kg/min]"].mean()
            max_vo2 = vo2_df["VO2[mL/kg/min]"].max()
            avg_hr = vo2_df["HR[bpm]"].mean()

            demo_row = demographics[demographics["Participant ID"] == participant_id].iloc[0]
            age = demo_row.get("Age", "N/A")
            sex = demo_row.get("Sex", "N/A")

            pss_row = pss[pss["Participant ID"] == participant_id].iloc[0, 1:]
            pss_vals = pss_row.str.strip().str.lower().value_counts()
            pss_total = (
                pss_vals.get("never", 0) * 0 +
                pss_vals.get("almost never", 1) * 1 +
                pss_vals.get("sometimes", 2) * 2 +
                pss_vals.get("fairly often", 3) * 3 +
                pss_vals.get("very often", 4) * 4
            )

            ifis_row = ifis[ifis["Participant ID"] == participant_id].iloc[0]
            perceived_fit = ifis_row[1].strip().lower()
            perceived_score = {"poor": 1, "average": 2, "good": 3, "very good": 4}.get(perceived_fit, 2)

            diet_row = diet[diet["Participant ID"] == participant_id].iloc[0]
            diet_score = diet_row[1:].str.strip().str.lower().value_counts().get("yes", 0) / len(diet_row[1:]) * 100

            weights = {"vo2": 0.4, "perceived_fit": 0.15, "stress": 0.15, "diet": 0.3}
            normalized_vo2 = avg_vo2 / 40 * 100
            normalized_stress = (40 - pss_total) / 40 * 100
            normalized_perceived = perceived_score / 4 * 100
            normalized_diet = diet_score

            optimization_score = (
                normalized_vo2 * weights["vo2"] +
                normalized_perceived * weights["perceived_fit"] +
                normalized_stress * weights["stress"] +
                normalized_diet * weights["diet"]
            )

            st.success(f"✅ Data for {participant_id} loaded.")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Max VO2", f"{max_vo2:.1f} mL/kg/min")
                st.metric("Avg HR", f"{avg_hr:.1f} bpm")
                st.metric("Diet Score", f"{diet_score:.1f}/100")
            with col2:
                st.metric("Avg VO2", f"{avg_vo2:.1f} mL/kg/min")
                st.metric("PSS Score", f"{pss_total}/40")
                st.metric("Optimization Score", f"{optimization_score:.1f}/100")

            st.markdown("### 🧠 Summary")
            st.info(f"""
{participant_id} reports average VO2 = {avg_vo2:.1f}, perceived fitness = '{perceived_fit}'.
Stress = {pss_total}/40, diet score = {diet_score:.1f}/100.
**Optimization Score** = {optimization_score:.1f}/100.
""")
        except Exception as e:
            st.error(f"❌ Could not process participant data: {e}")
    else:
        st.info("⬆️ Upload your VO2 .csv and enter a valid participant ID.")

# === TAB 2: LangChain Agent ===
with tab2:
    st.subheader("💬 Ask Questions About Your VO2 Report")

    if not api_key:
        st.error("🔐 OpenAI API key not found. Please check your `.env` file.")
    else:
        csv_file = st.file_uploader("Upload VO2 .csv again to query", type=["csv"], key="upload2")
        if csv_file:
            try:
                df = pd.read_csv(csv_file)
                st.dataframe(df.head())

                llm = ChatOpenAI(openai_api_key=api_key, model="gpt-4o", temperature=0.2)
                agent = create_pandas_dataframe_agent(llm, df, verbose=False, allow_dangerous_code=True)

                user_question = st.text_input("What do you want to know about your VO2 test?")
                if user_question:
                    with st.spinner("Thinking..."):
                        try:
                            response = agent.run(user_question)
                            st.write("🧠 Response:")
                            st.write(response)
                        except Exception as e:
                            st.error(f"❌ Agent error: {e}")
            except Exception as e:
                st.error(f"❌ Failed to read uploaded file: {e}")
        else:
            st.info("⬆️ Upload your VO2 CSV to ask questions.")

