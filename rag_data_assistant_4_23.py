import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate

import matplotlib.pyplot as plt

# Load API key
load_dotenv("/Users/swaraj/Downloads/QAC387 Project/ai-data-analysis-assistant/env")
api_key = os.getenv("OPENAI_API_KEY")

# Streamlit setup
st.set_page_config(page_title="VO₂max Data Assistant", layout="wide")
st.title("🧠 VO₂max Data Assistant with Expert Context")

# Upload file
uploaded_file = st.file_uploader("Upload your VO₂ dataset (CSV)", type="csv")
data_load_error = None
df = None

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        if df.empty or df.shape[1] < 2:
            data_load_error = "Uploaded file is empty or missing expected columns."
    except Exception as e:
        data_load_error = f"Error loading file: {e}"

    if data_load_error:
        st.error(data_load_error)
        st.stop()

# Summary + plotting
if df is not None and data_load_error is None:
    st.write("📊 Summary Statistics (mean, std, min, max):")
    summary = df.describe().loc[["mean", "std", "min", "max"]].T
    summary.reset_index(inplace=True)
    summary.rename(columns={"index": "Metric"}, inplace=True)
    st.dataframe(summary)

    key_cols = ['Time[s]', 'VO2[mL/kg/min]', 'HR[bpm]']
    plot_cols = [col for col in key_cols if col in df.columns]

    if len(plot_cols) >= 2:
        for col in plot_cols[1:]:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(df[plot_cols[0]], df[col], label=col)
            ax.set_xlabel(plot_cols[0])
            ax.set_ylabel(col)
            ax.set_title(f"{col} over {plot_cols[0]}")
            ax.legend()
            st.pyplot(fig)
    else:
        st.info("Not enough columns found for plotting.")

# Question input
question = st.text_input("Ask something about your VO₂ data (e.g., is it processed correctly?)")

# Analysis section
if question and df is not None and data_load_error is None:
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vectorstore = FAISS.load_local("vectorstore/faiss_index", embeddings, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever(search_type="similarity", k=3)
        context_docs = retriever.invoke(question)
        context_text = "\n\n".join([doc.page_content for doc in context_docs])

        # Subsample key columns
        data_cols = [col for col in key_cols if col in df.columns]
        if not data_cols:
            st.warning("Key columns for analysis not found.")
            st.stop()

        sampled_df = df[data_cols].iloc[::10]
        data_sample = sampled_df.to_csv(index=False)
        approx_tokens = len(data_sample) // 4
        st.caption(f"🧮 Estimated tokens passed to LLM: {approx_tokens}")

        prompt_template = PromptTemplate(
            input_variables=["data_sample", "question", "context_text"],
            template="""
You are a VO₂max data analysis assistant.
Use the expert VO₂max reporting recommendations below when analyzing the data.

Expert guidance:
{context_text}

Here is a sample of the uploaded dataset:
{data_sample}

User question: "{question}"

Using this, perform a deep statistical and physiological interpretation.

You must:
- Highlight meaningful changes in VO₂ or HR over time (e.g., rising/falling trends).
- Comment on whether the values are physiologically reasonable or unusual.
- Provide a short summary of what the data suggests about the subject's endurance performance.
- Recommend at least one follow-up test, transformation, or metric that could improve analysis.
"""
        )

        final_prompt = prompt_template.format(
            data_sample=data_sample,
            question=question,
            context_text=context_text
        )

        llm = ChatOpenAI(model="gpt-4", temperature=0, api_key=api_key)
        response = llm.invoke(final_prompt)
        st.markdown("### 💡 Assistant's Insight:")
        st.write(response.content)
    except Exception as e:
        st.error(f"Analysis failed: {e}")

# Feedback section
st.markdown("---")
st.write("Was this analysis helpful?")
rating = st.radio("Rate this app (1 = poor, 5 = excellent)", [1, 2, 3, 4, 5], horizontal=True)
feedback = st.text_area("Additional feedback or suggestions:")
if st.button("Submit Feedback"):
    st.success("✅ Feedback received. Thank you!")



 
