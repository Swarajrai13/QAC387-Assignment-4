import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from tabulate import tabulate

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

# --- Upload & Cache Data ---
if "df" not in st.session_state:
    st.session_state.df = None

uploaded_file = st.file_uploader("Upload your VO₂ dataset (CSV)", type="csv")
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        if df.empty or df.shape[1] < 2:
            st.error("Uploaded file is empty or missing expected columns.")
            st.stop()
        st.session_state.df = df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

df = st.session_state.df
if df is None:
    st.info("Please upload a VO₂ CSV to begin.")
    st.stop()

# --- Show Summary (using tabulate) and Plots ---
st.subheader("📊 Data Summary")
summary = df.describe().loc[["mean", "std", "min", "max"]].T.reset_index()
summary.rename(columns={"index": "Metric"}, inplace=True)
st.text(tabulate(summary, headers=summary.columns, tablefmt="github"))

key_cols = ['Time[s]', 'VO2[mL/kg/min]', 'HR[bpm]']
plot_cols = [c for c in key_cols if c in df.columns]

if len(plot_cols) >= 2:
    for col in plot_cols[1:]:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(df[plot_cols[0]], df[col])
        ax.set_xlabel(plot_cols[0])
        ax.set_ylabel(col)
        ax.set_title(f"{col} over {plot_cols[0]}")
        st.pyplot(fig)
else:
    st.info(f"Not enough key columns for plotting; found: {', '.join(df.columns)}")

# --- Example Questions (LLM-powered) ---
st.subheader("💬 Example Questions You Could Ask")
if "examples" not in st.session_state:
    col_list = ", ".join(df.columns.tolist())
    example_prompt = (
        f"You have a VO₂ dataset with these columns: {col_list}. "
        "Generate 5 example questions a user might ask to interpret this data."
    )
    llm = ChatOpenAI(model="gpt-4", temperature=0, api_key=api_key)
    resp = llm.invoke(example_prompt).content
    st.session_state.examples = [
        e.strip("0123456789. ") for e in resp.split("\n") if e.strip()
    ]

for i, q in enumerate(st.session_state.examples):
    # Use on_click to set `question` in session_state
    st.button(
        q,
        key=f"example_btn_{i}",
        on_click=lambda q=q: st.session_state.__setitem__("question", q)
    )

# --- Ask My Data Section ---
question = st.text_input(
    "Or type your own question:",
    value=st.session_state.get("question", "")
)

if question:
    # ... RAG retrieval & LLM analysis as before ...

    try:
        # RAG retrieval
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vectorstore = FAISS.load_local("vectorstore/faiss_index", embeddings,
                                       allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever(search_type="similarity", k=3)
        context_docs = retriever.invoke(question)
        context_text = "\n\n".join(doc.page_content for doc in context_docs)

        # Subsample key columns to manage token length
        data_cols = [c for c in key_cols if c in df.columns]
        sampled_df = df[data_cols].iloc[::10]
        data_sample = sampled_df.to_csv(index=False)
        approx_tokens = len(data_sample) // 4
        st.caption(f"🧮 Estimated tokens to LLM: {approx_tokens}")

        # Build prompt
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

        # ─── User Feedback Section ─────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Was this analysis helpful?")
        rating = st.radio("Rate this app (1 = poor, 5 = excellent)", [1, 2, 3, 4, 5], horizontal=True)
        feedback = st.text_area("Additional feedback or suggestions:")
        if st.button("Submit Feedback"):
            st.success("✅ Thanks for your feedback!")
            # Example: you could log to a file or database here
            # logging.info("User rating: %d | Feedback: %s", rating, feedback)

    except Exception as e:
        st.error(f"Analysis failed: {e}")





 
