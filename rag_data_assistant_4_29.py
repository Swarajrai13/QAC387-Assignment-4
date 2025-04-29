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
        st.session_state.df = df
    except Exception as e:
        st.error(f"Error loading file: {e}")

df = st.session_state.df
if df is None:
    st.info("Please upload a VO₂ CSV to begin.")
    st.stop()

# --- Show Summary (using tabulate) and Plot ---
st.subheader("Data Summary")
summary = df.describe().loc[["mean","std","min","max"]].T
# Use tabulate for a compact text table
st.text(tabulate(summary.reset_index(), headers=summary.reset_index().columns, tablefmt="github"))

# Plot each key metric separately
key_cols = ['Time[s]', 'VO2[mL/kg/min]', 'HR[bpm]']
plot_cols = [c for c in key_cols if c in df.columns]
for col in plot_cols[1:]:
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(df[plot_cols[0]], df[col])
    ax.set_xlabel(plot_cols[0])
    ax.set_ylabel(col)
    ax.set_title(f"{col} over {plot_cols[0]}")
    st.pyplot(fig)

# --- Example Questions (LLM-powered) ---
st.subheader("Example Questions You Could Ask")
if "examples" not in st.session_state:
    col_list = ", ".join(df.columns.tolist())
    example_prompt = (
        f"You have a VO₂ dataset with these columns: {col_list}. "
        "Generate 5 example questions a user might ask to interpret this data "
        "(e.g., relating VO₂ trends, heart rate changes, performance metrics)."
    )
    llm = ChatOpenAI(model="gpt-4", temperature=0, api_key=api_key)
    examples = llm.invoke(example_prompt).content.split("\n")
    st.session_state.examples = [e.strip("0123456789. ") for e in examples if e]
for q in st.session_state.examples:
    if st.button(q):
        st.session_state.question = q

# --- Ask My Data Section ---
question = st.text_input("Or type your own question:", st.session_state.get("question",""))
if question:
    # (RAG retrieval + analysis code here, unchanged)
    ...




 
