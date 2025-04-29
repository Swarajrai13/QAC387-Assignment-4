#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pickle
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load API key
load_dotenv("/Users/swaraj/Downloads/QAC387 Project/ai-data-analysis-assistant/env")
api_key = os.getenv("OPENAI_API_KEY")

# Load your VO2 Table 4 document
loader = TextLoader("vo2_processing_standards.txt")
docs = loader.load()

# Split the document into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = splitter.split_documents(docs)

# Embed and save with FAISS
embedding_model = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(split_docs, embedding=embedding_model)
vectorstore.save_local("vectorstore/faiss_index")

print("✅ RAG vector store created successfully.")

