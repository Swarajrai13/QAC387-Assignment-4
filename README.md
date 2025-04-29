# VO₂max Data Assistant with Expert RAG Integration

This Streamlit app leverages Retrieval-Augmented Generation (RAG) and OpenAI's GPT-4 to provide deep, context-aware analysis of your VO₂max datasets. It integrates **expert domain guidance** (from PMC10687136 Table 4) to ensure rigorous evaluation of data processing strategies and physiological insights.

---

## 🚀 New Features & Improvements (April 2025)

1. **RAG Pipeline Integration**  
   - Expert guidance document (`vo2_processing_standards.txt`) indexed via FAISS
   - Contextual retrieval ensures LLM answers refer to VO₂max reporting best practices

2. **Enhanced Data Handling & UX**  
   - Robust error handling for malformed or empty CSV uploads  
   - Data summary table (mean, std, min, max) replaces raw-preview for clarity  
   - Subsampling of every 10th row & key columns (`Time[s]`, `VO2[mL/kg/min]`, `HR[bpm]`) to respect model token limits

3. **Advanced Visualization**  
   - Individual Matplotlib plots for each key metric (separate panels)  
   - Automatic detection of required columns with user feedback if missing

4. **LLM Prompt Engineering**  
   - Deep statistical & physiological interpretation  
   - Trend highlighting, plausibility checks, and follow-up recommendations

5. **Testing & Validation**  
   - **Validation Log** (`validation_log.csv`) covering Functional, Input Validation, Code Accuracy, Output Invariance, Usability, and Edge Cases - in addition to tracking dates, actions, and outcomes  
   - In-app user feedback capture (1–5 star rating + comments)

---

## 📁 Repository Structure

```
├── README.md
├── vo2_processing_standards.txt    # Expert guidance for RAG
├── RAG_pipeline.py                # Build FAISS vector store
├── rag_data_assistant_4_29.py          # Main Streamlit app
├── validation_log.csv              # Testing & Validation Checklist + Detailed Validation Log
├── app_logs.txt                   # Runtime error & usage logs
└── vectorstore/                   # FAISS index files
```

---

## 📦 Installation

```bash
# Clone repo
git clone https://github.com/Swarajrai13/QAC387-Assignment-4.git
cd QAC387-Assignment-4

# Create & activate virtual env
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```  

**Note:** Ensure you have a `.env` file in the project root containing:
```env
OPENAI_API_KEY=sk-<your-key-here>
```

---

## ▶️ Running the App

1. **Generate the RAG vector store** (once, or whenever `vo2_processing_standards.txt` changes):
   ```bash
   python RAG_pipeline.py
   ```

2. **Launch Streamlit**:
   ```bash
   streamlit run rag_data_assistant_4_23.py
   ```

3. Upload your VO₂ CSV, ask a question, and review the insights.

---

## 🧪 Testing & Validation

- Review the **checklist.md** for implemented tests and pending items.  
- Consult **validation_log.md** for the detailed timeline of improvements.  
- User feedback collected in-app is appended to `app_logs.txt`.

---

## 🛠️ Next Steps

- Dynamic column mapping for non-standard VO₂ files  
- Support for categorical or multi-metric analyses  
- Batch processing for very large datasets  

---

**License**: MIT  
**Author**: Swaraj  