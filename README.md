# VO2 Optimization Assistant 🏃

This is a Streamlit web app that allows users to upload VO2 test data and receive an optimized fitness-recovery score and summary. The app combines data parsing, statistical analysis, and an LLM-powered question-answering assistant built using LangChain and OpenAI.

## 🔧 Features

- Upload your VO2 `.csv` test report
- Enter your participant ID to match survey data (stress, fitness, diet)
- Get a personalized Optimization Score based on training + recovery metrics
- Ask any question about your VO2 data using a built-in AI assistant

## ⚙️ How to Run

1. Clone this repo and `cd` into it.
2. Create a virtual environment and install dependencies:
   ```
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Create a `.env` file and add your OpenAI API key:
   ```
   OPENAI_API_KEY=sk-xxxxxxx
   ```
4. Place the required survey CSVs in the `/data` folder.
5. Run the app:
   ```
   streamlit run app.py
   ```

## 📁 Required Data Files (in `/data`)
- Demographics.csv
- Perceived Stress Scale (PSS).csv
- International Fitness Scale (IFIS).csv
- How healthy is your diet.csv

## ⚠️ Caution
- The LLM agent runs with `allow_dangerous_code=True` — avoid uploading sensitive data.
- LLM answers are based on available tabular data; they are not a replacement for medical advice.

## ✅ Submission Ready
Built for Assignment 3 of QAC387: LLM-powered data analysis.
