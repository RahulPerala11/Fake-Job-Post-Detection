import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import PyPDF2
import io

# Load the trained model and vectorizer
model = joblib.load("job_detection.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text.lower())
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# Custom CSS for styling the app
css = """
    <style>
        .main {
            background-color: #f1f1f1;
            font-family: 'Arial', sans-serif;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            border-radius: 5px;
        }
        .stButton > button:hover {
            background-color: #45a049;
        }
        .stTextInput, .stTextArea {
            background-color: #ffffff;
            border: 1px solid #ddd;
            padding: 10px;
            border-radius: 5px;
            font-size: 14px;
        }
        .stTextArea {
            min-height: 200px;
        }
        .stTitle {
            color: #333;
            font-size: 32px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 20px;
        }
    </style>
"""
st.markdown(css, unsafe_allow_html=True)

# Streamlit UI
st.title("Job Fraud Detection System")
job_description = st.text_area("Enter job description here:")
uploaded_file = st.file_uploader("Or upload a file (txt, csv, pdf):", type=["txt", "csv", "pdf"])

# Adjust threshold for fraud detection sensitivity
threshold = 0.1  # Custom threshold for fraud detection

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page].extract_text()
    return text

def extract_text_from_csv(file):
    df = pd.read_csv(file)
    if 'description' in df.columns:
        return ' '.join(df['description'].dropna().tolist())
    else:
        return 'No description column found.'

def extract_text_from_txt(file):
    return file.read().decode("utf-8")

# Analyze button functionality
if st.button("Analyze"):
    if job_description or uploaded_file:
        # Get input text either from textarea or uploaded file
        if job_description:
            input_text = job_description
        elif uploaded_file:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            if file_extension == 'pdf':
                input_text = extract_text_from_pdf(uploaded_file)
            elif file_extension == 'csv':
                input_text = extract_text_from_csv(uploaded_file)
            elif file_extension == 'txt':
                input_text = extract_text_from_txt(uploaded_file)
            else:
                st.error("Unsupported file format!")
                input_text = ""
        
        if input_text:
            cleaned_text = preprocess_text(input_text)
            st.write(f"Preprocessed Text: {cleaned_text}")
            
            # Vectorize the text
            cleaned_text_vectorized = vectorizer.transform([cleaned_text])
            st.write(f"Vectorized Text Shape: {cleaned_text_vectorized.shape}")
            
            # Predict probability if model supports it; otherwise, use a basic prediction
            if hasattr(model, "predict_proba"):
                prediction_prob = model.predict_proba(cleaned_text_vectorized)[0][1]  # Probability of fraud
                prediction = "Fraudulent" if prediction_prob > threshold else "Legitimate"
                st.write(f"Prediction: {prediction} (Fraud Probability: {prediction_prob:.2f})")
            else:
                # Fall back to simple prediction if predict_proba is not available
                prediction = model.predict(cleaned_text_vectorized)[0]
                st.write(f"Prediction: {'Fraudulent' if prediction == 1 else 'Legitimate'}")
