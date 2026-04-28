#python -m venv venv
#venv\Scripts\activate

#pip install streamlit tensorflow numpy
#streamlit run app.py

import os
# MUST stay at the very top
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import pickle
import re
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Sentiment Analysis", layout="centered")

st.title("🎬 Movie Review Sentiment Analysis")

# -------------------------------
# Load tokenizer and model
# -------------------------------
@st.cache_resource
def load_files():
    try:
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)

        # Added compile=False to bypass the quantization error
        model = load_model("lstm_model.h5", compile=False)

        return tokenizer, model, None

    except Exception as e:
        return None, None, str(e)

tokenizer, model, error = load_files()

# -------------------------------
# Error Handling
# -------------------------------
if error:
    st.error(f"❌ Error loading model: {error}")
    st.stop()
else:
    st.success("✅ Model & Tokenizer Loaded Successfully")

# -------------------------------
# Text cleaning function
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r'\bnot\s+(\w+)', r'not_\1', text)
    text = re.sub(r'[^a-zA-Z_!?]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -------------------------------
# Prediction function
# -------------------------------
def predict_sentiment(text):
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    # maxlen should match what you used during training (usually 100 or 200)
    padded = pad_sequences(seq, maxlen=100, padding='post', truncating='post')

    pred = model.predict(padded, verbose=0)[0][0]

    if pred > 0.5:
        return "Positive 😊", pred
    else:
        return "Negative 😞", pred

# -------------------------------
# UI Input
# -------------------------------
st.write("Type a movie review and check whether it's Positive or Negative.")
review = st.text_area("Enter your review here:", height=150)

if st.button("Predict"):
    if review.strip() == "":
        st.warning("⚠️ Please enter a review first.")
    else:
        with st.spinner("Analyzing sentiment..."):
            result, confidence = predict_sentiment(review)
            if "Positive" in result:
                st.success(f"Prediction: {result}")
            else:
                st.error(f"Prediction: {result}")
            st.write(f"Confidence Score: {confidence:.4f}")

