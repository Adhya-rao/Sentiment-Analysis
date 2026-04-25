#python -m venv venv

#venv\Scripts\activate
#streamlit run app.py

import streamlit as st
import pickle
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Sentiment Analysis", layout="centered")

st.title("🎬 Movie Review Sentiment Analysis")

# -------------------------------
# Load tokenizer and model (SAFE)
# -------------------------------
@st.cache_resource
def load_files():
    try:
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)

        
        model = load_model("lstm_model.keras")

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
    text = re.sub(r'[^a-zA-Z!?]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -------------------------------
# Prediction function
# -------------------------------
def predict_sentiment(text):
    text = clean_text(text)

    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100, padding='post', truncating='post')

    pred = model.predict(padded)[0][0]

    if pred > 0.5:
        return "Positive 😊", pred
    else:
        return "Negative 😞", pred

# -------------------------------
# UI Input
# -------------------------------
st.write("Type a movie review and check whether it's Positive or Negative.")

review = st.text_area("Enter your review here:", height=150)

# Example buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("Example Positive"):
        review = "This movie was absolutely amazing and I loved it!"

with col2:
    if st.button("Example Negative"):
        review = "This movie was boring and a complete waste of time."

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("Predict"):

    if review.strip() == "":
        st.warning("⚠️ Please enter a review first.")
    else:
        result, confidence = predict_sentiment(review)

        if "Positive" in result:
            st.success(f"Prediction: {result}")
        else:
            st.error(f"Prediction: {result}")

        st.write(f"Confidence Score: {confidence:.4f}")
