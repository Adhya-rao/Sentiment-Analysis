# 🎬 Movie Review Sentiment Analysis using LSTM

---

## 📌 Project Overview

This project performs **Sentiment Analysis on Movie Reviews** using a Deep Learning model (LSTM).

The application predicts whether a given movie review is:
- ✅ Positive
- ❌ Negative

A simple and interactive web app is built using **Streamlit**.

---
## Deployed Link 

  https://sentiment-analysis-upkuyjhiccmuy8bzudskeh.streamlit.app/

## 🚀 Features

- Text preprocessing and cleaning
- Tokenization using Keras Tokenizer
- LSTM-based deep learning model
- Real-time sentiment prediction
- Confidence score display
- Simple and user-friendly UI

---

## 🧠 Model Details

- Embedding Layer
- LSTM Layer
- Dense Layer (Sigmoid activation)

**Loss Function:** Binary Crossentropy  
**Optimizer:** Adam  

---

## 📂 Project Structure

sentiment-analysis/
│
├── app.py
├── tokenizer.pkl
├── lstm_model.h5
├── requirements.txt
└── README.md

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

git clone https://github.com/your-username/sentiment-analysis.git  
cd sentiment-analysis  

---

### 2️⃣ Create virtual environment

python -m venv venv  

---

### 3️⃣ Activate environment

**Windows:**  
venv\Scripts\activate  

**Mac/Linux:**  
source venv/bin/activate  

---

### 4️⃣ Install dependencies

pip install -r requirements.txt  

---

### 5️⃣ Run the application

streamlit run app.py  

---

## 🖥️ Application UI



(Add Screenshot Here)
<img width="982" height="745" alt="Screenshot 2026-04-26 185611" src="https://github.com/user-attachments/assets/4b5cd847-c8d0-4979-b4db-e967d8843521" />
<img width="1129" height="804" alt="Screenshot 2026-04-26 185527" src="https://github.com/user-attachments/assets/b3165b6b-22d3-48cf-961b-f7eff072ba21" />





---

## 📊 Sample Inputs & Outputs

| Input | Prediction |
|------|-----------|
| This movie is amazing | Positive 😊 |
| This movie is boring | Negative 😞 |
| not good | Negative 😞 |

---

## ⚠️ Limitations

- May struggle with sarcasm
- Limited understanding of complex sentences
- Performance depends on training data quality

---

## 🔮 Future Improvements

- Use Bidirectional LSTM
- Add Attention Mechanism
- Improve dataset balance
- Use advanced models like BERT

---

## 🧑‍💻 Technologies Used

- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Pickle

---

## 📌 Conclusion

This project demonstrates how deep learning can be used for text classification and deployed using a simple web application.

---

## 🙌 Acknowledgement

Dataset used: IMDB Movie Reviews Dataset

---

## 📬 Contact

For any queries or suggestions, feel free to reach out.
