import streamlit as st
import joblib
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# === Preprocessing ===
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# === Load model and label encoder ===
model = joblib.load("disaster_multiclass_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# === Streamlit UI ===
st.set_page_config(page_title="🌪️ Disaster Tweet Classifier", layout="centered")

st.markdown("<h1 style='text-align: center; color: #d63384;'>🌪️ Disaster Tweet Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter a tweet below to classify it as Earthquake, Flood, Fire, or Non-Disaster.</p>", unsafe_allow_html=True)

tweet = st.text_area("📝 Enter Tweet", height=150)

if st.button("🚀 Predict"):
    if tweet.strip() == "":
        st.warning("⚠️ Please enter a tweet.")
    else:
        processed = preprocess_text(tweet)
        prediction = model.predict([processed])[0]

        # If model returns encoded integer
        try:
            label = label_encoder.inverse_transform([prediction])[0]
        except ValueError:
            label = prediction

        color_map = {
            "Earthquake": "#ff5733",
            "Flood": "#3498db",
            "Fire": "#e74c3c",
            "Non-Disaster": "#2ecc71"
        }

        color = color_map.get(label, "#6c757d")

        st.markdown(
            f"<h3 style='color:{color}; text-align: center;'>Predicted Class: {label.upper()}</h3>",
            unsafe_allow_html=True
        )

# === Footer ===
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;font-size:13px'>Built with ❤️ using Streamlit</p>", unsafe_allow_html=True)
