# app.py

import streamlit as st
import joblib
import nltk
import re

# Download NLTK resources (only first time)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Preprocessing function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Load the model and encoder
model = joblib.load('disaster_multiclass_model.pkl')     
label_encoder = joblib.load('label_encoder.pkl')          
# Streamlit app layout
st.title("🌪️ Disaster Tweet Classifier")

tweet = st.text_area("📥 Paste a tweet to classify:", height=150)

if st.button("🚀 Predict"):
    if tweet.strip() == "":
        st.warning("Please enter some text.")
    else:
        processed = preprocess_text(tweet)
        prediction = model.predict([processed])[0]

        # Check type (model might return encoded integer or label string)
        try:
            label = label_encoder.inverse_transform([prediction])[0]
        except:
            label = prediction

        st.success(f"🧠 Predicted Class: **{label.upper()}**")
