import streamlit as st
import joblib
import re

# Load saved model and vectorizer
@st.cache_resource
def load_assets():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_assets()

# Text cleaning function
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    text = text.lower()
    return text

# Page Styling and Layout
st.set_page_config(page_title="IMDB Movie Review Sentiment Analyzer", page_icon="🎬", layout="centered")

st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.markdown("Enter a movie review below to find out whether the sentiment is *Positive* or *Negative* using NLP and Machine Learning.")

# User Input Text Area
user_review = st.text_area("✍️ Write your movie review here:", placeholder="Type or paste a review...")

if st.button("🔍 Analyze Sentiment"):
    if user_review.strip() == "":
        st.warning("Please enter a valid movie review before analyzing.")
    else:
        # Preprocess and vectorize
        cleaned = clean_text(user_review)
        transformed_input = vectorizer.transform([cleaned])
        
        # Predict
        prediction = model.predict(transformed_input)[0]
        confidence = model.predict_proba(transformed_input).max()
        
        st.markdown("---")
        st.subheader("Prediction Result")
        
        if prediction == 1:
            st.success(f"*Positive Review!* 😊 (Confidence: {confidence:.2f})")
        else:
            st.error(f"*Negative Review!* 😞 (Confidence: {confidence:.2f})")

# Footer info
st.markdown("---")
st.caption("Built with Python, Scikit-Learn, and Streamlit.")