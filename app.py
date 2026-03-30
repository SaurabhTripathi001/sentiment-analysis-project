import streamlit as st
import pickle
from preprocess import clean_text

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("Sentiment Analysis App 💬")

user_input = st.text_area("Enter your text:")

if st.button("Analyze"):
    cleaned = clean_text(user_input)
    vectorized = vectorizer.transform([cleaned])

    prediction = model.predict(vectorized)
    proba = model.predict_proba(vectorized)

    confidence = max(proba[0]) * 100

    if prediction[0] == "positive":
        st.success(f"Positive 😊 ({confidence:.2f}%)")
    elif prediction[0] == "negative":
        st.error(f"Negative 😡 ({confidence:.2f}%)")
    else:
        st.warning(f"Neutral 😐 ({confidence:.2f}%)")