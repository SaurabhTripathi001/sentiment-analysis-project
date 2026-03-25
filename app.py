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

    st.write("Sentiment:", prediction[0])