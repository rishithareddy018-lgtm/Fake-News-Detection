import streamlit as st
import pickle

# Load saved model and vectorizer
model = pickle.load(open("models/fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

st.title("📰 Fake News Detection System")

news = st.text_area("Enter a news article:")

if st.button("Predict"):
    if news.strip() != "":
        news_vector = vectorizer.transform([news])
        prediction = model.predict(news_vector)

        if prediction[0] == 0:
            st.error("❌ This looks like FAKE News.")
        else:
            st.success("✅ This looks like REAL News.")
    else:
        st.warning("Please enter some text.")