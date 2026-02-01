import streamlit as st
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure required NLTK resources are available
for resource in ["stopwords", "punkt", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Load trained pipeline
with open("product_category_pipe.pkl", "rb") as f:
    model = pickle.load(f)

def clean(doc: str) -> str:
    """Clean and preprocess text for prediction."""
    doc = doc.lower()
    doc = re.sub(r"http\S+|www\S+|https\S+", " ", doc)
    doc = re.sub(r"<.*?>", " ", doc)
    doc = re.sub(r"[^a-zA-Z.]", " ", doc)
    tokens = word_tokenize(doc)
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    clean_doc = " ".join(tokens)
    clean_doc = re.sub(r"\s+", " ", clean_doc).strip()
    return clean_doc

# Custom CSS
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to right, #ffecd2, #fcb69f);
    }
    .title {
        font-size: 40px;
        font-weight: bold;
        color: #2E86C1;
        text-align: center;
    }
    .prediction {
        font-size: 24px;
        color: #117A65;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# UI
st.markdown('<div class="title">🎨 E‑Commerce Product Categorization 🎨</div>', unsafe_allow_html=True)
st.write("Enter a product description or review, and the model will predict its category.")

user_input = st.text_area("✍️ Product Description", "")

if st.button("🔮 Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        cleaned_text = clean(user_input)
        try:
            prediction = model.predict([cleaned_text])[0]
            st.markdown(f'<div class="prediction">Predicted Category: {prediction}</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")