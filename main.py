import tensorflow as tf
from tensorflow.keras.models import load_model
from transformers import DistilBertTokenizer
import pandas as pd
import streamlit as st
import os
import gdown
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from textblob import TextBlob
import json

# ========= STREAMLIT CONFIG =========
st.set_page_config(page_title="GoStop BBQ Recommender", layout="centered")

# ========= PATH SETUP =========
CNN_BLSTM_MODEL_PATH = "distilbert_cnn_blstm_model.keras"
TOKENIZER_PATH = "distilbert_model/tokenizer"
# ========= CONFIG =========
MAX_LEN = 100

# ========= SIDEBAR =========
with st.sidebar:
    if os.path.exists("gostop.jpeg"):
        st.image(Image.open("gostop.jpeg"), use_container_width=True)
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# ========= DOWNLOAD SECTION =========
if not os.path.exists(TOKENIZER_PATH):
    with st.spinner("📦 Downloading tokenizer files from Google Drive..."):
        os.makedirs(TOKENIZER_PATH, exist_ok=True)
        urls = {
            "vocab.txt": "https://drive.google.com/uc?id=1VZcfhSxRxLTDyubONT733CLcZAg1oMK8",
            "config.json": "https://drive.google.com/uc?id=1QhCJwxkoqP4ooBjhJvFG2Ryokx9A-nvG",
            "tokenizer_config.json": "https://drive.google.com/uc?id=194U6dTZQQfkFLspPkT9Lt7iQlDlyerSr",
            "special_tokens_map.json": "https://drive.google.com/uc?id=1RCvAL5VXNuN1bYSkv8VrT80WPtZQ-k2u"
        }

        for fname, url in urls.items():
            dest = os.path.join(TOKENIZER_PATH, fname)
            gdown.download(url, dest, quiet=False)

            if fname.endswith(".json"):
                try:
                    with open(dest, "r", encoding="utf-8") as f:
                        json.load(f)
                except Exception as e:
                    st.error(f"❌ File {fname} rusak: {e}")
                    st.stop()

# ========= MENU DATA =========
menu_actual = [
    "soondubu jjigae", "prawn soondubu jjigae", "kimchi jjigae", "tofu jjigae",
    "samgyeopsal", "spicy samgyeopsal", "woo samgyup", "spicy woo samgyup",
    "bulgogi", "dak bulgogi", "spicy dak bulgogi", "meltique tenderloin", "odeng",
    "beef soondubu jjigae", "pork soondubu jjigae"
]

menu_categories = {
    "spicy": ["spicy samgyeopsal", "spicy woo samgyup", "spicy dak bulgogi", "kimchi jjigae", "budae jjigae"],
    "meat": ["samgyeopsal", "woo samgyup", "bulgogi", "dak bulgogi", "saeng galbi", "meltique tenderloin"],
    "soup": ["kimchi jjigae", "tofu jjigae", "budae jjigae", "soondubu jjigae", "beef soondubu jjigae", "pork soondubu jjigae", "prawn soondubu jjigae"],
    "seafood": ["prawn soondubu jjigae", "odeng"],
    "beef": ["bulgogi", "beef soondubu jjigae", "meltique tenderloin"],
    "pork": ["samgyeopsal", "spicy samgyeopsal", "pork soondubu jjigae"],
    "bbq": ["samgyeopsal", "woo samgyup", "bulgogi"],
    "non_spicy": ["tofu jjigae", "soondubu jjigae", "beef soondubu jjigae", "meltique tenderloin", "odeng"],
    "tofu_based": ["tofu jjigae", "soondubu jjigae", "beef soondubu jjigae", "pork soondubu jjigae"]
}

menu_aliases = {
    "soondubu": "soondubu jjigae",
    "suundobu": "soondubu jjigae",
    "beef soondubu": "beef soondubu jjigae",
    "pork soondubu": "pork soondubu jjigae",
    "soondubu jigae": "soondubu jjigae"
}

# ========= LOADERS =========
@st.cache_data
def load_data():
    if not os.path.exists("review_sentiment.csv"):
        st.error("❌ 'review_sentiment.csv' not found.")
        return None
    try:
        df = pd.read_csv("review_sentiment.csv")
        
        # Mengganti menu yang salah ketik sesuai dengan aliases yang ada
        df["menu"] = df["menu"].str.lower().replace(menu_aliases)

        # Menyaring review dengan sentiment positif
        df = df[df["sentiment"] == "positive"]

        # Menghitung statistik menu
        menu_stats = df.groupby("menu").agg(
            count=("menu", "count"),
            avg_sentiment=("compound_score", "mean")
        ).reset_index()

        # Normalisasi dan perhitungan skor menu
        scaler = MinMaxScaler()
        menu_stats[["count_norm", "sentiment_norm"]] = scaler.fit_transform(
            menu_stats[["count", "avg_sentiment"]]
        )
        menu_stats["score"] = (menu_stats["count_norm"] + menu_stats["sentiment_norm"]) / 2
        
        return menu_stats
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None

menu_stats = load_data()
if menu_stats is None:
    st.stop()

# ========= MODEL LOADER =========
@st.cache_resource
def load_model_and_tokenizer():
    # Memuat model CNN-BLSTM yang sudah dilatih
    sentiment_model = load_model(CNN_BLSTM_MODEL_PATH)
    
    return sentiment_model

# Load model
sentiment_model = load_model_and_tokenizer()

st.success("✅ All models and tokenizer loaded successfully!")

# ========= PREDICTION =========
def predict_sentiment(text, sentiment_model):
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding="max_length", max_length=MAX_LEN)
    preds = sentiment_model.predict(inputs['input_ids'], verbose=0)
    return int(preds[0][0] > 0.5)

# ========= UTILS =========
def correct_spelling(text):
    return str(TextBlob(text).correct())

def detect_category(text):
    text = text.lower()
    for keyword, category in keyword_aliases.items():
        if keyword in text:
            return category
    return None

def fuzzy_match_menu(text, menu_list):
    text = text.lower()
    for menu in menu_list:
        if all(word in text for word in menu.split()):
            return menu
    return None

def detect_negative_rule(text):
    negative_keywords = ["don't", "not", "dislike", "too", "hate", "worst", "bad"]
    return any(neg in text for neg in negative_keywords)

def is_category_only_input(text):
    words = text.lower().split()
    for word in words:
        if word not in keyword_aliases:
            return False
    return True

# ========= CHAT =========
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("👩‍🍳 GoStop Korean BBQ Menu Recommender")
st.markdown("Ask something like **'recommend me non-spicy food'** or **'how about odeng?'**")

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("You:", placeholder="Type your request here...")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    raw_input = user_input.lower()
    corrected_input = user_input if len(user_input.split()) <= 2 else correct_spelling(user_input)
    corrected_lower = corrected_input.lower()

    matched_menu = fuzzy_match_menu(raw_input, menu_actual)
    category = detect_category(raw_input)
    is_category_input = is_category_only_input(corrected_lower)
    explicit_negative = detect_negative_rule(raw_input)
    is_negative = False
    sentiment_pred = "SKIPPED"

    if matched_menu and not explicit_negative:
        is_negative = False
    elif matched_menu and explicit_negative:
        is_negative = True
    elif matched_menu:
        sentiment_pred = predict_sentiment(corrected_input, sentiment_model)
        is_negative = sentiment_pred == 0
    elif category and not explicit_negative and is_category_input:
        is_negative = False
    elif category and explicit_negative:
        is_negative = True
    else:
        sentiment_pred = predict_sentiment(corrected_input, sentiment_model)
        is_negative = sentiment_pred == 0

    show_mood = any(word in raw_input for word in ["love", "like", "want", "enjoy"]) and not is_negative
    sentiment_note = "😊 Awesome! You're
