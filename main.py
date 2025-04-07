import tensorflow as tf
import transformers
from transformers import DistilBertTokenizer, TFDistilBertModel
import pandas as pd
import streamlit as st
import os
import gdown
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from datetime import datetime
from textblob import TextBlob

# ========= STREAMLIT CONFIG =========
st.set_page_config(page_title="GoStop BBQ Recommender", layout="centered")

# Ganti path model sesuai dengan Google Drive
MODEL_PATH = "t5_model.keras"
CNN_BLSTM_MODEL_PATH = "distilbert_cnn_blstm_model.keras"
TOKENIZER_PATH = "distilbert_model/tokenizer"

# Jika model belum ada, unduh dari Google Drive
if not os.path.exists(MODEL_PATH):
    with st.spinner("📦 Downloading model from Google Drive... please wait..."):
        model_url = "https://drive.google.com/uc?id=17MCaqROy6CEyLY8XdjEGFLPuwGL0VbN3"
        gdown.download(model_url, MODEL_PATH, quiet=False)

# Jika file tokenizer belum ada, unduh dari Google Drive
if not os.path.exists(TOKENIZER_PATH):
    with st.spinner("📦 Downloading tokenizer from Google Drive... please wait..."):
        os.makedirs(TOKENIZER_PATH, exist_ok=True)

        vocab_url = "https://drive.google.com/uc?id=13bC7_29Vy7lJJ7NaO5uuRw4QaI9sE9Nl"
        config_url = "https://drive.google.com/uc?id=1rVyhwe_au-yUPo7Fq7rcYwqPd7X5zzZB"
        tokenizer_config_url = "https://drive.google.com/uc?id=1l5shBszGOPZXPwvESCU7zq9O-3DTIozj"
        special_tokens_url = "https://drive.google.com/uc?id=1pio78X3HAoEG9ejBGrs8fzbr2PorrZeK"

        gdown.download(vocab_url, os.path.join(TOKENIZER_PATH, "vocab.txt"), quiet=False)
        gdown.download(config_url, os.path.join(TOKENIZER_PATH, "config.json"), quiet=False)
        gdown.download(tokenizer_config_url, os.path.join(TOKENIZER_PATH, "tokenizer_config.json"), quiet=False)
        gdown.download(special_tokens_url, os.path.join(TOKENIZER_PATH, "special_tokens_map.json"), quiet=False)

# ========= CONFIG =========
MAX_LEN = 100



with st.sidebar:
    if os.path.exists("gostop.jpeg"):
        logo = Image.open("gostop.jpeg")
        st.image(logo, use_container_width=True)
    else:
        st.warning("⚠️ Logo 'gostop.jpeg' not found.")

    if st.button("🗑️ Clear Chat", key="clear"):
        st.session_state.chat_history = []
        st.rerun()

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
    "non spicy": "non_spicy", "non-spicy": "non_spicy", "not spicy": "non_spicy", "mild": "non_spicy",
    "grill": "bbq", "barbecue": "bbq", "bbq": "bbq",
    "hot soup": "soup", "warm soup": "soup",
    "hot": "spicy", "spicy": "spicy",
    "soup": "soup", "broth": "soup", "jjigae": "soup",
    "fish": "seafood", "prawn": "seafood", "seafood": "seafood",
    "beef": "beef", "pork": "pork", "meat": "meat",
    "tofu": "tofu_based"
}

# ========= LOADERS =========
@st.cache_data
def load_data():
    if not os.path.exists("review_sentiment.csv"):
        st.error("❌ 'review_sentiment.csv' not found.")
        return None
    try:
        df = pd.read_csv("review_sentiment.csv")
        df = df[df["sentiment"] == "positive"]
        df["menu"] = df["menu"].str.lower()
        df["menu"] = df["menu"].replace(menu_aliases)


        menu_stats = df.groupby("menu").agg(
            count=("menu", "count"),
            avg_sentiment=("compound_score", "mean")
        ).reset_index()

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

@st.cache_resource
def load_bert_model_and_tokenizer():
    from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dropout, Dense
    from tensorflow.keras.models import Model

    # Muat tokenizer dan model menggunakan file yang ada di folder lokal
    tokenizer = DistilBertTokenizer.from_pretrained(TOKENIZER_PATH)
    bert = TFDistilBertModel.from_pretrained("distilbert-base-uncased")
    bert.trainable = False

    bert_out_input = Input(shape=(MAX_LEN, 768), name="bert_output")
    x = Conv1D(128, kernel_size=5, activation='relu')(bert_out_input)
    x = MaxPooling1D(pool_size=2)(x)
    x = Bidirectional(LSTM(64, dropout=0.2, return_sequences=False))(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='sigmoid')(x)

    sentiment_model = Model(inputs=bert_out_input, outputs=output)

    # Muat bobot model CNN-BLSTM
    if os.path.exists(CNN_BLSTM_MODEL_PATH):
        sentiment_model.load_weights(CNN_BLSTM_MODEL_PATH)

    return tokenizer, bert, sentiment_model

def predict_sentiment(text, tokenizer, bert_model, sentiment_model, max_len=MAX_LEN):
    inputs = tokenizer(
        text,
        return_tensors="tf",
        truncation=True,
        padding="max_length",
        max_length=max_len
    )
    bert_output = bert_model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"]
    ).last_hidden_state

    preds = sentiment_model.predict(bert_output, verbose=0)
    return int(preds[0][0] > 0.5)

tokenizer, bert_model, bert_sentiment_model = load_bert_model_and_tokenizer()
