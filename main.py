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
        config_url = "https://drive.google.com/uc?id=1rVyhwe_au-yUPo7Fq7rcYwqPd7X5zzZB"
        vocab_url = "https://drive.google.com/uc?id=13bC7_29Vy7lJJ7NaO5uuRw4QaI9sE9Nl"
        gdown.download(config_url, os.path.join(TOKENIZER_PATH, 'config.json'), quiet=False)
        gdown.download(vocab_url, os.path.join(TOKENIZER_PATH, 'vocab.txt'), quiet=False)

# ========= CONFIG =========
MAX_LEN = 100

# ========= STREAMLIT CONFIG =========
st.set_page_config(page_title="GoStop BBQ Recommender", layout="centered")

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

# ========= LOADERS =========
@st.cache_data
def load_data():
    if not os.path.exists("review_sentiment.csv"):
        st.error("❌ 'review_sentiment.csv' not found.")
        return None
    try:
        df = pd.read_csv("review_sentiment.csv")
        df = df[df["sentiment"] == "positive"]
        df["menu"] = df["menu"].str.lower().str.replace("suundobu", "soondubu")
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
        sentiment_pred = predict_sentiment(corrected_input, tokenizer, bert_model, bert_sentiment_model)
        is_negative = sentiment_pred == 0
    elif category and not explicit_negative and is_category_input:
        is_negative = False
    elif category and explicit_negative:
        is_negative = True
    else:
        sentiment_pred = predict_sentiment(corrected_input, tokenizer, bert_model, bert_sentiment_model)
        is_negative = sentiment_pred == 0

    show_mood = any(word in raw_input for word in ["love", "like", "want", "enjoy"]) and not is_negative
    sentiment_note = "😊 Awesome! You're in a good mood! " if show_mood else "😕 No worries! I got you. " if is_negative else ""

    recommended = None
    if matched_menu:
        matched_menu = matched_menu.strip().lower()
        if is_negative:
            recommended = menu_stats[menu_stats["menu"] != matched_menu].sort_values("score", ascending=False).head(3)
            response = sentiment_note + f"Oops! You don't like <strong>{matched_menu.title()}</strong>? Try these instead:"
        elif matched_menu in menu_stats["menu"].values:
            row = menu_stats[menu_stats["menu"] == matched_menu].iloc[0]
            response = sentiment_note + f"🍽️ <strong>{matched_menu.title()}</strong> has <strong>{row['count']} reviews</strong> with average sentiment <strong>{row['avg_sentiment']:.2f}</strong>. Recommended! 🎉"
        elif matched_menu in menu_actual:
            response = f"🍽️ <strong>{matched_menu.title()}</strong> is on our menu! 🎉"
        else:
            recommended = menu_stats.sort_values("score", ascending=False).head(3)
            response = sentiment_note + "❌ Not sure about that menu. Here are our top 3 picks!"
    elif category and not matched_menu:
        matched = menu_categories.get(category, [])
        if is_negative:
            recommended = menu_stats[~menu_stats["menu"].isin(matched)].sort_values("score", ascending=False).head(3)
            response = f"🙅‍♂️ Avoiding <strong>{category.replace('_', ' ').title()}</strong>? Here are other ideas:"
        else:
            recommended = menu_stats[menu_stats["menu"].isin(matched)].sort_values("score", ascending=False).head(3)
            response = sentiment_note + f"🔥 You might like these <strong>{category.replace('_', ' ').title()}</strong> dishes:"
    else:
        recommended = menu_stats.sort_values("score", ascending=False).head(3)
        response = sentiment_note + "🤔 Couldn't find what you're looking for. Here's our top 3!"

    if recommended is not None:
        response += "<table><thead><tr><th>Rank</th><th>Menu</th><th>Sentiment</th><th>Reviews</th></tr></thead><tbody>"
        for idx, (_, row) in enumerate(recommended.iterrows(), 1):
            response += f"<tr style='text-align:center;'><td>{idx}</td><td>{row['menu'].title()}</td><td>{row['avg_sentiment']:.2f}</td><td>{int(row['count'])}</td></tr>"
        response += "</tbody></table>"

    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("bot", response))

    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(msg, unsafe_allow_html=True)
