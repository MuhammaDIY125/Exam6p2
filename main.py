import streamlit as st
from api_key import api_key_page
from chatbot import chatbot_page

# Инициализация session_state для API-ключа
if "api_key" not in st.session_state:
    st.session_state["api_key"] = None

st.sidebar.title("Навигация")
page = st.sidebar.radio("Перейти на:", ["API-ключ", "Чай-бот"])

if page == "API-ключ":
    api_key_page()
elif page == "Чай-бот":
    chatbot_page()
