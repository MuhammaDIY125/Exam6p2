import streamlit as st

def api_key_page():
    st.title("API-ключ")
    st.write("Введите ваш API-ключ, чтобы продолжить использование приложения.")
    
    api_key = st.text_input("API-ключ", type="password")
    
    if st.button("Сохранить ключ"):
        if api_key:
            st.session_state["api_key"] = api_key
            st.success("API-ключ успешно сохранён!")
        else:
            st.error("Пожалуйста, введите API-ключ.")
