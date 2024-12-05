import streamlit as st
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings

# Загрузка книги и ретривера
@st.cache_resource
def loading(api_key):
    loader = PyPDFLoader("chai.pdf")
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=api_key)
    vectorstore = FAISS.from_texts(
        [chunk.page_content for chunk in chunks], embeddings_model
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})

def chatbot_page():
    if not st.session_state["api_key"]:
        st.warning("Пожалуйста, введите API-ключ на странице 'API Key'.")
        st.stop()

    st.title("Чай-бот")

    # Инициализация истории чата
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "bot", "content": "Привет! Я могу ответить на вопросы по книге В. В. Похлёбкина 'Чай, его история, свойства и употребление'. Задавайте вопросы, и я постараюсь помочь!"}
        ]

    retriever = loading(st.session_state["api_key"])
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=st.session_state["api_key"], temperature=0.9)
    prompt_template = '''
    Ты - чат-бот по имени "чай-бот", который читал только одну книгу В. В. Похлёбкина "Чай, его история, свойства и употребление".

    Учитывая приведенный контекст и историю, ответь на вопрос в конце.

    История: {history}
    Контекст: {context}
    Вопрос: {question}
    '''
    prompt = PromptTemplate(input_variables=["history", "context", "question"], template=prompt_template)
    chain = prompt | llm

    # Интерфейс чата
    user_input = st.chat_input("Введите ваш вопрос:")
    if user_input:
        # Отображение вопроса пользователя
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Генерация ответа
        with st.spinner("Перечитываю книгу..."):
            context = "\n".join([doc.page_content for doc in retriever.get_relevant_documents(user_input)])
            history = "\n".join(
                f"Вопрос: {entry['content']}" if entry["role"] == "user" else f"Ответ: {entry['content']}"
                for entry in st.session_state.chat_history
            )
            result = chain.invoke({"history": history, "context": context, "question": user_input})

        # Отображение ответа от бота
        st.session_state.chat_history.append({"role": "bot", "content": result.content})

    # Рендер сообщений
    for entry in st.session_state.chat_history:
        if entry["role"] == "user":
            with st.chat_message("user"):
                st.markdown(entry["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(entry["content"])
