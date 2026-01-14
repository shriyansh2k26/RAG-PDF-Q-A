import streamlit as st
import os
from dotenv import load_dotenv
from huggingface_hub import login

# Embeddings & Vector DB
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# LLM
from langchain_groq import ChatGroq

# Document handling
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Prompts & chains (LATEST STYLE)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

# ---------------- UI ----------------
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("🤖 Intelligent PDF Assistant")

with st.sidebar:
    st.header("Settings")

    api_key = st.secrets.get("GROQ_API_KEY") if hasattr(st, "secrets") else None
    hf_token = st.secrets.get("HF_TOKEN") if hasattr(st, "secrets") else None

    api_key = api_key or st.text_input("Enter Groq API Key", type="password")
    hf_token = hf_token or st.text_input("HuggingFace Token", type="password")

    if hf_token:
        login(token=hf_token)

    session_id = st.text_input("Session ID", value="default")
    clear = st.button("Clear chat")

if clear:
    st.session_state.clear()
    st.rerun()

# ---------------- Memory ----------------
if "store" not in st.session_state:
    st.session_state.store = {}

def get_session_history(session_id: str):
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# ---------------- Embeddings ----------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ---------------- Main Logic ----------------
if api_key:

    llm = ChatGroq(api_key=api_key, model="llama-3.1-8b-instant")
    embeddings = load_embeddings()

    files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

    if files:

        with st.spinner("Processing documents..."):
            docs = []
            for f in files:
                path = f"temp_{f.name}"
                with open(path, "wb") as fp:
                    fp.write(f.getbuffer())
                docs.extend(PyPDFLoader(path).load())
                os.remove(path)

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = splitter.split_documents(docs)

            vectorstore = Chroma.from_documents(splits, embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        # ----------- Prompts -----------
        contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system", "Rewrite the user's question so it is standalone, using chat history."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        history_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_prompt
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer using ONLY the following context:\n\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        qa_chain = create_stuff_documents_chain(llm, qa_prompt)

        rag_chain = create_retrieval_chain(history_retriever, qa_chain)

        convo_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        # ----------- Chat UI -----------
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        if user_input := st.chat_input("Ask questions about your PDFs"):
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.chat_message("user").write(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    res = convo_chain.invoke(
                        {"input": user_input},
                        config={"configurable": {"session_id": session_id}}
                    )
                    answer = res["answer"]
                    st.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("Enter your Groq API key to begin.")
