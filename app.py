import streamlit as st
import os
from dotenv import load_dotenv
from huggingface_hub import login
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


load_dotenv()
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("🤖 Intelligent PDF Assistant")


with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Enter Groq API Key", type="password")
    hf_token = os.getenv("HF_TOKEN") or st.text_input("HuggingFace Token", type="password")
    
    if hf_token:
        login(token=hf_token)
    
    session_id = st.text_input("Session ID", value="default_session")
    clear_history = st.button("Clear Chat History")


if "store" not in st.session_state:
    st.session_state.store = {}

if clear_history:
    st.session_state.store = {}
    st.session_state.messages = []
    st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []


@st.cache_resource
def get_embeddings(token):
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"token": token}
    )


if api_key:
    llm = ChatGroq(groq_api_key=api_key, model="llama-3.1-8b-instant")
    embeddings = get_embeddings(hf_token)

    upload_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

    if upload_files:
        
        with st.status("Processing Documents..."):
            documents = []
            for upload_file in upload_files:
                temp_path = f"./temp_{upload_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(upload_file.getvalue())

                loader = PyPDFLoader(temp_path)
                documents.extend(loader.load())
                os.remove(temp_path) 

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(documents)
            
            
            vectorstore = Chroma.from_documents(chunks, embeddings)
            retriever = vectorstore.as_retriever()

        # Prompt setup
        contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given chat history and user question, rewrite the question so it makes sense without chat history."),
            MessagesPlaceholder("chat-history"),
            ("human", "{input}")
        ])
        history_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert assistant. Use the following context to answer.\n\n{context}"),
            MessagesPlaceholder("chat-history"),
            ("human", "{input}")
        ])
        qa_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_retriever, qa_chain)

        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        conversational_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat-history",
            output_messages_key="answer"
        )

        
        # Display existing messages
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        # User input
        if user_query := st.chat_input("Ask me anything about your documents..."):
            st.session_state.messages.append({"role": "user", "content": user_query})
            st.chat_message("user").write(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = conversational_chain.invoke(
                        {"input": user_query},
                        config={"configurable": {"session_id": session_id}}
                    )
                    response = result["answer"]
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.info("Please enter your Groq API key in the sidebar to start chatting.")
