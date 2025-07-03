import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

from dotenv import load_dotenv
load_dotenv()

os.environ['HF_TOKEN']=os.getenv('HF_TOKEN')
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# streamlit 
st.title("Conversational Rag With PDF upload")
api_key=st.text_input("Enter Your Groq Api Key",type="password")

if api_key:
    llm=ChatGroq(groq_api_key=api_key,model="gemma2-9b-it")
    # chat interface

    session_id=st.text_input("Session_id",value="default_session")

    # state management

    if 'store' not in st.session_state:
        st.session_state.store={} 

    upload_files=st.file_uploader("Chose a pdf file", type="pdf",accept_multiple_files=True) 

    #Process upload files
    if upload_files:
        document=[]
        for upload_file in upload_files:
            temppdf=f"./temp.pdf"
            with open(temppdf,"wb") as file:
                file.write(upload_file.getvalue())
                file_name=upload_file.name
            
            loader=PyPDFLoader(temppdf)
            docs=loader.load()
            document.extend(docs)
        
        # spit the documents
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=500)
        splits=text_splitter.split_documents(document)
        vectorstore=Chroma.from_documents(splits,embeddings)
        retriever=vectorstore.as_retriever()
    
        contextialize_q_system_prompt=(
            "Given a chat history and the latest user question "
            "which might reference context in the chat history"
            "formulate a standalone question which can be understood"
            "without the chat history .Do not answer the question"
            "just reformulate it if needed and otherwise return as it is"
        )

        contextialize_q_prompt=ChatPromptTemplate.from_messages(
            [
                ("system",contextialize_q_system_prompt),
                MessagesPlaceholder("chat-history"),
                ("human","{input}")
            ]

        )

        history_aware_retriever=create_history_aware_retriever(llm,retriever,contextialize_q_prompt)

        # Answer question prompt

        qa_system_prompt=(
            "You are assistant for question-answering task"
            "Use the following pieces of retrieved context to answer "
            " the question .If you don't know the answer ,say that you don't know "
            " Use three sentence maximum and keep the answer concise"
            "{context}"
        )

        qa_prompt=ChatPromptTemplate.from_messages(
            [
                ("system",qa_system_prompt),
                MessagesPlaceholder('chat-history'),
                ("human","{input}")
            ]
        )
        question_ans_chain=create_stuff_documents_chain(llm,qa_prompt)
        rag_chain=create_retrieval_chain(history_aware_retriever,question_ans_chain)

        def get_session_history(session_id:str):
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,input_messages_key="input",
            history_messages_key="chat-history",
            output_messages_key="answer"
        )

        user_input=st.text_input("Your Question")
        if user_input:
            session_history=get_session_history(session_id)
            response=conversational_rag_chain.invoke(
                {"input":user_input},
                config={
                    "configurable":{"session_id":session_id}
                }
            )
            st.write(st.session_state.store)
            st.success("Assistant:"+response['answer'])
            st.write('ChatHistory',session_history.messages)

else :
    st.warning("Please provide Groq Api key")
            


