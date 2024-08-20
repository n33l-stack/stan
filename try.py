import os 
import nltk
from dotenv import load_dotenv

from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder

from langchain_nomic import NomicEmbeddings
from langchain_community.retrievers import PineconeHybridSearchRetriever

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# from langchain.memory import ConversationBufferMemory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.chat_history import BaseChatMessageHistory

from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_core.runnables.history import RunnableWithMessageHistory

import streamlit as st 

load_dotenv()

st.set_page_config(page_title="STAN-bot v0.1", page_icon="👨‍💻")
st.write("i am working on some updates, down for now") 
