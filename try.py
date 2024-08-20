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
st.write("i am working on some updates, down for now" 
"""
#### PINECONE ####
pinecone_api_key = os.environ["PINECONE_API_KEY"]
pinecone_index_name = os.environ["PINECONE_INDEX_NAME"]

pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)

bm25_encoder = BM25Encoder().load("bm25_values.json")

nomic_embeddings = NomicEmbeddings(nomic_api_key=os.environ['NOMIC_API_KEY'], model='nomic-embed-text-v1.5', dimensionality=256)
retriever = PineconeHybridSearchRetriever(sparse_encoder=bm25_encoder, embeddings=nomic_embeddings, index=index)


#### LLM ####

llm = ChatGroq(
        api_key = os.environ['GROQ_API_KEY'], 
        model_name = "llama-3.1-70b-versatile",
        temperature = 0, 
    )


### Contextualize question ###
contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)


### Answer question ###
qa_system_prompt = """You work for University of Toronto's Information Commons Help Desk, Tier 1 IT support specialising in connecting and trouble shooting 
with UofT wireless network. 

Use the context: {context} as your knowledge base
to answer the question asked 
Make sure your answer is formatted properly, with proper line breaks use proper numerations and bullet points.  

You can also engage in fun conversation."""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


### Statefully manage chat history ###
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


import streamlit as st

st.set_page_config(page_title="STAN-bot v0.1", page_icon="👨‍💻")


st.title("STAN-bot v0.1")
st.subheader("Help Desk Tier 1 IT support")
st.write("Information Commons Help Desk")
st.write("Right now my knowledge base is limited to just Internet and Connectivity")

user_question =  st.chat_input("Ask a Question")


if "messages" not in st.session_state.keys():
    st.session_state["messages"] = [{"role": "assistant",
                                        "content": "Hello there, how can i help you?"}]

if "messages" in st.session_state.keys():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

if user_question is not None:
    st.session_state.messages.append({
        "role":"user",
        "content":user_question
    })

    with st.chat_message("user"):
        st.write(user_question)


if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Wait for it..."):
            query = str(user_question)
            output = conversational_rag_chain.invoke(
                {"input": query}, 
                config={'configurable': {'session_id': 'xyz123'}}
            )
            ai_response = output.get("answer")
            print(output.get("answer"))
            print(ai_response)
            st.write(f"{ai_response}")

    new_ai_message = {"role":"assistant","content": ai_response}
    st.session_state.messages.append(new_ai_message)"""
