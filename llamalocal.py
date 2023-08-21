import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory

llm = CTransformers(
    model="llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    config={"max_new_tokens": 128, "temperature": 0.01},
)


loader = DirectoryLoader("data/", glob="*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cuda"}
)

vectorstore = FAISS.from_documents(docs, embeddings)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

chain = ConversationalRetrievalChain.from_llm(
    memory=memory,
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
)

st.title("Health is wealth")


def conversation_chat(query):
    result = chain({"question": query, "chat_history": st.session_state["history"]})
    st.session_state["history"].append(((query, result["answer"])))
    return result["answer"]


def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state["history"] = []

    if "generated" not in st.session_state:
        st.session_state["generated"] = ["Hello! Ask me anything about 🤗"]

    if "past" not in st.session_state:
        st.session_state["past"] = ["Hey! 👋"]


def display_chat_history():
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key="abc", clear_on_submit=True):
            user = st.text_input(
                "Query:", placeholder="Question about health", key="input"
            )
            submit = st.form_submit_button(label="go")

        if submit and user:
            output = conversation_chat(user)
            st.session_state["past"].append(user)
            st.session_state["generated"].append(output)

    if st.session_state["generated"]:
        with reply_container:
            for i in range(len(st.session_state["generated"])):
                message(
                    st.session_state["past"][i],
                    is_user=True,
                    key=str(i) + "_user",
                    avatar_style="thumbs",
                )
                message(
                    st.session_state["generated"][i],
                    key=str(i),
                    avatar_style="fun-emoji",
                )


# Initialize session state
initialize_session_state()
# Display chat history
display_chat_history()
