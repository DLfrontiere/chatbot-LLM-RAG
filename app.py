# Import necessary libraries
import getpass 
import streamlit as st
import openai
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
import os
import tempfile
from loading import Loader
from embedding import EmbeddingModel
from langchain_community.document_loaders import PyPDFLoader
from answer_generation import AnswerGenerator
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
load_dotenv(Path("./api_key.env"))

# Set the title for the Streamlit app
st.title("RAG enhanced Chatbot")

# Set up the OpenAI API key from Streamlit input
api_key = st.text_input("Enter your OpenAI API Key:", type="password")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

accepted_files = ["pdf", "txt", "html","docx","doc"]

# Cached function to create a vectordb for the provided PDF files

@st.cache_data
def files_to_docs(files):

    with tempfile.TemporaryDirectory() as temp_dir:
        for file in files:
            file_path = os.path.join(temp_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            
        loader = Loader(temp_dir)
        docs = loader.load_documents(accepted_files)
        #docs = loader.summarize_docs(docs)
    
    return docs



@st.cache_data
def answer_gen(_docs):
    st.session_state['answer_generator'] = AnswerGenerator(EmbeddingModel(_docs).get_parent_retriever())


files = st.file_uploader("", type= accepted_files, accept_multiple_files=True)

if files:
    # Convert uploaded files to a list of file names and their sizes
    current_files_info = [(f.name, f.size) for f in files]

    # Get the previously processed files info from session state
    previous_files_info = st.session_state.get("previous_files_info", None)

    # Check if the current files are different from the previously processed ones
    if current_files_info != previous_files_info:
        processing_placeholder = st.empty()
        with processing_placeholder.container():
            with st.spinner(f"Processing {len(files)} file(s)..."):
                docs = files_to_docs(files)
                answer_gen(docs)
                st.session_state["vectordb"] = EmbeddingModel(docs).create_vector_store()
        processing_placeholder.empty()

        # Update the session state with the current files info
        st.session_state["previous_files_info"] = current_files_info


# Get the current prompt from the session state or set a default value
prompt = st.session_state.get("prompt", [{"role": "system", "content": "none"}])

# Get the user's question using Streamlit's chat input
question = st.chat_input("Ask anything")

# Handle the user's question
if question:

    for message in prompt:
        if message["role"] not in ["system"]:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    with st.chat_message("user"):
        st.write(question)

    
    answer_generator = st.session_state.get("answer_generator",None)

    if not answer_generator:
        with st.message("assistant"):
            st.write("You need to provide a PDF")
            st.stop()




    # Call ChatGPT with streaming and display the response as it comes
    with st.spinner(f"Retrieving answer.."):
        result = answer_generator.answer_prompt(question)

    with st.chat_message("assistant"):
        st.write(result)
    # Add the user's question to the prompt and display it
    prompt.append({"role": "user", "content": question})
    # Add the assistant's response to the prompt
    prompt.append({"role": "assistant", "content": result})


    
    # Store the updated prompt in the session state
    st.session_state["prompt"] = prompt

else:
    for message in prompt:
        if message["role"] not in ["system"]:
            with st.chat_message(message["role"]):
                st.write(message["content"])
