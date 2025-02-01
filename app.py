import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.schema import Document

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize session states
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "pdf_texts" not in st.session_state:
    st.session_state["pdf_texts"] = {}

if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = {}

# Function to process PDF files and extract text
def get_pdf_text(pdf_docs, step_tracker):
    step_tracker["Step 1"] = "Processing PDF Files"
    with st.spinner("Reading and extracting text from PDF files..."):
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            pdf_text = ""
            for page in pdf_reader.pages:
                pdf_text += page.extract_text()
            text += pdf_text  # Combine text from all PDFs into one
    step_tracker["Step 1"] = "PDF Files Processed"
    return text

# Function to split text into chunks
def get_text_chunks(text, step_tracker):
    step_tracker["Step 2"] = "Splitting Text into Chunks"
    with st.spinner("Splitting the extracted text into manageable chunks..."):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        chunks = text_splitter.split_text(text)
    step_tracker["Step 2"] = "Text Chunks Created"
    return chunks

# Function to create embeddings and store the vector store
def create_vector_store(pdf_text_chunks, step_tracker):
    step_tracker["Step 3"] = "Generating Embeddings"
    with st.spinner("Generating embeddings for the combined text..."):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(pdf_text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index_combined")  # Save vector store to disk
        st.session_state["vector_store"] = vector_store
    step_tracker["Step 3"] = "Embeddings Created"

# Function to create conversational chain for QA with focused prompt
def get_conversational_chain():
    prompt_template = """
    You are an AI bot specifically designed to assist with PDF document analysis.
    You can perform the following tasks:
    - Answer questions based on the content of uploaded PDF documents
    - Provide summaries of these documents

    Context:\n{context}\n
    Chat History:\n{history}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "history", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to convert text chunks into Document objects
def text_chunks_to_documents(chunks):
    return [Document(page_content=chunk) for chunk in chunks]

# Enhanced function to process user input with chat history and handle unsupported requests
def user_input(user_question):
    unsupported_requests = ["set a reminder", "play a game", "translate", "weather", "news"]

    # Check for unsupported requests
    if any(task in user_question.lower() for task in unsupported_requests):
        reply = "I'm here to help with questions based on the content of uploaded PDF documents. Unfortunately, I cannot assist with tasks like setting reminders, playing games, translating, or providing weather updates."
        st.session_state["chat_history"].append({"user": user_question, "bot": reply})
    else:
        # Proceed with the existing flow if the question is supported
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Load the combined vector store from session state
        vector_store = st.session_state["vector_store"]
        if vector_store is None:
            st.error("Vector store not found. Please re-upload and process the PDFs.")
            return
        
        # Retrieve relevant documents based on user question (Improved Search)
        docs = vector_store.similarity_search(user_question, k=5)  # Retrieve more results for better coverage
        
        docs = text_chunks_to_documents([doc.page_content for doc in docs])

        # Create a chat history string for context
        chat_history = "\n".join([f"User: {entry['user']}\nBot: {entry['bot']}" for entry in st.session_state["chat_history"]])

        # Process the question with the conversational chain
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "history": chat_history, "question": user_question}, return_only_outputs=True)

        # Add the response to chat history and display it
        reply = response["output_text"]
        st.session_state["chat_history"].append({"user": user_question, "bot": reply})

    # Display chat history in reverse order, latest at the top with a line separator
    for chat in reversed(st.session_state["chat_history"]):
        st.markdown(f"**🧔🏻‍♂️User:** {chat['user']}")
        st.markdown(f"**🤖Bot:**<br>{chat['bot']}", unsafe_allow_html=True)
        st.markdown("---")  # Add a line separator after each conversation

# Main app function with chat-like interface
def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF📝")

    # Initialize step tracker
    step_tracker = {
        "Step 1": "Not Started",
        "Step 2": "Not Started",
        "Step 3": "Not Started"
    }

    # Sidebar for file upload and processing
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        status_bar = st.empty()

        if st.button("Submit & Process"):
            if not pdf_docs:
                st.error("No files uploaded! Please upload a PDF to continue.")
                return

            # Store PDF files in session state to retain the uploaded files
            st.session_state["pdf_docs"] = pdf_docs

            status_bar.progress(0, text="Step 1: Processing PDF Files...")
            raw_text = get_pdf_text(pdf_docs, step_tracker)
            st.session_state["pdf_texts"] = raw_text  # Store PDF texts in session state
            status_bar.progress(33, text="Step 1 Completed: PDF Files Processed")

            pdf_text_chunks = get_text_chunks(raw_text, step_tracker)
            status_bar.progress(66, text="Step 2 Completed: Text Chunks Created")

            create_vector_store(pdf_text_chunks, step_tracker)
            status_bar.progress(100, text="Step 3 Completed: Embeddings Created & Stored")
            st.success("Processing Completed!")
            status_bar.empty()

        st.subheader("Processing Status:")
        for step, status in step_tracker.items():
            st.write(f"{step}: {status}")

    # Chat input for interacting with the bot
    user_question = st.text_input("Type your question here...")

    # Handle user question input
    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()
