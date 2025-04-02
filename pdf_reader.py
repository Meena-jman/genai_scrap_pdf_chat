# import os
# import streamlit as st
# import hashlib
# import time
# import google.generativeai as genai
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.prompts import PromptTemplate
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains import LLMChain
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from supabase import create_client, Client
# from dotenv import load_dotenv

# load_dotenv()

# # Supabase connection
# SUPABASE_URL = os.getenv("SUPABASE_URL")
# SUPABASE_KEY = os.getenv("SUPABASE_KEY")
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# genai.configure(api_key=GEMINI_API_KEY)

# supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# # Text extraction from PDFs
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text() or ""
#     return text

# # Extract text only from the landing page
# def extract_landing_page_data(url):
#     driver = webdriver.Chrome()
#     driver.get(url)
#     time.sleep(3)
#     try:
#         page_text = driver.find_element(By.TAG_NAME, "body").text
#     except:
#         page_text = "Could not extract data."
#     driver.quit()
#     return page_text

# # Split text into chunks
# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     return text_splitter.split_text(text)

# # Generate hash for deduplication
# def generate_hash(text):
#     return hashlib.sha256(text.encode()).hexdigest()

# # Store text chunks in Supabase
# def store_vectors_in_db(text_chunks, source):
#     full_text = "\n".join(text_chunks)
#     content_hash = generate_hash(full_text)
#     existing_entry = supabase.table("pdf_embeddings").select("pdf_hash").eq("pdf_hash", content_hash).execute()
#     if existing_entry.data:
#         print("Duplicate content detected. Skipping storage.")
#         return

#     for chunk in text_chunks:
#         response = genai.embed_content(model="models/embedding-001", content=chunk, task_type="retrieval_document")
#         vector = response["embedding"]
#         supabase.table("pdf_embeddings").insert({"text": chunk, "embedding": vector, "pdf_hash": content_hash, "source": source}).execute()

#     print("New content stored successfully!")

# # Search relevant chunks
# def search_vectors(user_query):
#     response = genai.embed_content(model="models/embedding-001", content=user_query, task_type="retrieval_query")
#     if "embedding" in response:
#         query_embedding = response["embedding"]
#         response = supabase.rpc("match_documents", {"query_embedding": query_embedding, "match_threshold": 0.7, "match_count": 5}).execute()
#         return [row["text"] for row in response.data] if response.data else []
#     return []

# # Conversational Chain using LangChain
# def get_conversational_chain():
#     prompt = PromptTemplate(
#         input_variables=["context", "question"],
#         template="Context: {context}\n\nQuestion: {question}\n\nAnswer:"
#     )
#     model = ChatGoogleGenerativeAI(
#         model="models/gemini-1.5-pro-latest",
#         google_api_key=GEMINI_API_KEY
#     )
#     return LLMChain(llm=model, prompt=prompt)

# # Process user queries
# def user_input(user_question):
#     relevant_chunks = search_vectors(user_question)
#     chain = get_conversational_chain()
#     response = chain({"context": "\n".join(relevant_chunks), "question": user_question}, return_only_outputs=True)
#     st.write("Reply: ", response.get("text", "No valid response"))

# # Streamlit UI
# def main():
#     st.set_page_config("Document & Web Scraper")
#     st.header("Explore Documents & Websites")
    
#     option = st.radio("Choose Input Type", ["Upload PDF", "Enter Website Domain"])
#     content_available = False
    
#     if option == "Upload PDF":
#         pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
#         if st.button("Submit PDF"):
#             with st.spinner("Processing PDF..."):
#                 raw_text = get_pdf_text(pdf_docs)
#                 text_chunks = get_text_chunks(raw_text)
#                 store_vectors_in_db(text_chunks, "PDF Upload")
#                 content_available = True
#                 st.success("PDF content stored!")
    
#     elif option == "Enter Website Domain":
#         domain = st.text_input("Enter Website URL")
#         if st.button("Scrape Website"):
#             with st.spinner("Scraping website..."):
#                 landing_page_text = extract_landing_page_data(domain)
#                 text_chunks = get_text_chunks(landing_page_text)
#                 store_vectors_in_db(text_chunks, "Web Scraping")
#                 content_available = True
#                 st.success("Website content stored!")
    
#     # Allow user to ask questions only if content is stored
#     user_question = st.text_input("Ask a question about the stored content", disabled=not content_available)
#     if user_question:
#         user_input(user_question)

# if __name__ == "__main__":
#     main()


import os
import streamlit as st
import hashlib
import time
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from selenium import webdriver
from selenium.webdriver.common.by import By
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# Supabase connection
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Text extraction from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

# Extract text only from the landing page
def extract_landing_page_data(url):
    driver = webdriver.Chrome()
    driver.get(url)
    time.sleep(3)
    try:
        page_text = driver.find_element(By.TAG_NAME, "body").text
    except:
        page_text = "Could not extract data."
    driver.quit()
    return page_text

# Split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# Generate hash for deduplication
def generate_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()

# Store text chunks in Supabase
def store_vectors_in_db(text_chunks, source):
    full_text = "\n".join(text_chunks)
    content_hash = generate_hash(full_text)
    existing_entry = supabase.table("pdf_embeddings").select("pdf_hash").eq("pdf_hash", content_hash).execute()
    if existing_entry.data:
        print("Duplicate content detected. Skipping storage.")
        return

    for chunk in text_chunks:
        response = genai.embed_content(model="models/embedding-001", content=chunk, task_type="retrieval_document")
        vector = response["embedding"]
        supabase.table("pdf_embeddings").insert({"text": chunk, "embedding": vector, "pdf_hash": content_hash, "source": source}).execute()

    print("New content stored successfully!")

# Search relevant chunks
def search_vectors(user_query):
    response = genai.embed_content(model="models/embedding-001", content=user_query, task_type="retrieval_query")
    if "embedding" in response:
        query_embedding = response["embedding"]
        response = supabase.rpc("match_documents", {"query_embedding": query_embedding, "match_threshold": 0.7, "match_count": 5}).execute()
        return [row["text"] for row in response.data] if response.data else []
    return []

# Conversational Chain using LangChain
def get_conversational_chain():
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    )
    model = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-pro-latest",
        google_api_key=GEMINI_API_KEY
    )
    return LLMChain(llm=model, prompt=prompt)

# Process user queries
def user_input(user_question):
    relevant_chunks = search_vectors(user_question)
    chain = get_conversational_chain()
    response = chain({"context": "\n".join(relevant_chunks), "question": user_question}, return_only_outputs=True)
    st.session_state.pdf_chat_history.append({"role": "assistant", "content": response.get("text", "No valid response")})
    st.chat_message("assistant").markdown(response.get("text", "No valid response"))

# Streamlit UI
def main():
    st.set_page_config("Document & Web Scraper")
    st.header("Explore Documents & Websites")
    
    if "pdf_chat_history" not in st.session_state:
        st.session_state.pdf_chat_history = []
    if "pdf_hash" not in st.session_state:
        st.session_state.pdf_hash = None
    
    option = st.radio("Choose Input Type", ["Upload PDF", "Enter Website Domain"])
    content_available = False
    
    if option == "Upload PDF":
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        if st.button("Submit PDF"):
            with st.spinner("Processing PDF..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                st.session_state.pdf_hash = generate_hash("\n".join(text_chunks))
                store_vectors_in_db(text_chunks, "PDF Upload")
                content_available = True
                st.success("PDF content stored! You can now chat with it.")
    
    elif option == "Enter Website Domain":
        domain = st.text_input("Enter Website URL")
        if st.button("Scrape Website"):
            with st.spinner("Scraping website..."):
                landing_page_text = extract_landing_page_data(domain)
                text_chunks = get_text_chunks(landing_page_text)
                store_vectors_in_db(text_chunks, "Web Scraping")
                content_available = True
                st.success("Website content stored!")
    
    # Sidebar for chat history
    with st.sidebar:
        st.header("Chat History")
        if st.button("New Chat"):
            st.session_state.pdf_chat_history = []
        for message in st.session_state.pdf_chat_history:
            st.text(f"{message['role'].capitalize()}: {message['content'][:50]}...")
    
    # Chat functionality
    for message in st.session_state.pdf_chat_history:
        with st.chat_message(message["role"]):
            st.markdown(f"<div style='text-align: left;'>{message['content']}</div>", unsafe_allow_html=True)
    
    user_query = st.chat_input("Ask something about the stored content...", disabled=not content_available)
    
    if user_query:
        st.session_state.pdf_chat_history.append({"role": "user", "content": user_query})
        st.chat_message("user").markdown(f"<div style='text-align: right;'>{user_query}</div>", unsafe_allow_html=True)
        user_input(user_query)

if __name__ == "__main__":
    main()
