

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
from datetime import datetime
import base64

load_dotenv()

# Supabase connection
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Mark all records as inactive when a new chat starts
def mark_previous_records_inactive():
    supabase.table("pdf_embeddings").update({"status": "inactive"}).neq("status", "inactive").execute()

# Extract text from PDFs
def get_pdf_text(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

# Extract text from websites
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
    
    # Check for duplicates
    existing_entry = supabase.table("pdf_embeddings").select("pdf_hash").eq("pdf_hash", content_hash).execute()
    if existing_entry.data:
        supabase.table("pdf_embeddings").update({"status": "active"}).eq("pdf_hash", content_hash).execute()

        return f"Duplicate content detected for {source}. Skipping storage."

    # Store each chunk separately
    for chunk in text_chunks:
        response = genai.embed_content(model="models/embedding-001", content=chunk, task_type="retrieval_document")
        vector = response["embedding"]
        supabase.table("pdf_embeddings").insert({
            "text": chunk,
            "embedding": vector,
            "pdf_hash": content_hash,
            "source": source,
            "status": "active"
        }).execute()
    
    return f"New content stored successfully for {source}!"

# Search relevant chunks
def search_vectors(user_query):
    response = genai.embed_content(model="models/embedding-001", content=user_query, task_type="retrieval_query")
    if "embedding" in response:
        query_embedding = response["embedding"]
        response = supabase.rpc("match_documents", {
            "query_embedding": query_embedding, "match_threshold": 0.7, "match_count": 5
        }).execute()
        return [row["text"] for row in response.data] if response.data else []
    return []

# Conversational Chain
def get_conversational_chain():
    prompt = PromptTemplate(input_variables=["context", "question"],
                            template="Context: {context}\n\nQuestion: {question}\n\nAnswer:")
    model = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro-latest", google_api_key=GEMINI_API_KEY)
    return LLMChain(llm=model, prompt=prompt)

# Process user query
def user_input(user_question):
    if not st.session_state.is_active:
        return "No active content. Please upload a PDF or scrape a website first."

    relevant_chunks = search_vectors(user_question)
    chain = get_conversational_chain()
    # response = chain({"context": "\n".join(relevant_chunks), "question": user_question}, return_only_outputs=True)
    response = chain.invoke({"context": "\n".join(relevant_chunks), "question": user_question})

    answer = response.get("text", "No valid response")

    # Append user query and response to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

    return answer
 
# Streamlit UI
def main():
    st.set_page_config("Document & Web Scraper")
    st.header("Explore Documents & Websites")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "conversation_records" not in st.session_state:
        st.session_state.conversation_records = []
    if "current_source" not in st.session_state:
        st.session_state.current_source = None
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = None
    if "is_active" not in st.session_state:
        st.session_state.is_active = False

    st.sidebar.markdown(
        """
        <h3 style="text-align: center; font-weight: bold;">Chat History</h3>
        """, 
        unsafe_allow_html=True
    )

    # New Chat Button
    if st.sidebar.button("New Chat", use_container_width=True):
        mark_previous_records_inactive()  # Mark all old records as inactive
        if st.session_state.current_source:
            st.session_state.conversation_records.append({
                "source": st.session_state.current_source,
                "history": st.session_state.chat_history
            })
        st.session_state.chat_history = []
        st.session_state.current_source = None
        st.session_state.uploaded_files = None
        st.session_state.is_active = False  # Reset active session
        st.rerun()

    # Display chat history
    st.sidebar.header("Chat History")
    for idx, record in enumerate(st.session_state.conversation_records):
        if st.sidebar.button(f"{record['source']}", key=f"chat_{idx}", use_container_width=True):
            st.session_state.chat_history = record["history"]
            st.session_state.current_source = record["source"]
            st.session_state.is_active = True  # Reactivate session
            st.rerun()
    option = st.radio("Choose Input Type", ["Upload PDF", "Enter Website Domain"])

    if option == "Upload PDF":
        uploaded_files = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        if st.button("Submit PDF"):
            with st.spinner("Processing PDFs..."):
                for pdf in uploaded_files:
                    raw_text = get_pdf_text(pdf)
                    text_chunks = get_text_chunks(raw_text)
                    source = pdf.name.replace(".pdf", "")
                    store_vectors_in_db(text_chunks, source)
                
                st.session_state.is_active = True
                st.success("PDFs stored! You can now chat with them.")

    elif option == "Enter Website Domain":
        domain = st.text_input("Enter Website URL")
        if st.button("Scrape Website"):
            with st.spinner("Scraping website..."):
                landing_page_text = extract_landing_page_data(domain)
                text_chunks = get_text_chunks(landing_page_text)
                st.session_state.current_source = domain.split("//")[-1].split("/")[0]
                store_vectors_in_db(text_chunks, "Web Scraping")
                st.session_state.is_active = True
                st.success("Website content stored!")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input("Ask something about the stored content...")
    if user_query:
        st.chat_message("user").markdown(user_query)
        answer = user_input(user_query)
        st.chat_message("assistant").markdown(answer)

     # Function to display PDF in Streamlit
    # Directory to store uploaded PDFs
    # Function to display PDF
    def display_pdf(file):
        base64_pdf = base64.b64encode(file.read()).decode("utf-8")
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px"></iframe>'
        return pdf_display
            
   # Display PDFs in sidebar
    if uploaded_files:
        with st.sidebar:
            st.subheader("Select a PDF to View")
            selected_file = st.selectbox("Choose a file", uploaded_files, format_func=lambda x: x.name)

        # Show selected PDF
        if selected_file:
            st.markdown(f"### Viewing: {selected_file.name}")
            st.markdown(display_pdf(selected_file), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
