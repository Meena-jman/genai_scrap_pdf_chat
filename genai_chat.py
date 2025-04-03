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

# Extracting text(PDF)
def get_pdf_text(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

# Extracting text(website)
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

# chunks generation
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# Hash generation
def generate_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()

# Chunks into Supabase
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

# Searching relevant chunks
def search_vectors(user_query):
    response = genai.embed_content(model="models/embedding-001", content=user_query, task_type="retrieval_query")
    if "embedding" in response:
        query_embedding = response["embedding"]
        response = supabase.rpc("match_documents", {
            "query_embedding": query_embedding, "match_threshold": 0.7, "match_count": 3
        }).execute()
        print(response.data)
        search_results = [row["text"] for row in response.data] if response.data  else []
        
        st.subheader("Sources")
        with st.expander("View souce"):
            for result in search_results:
                st.markdown(f"<div style='border: 1px solid #ccc; padding: 10px; margin: 5px; border-radius: 5px;'>{result}</div>", unsafe_allow_html=True)

        return [row["text"] for row in response.data] if response.data else []
    return []


def get_conversational_chain():
    prompt = PromptTemplate(input_variables=["context", "question"],
                            template="Answer only from these context. Answer within this, if not found then return as not found, Context: {context}\n\nQuestion: {question}\n\nAnswer:")
    model = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro-latest", google_api_key=GEMINI_API_KEY)
    return LLMChain(llm=model, prompt=prompt)

def user_input(user_question):
    if not st.session_state.is_active:
        return "No active content. Please upload a PDF or scrape a website first."

    relevant_chunks = search_vectors(user_question)
    chain = get_conversational_chain()
    response = chain.invoke({"context": "\n".join(relevant_chunks), "question": user_question})

    answer = response.get("text", "No valid response")

    st.session_state.chat_history.append({"role": "user", "content": user_question})
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

    return answer

def main():
    st.set_page_config("Document & Web Scraper")
    st.header("Explore Documents & Websites")

    # Initialize session state variables
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
    if "scraped_url" not in st.session_state:
        st.session_state.scraped_url = None
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = None

    st.sidebar.markdown("<h3 style='text-align: center; font-weight: bold;'>M Chat</h3>", unsafe_allow_html=True)
    
    if st.sidebar.button("New Chat", use_container_width=True):
        if st.session_state.current_source and st.session_state.chat_history:
            if not st.session_state.current_chat_id:
                st.session_state.current_chat_id = str(int(time.time()))
            
            existing_record = next((r for r in st.session_state.conversation_records 
                                  if r.get("chat_id") == st.session_state.current_chat_id), None)
            
            if existing_record:
                existing_record["history"] = st.session_state.chat_history.copy()
            else:
                st.session_state.conversation_records.append({
                    "chat_id": st.session_state.current_chat_id,
                    "source": st.session_state.current_source,
                    "history": st.session_state.chat_history.copy(),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
                })
        
        mark_previous_records_inactive()
        st.session_state.chat_history = []
        st.session_state.current_source = None
        st.session_state.uploaded_files = None
        st.session_state.is_active = False
        st.session_state.scraped_url = None
        st.session_state.current_chat_id = None  
        st.rerun()

    st.sidebar.subheader("Chat History")
    if st.session_state.conversation_records:
        sorted_records = sorted(st.session_state.conversation_records, 
                              key=lambda x: x.get("timestamp", ""), 
                              reverse=True)
        
        for record in sorted_records:
            display_name = f"{record['source']} - {record.get('timestamp', '')}"
            
            if st.sidebar.button(display_name, key=f"chat_{record['chat_id']}", use_container_width=True):
                st.session_state.chat_history = record["history"]
                st.session_state.current_source = record["source"]
                st.session_state.is_active = True
                st.session_state.current_chat_id = record["chat_id"]
                st.rerun()
    else:
        st.sidebar.write("No previous chats")

    option = st.radio("Choose Input Type", ["Upload PDF", "Enter Website Domain"])

    if option == "Upload PDF":
        fname=""
        uploaded_files = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        if st.button("Submit PDF"):
            with st.spinner("Processing PDFs..."):
                for pdf in uploaded_files:
                    fname+=pdf.name.replace(".pdf", "")+" "
                    raw_text = get_pdf_text(pdf)
                    text_chunks = get_text_chunks(raw_text)
                    source = pdf.name.replace(".pdf", "")
                    store_vectors_in_db(text_chunks, source)
                
                st.session_state.uploaded_files = uploaded_files
                st.session_state.is_active = True
                # st.session_state.current_source =uploaded_files[0].name.replace(".pdf", "")
                st.session_state.current_source =fname
                st.session_state.current_chat_id = str(int(time.time()))  # Generate new chat ID
                st.success("PDFs stored! You can now chat with them.")

        if st.session_state.uploaded_files:
            with st.sidebar:
                st.subheader("Select a PDF to View")
                selected_file = st.selectbox("Choose a file", st.session_state.uploaded_files, 
                                           format_func=lambda x: x.name)

            if selected_file:
                st.markdown(f"### Viewing: {selected_file.name}")
                selected_file.seek(0)
                base64_pdf = base64.b64encode(selected_file.read()).decode("utf-8")
                st.markdown(f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px"></iframe>', 
                           unsafe_allow_html=True)

    elif option == "Enter Website Domain":
        domain = st.text_input("Enter Website URL")
        if st.button("Scrape Website"):
            with st.spinner("Scraping website..."):
                landing_page_text = extract_landing_page_data(domain)
                text_chunks = get_text_chunks(landing_page_text)
                st.session_state.current_source = domain
                st.session_state.scraped_url = domain
                st.session_state.current_chat_id = str(int(time.time()))  
                store_vectors_in_db(text_chunks, "Web Scraping")
                st.session_state.is_active = True
                st.success("Website content stored!")

        if st.session_state.scraped_url:
            st.markdown(f"### View: {st.session_state.scraped_url}")
     
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input("Ask something about the stored content...")
    if user_query:
        st.chat_message("user").markdown(user_query)
        answer = user_input(user_query)
        st.chat_message("assistant").markdown(answer)

if __name__ == "__main__":
    main()