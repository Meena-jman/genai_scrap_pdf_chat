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

# # Extract text from PDFs
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text() or ""
#     return text

# # Extract text from websites
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
#         return "Duplicate content detected. Skipping storage."

#     for chunk in text_chunks:
#         response = genai.embed_content(model="models/embedding-001", content=chunk, task_type="retrieval_document")
#         vector = response["embedding"]
#         supabase.table("pdf_embeddings").insert({"text": chunk, "embedding": vector, "pdf_hash": content_hash, "source": source}).execute()
#     return "New content stored successfully!"

# # Search relevant chunks
# def search_vectors(user_query):
#     response = genai.embed_content(model="models/embedding-001", content=user_query, task_type="retrieval_query")
#     if "embedding" in response:
#         query_embedding = response["embedding"]
#         response = supabase.rpc("match_documents", {"query_embedding": query_embedding, "match_threshold": 0.7, "match_count": 5}).execute()
#         return [row["text"] for row in response.data] if response.data else []
#     return []

# # Conversational Chain
# def get_conversational_chain():
#     prompt = PromptTemplate(input_variables=["context", "question"],
#                             template="Context: {context}\n\nQuestion: {question}\n\nAnswer:")
#     model = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro-latest", google_api_key=GEMINI_API_KEY)
#     return LLMChain(llm=model, prompt=prompt)

# # Process user query
# def user_input(user_question):
#     relevant_chunks = search_vectors(user_question)
#     chain = get_conversational_chain()
#     response = chain({"context": "\n".join(relevant_chunks), "question": user_question}, return_only_outputs=True)
#     answer = response.get("text", "No valid response")
    
#     # Append user query and response to chat history
#     st.session_state.chat_history.append({"role": "user", "content": user_question})
#     st.session_state.chat_history.append({"role": "assistant", "content": answer})
    
#     return answer

# # Streamlit UI
# def main():
#     st.set_page_config("Document & Web Scraper")
#     st.header("Explore Documents & Websites")

#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = []
#     if "conversation_records" not in st.session_state:
#         st.session_state.conversation_records = []
#     if "current_source" not in st.session_state:
#         st.session_state.current_source = None
#     if "uploaded_files" not in st.session_state:
#         st.session_state.uploaded_files = None

#     if st.sidebar.button("New Chat"):
#         if st.session_state.current_source:
#             st.session_state.conversation_records.append({
#                 "source": st.session_state.current_source,
#                 "history": st.session_state.chat_history
#             })
#         st.session_state.chat_history = []
#         st.session_state.current_source = None
#         st.session_state.uploaded_files = None  # Clears uploaded files
#         st.rerun()

#     st.sidebar.header("Chat History")
#     for idx, record in enumerate(st.session_state.conversation_records):
#         if st.sidebar.button(record["source"], key=f"chat_{idx}"):
#             st.session_state.chat_history = record["history"]
#             st.session_state.current_source = record["source"]
#             st.rerun()

#     option = st.radio("Choose Input Type", ["Upload PDF", "Enter Website Domain"])
#     content_available = False

#     if option == "Upload PDF":
#         st.session_state.uploaded_files = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
#         if st.button("Submit PDF"):
#             with st.spinner("Processing PDF..."):
#                 raw_text = get_pdf_text(st.session_state.uploaded_files)
#                 text_chunks = get_text_chunks(raw_text)
#                 source = ", ".join([pdf.name.replace(".pdf", "") for pdf in st.session_state.uploaded_files])
#                 st.session_state.current_source = source
#                 store_vectors_in_db(text_chunks, "PDF Upload")
#                 content_available = True
#                 st.success("PDF content stored! You can now chat with it.")

#     elif option == "Enter Website Domain":
#         domain = st.text_input("Enter Website URL")
#         if st.button("Scrape Website"):
#             with st.spinner("Scraping website..."):
#                 landing_page_text = extract_landing_page_data(domain)
#                 text_chunks = get_text_chunks(landing_page_text)
#                 st.session_state.current_source = domain.split("//")[-1].split("/")[0]
#                 store_vectors_in_db(text_chunks, "Web Scraping")
#                 content_available = True
#                 st.success("Website content stored!")

#     for msg in st.session_state.chat_history:
#         with st.chat_message(msg["role"]):
#             st.markdown(msg["content"])

#     user_query = st.chat_input("Ask something about the stored content...")
#     if user_query:
#         st.chat_message("user").markdown(user_query)
#         answer = user_input(user_query)
#         st.chat_message("assistant").markdown(answer)

# if __name__ == "__main__":
#     main()


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

# # Extract text from PDFs
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text() or ""
#     return text

# # Extract text from websites
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

# # Store text chunks in Supabase with chunk-wise insertion
# def store_vectors_in_db(text_chunks, source):
#     full_text = "\n".join(text_chunks)
#     content_hash = generate_hash(full_text)
    
#     # Check for duplicate entry
#     existing_entry = supabase.table("pdf_embeddings").select("pdf_hash").eq("pdf_hash", content_hash).execute()
#     if existing_entry.data:
#         return "Duplicate content detected. Skipping storage."

#     # Store each chunk separately
#     for chunk in text_chunks:
#         response = genai.embed_content(model="models/embedding-001", content=chunk, task_type="retrieval_document")
#         vector = response["embedding"]
        
#         # Insert each chunk as a separate row
#         supabase.table("pdf_embeddings").insert({
#             "text": chunk,
#             "embedding": vector,
#             "pdf_hash": content_hash,
#             "source": source
#         }).execute()

#     return "New content stored successfully!"

# # Search relevant chunks
# def search_vectors(user_query):
#     response = genai.embed_content(model="models/embedding-001", content=user_query, task_type="retrieval_query")
#     if "embedding" in response:
#         query_embedding = response["embedding"]
#         response = supabase.rpc("match_documents", {
#             "query_embedding": query_embedding,
#             "match_threshold": 0.7,
#             "match_count": 5
#         }).execute()
#         return [row["text"] for row in response.data] if response.data else []
#     return []

# # Conversational Chain
# def get_conversational_chain():
#     prompt = PromptTemplate(
#         input_variables=["context", "question"],
#         template="Context: {context}\n\nQuestion: {question}\n\nAnswer:"
#     )
#     model = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro-latest", google_api_key=GEMINI_API_KEY)
#     return LLMChain(llm=model, prompt=prompt)

# # Process user query
# def user_input(user_question):
#     relevant_chunks = search_vectors(user_question)
#     chain = get_conversational_chain()
#     response = chain({
#         "context": "\n".join(relevant_chunks),
#         "question": user_question
#     }, return_only_outputs=True)
#     answer = response.get("text", "No valid response")

#     # Append user query and response to chat history
#     st.session_state.chat_history.append({"role": "user", "content": user_question})
#     st.session_state.chat_history.append({"role": "assistant", "content": answer})
    
#     return answer

# # Streamlit UI
# def main():
#     st.set_page_config("Document & Web Scraper")
#     st.header("Explore Documents & Websites")

#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = []
#     if "conversation_records" not in st.session_state:
#         st.session_state.conversation_records = []
#     if "current_source" not in st.session_state:
#         st.session_state.current_source = None
#     if "uploaded_files" not in st.session_state:
#         st.session_state.uploaded_files = None

#     # New Chat Button: Clears stored vectors from DB
#     if st.sidebar.button("New Chat"):
#         if st.session_state.current_source:
#             # Delete previous source data from database
#             supabase.table("pdf_embeddings").delete().eq("source", st.session_state.current_source).execute()
            
#             # Store the previous conversation before resetting
#             st.session_state.conversation_records.append({
#                 "source": st.session_state.current_source,
#                 "history": st.session_state.chat_history
#             })
        
#         # Reset the session states
#         st.session_state.chat_history = []
#         st.session_state.current_source = None
#         st.session_state.uploaded_files = None  
#         st.rerun()

#     st.sidebar.header("Chat History")
#     for idx, record in enumerate(st.session_state.conversation_records):
#         if st.sidebar.button(record["source"], key=f"chat_{idx}"):
#             st.session_state.chat_history = record["history"]
#             st.session_state.current_source = record["source"]
#             st.rerun()

#     option = st.radio("Choose Input Type", ["Upload PDF", "Enter Website Domain"])
#     content_available = False

#     if option == "Upload PDF":
#         st.session_state.uploaded_files = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
#         if st.button("Submit PDF"):
#             with st.spinner("Processing PDF..."):
#                 raw_text = get_pdf_text(st.session_state.uploaded_files)
#                 text_chunks = get_text_chunks(raw_text)
#                 source = ", ".join([pdf.name.replace(".pdf", "") for pdf in st.session_state.uploaded_files])
#                 st.session_state.current_source = source
#                 store_vectors_in_db(text_chunks, "PDF Upload")
#                 content_available = True
#                 st.success("PDF content stored! You can now chat with it.")

#     elif option == "Enter Website Domain":
#         domain = st.text_input("Enter Website URL")
#         if st.button("Scrape Website"):
#             with st.spinner("Scraping website..."):
#                 landing_page_text = extract_landing_page_data(domain)
#                 text_chunks = get_text_chunks(landing_page_text)
#                 st.session_state.current_source = domain.split("//")[-1].split("/")[0]
#                 store_vectors_in_db(text_chunks, "Web Scraping")
#                 content_available = True
#                 st.success("Website content stored!")

#     for msg in st.session_state.chat_history:
#         with st.chat_message(msg["role"]):
#             st.markdown(msg["content"])

#     user_query = st.chat_input("Ask something about the stored content...")
#     if user_query:
#         st.chat_message("user").markdown(user_query)
#         answer = user_input(user_query)
#         st.chat_message("assistant").markdown(answer)

# if __name__ == "__main__":
#     main()




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

# # Extract text from PDFs
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text() or ""
#     return text

# # Extract text from websites
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

# # Store text chunks in Supabase with chunk-wise insertion
# def store_vectors_in_db(text_chunks, source):
#     full_text = "\n".join(text_chunks)
#     content_hash = generate_hash(full_text)
    
#     # Check if this content already exists
#     existing_entry = supabase.table("pdf_embeddings").select("pdf_hash").eq("pdf_hash", content_hash).execute()
#     if existing_entry.data:
#         supabase.table("pdf_embeddings").update({"status": "active"}).eq("pdf_hash", content_hash).execute()
#         return "Duplicate content detected. Skipping storage."

#     for chunk in text_chunks:
#         response = genai.embed_content(model="models/embedding-001", content=chunk, task_type="retrieval_document")
#         vector = response["embedding"]

#         # Store with "active" status
#         supabase.table("pdf_embeddings").insert({
#             "text": chunk,
#             "embedding": vector,
#             "pdf_hash": content_hash,
#             "source": source,
#             "status": "active"  # New column added
#         }).execute()
    
#     return "New content stored successfully!"


# # Search relevant chunks
# def search_vectors(user_query):
#     response = genai.embed_content(model="models/embedding-001", content=user_query, task_type="retrieval_query")
#     if "embedding" in response:
#         query_embedding = response["embedding"]
#         response = supabase.rpc("match_documents", {
#             "query_embedding": query_embedding,
#             "match_threshold": 0.7,
#             "match_count": 5
#         }).execute()
#         return [row["text"] for row in response.data] if response.data else []
#     return []

# # Conversational Chain
# def get_conversational_chain():
#     prompt = PromptTemplate(
#         input_variables=["context", "question"],
#         template="For the uploaded context alone provide me the answer. Use only the provided content. Context: {context}\n\nQuestion: {question}\n\nAnswer:"
#     )
#     model = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro-latest", google_api_key=GEMINI_API_KEY)
#     return LLMChain(llm=model, prompt=prompt)

# # Process user query
# def user_input(user_question):
#     relevant_chunks = search_vectors(user_question)
#     chain = get_conversational_chain()
#     response = chain({
#         "context": "\n".join(relevant_chunks),
#         "question": user_question
#     }, return_only_outputs=True)
#     answer = response.get("text", "No valid response")

#     # Append user query and response to chat history
#     st.session_state.chat_history.append({"role": "user", "content": user_question})
#     st.session_state.chat_history.append({"role": "assistant", "content": answer})
    
#     return answer

# # Streamlit UI
# def main():
#     st.set_page_config("Document & Web Scraper")
#     st.header("Explore Documents & Websites")

#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = []
#     if "conversation_records" not in st.session_state:
#         st.session_state.conversation_records = []
#     if "current_source" not in st.session_state:
#         st.session_state.current_source = None
#     if "uploaded_files" not in st.session_state:
#         st.session_state.uploaded_files = None

#     # New Chat Button: Clears stored vectors from DB
#     if st.sidebar.button("New Chat"):
#     # Update all previous records to "inactive"
#         supabase.table("pdf_embeddings").update({"status": "inactive"}).neq("status", "inactive").execute()

#         # Store the previous conversation before resetting
#         if st.session_state.current_source:
#             st.session_state.conversation_records.append({
#                 "source": st.session_state.current_source,
#                 "history": st.session_state.chat_history
#             })

#         # Reset session states
#         st.session_state.chat_history = []
#         st.session_state.current_source = None
#         st.session_state.uploaded_files = None  
#         st.rerun()


#     st.sidebar.header("Chat History")
#     for idx, record in enumerate(st.session_state.conversation_records):
#         if st.sidebar.button(record["source"], key=f"chat_{idx}"):
#             st.session_state.chat_history = record["history"]
#             st.session_state.current_source = record["source"]
#             st.rerun()

#     option = st.radio("Choose Input Type", ["Upload PDF", "Enter Website Domain"])
#     content_available = False

#     if option == "Upload PDF":
#         st.session_state.uploaded_files = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
#         if st.button("Submit PDF"):
#             with st.spinner("Processing PDF..."):
#                 raw_text = get_pdf_text(st.session_state.uploaded_files)
#                 text_chunks = get_text_chunks(raw_text)
#                 source = ", ".join([pdf.name.replace(".pdf", "") for pdf in st.session_state.uploaded_files])
#                 st.session_state.current_source = source
#                 store_vectors_in_db(text_chunks, "PDF Upload")
#                 content_available = True
#                 st.success("PDF content stored! You can now chat with it.")

#     elif option == "Enter Website Domain":
#         domain = st.text_input("Enter Website URL")
#         if st.button("Scrape Website"):
#             with st.spinner("Scraping website..."):
#                 landing_page_text = extract_landing_page_data(domain)
#                 text_chunks = get_text_chunks(landing_page_text)
#                 st.session_state.current_source = domain.split("//")[-1].split("/")[0]
#                 store_vectors_in_db(text_chunks, "Web Scraping")
#                 content_available = True
#                 st.success("Website content stored!")

#     for msg in st.session_state.chat_history:
#         with st.chat_message(msg["role"]):
#             st.markdown(msg["content"])

#     user_query = st.chat_input("Ask something about the stored content...")
#     if user_query:
#         st.chat_message("user").markdown(user_query)
#         answer = user_input(user_query)
#         st.chat_message("assistant").markdown(answer)

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

# Extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
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
    existing_entry = supabase.table("pdf_embeddings").select("pdf_hash").eq("pdf_hash", content_hash).execute()
    if existing_entry.data:
        return "Duplicate content detected. Skipping storage."

    for chunk in text_chunks:
        response = genai.embed_content(model="models/embedding-001", content=chunk, task_type="retrieval_document")
        vector = response["embedding"]
        supabase.table("pdf_embeddings").insert({"text": chunk, "embedding": vector, "pdf_hash": content_hash, "source": source}).execute()
    return "New content stored successfully!"

# Search relevant chunks
def search_vectors(user_query):
    response = genai.embed_content(model="models/embedding-001", content=user_query, task_type="retrieval_query")
    if "embedding" in response:
        query_embedding = response["embedding"]
        response = supabase.rpc("match_documents", {"query_embedding": query_embedding, "match_threshold": 0.7, "match_count": 5}).execute()
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
    relevant_chunks = search_vectors(user_question)
    chain = get_conversational_chain()
    response = chain({"context": "\n".join(relevant_chunks), "question": user_question}, return_only_outputs=True)
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

    if st.sidebar.button("New Chat"):
        if st.session_state.current_source:
            st.session_state.conversation_records.append({
                "source": st.session_state.current_source,
                "history": st.session_state.chat_history
            })
        st.session_state.chat_history = []
        st.session_state.current_source = None
        st.session_state.uploaded_files = None  # Clears uploaded files
        st.rerun()

    st.sidebar.header("Chat History")
    for idx, record in enumerate(st.session_state.conversation_records):
        if st.sidebar.button(record["source"], key=f"chat_{idx}"):
            st.session_state.chat_history = record["history"]
            st.session_state.current_source = record["source"]
            st.rerun()

    option = st.radio("Choose Input Type", ["Upload PDF", "Enter Website Domain"])
    content_available = False

    if option == "Upload PDF":
        st.session_state.uploaded_files = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        if st.button("Submit PDF"):
            with st.spinner("Processing PDF..."):
                raw_text = get_pdf_text(st.session_state.uploaded_files)
                text_chunks = get_text_chunks(raw_text)
                source = ", ".join([pdf.name.replace(".pdf", "") for pdf in st.session_state.uploaded_files])
                st.session_state.current_source = source
                store_vectors_in_db(text_chunks, "PDF Upload")
                content_available = True
                st.success("PDF content stored! You can now chat with it.")

    elif option == "Enter Website Domain":
        domain = st.text_input("Enter Website URL")
        if st.button("Scrape Website"):
            with st.spinner("Scraping website..."):
                landing_page_text = extract_landing_page_data(domain)
                text_chunks = get_text_chunks(landing_page_text)
                st.session_state.current_source = domain.split("//")[-1].split("/")[0]
                store_vectors_in_db(text_chunks, "Web Scraping")
                content_available = True
                st.success("Website content stored!")

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