import streamlit as st
import os
from PyPDF2 import PdfReader

def main():
    st.set_page_config(layout="wide")
    
    # Sidebar for chat history
    with st.sidebar:
        st.header("Chat History")
        chat_history = st.session_state.get("chat_history", [])
        for chat in chat_history:
            st.text(chat)
    
    # Main UI layout
    st.title("PDF & Web Chat Application")
    col1, col2, col3 = st.columns([2, 3, 2])
    
    with col2:
        st.subheader("Choose an Option")
        option = st.radio("", ["Upload a PDF", "Enter a Web Domain"])
        
        if option == "Upload a PDF":
            uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
            if uploaded_file:
                save_uploaded_file(uploaded_file)
                st.success(f"Uploaded: {uploaded_file.name}")
                st.session_state["uploaded_files"] = st.session_state.get("uploaded_files", [])
                st.session_state["uploaded_files"].append(uploaded_file.name)
        
        elif option == "Enter a Web Domain":
            domain = st.text_input("Enter Web Domain")
            if st.button("Fetch Content"):
                st.session_state["web_content"] = f"Content from {domain} will be fetched here."
    
    # Chat input at the bottom
    chat_input = st.text_input("Chat here:")
    if st.button("Send"):
        if chat_input:
            chat_history.append(f"User: {chat_input}")
            st.session_state["chat_history"] = chat_history
    
    # Right Sidebar for displaying uploaded PDFs
    with st.sidebar:
        if "uploaded_files" in st.session_state:
            st.header("Uploaded Files")
            for file in st.session_state["uploaded_files"]:
                if st.button(file):
                    st.session_state["selected_file"] = file
                    display_pdf(file)

def save_uploaded_file(uploaded_file):
    with open(os.path.join("uploads", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())

def display_pdf(file_name):
    file_path = os.path.join("uploads", file_name)
    with open(file_path, "rb") as f:
        pdf_reader = PdfReader(f)
        pdf_text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        st.text_area("PDF Content", pdf_text, height=300)

if __name__ == "__main__":
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    main()

# import streamlit as st
# import os
# import webbrowser
# from PyPDF2 import PdfReader

# def main():
#     st.set_page_config(layout="wide")
    
#     # Initialize session state for conversation history
#     if "conversation_history" not in st.session_state:
#         st.session_state.conversation_history = []
#     if "first_question_asked" not in st.session_state:
#         st.session_state.first_question_asked = False
#     if "query" not in st.session_state:
#         st.session_state.query = ""
#     if "current_chat" not in st.session_state:
#         st.session_state.current_chat = None
#     if "chat_mode" not in st.session_state:
#         st.session_state.chat_mode = None
#     if "viewing_content" not in st.session_state:
#         st.session_state.viewing_content = False
    
#     # Sidebar for chat history
#     with st.sidebar:
#         st.header("Chat History")
#         if st.button("New Chat"):
#             st.session_state.current_chat = None
#             st.session_state.conversation_history = []
#             st.session_state.chat_mode = None
#             st.session_state.viewing_content = False
        
#         for i, (q, a) in enumerate(st.session_state.conversation_history):
#             with st.expander(f"**Q{i+1}:** {q}"):
#                 st.write(f"**A{i+1}:** {a}")
    
#     # Main UI layout
#     st.title("PDF & Web Chat Application")
#     col1, col2, col3 = st.columns([2, 3, 2])
    
#     with col2:
#         if not st.session_state.chat_mode:
#             st.subheader("Choose an Option")
#             chat_option = st.radio("Select an option:", ["Upload a PDF", "Enter a Web Domain"], horizontal=True)
            
#             if chat_option == "Upload a PDF":
#                 st.session_state.chat_mode = "pdf"
#                 st.session_state.conversation_history = []
#             elif chat_option == "Enter a Web Domain":
#                 st.session_state.chat_mode = "web"
#                 st.session_state.conversation_history = []
        
#         elif st.session_state.chat_mode == "pdf":
#             uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
#             if uploaded_file:
#                 save_uploaded_file(uploaded_file)
#                 st.success(f"Uploaded: {uploaded_file.name}")
#                 st.session_state.current_chat = uploaded_file.name
#                 st.session_state.chat_mode = "chat"
        
#         elif st.session_state.chat_mode == "web":
#             domain = st.text_input("Enter Web Domain")
#             if st.button("Fetch Content") and domain:
#                 st.session_state.current_chat = domain
#                 st.session_state.chat_mode = "chat"
    
#     # Chat input at the bottom
#     if st.session_state.chat_mode == "chat":
#         if not st.session_state.viewing_content:
#             query = st.text_input("Chat here:", value=st.session_state.query, key="chat_input")
#             if st.button("Send"):
#                 if query.strip():
#                     answer = f"Response for: {query}"
#                     st.session_state.conversation_history.append((query, answer))
#                     st.session_state.first_question_asked = True
#                     st.session_state.query = ""
        
#         if st.session_state.first_question_asked and not st.session_state.viewing_content:
#             st.subheader("Follow-up Question:")
#             follow_up_query = st.text_input("Ask a follow-up question:", key="follow_up_input")
#             if st.button("Ask Follow-up"):
#                 if follow_up_query.strip():
#                     follow_up_answer = f"Follow-up response for: {follow_up_query}"
#                     st.session_state.conversation_history.append((follow_up_query, follow_up_answer))
#                     st.session_state.query = ""
#                 else:
#                     st.error("Please enter a follow-up question.")
    
#     # Right Sidebar for displaying uploaded PDFs or website link
#     with col3:
#         if st.session_state.current_chat:
#             st.header("Chat Details")
#             if st.session_state.chat_mode == "chat":
#                 if ".pdf" in st.session_state.current_chat:
#                     if st.button("View File"):
#                         st.session_state.viewing_content = not st.session_state.viewing_content
#                     if st.session_state.viewing_content:
#                         display_pdf(st.session_state.current_chat)
#                 else:
#                     if st.button("View Website"):
#                         open_website(st.session_state.current_chat)
#                         st.session_state.viewing_content = not st.session_state.viewing_content


# def save_uploaded_file(uploaded_file):
#     with open(os.path.join("uploads", uploaded_file.name), "wb") as f:
#         f.write(uploaded_file.getbuffer())

# def display_pdf(file_name):
#     file_path = os.path.join("uploads", file_name)
#     with open(file_path, "rb") as f:
#         pdf_reader = PdfReader(f)
#         pdf_text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
#         st.text_area("PDF Content", pdf_text, height=700)

# def open_website(domain):
#     webbrowser.open(f"https://{domain}")
#     st.success(f"Opening {domain} in browser...")

# if __name__ == "__main__":
#     if not os.path.exists("uploads"):
#         os.makedirs("uploads")
#     main()
