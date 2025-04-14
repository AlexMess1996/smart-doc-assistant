import streamlit as st
from utils import process_document, ask_question

# Page config
st.set_page_config(page_title="Smart Document Assistant", page_icon="🤖", layout="centered")

st.markdown(
    "<h1 style='text-align: center; color: #4A90E2;'>📄 Smart Document Assistant</h1>", 
    unsafe_allow_html=True
)

# Sidebar: Upload and instructions
with st.sidebar:
    st.header("🗂️ Upload your document")
    uploaded_file = st.file_uploader("Choose a PDF or TXT file", type=["pdf", "txt"])

    st.markdown("---")
    st.markdown("💡 **Tip:** After uploading a file, ask any question related to its content. Your chat history will be saved.")

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "docs" not in st.session_state:
    st.session_state.docs = None

# Process uploaded document
if uploaded_file:
    with st.spinner("Processing your document..."):
        st.session_state.docs = process_document(uploaded_file)
        st.session_state.chat_history = []
    st.success(f"✅ File '{uploaded_file.name}' uploaded and processed.")

# Chat interface
st.markdown("---")
chat_container = st.container()

if st.session_state.docs:
    user_input = st.chat_input("💬 Ask something about the document...")

    if user_input:
        with st.spinner("Thinking... generating your answer..."):
            answer, chunk_used = ask_question(st.session_state.docs, user_input)

        st.session_state.chat_history.append({
            "question": user_input,
            "answer": answer,
            "chunk": chunk_used
        })

# Display chat history
with chat_container:
    if st.session_state.chat_history:
        for i, msg in enumerate(st.session_state.chat_history):
            with st.chat_message("user"):
                st.markdown(f"**You:** {msg['question']}")

            with st.chat_message("assistant"):
                st.markdown(f"**Assistant:** {msg['answer']}")
                with st.expander("📖 Show document context"):
                    st.markdown(msg["chunk"])
    elif st.session_state.docs:
        st.info("👆 Ask your first question about the uploaded document.")
    else:
        st.info("👈 Upload a document to get started.")

# Footer
st.markdown("---")

