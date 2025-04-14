import streamlit as st
from utils import process_document, ask_question

st.set_page_config(page_title="Smart Document Assistant", layout="wide")
st.title("Smart Document Assistant")

uploaded_file = st.file_uploader("Upload your document (PDF or TXT)", type=["pdf","txt"])

if uploaded_file:
    st.success("File uploaded successfully!")
    docs = process_document(uploaded_file)


    question = st.text_input("Ask a question about the document:")
    if question:
        answer = ask_question(docs, question)
        st.markdown(f"**Answer:** {answer}")