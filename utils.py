from transformers import pipeline
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
import os
import tempfile


qa_model = pipeline("Question-answering", model="distilbert-base-uncased-distilled-squad")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def process_document(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    if uploaded_file.name.endswith(".pdf"):
        from langchain.document_loaders import PyPDFLoader
        loader = PyPDFLoader(tmp_path)
    else:
        from langchain.document_loaders import TextLoader
        loader = TextLoader(tmp_file)
    
    raw_docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(raw_docs)
    return docs

def ask_question(docs, question):
    texts = [doc.page_content for doc in docs ]
    embeddings = embedder.encode(texts)

    from sklearn.metrics.pairwise import cosine_similarity
    question_embedding = embedder.encode([question])
    similarities = cosine_similarity(question_embedding, embeddings)[0]

    top_index =similarities.argmax()
    context = texts[top_index]

    result = qa_model(question=question, context=context)
    return result["answer"]