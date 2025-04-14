import tempfile
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity

# Load free Hugging Face models
qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def process_document(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(tmp_path)
    else:
        loader = TextLoader(tmp_path)

    raw_docs = loader.load()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(raw_docs)
    return docs

def ask_question(docs, question):
    texts = [doc.page_content for doc in docs]
    embeddings = embedder.encode(texts)
    question_embedding = embedder.encode([question])

    similarities = cosine_similarity(question_embedding, embeddings)[0]
    top_index = similarities.argmax()
    context = texts[top_index]

    result = qa_model(question=question, context=context)
    return result["answer"]
