import tempfile
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity

# Load free Hugging Face models
#the model is too small
#qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")

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

def ask_question(docs, question, top_k=3):
    texts = [doc.page_content for doc in docs]
    embeddings = embedder.encode(texts)
    question_embedding = embedder.encode([question])

    similarities = cosine_similarity(question_embedding, embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]  # Get top K similar chunks

    # Combine top chunks into one big context
    context = "\n\n".join([texts[i] for i in top_indices])

    result = qa_model(question=question, context=context)
    return result["answer"], context

