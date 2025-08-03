import openai
import tiktoken
from PyPDF2 import PdfReader
import requests
from config import OPENAI_API_KEY
from vector_store import upsert_vectors, search_vectors

openai.api_key = OPENAI_API_KEY

def extract_text_from_pdf(url):
    response = requests.get(url)
    with open("temp.pdf", "wb") as f:
        f.write(response.content)

    reader = PdfReader("temp.pdf")
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

def chunk_text(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def embed_chunks(chunks):
    embeddings = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=chunks
    )["data"]

    return [
        (f"chunk-{i}", emb["embedding"], {"text": chunks[i]})
        for i, emb in enumerate(embeddings)
    ]

def retrieve_context(query, top_k=3):
    query_vector = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=query
    )["data"][0]["embedding"]

    results = search_vectors(query_vector, top_k=top_k)
    return " ".join([match["metadata"]["text"] for match in results["matches"]])

def ask_gpt4(question, context):
    prompt = f"""Answer the question based on the context below:\n\nContext:\n{context}\n\nQuestion: {question}"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response["choices"][0]["message"]["content"]
