import pinecone
from config import PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pinecone.Index(PINECONE_INDEX)

def upsert_vectors(vectors):
    index.upsert(vectors)

def search_vectors(vector, top_k=5):
    return index.query(vector=vector, top_k=top_k, include_metadata=True)
