import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
