import os

# Project path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PDF_DIR = os.path.join(DATA_DIR, "pdf")
TEXT_DIR = os.path.join(DATA_DIR, "text")
CSV_DIR = os.path.join(DATA_DIR, "csv")
INDEX_DIR = os.path.join(PROJECT_ROOT, "index")

# Text split
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Vector store
VECTOR_DB_TYPE = "faiss"  # Options: "faiss", "chroma"

# Embedding
EMBEDDING_MODEL = "openai"  # Options: "openai", "huggingface"
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"
HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 1536 if EMBEDDING_MODEL == "openai" else 384

# LLM
LLM_MODEL = "gpt-3.5-turbo"
TEMPERATURE = 0.7
MAX_TOKENS = 512

# Indexing
TOP_K = 5
