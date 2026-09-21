"""
Config file - copy to config.py and edit
"""

# Ollama server URL
# Local: http://localhost:11434
# Remote: http://your-server-ip:port
OLLAMA_SERVER = "http://localhost:11434"

# Models (must be available on your Ollama server)
# Run: ollama pull llama3.2
# Run: ollama pull mxbai-embed-large
CHAT_MODEL = "llama3.2"                    # or any chat model you have
EMBED_MODEL = "mxbai-embed-large:latest"   # for embeddings

# Paths
DOCS_FOLDER = "./docs"
CACHE_DIR = "./cache"

# Governs main.py's cache-read only; app.py never caches and rebuilds the
# vector store on every launch. The cache is a FAISS index directory
# (save_local/load_local), which is pickle-backed internally, not a plain
# file. Set False to always rebuild from docs/ instead of reading it.
USE_CACHE = True

# Retrieval
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120   # was 300, which overlapped 43% of every chunk
RETRIEVER_K = 8       # was 12

# Generation
# Low on purpose. The system prompt tells the model to answer only from the
# retrieved documents, and a high temperature is how it starts inventing.
TEMPERATURE = 0.1
