# 🎪 Marbet Event Assistant

RAG-powered chatbot for answering questions about company events using LangChain and local LLMs.

**University Project** - Breda University of Applied Sciences  
**Authors**: Oleksii Krasnoshtanov, Danil Sysenko

## What it does

- Loads PDF and TXT documents about events
- Creates vector embeddings for semantic search
- Answers questions using retrieval-augmented generation (RAG)
- **Shows source documents used for each answer (XAI - Explainable AI)**
- Provides web interface (Gradio) and CLI mode

## Tech Stack

- **LangChain** - RAG framework
- **Ollama** - Local LLM inference
- **FAISS** - Vector similarity search
- **Gradio** - Web UI

> Note: Originally developed using university GPU server. Works with any Ollama setup.

## Quick Start

### 1. Install Ollama

You need an Ollama server running locally or on a remote machine.

~~~bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull required models
ollama pull llama3.2
ollama pull mxbai-embed-large

# Start server (runs on localhost:11434 by default)
ollama serve
~~~

### 2. Install dependencies

~~~bash
pip install -r requirements.txt
~~~

### 2. Setup config

~~~bash
cp config_example.py config.py
# Edit config.py with your Ollama server URL
~~~

### 3. Add your documents

Put PDF and TXT files in the `docs/` folder.

### 4. Run

~~~bash
# Command line (simple)
python main.py

# Web interface (optional)
python app.py
~~~

## Project Structure

~~~
marbet-event-assistant/
├── README.md
├── requirements.txt
├── config_example.py
├── config.py           ← your settings (gitignored)
├── main.py             ← CLI version (~100 lines)
├── app.py              ← Web UI version (optional)
└── docs/               ← put your documents here
~~~

## How RAG Works

~~~
1. Load Documents
   PDF/TXT files → split into chunks

2. Create Embeddings  
   Chunks → vector embeddings (Ollama)

3. Store in FAISS
   Embeddings → vector database

4. Query Processing
   User question → find similar chunks → send to LLM with context

5. Generate Answer
   LLM reads context + question → generates answer
~~~

## Configuration

| Parameter | Description |
|-----------|-------------|
| `OLLAMA_SERVER` | URL of Ollama server |
| `CHAT_MODEL` | LLM model for chat |
| `EMBED_MODEL` | Model for embeddings |
| `DOCS_FOLDER` | Path to documents |

## Screenshots

*Add screenshots of your Gradio interface here*

## License

MIT
