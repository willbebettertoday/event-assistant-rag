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

```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull required models
ollama pull llama3.2
ollama pull mxbai-embed-large

# Start server (runs on localhost:11434 by default)
ollama serve
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Setup config

```bash
cp config_example.py config.py
# Edit config.py with your Ollama server URL
```

### 4. Add your documents

Put PDF and TXT files in the `docs/` folder.

### 5. Run

```bash
# Command line (simple)
python main.py

# Web interface (with XAI visualization)
python app.py
```

## Project Structure

```
marbet-event-assistant/
├── README.md
├── requirements.txt
├── config_example.py
├── config.py           ← your settings (gitignored)
├── main.py             ← CLI version
├── app.py              ← Web UI with XAI visualization
└── docs/               ← put your documents here
```

## How RAG Works

```
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
```

## Configuration

| Parameter | Description |
|-----------|-------------|
| `OLLAMA_SERVER` | URL of Ollama server |
| `CHAT_MODEL` | LLM model for chat |
| `EMBED_MODEL` | Model for embeddings |
| `DOCS_FOLDER` | Path to documents |

## Known Limitations & Future Improvements

- **Temperature setting (0.8)** is higher than typical for QA systems (0.1-0.3 recommended) — may cause occasional hallucinations
- **Relevance scores in XAI visualization** are based on retrieval order, not actual similarity distances from FAISS
- **Vectorstore caching** is implemented only in CLI version
- **No persistent storage** for web UI — cache rebuilds on each restart

These trade-offs were made for simplicity within the project scope.

## License

MIT
