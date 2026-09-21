# 🎪 Marbet Event Assistant

A RAG chatbot that answers questions about event documents through a local Ollama server, and shows, with real FAISS similarity scores, which chunks backed each answer.

University project at Breda University of Applied Sciences. Authors: Oleksii Krasnoshtanov, Danil Sysenko.

[![ci](https://github.com/willbebettertoday/event-assistant-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/willbebettertoday/event-assistant-rag/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)

## Screenshot

Not included yet. Running the Gradio interface needs a local Ollama server with the chat and embedding models pulled, and I did not have one available while writing this page. The interface is a chat panel next to the source-similarity chart described under "How it works" below.

## Problem

Event staff and attendees get handed a folder of PDFs and text files, agendas, venue details, policies, and have to search them by hand to answer a question. Oleksii Krasnoshtanov and I built this as a university project at Breda University of Applied Sciences, for an event-management case the code calls Marbet, to answer those questions directly from the documents instead. A chatbot that only answers is not enough for that: whoever asks needs to see which document backed the answer, not just trust the model.

## Approach

The app loads PDFs and text files from a folder, splits them into chunks, embeds them with an Ollama model, and stores them in a FAISS index. A LangChain `ConversationalRetrievalChain` retrieves the closest chunks for each question and answers from them only. `main.py` (CLI) and `app.py` (Gradio) used to each carry their own copy of that pipeline, with the same chunk size, overlap, k and temperature hard-coded twice; both are now thin entry points over `src/pipeline.py`, and the sampling temperature came down from 0.8 to 0.1 because the system prompt already restricts answers to the retrieved documents.

The technical choice worth calling out is how the source chart is scored. It used to rank each retrieved chunk by its position in the result list (`1.0 - (i * 0.08)`), so it drew the same 1.00, 0.92, 0.84 ramp for every question regardless of what was actually retrieved. It now retrieves through `similarity_search_with_score` and turns each real FAISS L2 distance `d` into `1 / (1 + d)` (`src/retrieval.py`), so two equally close chunks get equal bars and closer chunks always outscore farther ones. What that number does and does not mean is covered under "Validation and limitations" below.

A second, subtler defect sat underneath the first one. `ConversationalRetrievalChain` condenses a follow-up question into a standalone one before it retrieves, so from the second question in a conversation onward, the documents behind the answer were not necessarily the documents for the raw turn the chart was scoring. The chain now returns that generated question (`return_generated_question=True`), and the chart re-retrieves and scores with the same question the chain actually used.

Vector store persistence changed too, and this is worth stating precisely rather than as a clean-up story: it used to `pickle.dump` a live vector store object directly, and now uses FAISS's own `save_local`/`load_local`. That format is still pickle-backed internally, so this did not remove pickle from the read path. What changed is that a hand-rolled serialisation of a live object was replaced by the library's own documented format, and reading it back now requires passing `allow_dangerous_deserialization=True` explicitly at the one place that loads it (`src/pipeline.py`), which is a reasoned, scoped opt-in rather than a hidden risk.

## Results

| Check | Result | Condition |
|---|---|---|
| Unit tests | 20 passed | `pytest -q`, against fakes for FAISS and Ollama (`tests/conftest.py`), no live server involved |
| Lint | 0 issues | `ruff check .` |
| CI | green | GitHub Actions, Python 3.11 and 3.13 (`.github/workflows/ci.yml`) |

No retrieval or answer-quality benchmark was run; see "Validation and limitations".

## How it works

1. `load_documents` reads every PDF and TXT file under `docs/` (`src/pipeline.py`).
2. `split_documents` cuts them into chunks of `CHUNK_SIZE` (700) with `CHUNK_OVERLAP` (120) characters of overlap.
3. `build_vectorstore` embeds the chunks with an Ollama embedding model and stores them in a FAISS index.
4. `main.py` caches that index to disk with FAISS's `save_local`/`load_local`; `app.py` rebuilds it fresh on every launch and never caches.
5. `build_chain` wraps the index in a `ConversationalRetrievalChain`: for a follow-up question it first condenses the turn into a standalone one, then retrieves the `RETRIEVER_K` (8) closest chunks for that question and answers from them only, at `TEMPERATURE` 0.1.
6. For the chart, `retrieval_query` picks that generated question over the raw turn, and `retrieve_with_scores` re-runs it through `similarity_search_with_score` to get the real FAISS distances behind the answer.
7. `distances_to_similarity` turns each distance `d` into `1 / (1 + d)`, and `visualize_sources` (`app.py`) bar-charts the top chunks by that number.

## Reproduce it

### Run the tests

No Ollama server needed for this part; `src/pipeline.py` imports LangChain, FAISS and Ollama lazily, and the tests stub what they touch.

```bash
git clone https://github.com/willbebettertoday/event-assistant-rag.git
cd event-assistant-rag
pip install pytest ruff numpy pandas scikit-learn
ruff check .
pytest -q
```

I ran this exact sequence on Python 3.13 while writing this page (20 passed, lint clean). CI runs the same on 3.11 and 3.13.

### Run the app

This needs a local Ollama server with both models pulled, which I did not have running while writing this page, so treat the commands below as verified individually (dependency resolution, config, entry points) rather than as an end-to-end run.

```bash
# 1. Ollama, with the models this project uses
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2
ollama pull mxbai-embed-large
ollama serve

# 2. Project dependencies
pip install -r requirements.txt

# 3. Config
cp config_example.py config.py
# edit config.py if your Ollama server is not on localhost:11434

# 4. Documents: drop PDF or TXT files into docs/ (created automatically on first run)

# 5. Run
python main.py    # CLI, caches the vector store to cache/
python app.py      # Gradio UI, prints a local URL, shows the source chart
```

Defaults live in `config_example.py`:

| Setting | Default | Note |
|---|---|---|
| `OLLAMA_SERVER` | `http://localhost:11434` | |
| `CHAT_MODEL` | `llama3.2` | |
| `EMBED_MODEL` | `mxbai-embed-large:latest` | |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 700 / 120 | overlap was 300, which overlapped 43% of every chunk |
| `RETRIEVER_K` | 8 | was 12 |
| `TEMPERATURE` | 0.1 | was 0.8; low because the system prompt already restricts answers to the retrieved documents |
| `USE_CACHE` | `True` | governs `main.py` only; `app.py` always rebuilds |

## Validation and limitations

The source chart's bar heights are `1 / (1 + FAISS L2 distance)` for each retrieved chunk (`distances_to_similarity` in `src/retrieval.py`). That number is a similarity: how close the chunk's embedding is to the question's embedding inside FAISS's index. It is not a probability, and it is not a measurement of how much that chunk caused the model's answer. The LLM can still lean on one chunk unevenly, ignore a retrieved chunk entirely, or blend information across several within the limits the system prompt sets. What the chart shows honestly is which chunks were retrieved for the question actually used and how close each one was; what it does not show is how the model used them.

I have not run a retrieval benchmark against this project's documents, so there is no recall@k number and no precision figure, nothing to say how often the right chunk actually lands in the top k. Answer quality also depends on which chat and embedding models are pulled on whatever Ollama server the app points at (`OLLAMA_SERVER`, `CHAT_MODEL`, `EMBED_MODEL` are all config), so results will differ across machines. Like any RAG system, an answer is only as good as the documents in `docs/`: if the information is not there, or is split across chunks awkwardly, the model has nothing to answer from beyond the instruction to say so.

The 20 tests in `tests/` cover the pipeline glue (chunking, caching, and the retrieval-scoring math) and pin the two regressions described above on purpose: the rank-based chart score and the raw-turn retrieval bug. They run against fakes for FAISS and Ollama (`tests/conftest.py`), not a live server or a real index, so they say nothing about answer quality or about behaviour against a real Ollama server.

## Tech stack

- LangChain (`langchain`, `langchain-community`, `langchain-ollama`) for document loading, chunking and the retrieval chain
- Ollama for local chat and embedding inference
- FAISS (`faiss-cpu`) for the vector index
- Gradio for the web interface
- matplotlib for the source chart
- pytest and ruff for tests and linting, run on Python 3.11 and 3.13 in CI

## Project structure

```
event-assistant-rag/
├── .github/workflows/ci.yml   # lint + test on Python 3.11 and 3.13
├── .gitignore
├── LICENSE
├── README.md
├── app.py                     # Gradio entry point, with the source chart
├── config_example.py          # copy to config.py and edit
├── main.py                    # CLI entry point
├── pyproject.toml             # ruff and pytest config
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── pipeline.py            # shared pipeline: load, split, embed, cache, chain
│   └── retrieval.py           # FAISS-distance similarity scoring for the chart
└── tests/
    ├── __init__.py
    ├── conftest.py            # fakes for FAISS, Ollama and config
    ├── test_pipeline.py
    └── test_retrieval.py
```

`docs/`, `cache/` and `config.py` are created at runtime and are gitignored; they are not part of the repository.

## License

MIT, copyright Oleksii Krasnoshtanov and Danil Sysenko. See [LICENSE](LICENSE).
