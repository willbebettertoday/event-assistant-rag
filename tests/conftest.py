"""Fakes for the RAG tests. No Ollama server and no real embeddings."""

import sys
import types

import pytest


def _build_config_stub():
    cfg = types.ModuleType("config")
    cfg.OLLAMA_SERVER = "http://localhost:11434"
    cfg.CHAT_MODEL = "llama3.2"
    cfg.EMBED_MODEL = "mxbai-embed-large:latest"
    cfg.DOCS_FOLDER = "./docs"
    cfg.CACHE_DIR = "./cache"
    cfg.USE_CACHE = True
    return cfg


sys.modules.setdefault("config", _build_config_stub())


class FakeDocument:
    def __init__(self, source, page=None, content="text"):
        self.page_content = content
        self.metadata = {"source": source}
        if page is not None:
            self.metadata["page"] = page


class FakeVectorStore:
    """Stands in for FAISS. Returns the pairs it was constructed with."""

    def __init__(self, pairs):
        self._pairs = pairs
        self.last_call = None

    def similarity_search_with_score(self, query, k=4):
        self.last_call = (query, k)
        return self._pairs[:k]


@pytest.fixture
def fake_docs():
    return [
        (FakeDocument("/docs/agenda.pdf", page=2), 0.10),
        (FakeDocument("/docs/venue.pdf", page=0), 0.45),
        (FakeDocument("/docs/notes.txt"), 1.30),
    ]


@pytest.fixture
def fake_store(fake_docs):
    return FakeVectorStore(fake_docs)
