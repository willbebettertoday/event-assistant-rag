"""Tests for src/pipeline.py. No embeddings server involved."""

import sys
import types

from src.pipeline import load_vectorstore, save_vectorstore


class RecordingStore:
    def __init__(self):
        self.saved_to = None

    def save_local(self, path):
        self.saved_to = path


def test_saving_uses_faiss_save_local_not_pickle(tmp_path):
    """save_vectorstore must delegate to FAISS's own save_local.

    FAISS ships save_local/load_local, so there is no reason for this
    module to hand-roll its own serialisation of the vector store object.
    """
    store = RecordingStore()
    save_vectorstore(store, str(tmp_path / "cache"))
    assert store.saved_to == str(tmp_path / "cache")


def test_no_module_hand_rolls_serialisation_of_the_vectorstore():
    """No entry point, and no module here, should reinvent this on its own.

    FAISS's own save_local/load_local is still pickle-backed internally
    (see load_vectorstore's docstring for why that is an accepted, scoped
    risk rather than an eliminated one). What this guards against is a
    *second*, hand-rolled serialisation path reappearing alongside it,
    under pickle's own dump/dumps/load/loads or a third-party stand-in
    such as joblib or dill.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    banned = ("pickle.dump", "pickle.load", "pickle.dumps", "pickle.loads", "joblib", "dill")
    for name in ("main.py", "app.py", "src/pipeline.py"):
        text = (root / name).read_text(encoding="utf-8")
        for token in banned:
            assert token not in text, f"{name} uses {token} to (de)serialise the vector store"


def test_load_vectorstore_returns_none_when_no_cache_exists(tmp_path):
    """The interface promises None on a miss, without ever touching FAISS."""
    assert load_vectorstore(str(tmp_path / "no-such-cache"), embeddings=object()) is None


def test_load_vectorstore_reads_an_existing_cache_with_deserialization_allowed(
    tmp_path, monkeypatch
):
    """Pins the call shape of the one branch that actually reads the cache.

    This is the branch that regressed before: load_local was called with
    allow_dangerous_deserialization=False, which makes LangChain raise
    unconditionally rather than attempt the load, so the cache was written
    and never read. CI has no langchain_community installed (see the
    module docstring), so this stubs the same import main.py and app.py
    make, the way tests/conftest.py already stubs config, and checks the
    arguments load_local receives rather than trusting the real library.
    """
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "index.faiss").touch()

    calls = []

    class FakeFAISS:
        @staticmethod
        def load_local(path, embeddings, allow_dangerous_deserialization=False):
            calls.append((path, embeddings, allow_dangerous_deserialization))
            return "fake-vectorstore"

    fake_package = types.ModuleType("langchain_community")
    fake_vectorstores = types.ModuleType("langchain_community.vectorstores")
    fake_vectorstores.FAISS = FakeFAISS
    monkeypatch.setitem(sys.modules, "langchain_community", fake_package)
    monkeypatch.setitem(sys.modules, "langchain_community.vectorstores", fake_vectorstores)

    embeddings = object()
    result = load_vectorstore(str(cache_dir), embeddings)

    assert result == "fake-vectorstore"
    assert calls == [(str(cache_dir), embeddings, True)]
