"""Tests for src/pipeline.py. No embeddings server involved."""

from src.pipeline import save_vectorstore


class RecordingStore:
    def __init__(self):
        self.saved_to = None
        self.pickled = False

    def save_local(self, path):
        self.saved_to = path


def test_saving_uses_faiss_save_local_not_pickle(tmp_path):
    """Loading a pickle runs whatever code is inside it.

    FAISS ships save_local/load_local, so there is no reason to accept that
    risk for a cache file.
    """
    store = RecordingStore()
    save_vectorstore(store, str(tmp_path / "cache"))
    assert store.saved_to == str(tmp_path / "cache")
    assert not store.pickled


def test_no_module_imports_pickle_for_the_vectorstore():
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    for name in ("main.py", "app.py", "src/pipeline.py"):
        text = (root / name).read_text(encoding="utf-8")
        assert "pickle.dump" not in text, f"{name} still pickles the vector store"
        assert "pickle.load" not in text, f"{name} still unpickles the vector store"
