"""Tests for src/retrieval.py."""

from pathlib import Path

import pytest

from src.retrieval import (
    distances_to_similarity,
    retrieval_query,
    retrieve_with_scores,
    source_label,
)

from .conftest import FakeDocument


class TestRetrieveWithScores:
    def test_returns_document_score_pairs(self, fake_store):
        out = retrieve_with_scores(fake_store, "when does it start", k=2)
        assert len(out) == 2
        assert out[0][1] == pytest.approx(0.10)

    def test_forwards_k_to_the_store(self, fake_store):
        retrieve_with_scores(fake_store, "q", k=3)
        assert fake_store.last_call == ("q", 3)


class TestDistancesToSimilarity:
    def test_a_zero_distance_is_a_perfect_match(self):
        assert distances_to_similarity([0.0]) == [pytest.approx(1.0)]

    def test_similarity_decreases_as_distance_grows(self):
        out = distances_to_similarity([0.1, 0.45, 1.3])
        assert out == sorted(out, reverse=True)
        assert all(0.0 < s <= 1.0 for s in out)

    def test_scores_reflect_the_distances_not_the_rank(self):
        """The old code returned 1.0, 0.92, 0.84 whatever the distances were.

        Two documents at the same distance must score the same, and the gap
        between scores must follow the gap between distances.
        """
        assert distances_to_similarity([0.5, 0.5]) == [
            pytest.approx(1 / 1.5),
            pytest.approx(1 / 1.5),
        ]
        near, far = distances_to_similarity([0.01, 5.0])
        assert near - far > 0.5

    def test_empty_input_gives_empty_output(self):
        assert distances_to_similarity([]) == []


class TestRetrievalQuery:
    def test_prefers_the_generated_question_when_present(self):
        result = {"generated_question": "when does the opening ceremony start"}
        assert retrieval_query(result, "and then?") == "when does the opening ceremony start"

    def test_falls_back_when_the_key_is_absent(self):
        """No condensation happens on the first turn of a conversation."""
        assert retrieval_query({"answer": "..."}, "when does it start") == "when does it start"

    def test_falls_back_when_the_value_is_an_empty_string(self):
        assert retrieval_query({"generated_question": ""}, "when does it start") == (
            "when does it start"
        )

    def test_falls_back_when_the_value_is_none(self):
        assert retrieval_query({"generated_question": None}, "when does it start") == (
            "when does it start"
        )

    def test_a_none_result_does_not_raise(self):
        assert retrieval_query(None, "when does it start") == "when does it start"


class TestNoRankBasedScoring:
    """Guards against the original defect reappearing inline in app.py.

    The bug this task fixes was `relevance = 1.0 - (i * 0.08)`: a chart
    height derived from a document's position in the result list instead
    of its real FAISS distance. The tests above only exercise
    src/retrieval.py, so a future edit could reintroduce the same
    rank-based arithmetic directly inside app.py's visualize_sources and
    nothing here would notice. This test reads the source text of app.py
    to close that gap.
    """

    def test_app_scores_via_distances_to_similarity_not_rank(self):
        app_source = (Path(__file__).resolve().parent.parent / "app.py").read_text(
            encoding="utf-8"
        )
        assert "* 0.08" not in app_source
        assert "1.0 - i" not in app_source
        assert "distances_to_similarity" in app_source


class TestSourceLabel:
    def test_includes_the_page_for_a_pdf(self):
        assert source_label(FakeDocument("/docs/agenda.pdf", page=2)) == "agenda.pdf (p.3)"

    def test_omits_the_page_when_absent(self):
        assert source_label(FakeDocument("/docs/notes.txt")) == "notes.txt"

    def test_truncates_a_long_name(self):
        label = source_label(FakeDocument("/docs/" + "x" * 60 + ".pdf"))
        assert len(label) <= 29
        assert "..." in label

    def test_missing_source_metadata_does_not_raise(self):
        doc = FakeDocument("/docs/a.pdf")
        doc.metadata = {}
        assert source_label(doc) == "Unknown"
