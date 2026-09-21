"""Tests for src/retrieval.py."""

import pytest

from src.retrieval import distances_to_similarity, retrieve_with_scores, source_label

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
