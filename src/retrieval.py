"""Retrieval with real similarity scores.

FAISS returns an L2 distance for every hit. Reporting those, rather than a
number derived from a document's position in the list, is what makes the
source chart an explanation instead of a decoration.
"""

import os

MAX_LABEL_STEM = 25


def retrieve_with_scores(vectorstore, question, k=8):
    """Return (document, distance) pairs. Smaller distance means closer."""
    return vectorstore.similarity_search_with_score(question, k=k)


def distances_to_similarity(distances):
    """Map L2 distances to (0, 1]. 1.0 is an exact match.

    Monotonically decreasing in distance, so equal distances always give
    equal scores. That is the property the previous rank-based score broke.
    """
    return [1.0 / (1.0 + float(d)) for d in distances]


def retrieval_query(result, fallback):
    """The question the chain actually retrieved with.

    ConversationalRetrievalChain condenses a follow-up into a standalone
    question and retrieves with that. Scoring the raw turn instead would
    chart a different document set than the answer came from.
    """
    if isinstance(result, dict):
        generated = result.get("generated_question")
        if isinstance(generated, str) and generated:
            return generated
    return fallback


def source_label(document):
    """Human readable label for a retrieved chunk."""
    source = document.metadata.get("source")
    if not source:
        return "Unknown"

    name = os.path.basename(source)
    if len(name) > MAX_LABEL_STEM:
        name = name[: MAX_LABEL_STEM - 3] + "..."

    page = document.metadata.get("page")
    if page is not None:
        return f"{name} (p.{int(page) + 1})"
    return name
