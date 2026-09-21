"""Marbet Event Assistant, web interface.

Authors: Oleksii Krasnoshtanov, Danil Sysenko
University: Breda University of Applied Sciences

This is the Gradio app, which also shows which chunks each answer came
from. For a terminal session, run main.py.
"""

import gradio as gr
import matplotlib
import numpy as np
from config import (
    CHAT_MODEL,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCS_FOLDER,
    EMBED_MODEL,
    OLLAMA_SERVER,
    RETRIEVER_K,
    TEMPERATURE,
)

from src.pipeline import build_chain, build_vectorstore, load_documents, split_documents

matplotlib.use("Agg")  # Gradio renders server side; no interactive backend
import matplotlib.pyplot as plt  # noqa: E402

from src.retrieval import (  # noqa: E402
    distances_to_similarity,
    retrieval_query,
    retrieve_with_scores,
    source_label,
)

SYSTEM_PROMPT = """You are a helpful event assistant.
Answer questions based ONLY on the provided documents.
If the answer is not in the documents, say you don't have that information."""

# Global variables
qa_chain = None
vectorstore = None
chat_history = []


def setup_chain():
    """Load documents and create QA chain"""
    from langchain_ollama import OllamaEmbeddings

    print("Loading documents...")
    all_docs = load_documents(DOCS_FOLDER)
    print(f"Loaded {len(all_docs)} documents")

    print("Creating chunks...")
    chunks = split_documents(all_docs, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"Created {len(chunks)} chunks")

    print("Building vector store...")
    embeddings = OllamaEmbeddings(base_url=OLLAMA_SERVER, model=EMBED_MODEL)
    vectorstore = build_vectorstore(chunks, embeddings)

    chain = build_chain(
        vectorstore, OLLAMA_SERVER, CHAT_MODEL, TEMPERATURE, SYSTEM_PROMPT, RETRIEVER_K
    )

    print("Ready!")
    return chain, vectorstore


def visualize_sources(scored_docs):
    """Bar chart of real retrieval similarity per source chunk.

    scored_docs is a list of (document, distance) pairs from
    retrieve_with_scores. The heights are computed from the distances FAISS
    returned, so two equally relevant chunks get equal bars.
    """
    if not scored_docs:
        return None

    labels = [source_label(doc) for doc, _ in scored_docs][:10]
    scores = distances_to_similarity([dist for _, dist in scored_docs])[:10]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels, scores, color=plt.cm.Blues(np.linspace(0.5, 1.0, len(labels))))
    ax.set_xlim(0, 1)
    ax.set_xlabel("similarity (1 / (1 + FAISS L2 distance))")
    ax.set_title("Which chunks the answer was drawn from")
    ax.invert_yaxis()
    fig.tight_layout()
    try:
        return fig
    finally:
        plt.close(fig)


def respond(message, history):
    """Handle user message and return response with sources"""
    global chat_history

    if not message.strip():
        return "", history, None

    # Get response
    result = qa_chain.invoke({"question": message, "chat_history": chat_history})

    # Update history
    chat_history.append((message, result["answer"]))
    if len(chat_history) > 10:
        chat_history = chat_history[-10:]

    # Create visualization from the real retrieval scores. The chain's
    # source_documents carry no scores, so re-retrieve through the
    # vectorstore directly to get the FAISS distances behind the chart.
    # The chain condenses follow-up questions before retrieving, so score
    # the question it actually used rather than the raw turn.
    query = retrieval_query(result, message)
    scored_docs = retrieve_with_scores(vectorstore, query, k=RETRIEVER_K)
    fig = visualize_sources(scored_docs)

    # Update chat display
    history = history + [[message, result["answer"]]]

    return "", history, fig


def clear_chat():
    """Clear chat history"""
    global chat_history
    chat_history = []
    return [], None


# Create Gradio interface
def create_app():
    with gr.Blocks(title="Event Assistant", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🎪 Event Assistant

        Ask questions about the documents. Sources are shown for transparency (XAI).
        """)

        with gr.Row():
            # Chat column
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=500, show_copy_button=True)

                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Ask a question...",
                        container=False,
                        scale=4,
                        show_label=False
                    )
                    submit_btn = gr.Button("Ask", variant="primary", scale=1)

                clear_btn = gr.Button("Clear Chat")

            # Sources column (XAI)
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Source Documents (XAI)")
                source_plot = gr.Plot(label="Where did the answer come from?")

                gr.Markdown("""
                *The chart shows which documents were used to generate the answer.
                Higher score = more relevant to your question.*
                """)

        # Events
        msg.submit(respond, [msg, chatbot], [msg, chatbot, source_plot])
        submit_btn.click(respond, [msg, chatbot], [msg, chatbot, source_plot])
        clear_btn.click(clear_chat, None, [chatbot, source_plot])

    return app


if __name__ == "__main__":
    qa_chain, vectorstore = setup_chain()
    app = create_app()
    app.launch(share=True)
