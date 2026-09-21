"""
Marbet Event Assistant - Web Interface with XAI

Shows source documents used for answers (Explainable AI)

Run: python app.py
"""

import gradio as gr
import matplotlib
import numpy as np
from config import CHAT_MODEL, DOCS_FOLDER, EMBED_MODEL, OLLAMA_SERVER
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFDirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM

matplotlib.use("Agg")  # Gradio renders server side; no interactive backend
import matplotlib.pyplot as plt  # noqa: E402

from src.retrieval import distances_to_similarity, retrieve_with_scores, source_label  # noqa: E402

SYSTEM_PROMPT = """You are a helpful event assistant.
Answer questions based ONLY on the provided documents.
If the answer is not in the documents, say you don't have that information."""

# Global variables
qa_chain = None
vectorstore = None
chat_history = []

def setup_chain():
    """Load documents and create QA chain"""
    print("Loading documents...")
    all_docs = []

    # PDFs
    try:
        pdf_loader = PyPDFDirectoryLoader(DOCS_FOLDER, extract_images=False)
        all_docs.extend(pdf_loader.load())
    except Exception as e:
        print(f"PDF error: {e}")

    # TXT
    try:
        txt_loader = DirectoryLoader(
            DOCS_FOLDER, glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        all_docs.extend(txt_loader.load())
    except Exception as e:
        print(f"TXT error: {e}")

    print(f"Loaded {len(all_docs)} documents")

    # Split
    print("Creating chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=300)
    chunks = splitter.split_documents(all_docs)
    print(f"Created {len(chunks)} chunks")

    # Embeddings and vectorstore
    print("Building vector store...")
    embeddings = OllamaEmbeddings(base_url=OLLAMA_SERVER, model=EMBED_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Chain
    retriever = vectorstore.as_retriever(search_kwargs={"k": 12})
    llm = OllamaLLM(
        base_url=OLLAMA_SERVER,
        model=CHAT_MODEL,
        temperature=0.8,
        system=SYSTEM_PROMPT
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
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
    result = qa_chain.invoke({
        "question": message,
        "chat_history": chat_history
    })

    # Update history
    chat_history.append((message, result["answer"]))
    if len(chat_history) > 10:
        chat_history = chat_history[-10:]

    # Create visualization from the real retrieval scores. The chain's
    # source_documents carry no scores, so re-retrieve through the
    # vectorstore directly to get the FAISS distances behind the chart.
    scored_docs = retrieve_with_scores(vectorstore, message, k=12)
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
