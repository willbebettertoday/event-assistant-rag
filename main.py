"""Marbet Event Assistant, command line interface.

Authors: Oleksii Krasnoshtanov, Danil Sysenko
University: Breda University of Applied Sciences

This is the CLI. For the web interface with the source chart, run app.py.
"""

import os

from src.pipeline import (
    build_chain,
    build_vectorstore,
    load_documents,
    load_vectorstore,
    save_vectorstore,
    split_documents,
)

# Load config
try:
    from config import (
        CACHE_DIR,
        CHAT_MODEL,
        CHUNK_OVERLAP,
        CHUNK_SIZE,
        DOCS_FOLDER,
        EMBED_MODEL,
        OLLAMA_SERVER,
        RETRIEVER_K,
        TEMPERATURE,
        USE_CACHE,
    )
except ImportError:
    print("config.py not found! Copy config_example.py to config.py")
    exit(1)

# Create folders
os.makedirs(DOCS_FOLDER, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


def rebuild_vectorstore():
    """Load documents from disk, chunk them and build a fresh vector store."""
    from langchain_ollama import OllamaEmbeddings

    print("Loading documents...")
    all_docs = load_documents(DOCS_FOLDER)

    if len(all_docs) == 0:
        print("No documents found! Add PDF or TXT files to docs/ folder.")
        exit(1)

    print("Splitting into chunks...")
    chunks = split_documents(all_docs, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"  Created {len(chunks)} chunks")

    print("Creating embeddings (this may take a minute)...")
    embeddings = OllamaEmbeddings(base_url=OLLAMA_SERVER, model=EMBED_MODEL)

    vectorstore = build_vectorstore(chunks, embeddings)

    print("Saving to cache...")
    save_vectorstore(vectorstore, CACHE_DIR)

    return vectorstore


def get_vectorstore():
    """Load the cached vector store, or build one if there is none to load."""
    from langchain_ollama import OllamaEmbeddings

    if USE_CACHE:
        embeddings = OllamaEmbeddings(base_url=OLLAMA_SERVER, model=EMBED_MODEL)
        print("Loading from cache...")
        try:
            vectorstore = load_vectorstore(CACHE_DIR, embeddings)
        except Exception as e:
            print(f"  Cache error: {e}")
            vectorstore = None

        if vectorstore is not None:
            print("  Loaded successfully!")
            return vectorstore
        print("  No cache found. Rebuilding...")

    return rebuild_vectorstore()


def main():
    # Load or build vectorstore
    vectorstore = get_vectorstore()

    # Setup chain
    qa_chain = build_chain(vectorstore, OLLAMA_SERVER, CHAT_MODEL, TEMPERATURE, RETRIEVER_K)

    # Chat loop
    print()
    print("=" * 50)
    print("EVENT ASSISTANT".center(50))
    print("=" * 50)
    print("Commands: 'exit' - quit, 'rebuild' - reload documents")
    print()

    chat_history = []

    while True:
        query = input("You: ").strip()

        if not query:
            continue

        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        # Rebuild cache if requested
        if query.lower() == "rebuild":
            print("Rebuilding vectorstore...")
            vectorstore = rebuild_vectorstore()
            qa_chain = build_chain(vectorstore, OLLAMA_SERVER, CHAT_MODEL, TEMPERATURE, RETRIEVER_K)
            print("Done! You can ask questions now.\n")
            continue

        try:
            result = qa_chain.invoke({"question": query, "chat_history": chat_history})

            print(f"\nAssistant: {result['answer']}")

            # Show sources
            sources = result.get("source_documents", [])
            if sources:
                source_names = set()
                for doc in sources[:5]:
                    name = os.path.basename(doc.metadata.get("source", "Unknown"))
                    source_names.add(name)
                print(f"Sources: {', '.join(source_names)}")
            print()

            # Update history
            chat_history.append((query, result["answer"]))
            if len(chat_history) > 10:
                chat_history = chat_history[-10:]

        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
