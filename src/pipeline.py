"""The retrieval-augmented generation pipeline shared by main.py and app.py.

Both entry points used to build this by hand, with copy-pasted loaders and
the same hard-coded chunk size, overlap, k and temperature. Collecting it
here means there is exactly one place that builds a chain, and exactly one
place that decides how the vector store is cached.

This module does not remove pickle from the cache path. It replaces a
hand-rolled use of Python's pickle module to serialise and deserialise a
live vector store object with FAISS's own save_local/load_local, which is
a documented format rather than an ad hoc one, but that format still
pickles the docstore internally. The improvement is a known, versioned
format and an explicit opt-in (allow_dangerous_deserialization) at the one
place that reads it, not the removal of deserialisation risk. See
load_vectorstore for the trust boundary that makes accepting that risk
reasonable here.

LangChain, FAISS and Ollama are imported lazily, inside the functions that
need them, so importing this module costs nothing. tests/test_pipeline.py
imports it without any of those packages installed.
"""

import os

SYSTEM_PROMPT = """You are a helpful event assistant.
Answer questions based ONLY on the provided documents.
If the answer is not in the documents, say you don't have that information."""


def load_documents(docs_folder):
    """Load all PDF and TXT documents from docs_folder."""
    from langchain_community.document_loaders import (
        DirectoryLoader,
        PyPDFDirectoryLoader,
        TextLoader,
    )

    all_docs = []

    try:
        pdf_loader = PyPDFDirectoryLoader(docs_folder, extract_images=False)
        pdf_docs = pdf_loader.load()
        all_docs.extend(pdf_docs)
        print(f"  Loaded {len(pdf_docs)} PDF pages")
    except Exception as e:
        print(f"  PDF error: {e}")

    try:
        txt_loader = DirectoryLoader(
            docs_folder,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        txt_docs = txt_loader.load()
        all_docs.extend(txt_docs)
        print(f"  Loaded {len(txt_docs)} TXT files")
    except Exception as e:
        print(f"  TXT error: {e}")

    return all_docs


def split_documents(docs, chunk_size, chunk_overlap):
    """Split documents into chunks of chunk_size with chunk_overlap."""
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)


def build_vectorstore(chunks, embeddings):
    """Build a FAISS vector store from already-split chunks."""
    from langchain_community.vectorstores import FAISS

    return FAISS.from_documents(chunks, embeddings)


def save_vectorstore(vectorstore, cache_dir):
    """Persist the vector store with FAISS's own save_local.

    Not a hand-rolled pickle of the vector store object: save_local is
    FAISS's documented, versioned format for exactly this. It still writes
    a pickled docstore inside cache_dir; see load_vectorstore for what that
    means for the read side.
    """
    vectorstore.save_local(cache_dir)


def load_vectorstore(cache_dir, embeddings):
    """Load a cached vector store, or return None when no cache exists.

    FAISS's local format stores its docstore as a pickle, so loading it
    runs whatever code that pickle contains. allow_dangerous_deserialization
    must be True for load_local to work at all; passing False does not make
    loading safe, it makes loading raise unconditionally. Accepting that risk
    here is reasonable only because this application is the sole writer of
    cache_dir: it is written by save_vectorstore above and by nothing else.
    It would not be reasonable for a cache directory that is shared, synced
    from another machine, or received from anyone else.
    """
    index_file = os.path.join(cache_dir, "index.faiss")
    if not os.path.exists(index_file):
        return None

    from langchain_community.vectorstores import FAISS

    return FAISS.load_local(cache_dir, embeddings, allow_dangerous_deserialization=True)


def build_chain(
    vectorstore, ollama_server, chat_model, temperature, k, system_prompt=SYSTEM_PROMPT
):
    """Build the conversational retrieval chain used by both entry points.

    system_prompt defaults to the module-level SYSTEM_PROMPT above, the last
    thing that used to be copy-pasted between main.py and app.py verbatim;
    both entry points now just take the default instead of declaring it.

    return_generated_question=True is required here, not optional: the
    chain condenses a follow-up turn into a standalone question before
    retrieving, and callers that want to score the documents behind an
    answer need that condensed question rather than the raw turn.
    """
    from langchain.chains import ConversationalRetrievalChain
    from langchain_ollama import OllamaLLM

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    llm = OllamaLLM(
        base_url=ollama_server,
        model=chat_model,
        temperature=temperature,
        system=system_prompt,
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        return_generated_question=True,
    )
