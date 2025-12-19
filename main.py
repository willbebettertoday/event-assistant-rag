"""
Marbet Event Assistant - Simple CLI Version

Authors: Oleksii Krasnoshtanov, Danil Sysenko
University: Breda University of Applied Sciences
"""

import os
import pickle
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import DirectoryLoader, PyPDFDirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM

# Load config
try:
    from config import OLLAMA_SERVER, CHAT_MODEL, EMBED_MODEL, DOCS_FOLDER, CACHE_DIR
except ImportError:
    print("config.py not found! Copy config_example.py to config.py")
    exit(1)

# Create folders
os.makedirs(DOCS_FOLDER, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

SYSTEM_PROMPT = """You are a helpful event assistant.
Answer questions based ONLY on the provided documents.
If the answer is not in the documents, say you don't have that information."""

CACHE_FILE = os.path.join(CACHE_DIR, "vectorstore.pkl")


def load_documents():
    """Load all PDF and TXT documents"""
    all_docs = []
    
    # Load PDFs
    try:
        pdf_loader = PyPDFDirectoryLoader(DOCS_FOLDER, extract_images=False)
        pdf_docs = pdf_loader.load()
        all_docs.extend(pdf_docs)
        print(f"  Loaded {len(pdf_docs)} PDF pages")
    except Exception as e:
        print(f"  PDF error: {e}")
    
    # Load TXT files
    try:
        txt_loader = DirectoryLoader(
            DOCS_FOLDER,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        txt_docs = txt_loader.load()
        all_docs.extend(txt_docs)
        print(f"  Loaded {len(txt_docs)} TXT files")
    except Exception as e:
        print(f"  TXT error: {e}")
    
    return all_docs


def build_vectorstore():
    """Build vectorstore from documents"""
    print("Loading documents...")
    all_docs = load_documents()
    
    if len(all_docs) == 0:
        print("No documents found! Add PDF or TXT files to docs/ folder.")
        exit(1)
    
    # Split into chunks
    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=300)
    chunks = splitter.split_documents(all_docs)
    print(f"  Created {len(chunks)} chunks")
    
    # Create embeddings
    print("Creating embeddings (this may take a minute)...")
    embeddings = OllamaEmbeddings(base_url=OLLAMA_SERVER, model=EMBED_MODEL)
    
    # Create vectorstore
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Save to cache
    print("Saving to cache...")
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(vectorstore, f)
    
    return vectorstore


def load_vectorstore():
    """Load vectorstore from cache or build new one"""
    
    # Check if cache exists
    if os.path.exists(CACHE_FILE):
        print("Loading from cache...")
        try:
            with open(CACHE_FILE, 'rb') as f:
                vectorstore = pickle.load(f)
            print("  Loaded successfully!")
            return vectorstore
        except Exception as e:
            print(f"  Cache error: {e}")
            print("  Rebuilding...")
    
    # Build new vectorstore
    return build_vectorstore()


def main():
    # Load or build vectorstore
    vectorstore = load_vectorstore()
    
    # Setup retriever and LLM
    retriever = vectorstore.as_retriever(search_kwargs={"k": 12})
    llm = OllamaLLM(
        base_url=OLLAMA_SERVER,
        model=CHAT_MODEL,
        temperature=0.8,
        system=SYSTEM_PROMPT
    )
    
    # Create QA chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
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
            vectorstore = build_vectorstore()
            retriever = vectorstore.as_retriever(search_kwargs={"k": 12})
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
            )
            print("Done! You can ask questions now.\n")
            continue
        
        try:
            result = qa_chain.invoke({
                "question": query,
                "chat_history": chat_history
            })
            
            print(f"\nAssistant: {result['answer']}")
            
            # Show sources
            sources = result.get("source_documents", [])
            if sources:
                source_names = set()
                for doc in sources[:5]:
                    name = os.path.basename(doc.metadata.get('source', 'Unknown'))
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