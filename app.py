"""
Marbet Event Assistant - Web Interface with XAI

Shows source documents used for answers (Explainable AI)

Run: python app.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import DirectoryLoader, PyPDFDirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM

from config import OLLAMA_SERVER, CHAT_MODEL, EMBED_MODEL, DOCS_FOLDER

SYSTEM_PROMPT = """You are a helpful event assistant.
Answer questions based ONLY on the provided documents.
If the answer is not in the documents, say you don't have that information."""

# Global variables
qa_chain = None
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
    return chain


def visualize_sources(source_docs):
    """
    Create bar chart showing source document relevance.
    This provides XAI - users can see WHERE the answer came from.
    """
    if not source_docs:
        return None
    
    # Get unique sources with their relevance
    sources = {}
    for i, doc in enumerate(source_docs[:10]):
        name = os.path.basename(doc.metadata.get('source', 'Unknown'))
        
        # Truncate long names
        if len(name) > 25:
            name = name[:22] + "..."
        
        # Add page number if PDF
        page = doc.metadata.get('page', None)
        if page is not None:
            name = f"{name} (p.{page})"
        
        # Higher position = more relevant
        relevance = 1.0 - (i * 0.08)
        
        if name not in sources:
            sources[name] = relevance
    
    # Create plot
    names = list(sources.keys())
    scores = list(sources.values())
    colors = plt.cm.Blues(np.linspace(0.5, 1.0, len(names)))
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(names, scores, color=colors)
    
    plt.xlabel('Relevance Score')
    plt.title('Source Documents Used (XAI)')
    plt.xlim(0, 1.1)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    
    # Add score labels
    for bar, score in zip(bars, scores):
        plt.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                 f'{score:.2f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    # Save to temp file
    path = f"./source_viz_{len(source_docs)}.png"
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    
    return path


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
    
    # Create visualization
    viz_path = visualize_sources(result.get("source_documents", []))
    
    # Update chat display
    history = history + [[message, result["answer"]]]
    
    return "", history, viz_path


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
                source_image = gr.Image(label="Where did the answer come from?", type="filepath")
                
                gr.Markdown("""
                *The chart shows which documents were used to generate the answer.
                Higher score = more relevant to your question.*
                """)
        
        # Events
        msg.submit(respond, [msg, chatbot], [msg, chatbot, source_image])
        submit_btn.click(respond, [msg, chatbot], [msg, chatbot, source_image])
        clear_btn.click(clear_chat, None, [chatbot, source_image])
    
    return app


if __name__ == "__main__":
    qa_chain = setup_chain()
    app = create_app()
    app.launch(share=True)