# Folder: Basketball RAG Agent
# Description: Optimized code for CPU speed and statistical accuracy (English Version).

import os
import torch
import sys
import re
import shutil
from glob import glob 
from typing import List
import multiprocessing

from langchain_community.llms import LlamaCpp 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- System Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GGUF_MODEL_PATH = "Phi-3-mini-4k-instruct-Q4_K_M.gguf" 
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
ARTICLES_DIR = './basketball_articles'
CHROMA_DB_DIR = './chroma_db'

def setup_rag_index():
    """Creates or loads the ChromaDB vector store."""
    print("Checking database...") 
    
    if os.path.exists(CHROMA_DB_DIR):
        print(f"  > Found existing database at {CHROMA_DB_DIR}")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_ID)
        vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
        return vectorstore

    print("  > Database not found. Building database...")
    if not os.path.exists(ARTICLES_DIR):
        os.makedirs(ARTICLES_DIR)
        print(f"WARNING: Folder {ARTICLES_DIR} created but is empty.")
        return None
    
    all_files = glob(os.path.join(ARTICLES_DIR, "*.txt"))
    if not all_files:
        print("WARNING: No .txt files found for processing.")
        return None

    final_docs = []
    # Large chunk size for Box Scores to prevent team stats from splitting
    box_splitter = RecursiveCharacterTextSplitter(chunk_size=2800, chunk_overlap=0)
    # Smaller for articles for better retrieval targeting
    article_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    print(f"Processing {len(all_files)} files...")

    for file_path in all_files:
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            raw_docs = loader.load()
            
            if "_box.txt" in file_path:
                splitted = box_splitter.split_documents(raw_docs)
            else:
                splitted = article_splitter.split_documents(raw_docs)
            
            for doc in splitted:
                doc.metadata['source'] = file_path
            
            final_docs.extend(splitted)
        except Exception as e:
            print(f"Error in file {file_path}: {e}")
    
    print(f"  > Creating Embeddings for {len(final_docs)} chunks...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_ID)
    
    vectorstore = Chroma.from_documents(
        documents=final_docs, 
        embedding=embeddings, 
        persist_directory=CHROMA_DB_DIR
    )
    vectorstore.persist()
    print("--- Indexing Completed ---")
    return vectorstore

def load_local_llm():
    """Loads the Phi-3 model with CPU optimizations."""
    if not os.path.exists(GGUF_MODEL_PATH):
        print(f"ERROR: Model file {GGUF_MODEL_PATH} not found.")
        return None

    print(f"--- Loading model on {DEVICE} ---")
    
    # Use multiple threads for speed
    num_cores = multiprocessing.cpu_count()
    cpu_threads = 6 if num_cores >= 8 else 4
    
    return LlamaCpp(
        model_path=GGUF_MODEL_PATH,
        temperature=0.01, 
        max_tokens=150,  # Short answers = higher speed
        n_ctx=2560,      # Balance between memory and context
        n_gpu_layers=-1 if DEVICE.type == 'cuda' else 0, 
        streaming=False,
        verbose=False,
        n_batch=1024,     # Accelerates RAG prompt processing
        n_threads=cpu_threads, 
        # Stop words to prevent the model from hallucinating new questions
        stop=["<|end|>", "<|user|>", "Question:", "\nQuestion:"] 
    )

def create_qa_chain(llm, vectorstore):
    """Creates the RAG QA chain."""
    
    # Prompt template for Phi-3 standards
    template = """<|user|>
    You are an expert Basketball Analyst. Use the provided Context to answer the question briefly and accurately.
    
    GUIDELINES:
    1. SUMMARY: Describe game flow and final score using the narrative articles.
    2. STATS: Use Box Score lines (PLAYER: Name... STATS: PTS...). Match names by LAST name (e.g. 'S. Lee' is 'Saben Lee').
    3. If information is not in context, say 'Information not found.'

    Context:
    {context}

    Question: {question}<|end|>
    <|assistant|>
    """
    
    custom_prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # k=2 is enough with large chunk sizes
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) 

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": custom_prompt},
        return_source_documents=True 
    )

if __name__ == "__main__":
    # Ensure correct terminal buffering
    sys.stdout.reconfigure(line_buffering=True)

    vectorstore = setup_rag_index()
    if not vectorstore: 
        print("Database issue. Exiting.")
        exit()

    llm = load_local_llm()
    if not llm: 
        print("Model loading issue. Exiting.")
        exit()

    qa_agent = create_qa_chain(llm, vectorstore)

    print("\n" + "="*30)
    print("🏀 BASKETBALL AI AGENT READY")
    print("="*30)
    
    while True:
        user_input = input("\n Ask a question or press enter to exit: ")
        if user_input.lower() in ['exit', 'quit']: 
            break
        
        print("Thinking...")
        try:
            result = qa_agent.invoke({"query": user_input})
            answer = result['result'].strip()
            
            # Cleanup if "Answer:" is repeated in output
            if answer.lower().startswith('answer:'):
                answer = answer.split('Answer:')[-1].strip()

            print(f"\n\n{answer}")
            
            # Display source files used
            sources = [os.path.basename(doc.metadata.get('source', '')) for doc in result['source_documents']]
            print(f"\n[Sources]: {list(set(sources))}")
            
        except Exception as e:
            print(f"Error: {e}")