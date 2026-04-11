# Αρχείο: main.py
# Περιγραφή: Hybrid RAG Agent (Journalist for Summaries & Strict Analyst for Stats)

import os
import torch
import sys
import shutil
from glob import glob 
import multiprocessing

from langchain_community.llms import LlamaCpp 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

# --- System Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GGUF_MODEL_PATH = "Phi-3-mini-4k-instruct-Q4_K_M.gguf" 
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_PATH = './chroma_db'

# Αρχικοποίηση Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_ID, 
    model_kwargs={'device': DEVICE}
)

def setup_rag_index(rebuild=False):
    if rebuild and os.path.exists(CHROMA_PATH):
        print("Cleaning up old index...")
        shutil.rmtree(CHROMA_PATH)

    if not os.path.exists(CHROMA_PATH):
        print("🚀 Creating new index from scratch...")
        
        data_folders = [
            'data/summaries',       
            'data/box_scores',      
            'data/global_metadata'  
        ]

        documents = []
        for folder in data_folders:
            folder_path = os.path.join(os.getcwd(), folder)
            if not os.path.exists(folder_path):
                print(f"⚠️ Warning: Folder '{folder}' not found.")
                continue
            
            files = glob(os.path.join(folder_path, "*.txt")) + glob(os.path.join(folder_path, "*.csv"))
            
            for file_path in files:
                try:
                    if file_path.endswith('.csv'):
                        loader = CSVLoader(file_path=file_path, encoding='utf-8')
                    else:
                        loader = TextLoader(file_path=file_path, encoding='utf-8')
                    documents.extend(loader.load())
                    print(f"✅ Loaded: {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"❌ Error loading {file_path}: {e}")

        if not documents:
            print("❌ No documents found!")
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)
        
        vectorstore = Chroma.from_documents(
            documents=docs, 
            embedding=embeddings, 
            persist_directory=CHROMA_PATH
        )
        print("✨ Indexing complete!")
        # Χρήση MMR για καλύτερη κάλυψη δεδομένων (CSV + TXT)
        return vectorstore.as_retriever(
            search_type="mmr", 
            search_kwargs={"k": 12, "fetch_k": 20}
        )
    
    else:
        print("📂 Loading existing index from disk...")
        vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
        return vectorstore.as_retriever(
            search_type="mmr", 
            search_kwargs={"k": 12, "fetch_k": 20}
        )

def load_local_llm():
    if not os.path.exists(GGUF_MODEL_PATH):
        print(f"❌ Model nott found at {GGUF_MODEL_PATH}")
        return None

    print(f"--- Loading model on {DEVICE} ---")
    num_cores = multiprocessing.cpu_count()
    cpu_threads = max(1, num_cores - 2) 
    
    return LlamaCpp(
        model_path=GGUF_MODEL_PATH,
        temperature=0.2, # Χαμηλότερο για μεγαλύτερη ακρίβεια στα νούμερα
        max_tokens=512,      
        n_ctx=4096,      
        n_gpu_layers=0, 
        streaming=False, 
        verbose=False,
        n_batch=512,     
        n_threads=cpu_threads, 
        stop=["<|end|>", "<|user|>", "Question:"] 
    )

def create_qa_chain(llm, retriever_obj):
    qa_template = (
        "<|user|>\n"
        "You are a Basketball Journalist and Data Analyst. Answer using ONLY the provided Context.\n\n"
        "RULES:\n"
        "1. GREETINGS: If the user says 'Hello', respond politely and ask for basketball questions.\n"
        "2. SUMMARIES: For a 'summary', write a narrative paragraph based on TXT files.\n"
        "3. STATS: For 'stats', list EVERY metric from the CSV (PTS, REB, AST, STL, TO, PIR, 2FG, 3FG, FT, MIN). "
        "DO NOT guess full names. Use the exact format found in the data (e.g., if it says K. Nunn, write K. Nunn).\n"
        "4. MISSING DATA: If info is missing, say 'Information not found'.\n"
        "5. GENERAL INFO: Answer about coaches/arenas strictly from context.\n"
        "6. JOURNALISTIC COMMENT: After listing a player's stats,ALWAYS add a  comment based ONLY on those numbers.\n"
        "7. DO NOT repeat these instructions in your response.\n\n"
        "Context:\n"
        "{context}\n\n"
        "Question: {question}<|end|>\n"
        "<|assistant|>"
    )
    QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["context", "question"])

    condense_template = (
        "<|user|>\n"
        "Rephrase the follow-up question to be a standalone search query.\n"
        "Chat History: {chat_history}\n"
        "Follow-up: {question}<|end|>\n"
        "<|assistant|>"
    )
    CONDENSE_PROMPT = PromptTemplate(template=condense_template, input_variables=["chat_history", "question"])

    memory = ConversationBufferWindowMemory(
        k=2, 
        memory_key="chat_history", 
        return_messages=True, 
        output_key='answer'
    )
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever_obj, 
        memory=memory,
        condense_question_prompt=CONDENSE_PROMPT,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True
    )

if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)

    print("\n" + "="*30)
    user_rebuild = input("Rebuild database? (y/n): ").strip().lower()
    rebuild_flag = True if user_rebuild == 'y' else False

    retriever = setup_rag_index(rebuild=rebuild_flag)
    if not retriever: exit()

    llm = load_local_llm()
    if not llm: exit()

    qa_agent = create_qa_chain(llm, retriever)

    print("\n" + "="*30)
    print("🏀 BASKETBALL AI AGENT READY")
    print("="*30)
    
    while True:
        user_input = input("\nAsk a question: ")
        if user_input.lower() in ['exit', 'quit']: break
        
        print("Thinking...")
        try:
            result = qa_agent.invoke({"question": user_input})
            answer = result['answer'].strip()
            print(f"\n[RESPONSE]:\n{answer}")
            
            sources = [os.path.basename(doc.metadata.get('source', '')) for doc in result['source_documents']]
            print(f"\n[Sources Used]: {list(set(sources))}")
        except Exception as e:
            print(f"\nError: {e}")