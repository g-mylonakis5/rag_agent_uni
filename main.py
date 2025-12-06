import os
import torch
import sys
import re
from typing import List

# LangChain Core/Community Imports
from langchain_community.llms import LlamaCpp 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from glob import glob 

# NOTE: We use LlamaCpp and GGUF for speed/stability instead of slow HuggingFacePipeline

# --- CONFIGURATION (SETTINGS) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. LLM (Using GGUF - Switched to Mistral 7B for stability and better reasoning)
GGUF_MODEL_PATH = "mistral-7b-instruct-v0.2.Q4_K_S.gguf" 

# 2. Embedding Model (Required for RAG)
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# 3. Paths
ARTICLES_DIR = './basketball_articles'
CHROMA_DB_DIR = './chroma_db'


# --- 1. RAG INDEXING (LOAD AND STORE DATA) ---

def setup_rag_index():
    """Reads articles, converts them to vectors (vectors) and stores them in ChromaDB."""
    print("--- 1. RAG INDEXING: Starting data loading ---") 
    
    if not os.path.exists(ARTICLES_DIR):
        print(f"Error: Folder {ARTICLES_DIR} does not exist. Create it and place .txt files.")
        return None
    
    # 1. Find all .txt files
    all_files = glob(os.path.join(ARTICLES_DIR, "*.txt"))
    if not all_files:
        print("No articles found. Please add .txt files to the folder.")
        return None

    # 2. Load each file separately with TextLoader
    documents = []
    for file_path in all_files:
        try:
            # Use TextLoader with utf-8 for Greek characters
            loader = TextLoader(file_path, encoding='utf-8')
            documents.extend(loader.load())
        except Exception as e:
            print(f"Warning: Failed to load file {file_path}. Error: {e}", file=sys.stderr)
    
    if not documents:
        print("No valid documents found for processing. Check character encoding.", file=sys.stderr)
        return None
        
    # Text splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    
    # Load Embedding Model
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_ID)
    
    # Create and store Vector Store
    vectorstore = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings, 
        persist_directory=CHROMA_DB_DIR
    )
    vectorstore.persist()
    print("--- RAG Indexing Completed ---")
    return vectorstore

# --- 2. LLM SETUP (LlamaCpp GGUF LOADING) ---

def load_local_llm():
    """Loads the quantized LLM (GGUF) via LlamaCpp for fast CPU execution."""
    print(f"--- 2. LLM SETUP: Loading quantized model from {GGUF_MODEL_PATH} ---")
    
    if not os.path.exists(GGUF_MODEL_PATH):
        print(f"Error: GGUF model file not found: {GGUF_MODEL_PATH}", file=sys.stderr)
        return None

    try:
        # LlamaCpp wrapper configuration
        llm = LlamaCpp(
            model_path=GGUF_MODEL_PATH,
            temperature=0.05, # Kept low for accuracy
            max_tokens=2048, 
            n_gpu_layers=-1 if DEVICE.type == 'cuda' else 0,
            n_ctx=4096, 
            streaming=False, # Disabled streaming for stability
            verbose=False,
            n_batch=512,     # Batch size for stability
        )
        print("--- LLM LlamaCpp loaded successfully ---")
        return llm
    except Exception as e:
        print(f"Error loading LlamaCpp: {e}", file=sys.stderr)
        print("Check if you have installed llama-cpp-python correctly.", file=sys.stderr)
        return None

# --- 3. AGENT EXECUTION (RETRIEVALQA CHAIN - THE STABLE WAY) ---

def create_qa_chain(llm, vectorstore):
    """Defines the stable RAG chain (RetrievalQA Chain) of LangChain."""
    # CUSTOM PROMPT TEMPLATEd - FINAL VERSION FOR ACCURACY AND BOX SCORES
    template = """
    You are an extremely meticulous sports data analyst. Your goal is to extract EXACT player statistics and game information from the provided CONTEXT.
    
    RULES:
    1. STRICTLY use ONLY the facts present in the Context. If a name or stat is not present, DO NOT GUESS (NO HALLUCINATION).
    2. The Context contains structured data (e.g., PLAYER: N. Mitoglou (Panathinaikos) - STATS: PTS: 9, REB: 6). You MUST use this structured format for stat extraction.
    3. Before providing any statistics for a player, you MUST VERIFY the full player name and the specific team/game context within the context. If they do not match, state 'Information not found in the provided articles.'
    4. When extracting statistics, provide them as a concise list of KEY-VALUE pairs (e.g., "PTS: 29, Rebounds: 8, 3FG%: 83.3%").
    5. If the user asks for the SCORE, provide only the final score in the format: "FINAL SCORE: Team A Score A - Team B Score B".
    6. Answer in English.

    Context:
    {context}

    Question: {question}
    Answer:
    """
    custom_prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # Create Retriever (Retrieval)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Create the Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": custom_prompt},
        return_source_documents=True # Returns the articles used
    )
    return qa_chain


# --- MAIN CODE (MAIN CODE) ---

if __name__ == "__main__":
    # 1. Prepare RAG Index
    vectorstore = setup_rag_index()
    if vectorstore is None:
        exit()

    # 2. Load LLM 
    llm = load_local_llm()
    if llm is None:
        exit()

    # 3. Create Agent Chain (The stable RAG Chain)
    qa_agent = create_qa_chain(llm, vectorstore)

    # 4. Question and Answer Loop
    print("\n--- STABLE RAG AGENT READY (RAG only). Enter questions (or 'exit' to quit) ---")
    print("Example 1: What was the score of Panathinaikos-Partizan?")
    print("Example 2: What were Vezenkov's points and total rebounds in the Olympiacos-Zvezda game?")
    
    while True:
        user_input = input("Question: ")
        if user_input.lower() == 'exit':
            break
            
        # Execute the Agent
        print("Agent thinking...")
        
        try:
            # We call the RetrievalQA chain
            result = qa_agent.invoke({"query": user_input})

            print("\n[AGENT RESPONSE]")
            
            clean_output = result['result'].strip()
            
            # Simple cleaning for LlamaCpp output
            if clean_output.lower().startswith('answer:'):
                clean_output = clean_output.split('Answer:')[-1].strip()
            if clean_output.lower().startswith('question:'):
                clean_output = clean_output.split('Answer:')[-1].strip()

            # --- Score Extraction (Feature) ---
            # NOTE: We keep the score extraction logic here
            score_pattern = r"(\b[A-Za-z]+\s?[A-Za-z]*\s?[A-Za-z]*\s?[A-Za-z]*\s?[A-Za-z]*)\s(\d+)\s-\s(\b[A-Za-z]+\s?[A-Za-z]*\s?[A-Za-z]*\s?[A-Za-z]*\s?[A-Za-z]*)\s(\d+)"
            match = re.search(score_pattern, clean_output, re.IGNORECASE)
            
            if match:
                team_a = match.group(1).strip()
                score_a = match.group(2).strip()
                team_b = match.group(3).strip()
                score_b = match.group(4).strip()
                
                print(f"FINAL SCORE: {team_a} {score_a} - {team_b} {score_b}")
            else:
                print(clean_output)
            
            # Show sources (optional)
            sources = [doc.metadata.get('source', 'Unknown Source') for doc in result['source_documents']]
            print("\n[Used Article Sources]:")
            print(set(sources))
            print("-" * 50)
            
        except Exception as e:
            print(f"An error occurred during response generation: {e}", file=sys.stderr)
            print("-" * 50)