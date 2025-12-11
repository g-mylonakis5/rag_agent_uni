#imports
import os
import torch
import sys
import re
import shutil
from glob import glob 
from typing import List


from langchain_community.llms import LlamaCpp 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") #running on cpu 
GGUF_MODEL_PATH = "mistral-7b-instruct-v0.2.Q4_K_S.gguf" 
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
ARTICLES_DIR = './basketball_articles'
CHROMA_DB_DIR = './chroma_db'

# Rag indexing 
def setup_rag_index():
    print("--- 1. RAG INDEXING: Checking database ---") 
    
    
    if os.path.exists(CHROMA_DB_DIR):
        print(f"  > Found existing database at {CHROMA_DB_DIR}")
        print("  > Loading vector store... ")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_ID)
        vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
        return vectorstore

    #in case that database doesn't exist/not found 
    print("  > Database not found. Building from scratch...")
    
    if not os.path.exists(ARTICLES_DIR):
        print(f"Error: Folder {ARTICLES_DIR} does not exist.")
        return None
    
    all_files = glob(os.path.join(ARTICLES_DIR, "*.txt"))
    if not all_files:
        print("No articles found.")
        return None

    final_docs = []

    #2200 chars for box scores 
    box_splitter = RecursiveCharacterTextSplitter(chunk_size=2200, chunk_overlap=0)
    #1000 chars for game articles 
    article_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

    print(f"Found {len(all_files)} files. Processing...")

    for file_path in all_files:
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            raw_docs = loader.load()
            
            if "_box.txt" in file_path:
                print(f"  > Box Score: {os.path.basename(file_path)}")
                splitted = box_splitter.split_documents(raw_docs)
            else:
                print(f"  > Article: {os.path.basename(file_path)}")
                splitted = article_splitter.split_documents(raw_docs)
            
            for doc in splitted:
                doc.metadata['source'] = file_path
            
            final_docs.extend(splitted)

        except Exception as e:
            print(f"Warning: Failed to load file {file_path}. Error: {e}", file=sys.stderr)
    
    print(f"  > Embedding {len(final_docs)} chunks...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_ID)
    
    vectorstore = Chroma.from_documents(
        documents=final_docs, 
        embedding=embeddings, 
        persist_directory=CHROMA_DB_DIR
    )
    vectorstore.persist()
    print("--- RAG Indexing Completed ---")
    return vectorstore

#setup
def load_local_llm():
    if not os.path.exists(GGUF_MODEL_PATH):
        print(f"Error: Model {GGUF_MODEL_PATH} not found.")
        return None

    print(f"--- Loading model on {DEVICE} ---")
    
    return LlamaCpp(
        model_path=GGUF_MODEL_PATH,
        temperature=0.01, #low temperature for accurate results strictly from the articles provided 
        max_tokens=600,  #Increased slightly for longer/better summaries
        n_ctx=4096,      
        n_gpu_layers=-1 if DEVICE.type == 'cuda' else 0, 
        streaming=False,
        verbose=False,
        n_batch=512,
    )

#Appropriate prompt for summaries and stat retrieval (used Gemini to find the best prompt)
def create_qa_chain(llm, vectorstore):
    
    # prompt
    template = """
    You are an expert Sports Journalist and Data Analyst. 
    
    INSTRUCTIONS:
    1. **IF ASKED FOR A SUMMARY / "WHAT HAPPENED":**
       - You MUST read the narrative Article text.
       - Describe the game flow: Who won? Was it close? Who made the key plays?
       - Mention the Final Score.
       - Do NOT just list numbers. Tell the story.

    2. **IF ASKED FOR SPECIFIC STATS (e.g., "Who led?", "Points", "Leading Scorer"):**
       - Look strictly at the Box Score lines: "PLAYER: Name... STATS:..."
       - For "Leading Scorer", specifically compare the "PTS" values of ALL players in the context.
       - You MUST find the single highest value to identify the leader.
       
    3.   **IF ASKED FOR GENERAL KNOWLEDGE:**
       -Use the articles provided to you to answer CORRECTLY to questions such as "Who is the head coach of Olympiacos?"

    4. **GENERAL:**
       - Use ONLY the provided context.
       - If the info is missing, say "Information not found."

    Context:
    {context}

    Question: {question}
    Answer:
    """
    
    custom_prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) # 1 game article chunk + 1 box score chunk 

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": custom_prompt},
        return_source_documents=True 
    )
    return qa_chain

#execution
if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)

    vectorstore = setup_rag_index()
    if not vectorstore: exit()

    llm = load_local_llm()
    if not llm: exit()

    qa_agent = create_qa_chain(llm, vectorstore)

    print("\n--- BASKETBALL RAG AGENT ---")
    
    while True:
        user_input = input("\nQuestion (or 'exit'): ")
        if user_input.lower() == 'exit': break
        
        print("Agent thinking...")
        try:
            result = qa_agent.invoke({"query": user_input})
            clean_output = result['result'].strip()
            
            if clean_output.lower().startswith('answer:'):
                clean_output = clean_output.split('Answer:')[-1].strip()

            print(f"\n[RESPONSE]:\n{clean_output}")
            
            sources = [os.path.basename(doc.metadata.get('source', '')) for doc in result['source_documents']]
            print(f"[Source File]: {list(set(sources))}") #tells the user which files were used by the model for each prompt 
            
        except Exception as e:
            print(f"Error: {e}")