# 
#  Hybrid RAG Agent (Journalist for Summaries & Strict Analyst for Stats)

import os
import torch
import sys
import shutil
from glob import glob 
import multiprocessing
#imports 
from langchain_community.llms import LlamaCpp 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GGUF_MODEL_PATH = "Phi-3-mini-4k-instruct-Q4_K_M.gguf" 
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_PATH = './chroma_db'

# Embeddings 
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_ID, 
    model_kwargs={'device': DEVICE}
)

def setup_rag_index(rebuild=False):
    if rebuild and os.path.exists(CHROMA_PATH):
        print("Cleaning up old index...")
        shutil.rmtree(CHROMA_PATH)

    if not os.path.exists(CHROMA_PATH):
        print(" Creating new index...")
        
        data_folders = [
            'data/summaries',       
            'data/box_scores',      
            'data/global_metadata'  
        ]

        documents = []
        for folder in data_folders:
            folder_path = os.path.join(os.getcwd(), folder)
            if not os.path.exists(folder_path):
                continue
            
            files = glob(os.path.join(folder_path, "*.txt")) + glob(os.path.join(folder_path, "*.csv"))
            
            for file_path in files:
                try:
                    # Declaring type based on file 
                    if 'summaries' in folder:
                        file_description = "detailed match narrative, game report, storytelling, and play-by-play summary"
                    elif 'global_metadata' in folder:
                        file_description = "arenas,stadiums,venues,locations, head coaches and general league information"
                    elif 'box_scores' in folder:
                        file_description = "match statistics boxscore"
                    else:
                        file_description = "data"

                    if file_path.endswith('.csv'):
                        loader = CSVLoader(file_path=file_path, encoding='utf-8')
                    else:
                        loader = TextLoader(file_path=file_path, encoding='utf-8')

                    loaded_docs = loader.load()
                    for doc in loaded_docs:
                        filename = os.path.basename(file_path)
                        # Adding right info to content 
                        doc.page_content = f"Type: {file_description}\nContent: {doc.page_content}"
                    
                    documents.extend(loaded_docs)
                    print(f" Loaded ({folder}): {os.path.basename(file_path)}")
                except Exception as e:
                    print(f" Error loading {file_path}: {e}")
                except Exception as e:
                    print(f" Error loading {file_path}: {e}")
        if not documents:
            print(" No documents found!")
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1100, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        
        vectorstore = Chroma.from_documents(
            documents=docs, 
            embedding=embeddings, 
            persist_directory=CHROMA_PATH
        )
        print(" Indexing complete.")
        return vectorstore.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 4}
        )
    
    else:
        print(" Loading existing index from disk...")
        vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
        return vectorstore.as_retriever(
            search_type="similarity", #tried mmr but similarity is better 
            search_kwargs={"k": 4}
        )

def load_local_llm():
    if not os.path.exists(GGUF_MODEL_PATH):
        print(f" Model not found at {GGUF_MODEL_PATH}")
        return None

    print(f"Loading model ")
    
    # Threads based on CPU
    num_cores = multiprocessing.cpu_count()
    cpu_threads = max(1, num_cores - 2) 
    #LLamaCpp configuration
    return LlamaCpp(
        model_path=GGUF_MODEL_PATH,
        temperature=0.01,
        max_tokens=750,
        n_ctx=2048,
        n_gpu_layers=0,
        n_batch=512,
        n_threads=cpu_threads, 
        repeat_penalty=1.15,
        verbose=False
    )
def create_qa_chain(llm, retriever_obj):
    qa_template = (
        "<|user|>\n"
        "You are a EuroLeague Master Analyst & Journalist. Use the provided Context to answer the question with absolute accuracy.\n\n"
        
        "###  MANDATORY RULES (DO NOT IGNORE):\n"
        "1. **NO CITATIONS**: Never mention filenames like '.txt', '.csv', or '(source: ...)' in your text. Write like a professional news report.\n"
        "2. **MATCH ISOLATION**: Identify the teams in the question. If the Context contains info about OTHER teams , DISCARD it. Only use data for the specific match requested.\n"
        "3. **GEOGRAPHY & NAMES**: Use names EXACTLY as they appear in the Context. NEVER invent, combine, or add prefixes to names (e.g., if CSV says 'Sasha Vezenkov', do NOT write 'Aaron Vezenkov')."
        "4. **PROFESSIONAL BALANCE**: For summaries, provide a solid narrative (1-2 paragraphs), BASED ON THE .TXT FILES. For specific questions (coaches, scores), be brief (1-2 sentences)."
        
                
        "### INTENT MATCHING:\n"
        "- If the user asks a SPECIFIC question (e.g., 'Who is the coach', 'What is the score'), answer ONLY that question  using the metadata or the stats.\n"
        "- ONLY provide a 2-paragraph summary if the user explicitly asks for a 'summary', 'highlights', or 'what happened'.\n"
        
        "###  1. SUMMARIES & NARRATIVES (.txt files)\n"
        "When asked for a 'summary' or 'what happened':\n"
        "- **Storytelling**: Provide a detailed 2-paragraph narrative based on the '.txt' summary.\n"
        "- **Key Events**: Include the final score, winner, scoring swings, and crucial plays in the final minute (e.g., misses, clutch shots), if available to you..\n\n"
        
        "###  2. PLAYER STATISTICS (.csv files)\n"
        "When asked for stats or individual performance:\n"
        "- **Full Boxscore**: Provide ALL stats from the '.csv' (PTS, REB, AST, 3FG, PIR, etc.).\n"
        "- **Hybrid Comment**: \n"
        "  * If the player is in the '.txt' narrative, link their stats to their actions (e.g., 'scored 12 points, including the game-tying shot').\n"
        "  * If NOT in the '.txt', provide a journalistic comment on their efficiency based on the numbers.\n\n"
        
        "###  3. ARENAS & COACHES (global_metadata files)\n"
        "- **Metadata Reference**: Always check 'global_metadata' for the correct head coaches and stadium names (e.g., 'Head coach of Olympiacos is Giorgos Bartzokas').\n\n"
        
        "**Strict Rule**: If information is missing from the Context, state you don't have enough data. Do NOT hallucinate.\n\n"
        "Context:\n"
        "{context}\n\n"
        "Question: {question}<|end|>\n"
        "<|assistant|>"
    )
    QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["context", "question"])

    condense_template = (
    "<|user|>\n"
    "You are an expert search query generator for a basketball RAG system. "
    "Your goal is to create a single, optimized search query based on the Chat History and the Follow-up Question.\n\n"
    
    "### LOGIC RULES:\n"
    "1. **Subject Switch**: If the follow-up question refers to a DIFFERENT match or team than the history, IGNORE all previous player names or specific stats from the history.\n"
    "2. **Keyword Injection**: \n"
    "- If the user asks for a 'summary', query MUST start with: 'detailed match narrative report summary'.\n"
    "- If the user asks for 'stats', query MUST start with: 'player boxscore statistics csv'.\n"
    "- If the user asks 'where' or 'location', query MUST include: 'arena stadium venue location'.\n"
    "3. **Entity Extraction**: Always include the full names of the teams involved (e.g., 'Olympiacos', 'Panathinaikos').\n"
    "4. **No Conversational Filler**: Output ONLY the search query. No 'Here is your query' or explanations.\n\n"
    "  If the user asks 'where' or about a location, include keywords: 'arena stadium venue location'."
    
    "Chat History: {chat_history}\n"
    "Follow-up Question: {question}<|end|>\n"
    "<|assistant|>"
)
    CONDENSE_PROMPT = PromptTemplate(template=condense_template, input_variables=["chat_history", "question"])
    #memory in conversation
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
    print("Agent ready. Ask a question...")
    print("="*30)
    
    while True:
        user_input = input("\nAsk a question: ")
        if user_input.lower() in ['exit', 'quit']: break
        
        print("Thinking...")
        try:
            result = qa_agent.invoke({"question": user_input})
            answer = result['answer'].strip()
            print(f"\n\n{answer}")
            
            sources = [os.path.basename(doc.metadata.get('source', '')) for doc in result['source_documents']]
            print(f"\n[Sources Used]: {list(set(sources))}")
        except Exception as e:
            print(f"\nError: {e}")