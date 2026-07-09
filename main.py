# Hybrid RAG & Advanced Conversational Agent
# Developed for EuroLeague Analysis & LLM Agent Security Testing (RCE Exploit Bed)

import os
import torch
import sys
import shutil
import csv
from glob import glob 

# Third-party LangChain & Google GenAI Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

# System Hardware & Vector DB Database Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_PATH = './chroma_db'

# Initialize Text Embedding Model
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_ID, 
    model_kwargs={'device': DEVICE}
)

def setup_rag_index(rebuild=False):
    """Handles Vector Store indexing, document loading, and retriever configuration."""
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

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        final_docs = []
        
        for folder in data_folders:
            folder_path = os.path.join(os.getcwd(), folder)
            if not os.path.exists(folder_path):
                continue
            
            files = glob(os.path.join(folder_path, "*.txt")) + glob(os.path.join(folder_path, "*.csv"))
            
            for file_path in files:
                try:
                    # Append domain metadata context tags dynamically
                    if 'summaries' in folder:
                        file_description = "NARRATIVE MATCH REPORT SUMMARY GAME HIGHLIGHTS STORYTELLING"
                    elif 'global_metadata' in folder:
                        file_description = "GLOBAL LEAGUE METADATA ARENAS COACHES STADIUMS"
                    elif 'box_scores' in folder:
                        file_description = "STATISTICS BOXSCORE NUMBERS PLAYER STATS CSV"
                    else:
                        file_description = "DATA"

                    # Differentiate loaders based on file extensions
                    if file_path.endswith('.csv'):
                        loader = CSVLoader(file_path=file_path, encoding='utf-8')
                    else:
                        loader = TextLoader(file_path=file_path, encoding='utf-8')

                    loaded_docs = loader.load()
                    match_name = os.path.basename(file_path).replace('.csv', '').replace('.txt', '').replace('_', ' ').title()
                    
                    # Embed system prompt metadata directly inside text page content
                    for doc in loaded_docs:
                        doc.metadata['source'] = doc.metadata.get('source', '').replace('\\', '/')
                        doc.page_content = f"Team Matchup Context: {match_name}\nData Format: {file_description}\nInformation Content: {doc.page_content}"
                    
                    if file_path.endswith('.csv'):
                        final_docs.extend(loaded_docs)
                    else:
                        splitted = text_splitter.split_documents(loaded_docs)
                        final_docs.extend(splitted)

                    print(f" Loaded ({folder}): {os.path.basename(file_path)}")
                except Exception as e:
                    print(f" Error loading {file_path}: {e}")
                
        if not final_docs:
            print(" No documents found!")
            return None

        # Build and persist Chroma Vector DB store
        vectorstore = Chroma.from_documents(
            documents=final_docs, 
            embedding=embeddings, 
            persist_directory=CHROMA_PATH
        )
        vectorstore.persist()
        print(" Indexing complete.")
    
    else:
        print(" Loading existing index from disk...")
        vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
        
    # Configure MMR Retriever to maximize relevance diversity
    retriever = vectorstore.as_retriever(
        search_type="mmr", 
        search_kwargs={
            "k": 5,             
            "fetch_k": 15,      
            "lambda_mult": 0.5  
        }
    )
    return retriever

def load_gemini_llm():
    """Initializes the remote Generative AI model with structured inference boundaries."""
    print("Connecting to Gemini API (gemini-3.1-flash-lite)...")
    return ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0.1,
        max_tokens=1000,
        disable_streaming=False
    )

def create_qa_chain(llm, retriever_obj):
    """Assembles a standard RAG LCEL prompt-driven inference execution layout."""
    qa_template = (
        "You are a EuroLeague Master Analyst & Journalist. Use the provided Context to answer the question with absolute accuracy.\n\n"
        "### CRITICAL MANDATORY RULES:\n"
        "1. **STRICT QUESTION FOCUS**: Answer only and precisely what the user is asking. If asked about capacity, location, arena, or coach, answer directly based on the context in one clean sentence. Do not add non-related facts.\n"
        "2. **GAME SUMMARIES ONLY**: IF AND ONLY IF the user is asking for a match summary or game highlights, you MUST start your response with the final score (e.g., 'Final Score: Team A XX - XX Team B').\n"
        "3. **GENERAL QUESTIONS & METADATA**: If the question is about a head coach, stadium, arena, capacity, or location, DO NOT include any final score line or match information. Jump straight into the direct plain answer.\n"
        "4. **STRICT NO ASTERISKS RULE**: NEVER use asterisks (*) for bullet points or bold text anywhere. Output only clean, raw plain text sentences.\n"
        "5. **NO CITATIONS**: Never mention filenames like '.txt', '.csv', or '(source: ...)'.\n\n"
        "Context:\n"
        "{context}\n\n"
        "Question: {input}\n"
    )
    QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["context", "input"])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Modular modern LCEL chain composition
    rag_chain = (
        {"context": retriever_obj | format_docs, "input": RunnablePassthrough()}
        | QA_PROMPT
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

if __name__ == "__main__":
    chat_history_manual = []
    sys.stdout.reconfigure(line_buffering=True)

    print("\n" + "="*30)
    user_rebuild = input("Rebuild database? (y/n): ").strip().lower()
    rebuild_flag = True if user_rebuild == 'y' else False

    # Agent system bootstrap execution
    retriever = setup_rag_index(rebuild=rebuild_flag)
    if not retriever: exit()

    llm = load_gemini_llm()
    if not llm: exit()

    qa_agent = create_qa_chain(llm, retriever)

    print("\n" + "="*30)
    print("Agent ready. Ask a question...")
    print("="*30)
    
    while True:
        user_input = input("\nAsk a question: ")
        if user_input.lower() in ['exit', 'quit']: break
        
        print("Thinking...")
        
        stats_keywords = ['best player', 'highest pir', 'total points', 'average', 'καλυτερος παικτης', 'καλύτερος παίκτης', 'σκορερ', 'στατιστικα σε ολα', 'scorer', 'best scorer', 'points per game', 'points', 'statline', 'stats']
        
       
       # --- ROUTE A: NATIVE CODE-DRIVEN RAG (BOX SCORE ANALYST ENGINE) ----------
        
        if any(keyword in user_input.lower() for keyword in stats_keywords):
            try:
                # 1. Clean query input tokens to isolate explicit target names
                clean_input = user_input.lower()
                user_input_clean = user_input.lower()
                
                # Filter out registered team tokens from query
                teams_list = ['olympiacos', 'panathinaikos', 'real', 'partizan', 'bayern', 'dubai', 'barcelona', 'barca', 'zvezda', 'maccabi', 'paris', 'armani', 'baskonia', 'valencia','efes','virtus','asvel','zalgiris','hapoel','monaco','fenerbahce']
                
                # Find teams and maintain their structural appearance order (Home vs Away)
                found_teams_with_positions = []
                for team in teams_list:
                    pos = user_input_clean.find(team)
                    if pos != -1:
                        actual_team_name = 'barcelona' if team == 'barca' else team
                        found_teams_with_positions.append((pos, actual_team_name))
                
                found_teams_with_positions.sort()
                found_teams = [team_name for _, team_name in found_teams_with_positions]
                found_teams = list(dict.fromkeys(found_teams)) # Remove potential duplicates
                
                # Strip out questions phrases and structural padding text
                for word in stats_keywords + teams_list + ['game', 'match', 'in', 'the', 'what', 'were', 'stats', 'statline', 'player', 'for', 'of', '?', "'s"]:
                    clean_input = clean_input.replace(word, " ")
                
                # Collect remaining query string elements (e.g., ['fournier'])
                player_words = [w.strip() for w in clean_input.split() if len(w.strip()) > 2]

                # Resolve runtime file paths natively to circumvent environment boundaries
                box_scores_dir = os.path.abspath(os.path.join(os.getcwd(), 'data', 'box_scores'))
                csv_files = glob(os.path.join(box_scores_dir, '*.csv'))
                
                print(f"\n[DEBUG]: Ψάχνω αρχεία στο: {box_scores_dir}")
                
                raw_data_output = ""
                
                # Determine strict file filtering based on how many teams were extracted
                target_specific_file = None
                if len(found_teams) == 2:
                    # Enforce strict Home_Away order match to isolate a single game file
                    target_specific_file = f"{found_teams[0]}_{found_teams[1]}.csv"
                
                # 2. Iterate and scan structural spreadsheet data directly
                matched_files_count = 0
                for file_path in csv_files:
                    filename = os.path.basename(file_path).lower()
                    match_file = False
                    
                    if target_specific_file:
                        if filename == target_specific_file:
                            match_file = True
                    else:
                        if found_teams:
                            if any(team in filename for team in found_teams):
                                match_file = True
                        else:
                            match_file = True # Fallback if no teams stated at all
                            
                    if match_file:
                        matched_files_count += 1
                        with open(file_path, mode='r', encoding='utf-8') as f:
                            reader = csv.DictReader(f)
                            for row in reader:
                                player_name_in_row = row.get('Player', '').lower()
                                
                                # Evaluate fuzzy text containment condition matches
                                match_player = False
                                if player_words:
                                    if any(word in player_name_in_row for word in player_words):
                                        match_player = True
                                else:
                                    match_player = True # Default fallback if player unstated
                                
                                if match_player:
                                    raw_data_output += f"Match: {row.get('Match')}, Team: {row.get('Team')}, Player: {row.get('Player')}, MIN: {row.get('MIN')}, PTS: {row.get('PTS')}, 2FG: {row.get('2FG')}, 3FG: {row.get('3FG')}, FT: {row.get('FT')}, REB: {row.get('REB')}, AST: {row.get('AST')}, STL: {row.get('STL')}, TO: {row.get('TO')}, PIR: {row.get('PIR')}\n"

                print(f"[DEBUG]: Σκαναρίστηκαν {matched_files_count} αρχεία CSV βάσει των κριτηρίων.")

                # 3. Refine compiled textual rows using the generative AI instance
                if not raw_data_output.strip():
                    print("\nCould not find specific statistics for this query. Please check data files.")
                else:
                    refine_prompt = (
                        f"You are a professional sports journalist. Convert the following raw basketball statistics into a single, smooth, natural plain text sentence without asterisks or bold formatting.\n"
                        f"CRITICAL: Focus ONLY on the requested matchup and do not mix up separate games.\n\n"
                        f"Raw Statistics:\n{raw_data_output}"
                    )
                    refine_chain = llm | StrOutputParser()
                    final_speech = refine_chain.invoke([HumanMessage(content=refine_prompt)])
                    print(f"\n\n{final_speech.strip()}")
            except Exception as e:
                print(f"\nTool Error: {e}")
                
      
        # --- ROUTE B: UNIFIED CONVERSATIONAL RAG (NARRATIVE GRAPH PATH) ----------
        
        else:
            try:
                user_input_clean = user_input.lower()
                
                # 1. Compress conversational history tokens to expand contextual query
                history_str = "\n".join(chat_history_manual[-4:])
                
                query_generation_prompt = (
                    f"You are an expert search query optimizer for a EuroLeague RAG system.\n"
                    f"Analyze the Chat History and the User's current Input. Generate a single, highly optimized standalone search query that resolves pronouns and fixes spelling.\n"
                    f"Example: If history is about 'Olympiacos' and Input is 'what is the capacity?', output: 'Olympiacos arena stadium capacity'.\n"
                    f"Example 2: If Input is 'Barca', output: 'FC Barcelona basketball arena coach stadium metadata'.\n\n"
                    f"Chat History:\n{history_str}\n\n"
                    f"User Input: {user_input}\n"
                    f"Output ONLY the raw optimized keywords. No quotes, no explanations."
                )
                
                query_chain = llm | StrOutputParser()
                optimized_query = query_chain.invoke([HumanMessage(content=query_generation_prompt)]).strip()
                
                if not optimized_query: optimized_query = user_input

                # 2. Extract specific matchup attributes to enforce direct cache file routing
                teams_list = ['olympiacos', 'panathinaikos', 'real', 'partizan', 'bayern', 'dubai', 'barcelona', 'barca', 'zvezda', 'maccabi', 'paris', 'armani', 'baskonia', 'valencia','efes','virtus','asvel','zalgiris','hapoel','monaco','fenerbahce']
                
                found_teams_with_positions = []
                for team in teams_list:
                    pos = user_input_clean.find(team)
                    if pos != -1:
                        actual_team_name = 'barcelona' if team == 'barca' else team
                        found_teams_with_positions.append((pos, actual_team_name))
                
                found_teams_with_positions.sort()
                mentioned_teams = [team_name for _, team_name in found_teams_with_positions]
                mentioned_teams = list(dict.fromkeys(mentioned_teams))

                context_content = ""
                source_file_used = "ChromaDB Optimized Search"
                
                # Matchup routing override to bypass vector index noise on direct summaries
                if len(mentioned_teams) == 2 and any(w in user_input_clean for w in ['summary', 'summarize', 'game', 'match', 'score', 'result']):
                    home_team = mentioned_teams[0]
                    away_team = mentioned_teams[1]
                    expected_filename = f"{home_team}_{away_team}.txt"
                    target_path = os.path.join("data", "summaries", expected_filename)
                    
                    if not os.path.exists(target_path):
                        expected_filename = f"{away_team}_{home_team}.txt"
                        target_path = os.path.join("data", "summaries", expected_filename)
                        
                    if os.path.exists(target_path):
                        with open(target_path, "r", encoding="utf-8") as f:
                            context_content = f.read()
                        source_file_used = expected_filename
                
                # 3. Standard Vector Embeddings retrieval fallback path
                if not context_content:
                    source_documents = retriever.invoke(optimized_query)
                    context_content = "\n\n".join([doc.page_content for doc in source_documents])
                    if source_documents:
                        source_file_used = ", ".join(list(set([os.path.basename(doc.metadata.get('source', '')) for doc in source_documents])))

                # 4. Synthesize finalized plain text output and keep memory array fresh
                qa_prompt = (
                    f"You are a EuroLeague Master Analyst & Journalist. Use the provided Context and Chat History to answer the user's Question with absolute accuracy.\n\n"
                    f"### CRITICAL MANDATORY RULES:\n"
                    f"1. **STRICT QUESTION FOCUS**: Answer only and precisely what the user is asking. If asked about a coach or stadium, give only that in one plain text sentence.\n"
                    f"2. **GAME SUMMARIES ONLY**: IF AND ONLY IF the user is asking for a match summary, start with the final score exactly as written in the context.\n"
                    f"3. **GENERAL QUESTIONS & METADATA**: Do not include final scores for questions about arenas or coaches.\n"
                    f"4. **STRICT NO ASTERISKS RULE**: NEVER use asterisks (*) anywhere. Output clean raw text.\n"
                    f"5. **NO FILENAMES**: Do not mention filenames like '.txt' or '.csv'.\n\n"
                    f"Chat History:\n{history_str}\n\n"
                    f"Context:\n{context_content}\n\n"
                    f"Question: {user_input}"
                )
                
                qa_response_chain = llm | StrOutputParser()
                answer = qa_response_chain.invoke([HumanMessage(content=qa_prompt)])
                
                print(f"\n\n{answer.strip()}")
                print(f"\n[Sources Used]: ['{source_file_used}'] (Optimized Search Query: '{optimized_query}')")
                
                chat_history_manual.append(f"User: {user_input}")
                chat_history_manual.append(f"Agent: {answer.strip()}")
                
            except Exception as e:
                print(f"\nError: {e}")