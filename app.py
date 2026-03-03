import streamlit as st
import main
import os
import shutil

# Ρύθμιση σελίδας
st.set_page_config(page_title="Thesis RAG Agent", page_icon="🏀", layout="wide")
st.title("🏀 EuroLeague AI Agent (Web Scraper)")
st.markdown("*Thesis Prototype: Automated Web Scraping & RAG Analysis*")

# --- ΠΛΑΪΝΗ ΜΠΑΡΑ (SIDEBAR) ---
with st.sidebar:
    st.header("Προσθήκη Δεδομένων")
    
    # ΕΠΙΛΟΓΗ 1: Προσθήκη ενός Link (Game URL)
    with st.expander("➕ Προσθήκη Αγώνα (URL)"):
        url = st.text_input("Game URL:", placeholder="https://...", key="single_url")
        if st.button("Add & Scrape Game"):
            if url and url not in main.GAME_URLS:
                main.GAME_URLS.append(url)
                st.cache_resource.clear() # Καθαρισμός μνήμης για να ξανατρέξει το indexing
                st.toast("Το URL προστέθηκε! Γίνεται ενημέρωση...", icon="🔄")
                st.rerun()

    # ΕΠΙΛΟΓΗ 2: Αυτόματη εύρεση από Πρόγραμμα (Schedule)
    with st.expander("🕵️ Αυτόματη Εύρεση (Schedule)"):
        st.caption("Βάλε το link της σελίδας Προγράμματος μιας ομάδας.")
        sched_url = st.text_input("Schedule URL:", placeholder="https://.../schedule", key="sched_url")
        
        if st.button("🔍 Find All Games"):
            if sched_url:
                with st.spinner("Γίνεται σάρωση της σελίδας..."):
                    found = main.crawl_schedule_page(sched_url)
                
                if found:
                    st.success(f"Βρέθηκαν {len(found)} αγώνες!")
                    # Προσθήκη στη λίστα
                    for link in found:
                        if link not in main.GAME_URLS:
                            main.GAME_URLS.append(link)
                    st.cache_resource.clear()
                    st.rerun()
                else:
                    st.error("Δεν βρέθηκαν links.")

    st.divider()
    
    # Εμφάνιση ενεργών πηγών
    if main.GAME_URLS:
        st.info(f"Ενεργά Web Links: {len(main.GAME_URLS)}")
        
    # Κουμπί Reset (Διαγραφή Βάσης)
    if st.button("⚠️ Διαγραφή Βάσης & Reset"):
        main.GAME_URLS = []
        st.cache_resource.clear()
        if os.path.exists("./chroma_db"): 
            shutil.rmtree("./chroma_db")
        st.rerun()

# --- ΑΡΧΙΚΟΠΟΙΗΣΗ ΣΥΣΤΗΜΑΤΟΣ ---
@st.cache_resource
def get_chain():
    # Αυτή η συνάρτηση καλείται μόνο μία φορά (εκτός αν πατήσουμε κουμπί που καθαρίζει την cache)
    vectorstore = main.setup_rag_index() 
    if not vectorstore: return None
    llm = main.load_local_llm()
    return main.create_qa_chain(llm, vectorstore)

# --- ΚΥΡΙΩΣ ΕΦΑΡΜΟΓΗ ---
try:
    with st.spinner("Φόρτωση AI Agent & Βάσης Δεδομένων..."):
        chain = get_chain()

    if not chain:
        st.warning("Η βάση είναι κενή. Πρόσθεσε ένα URL ή έλεγξε τα .txt αρχεία.")
        st.stop()
    else:
        st.toast("Το σύστημα είναι έτοιμο!", icon="✅")

    # Chat Interface History
    if "messages" not in st.session_state: st.session_state.messages = []

    # Εμφάνιση προηγούμενων μηνυμάτων
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    # Λήψη νέας ερώτησης
    if prompt := st.chat_input("Ρώτησε κάτι για τα παιχνίδια..."):
        # 1. Εμφάνιση ερώτησης χρήστη
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        # 2. Απάντηση AI
        with st.chat_message("assistant"):
            msg_container = st.empty()
            msg_container.markdown("▌ *Σκέφτεται...*")
            
            try:
                result = chain.invoke({"query": prompt})
                answer = result['result'].split("Answer:")[-1].strip()
                
                # Εξαγωγή Πηγών (Sources)
                sources = []
                if 'source_documents' in result:
                    for doc in result['source_documents']:
                        src = doc.metadata.get('source', 'Unknown')
                        # Αν είναι URL κρατάμε το link, αλλιώς το όνομα αρχείου
                        if src.startswith('http'):
                            sources.append(src)
                        else:
                            sources.append(os.path.basename(src))
                
                unique_sources = list(set(sources))
                final_resp = f"{answer}\n\n---\n*📚 Πηγές:*\n"
                for s in unique_sources:
                    final_resp += f"- `{s}`\n"
                
                msg_container.markdown(final_resp)
                st.session_state.messages.append({"role": "assistant", "content": final_resp})
                
            except Exception as e:
                msg_container.error(f"Σφάλμα: {e}")

except Exception as e:
    st.error(f"Κρίσιμο Σφάλμα: {e}")