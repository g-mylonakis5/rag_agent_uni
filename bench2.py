"""
EuroLeague RAG Agent - Security Benchmarking Engine
Developed for Automated Exploit Testing & Utility Evaluation

ARCHITECTURE OVERVIEW:
This script acts as the automated testing harness for the EuroLeague RAG Agent.
It iterates through a structured JSON payload of test cases (Legitimate & Adversarial),
injects them into the agent under specific security phases (1-5), and meticulously 
records the latency, agent response, and internal system logs.

KEY FEATURES:
- STDOUT Trapping: Captures internal print statements (e.g., AST Firewall alerts, Code REPL outputs) 
  to provide full visibility into the agent's internal state.
- Unbiased Raw Logging: Exports data directly to CSV with a 'PENDING_MANUAL_REVIEW' flag, 
  ensuring the researcher accurately calculates the CARS index without automated LLM bias.
"""

import json
import csv
import argparse
import time
import os
import sys
import io
from contextlib import redirect_stdout

# Import the core routing bridge, phase setter, and initializers from the main agent module
try:
    from main import ask_agent, set_defense_phase, setup_rag_index, load_gemini_llm
except ImportError:
    print("[ERROR]: Could not import functions from 'main.py'. Ensure you are running from the project root.")
    sys.exit(1)

def load_test_cases(file_path="benchmarks/tests.json"):
    """
    Loads the structured benchmark test suite from disk.
    The JSON file contains both Task Completion (Utility) queries and Threat Vectors (ASR).
    """
    if not os.path.exists(file_path):
        print(f"[ERROR]: Benchmark file '{file_path}' not found.")
        sys.exit(1)
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("test_cases", [])

def save_to_csv(data_list, filename):
    """
    Exports row-by-row evaluation logs to a CSV artifact.
    This artifact is used by the researcher to perform manual annotation 
    for the Security Index (SI) and Task Completion Rate (TCR) metrics.
    """
    if not data_list:
        return
    headers = data_list[0].keys()
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data_list)

def run_evaluation(phase, test_file="benchmarks/tests.json", target_ids=None):
    """
    Core execution loop for the benchmark test suite.
    Executes queries against the specified defense phase, tracking performance metrics
    and trapping system-level outputs for deep-dive security analysis.
    """
    print("=" * 70)
    print(f"[STARTING] EUROLEAGUE RAG BENCHMARK | PHASE {phase}" + (f" | TARGETS: {target_ids}" if target_ids else ""))
    print("=" * 70)

    # 1. Setup Defense Phase & Initialize RAG / LLM instances
    set_defense_phase(phase)
    print("[INIT]: Loading ChromaDB Retriever and Gemini LLM for evaluation...")
    eval_retriever = setup_rag_index(rebuild=False)
    eval_llm = load_gemini_llm()
    # ------------------------------------------------

    tests = load_test_cases(test_file)

    # 2. Filter test suite if specific Target IDs are provided via CLI arguments
    if target_ids:
        id_list = [x.strip() for x in target_ids.split(",")]
        tests = [t for t in tests if t.get("id") in id_list]
        if not tests:
            print(f"[ERROR]: None of the target IDs '{target_ids}' were found in '{test_file}'.")
            sys.exit(1)

    results_log = []
    start_time = time.time()

    # 3. Benchmark Execution Loop
    for i, test in enumerate(tests, 1):
        t_id = test.get("id", f"test_{i}")
        category = test.get("category", "Unknown")
        query = test.get("query", "")

        print("\n" + "~" * 70)
        print(f"[{i:02d}/{len(tests)}] TEST ID: {t_id} | CATEGORY: {category}")
        print(f"QUERY: {query}")
        print("~" * 70)

        t0 = time.time()
        response = ""
        status = "ERROR"

        # --- SYSTEM-LEVEL AUDITING (STDOUT TRAP) ---
        # The io.StringIO trap is critical for security auditing. 
        # It captures underlying print() statements from main.py, such as QCSF intent blocks 
        # or DSAG AST validation alerts, providing transparent proof of the defense mechanism.
        f_trap = io.StringIO()

        try:
            with redirect_stdout(f_trap):
                response, status = ask_agent(
                    query, 
                    phase=phase, 
                    retriever_instance=eval_retriever, 
                    llm_instance=eval_llm
                )
        except Exception as e:
            status = "CRASH_EXCEPTION"
            response = f"[Execution Exception caught by Benchmark Guard]: {str(e)}"

        latency = round(time.time() - t0, 3)
        
        # Display the trapped REPL/System output to the console for real-time monitoring
        console_output = f_trap.getvalue()
        if console_output:
            print(console_output, end="")

        # Append raw execution data. The Status Code is intentionally left as 'PENDING_MANUAL_REVIEW' 
        # to ensure academic rigor through manual human evaluation (True Positives, False Positives, etc.)
        results_log.append({
            "Test_ID": t_id,
            "Category": category,
            "Defense_Phase": phase,
            "Status_Code": "PENDING_MANUAL_REVIEW",
            "Execution_Status": status,
            "Latency_sec": latency,
            "Query_Preview": query,
            "Agent_Response": str(response).strip(),
            "Console_Output": str(console_output).strip()
        })

        # Brief pause to prevent rate-limiting from the LLM Provider (Google Gemini API)
        time.sleep(1)

    total_duration = round(time.time() - start_time, 2)

    print("\n" + "=" * 70)
    print(f"EXECUTION COMPLETE (PHASE {phase})" + (f" [TARGETS: {target_ids}]" if target_ids else ""))
    print("=" * 70)
    print(f"Total Execution Time         : {total_duration}s")
    print(f"Total Test Cases Executed    : {len(results_log)}")
    print("=" * 70)

    # --- EXPORT ARTIFACT REPORT ---
    suffix = ""
    if target_ids:
        id_list = [x.strip() for x in target_ids.split(",")]
        suffix = f"_{id_list[0]}" if len(id_list) == 1 else "_custom_group"
        
    output_filename = f"benchmark_results_phase{phase}{suffix}.csv"
    save_to_csv(results_log, output_filename)
    print(f"\n[ARTIFACT GENERATED]: Raw logs saved to -> '{output_filename}'\n")

if __name__ == "__main__":
    # Command Line Interface configuration for flexible benchmarking
    parser = argparse.ArgumentParser(description="Raw Execution Engine for Manual Annotation")
    
    # Allows switching between the Baseline (1) and the various Mitigation Architectures (2-6)
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3, 4, 5, 6], help="Defense Phase to evaluate (1 to 6)")
    parser.add_argument("--tests", type=str, default="benchmarks/tests.json", help="Path to the JSON test suite")
    
    # Supports targeted testing for debugging specific Edge Cases or False Positives
    parser.add_argument("--id", "--ids", dest="target_ids", type=str, default=None, help="Run a specific test ID or comma-separated list of IDs")
    args = parser.parse_args()

    run_evaluation(phase=args.phase, test_file=args.tests, target_ids=args.target_ids)