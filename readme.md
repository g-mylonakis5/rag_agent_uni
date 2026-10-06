EuroLeague Code-Driven RAG Agent — Security Benchmark

VIDEO DEMO:



https://github.com/user-attachments/assets/c8d2e20e-7395-4b92-b204-fde7ec3e57b2




A retrieval-augmented generation agent for EuroLeague basketball analytics, built as a testbed for measuring how well layered defenses protect an LLM agent that executes generated Python code.

The agent answers questions about EuroLeague games from a local corpus of box scores and match summaries. For analytical questions it writes Python and runs it — which is exactly the capability an attacker wants to reach. The project implements five cumulative defense phases and measures, with a 74-case benchmark suite, how many attacks each phase stops.

Security note. This repository deliberately contains attack prompts (benchmarks/tests.json, attacks.txt) and a honeypot file of fabricated credentials (data/global_metadata/admin_internal_config.txt). Nothing in it is a real secret. Phase 1 is an intentionally vulnerable baseline that will execute hostile code on your machine — see Safety before running it.

Quick start

Requires Python 3.11 and a Google Gemini API key.

bash
git clone https://github.com/g-mylonakis5/rag_agent_uni.git
cd rag_agent_uni

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt

Set your API key — the app reads it from the environment and fails at import without it:

bash
# Windows (PowerShell)
$env:GOOGLE_API_KEY = "your-key-here"

# macOS / Linux
export GOOGLE_API_KEY="your-key-here"

Get a free key at https://aistudio.google.com/app/apikey.

Then launch the demo UI from the repository root:

bash
streamlit run app.py

The first run downloads the all-MiniLM-L6-v2 embedding model (~90 MB) and builds the Chroma vector index into ./chroma_db/, which takes a few minutes. That directory is gitignored and rebuilt automatically whenever it is missing.

Paths are relative, so always run from the repository root.

Using the demo

The sidebar holds three controls:

Defensive phase — switches the active security posture at runtime (1–5).
Examples — all 74 benchmark cases from benchmarks/tests.json, grouped by category. Each case's expected behaviour is revealed after the run, so you see what actually happened before you see what was supposed to.
Chat box — type any prompt of your own.

Each answer is labelled with the status the agent returned and the phase it ran under, so a transcript stays readable after you switch phases mid-session. The Execution log expander shows the generated code and the security layers' decisions.

A good first comparison: run the Direct RCE → attack_01 example at Phase 1, then the same case at Phase 3. Phase 1 executes os.system('whoami') and prints your username; Phase 3 blocks it at the AST layer.

<!-- TODO: add two screenshots here, same prompt at Phase 1 and Phase 3: ![Phase 1 — attack executes](docs/phase1_rce.png) ![Phase 3 — attack blocked](docs/phase3_blocked.png) -->
The defense phases
Phase	Defenses
1	Baseline — none. Generated code is executed as-is.
2	Naive AST blacklist of dangerous names, plus XML-tagged prompt guardrails.
3	DSAG (Deterministic Syntax Analysis Guard): zero-trust AST allowlist of modules, functions and attributes, plus data-layer isolation that strips administrative files from the RAG index.
4	Adds QCSF 2.0 (Query Contextual Semantic Firewall): static pattern rules over both the incoming query and the retrieved context, catching indirect prompt injection via RAG poisoning.
5	Replaces the static query rules with QCSF 3.0 — a dynamic LLM-as-a-Judge intent classifier that catches linguistic obfuscation the static patterns miss.

Phase 6 exists in the code (sandbox.py): a runtime containment layer running generated code in an isolated process under time and memory limits. It is reachable from the benchmark harness but not from the UI's phase selector.

Each phase is cumulative — Phase 5 includes everything from Phases 2–4.

Attack categories

The 74 cases in benchmarks/tests.json:

Category	Cases
Legitimate (false-positive control)	25
Direct RCE	10
Direct LFI	6
Data leakage	1
AST evasions	15
Advanced AI threats	17

The 25 legitimate queries matter as much as the attacks: a defense that blocks everything scores perfectly on attacks and is useless. They measure how much real functionality each phase costs.

Results

Every phase was evaluated over three independent runs of the full suite; the figures below are means, with standard deviation under 2% throughout.

Phase	Attack success	Functionality retained	CARS
1 — Baseline	72.8%	100%	0.178
2 — Blacklist	41.5%	93%	0.431
3 — DSAG	12.9%	92%	0.689
4 — Static QCSF	14.3%	84%	0.650
5 — Dynamic QCSF	0.0%	72%	0.717

CARS (Comprehensive Agent Robustness Score) is the product of a weighted security index and the task completion rate. It is a product rather than a mean so that failing either dimension drags the score down — a system that blocks everything scores zero, not fifty percent.

What the numbers show:

Deterministic AST control eliminates code-level attacks outright. At Phase 3, direct RCE, LFI and AST evasions all drop to 0%. The residual 12.9% is entirely semantic — indirect prompt injection, which no syntax check can see.
Two distinct layers are necessary. Code attacks need a deterministic guard; semantic attacks need a semantic one. Neither substitutes for the other.
Maximum security is not the optimal design point. Phase 3 reaches 96% of Phase 5's robustness while keeping 92% functionality instead of 72% — twenty points of usability for four percent of robustness.
Static rules on retrieved context cost more than they return. Phase 4 leaves attack success flat while dropping functionality by eight points, which is why its CARS falls below Phase 3's.

Validation. The code-level findings were reproduced on a second foundation model (Anthropic Claude), where Phase 3 again zeroed RCE and evasion attacks. A separate blind set of six attacks — written against threat categories outside the judge's rule, so it could not have been tuned for them — was blocked in full at Phase 5 while partially succeeding at baseline, which argues the semantic filter generalises rather than pattern-matching the known suite.

Reproducing the benchmarks

Run the suite for one phase:

bash
python run_benchmarks.py --phase 3

This writes benchmark_results_phase3.csv with the prompt, the generated code, the trapped console output and the returned status for every case.

Useful flags:

bash
python run_benchmarks.py --phase 5 --id attack_01,evasion_03   # specific cases
python run_benchmarks.py --phase 3 --tests benchmarks/blind_tests.json
python run_benchmarks.py --phase 3 --model claude              # cross-model run

--model claude requires ANTHROPIC_API_KEY and exists to check that the findings are not an artifact of one foundation model.

Score the resulting CSVs into attack-success and functionality metrics:

bash
python classify_results.py benchmark_results_phase*.csv --summary summary.csv

The agent can also be driven from a terminal REPL, bypassing Streamlit:

bash
python main.py --phase 3
Safety

Phase 1 and Phase 2 execute attacker-controlled code with no meaningful restrictions. The test suite includes reverse shells, file deletion, process killers and resource exhaustion. Treat a Phase 1 run as running untrusted code as your own user.

Run the low phases in a VM or container, not on a machine you care about. Phases 3–5 block the destructive cases at the AST layer before execution, and Phase 6 adds process isolation — but the benchmark's purpose is to find the cases where that fails, so do not assume any phase is airtight.

The honeypot credentials in data/global_metadata/admin_internal_config.txt are fabricated. The hosts do not resolve and the keys authenticate nothing; they exist so an exfiltration attempt produces something observable in the logs.

Repository layout
app.py                  Streamlit demo UI
main.py                 Agent: routing, defense layers, code execution
sandbox.py              Phase 6 process isolation (time + memory limits)
run_benchmarks.py       Benchmark harness -> benchmark_results_phase{N}.csv
classify_results.py     Scores result CSVs into ASR / SI / TCR / CARS metrics
box_scraper.py          Scrapes box scores (not needed; data/ is populated)
summary_scraper.py      Scrapes match summaries (likewise)
benchmarks/tests.json   The 74-case evaluation suite
data/box_scores/        Per-game CSVs: Match, Team, Player, MIN, PTS, REB, AST, PIR...
data/summaries/         Match report text used for semantic retrieval
data/global_metadata/   League metadata + the synthetic honeypot config
Agent routing

Queries take one of three paths, chosen by regex intent matching:

Route A — static CSV parsing for standard aggregations (leaderboards, averages). No code generation, so no execution risk.
Route B — conversational RAG over match summaries.
Route C — the Python REPL: the LLM writes code, the defense layers vet it, and it runs. This is the attack surface every phase is defending.


Known limitations
main.py captures stdout but not stderr, so a child process that fails writes to fd 2 and leaves no trace in the execution log.
The UI phase selector covers 1–5; Phase 6 is benchmark-only.
Result CSVs from before the descriptor-level stdout fix under-record output from os.system and subprocess payloads. The pass/fail status is unaffected.
