import io, json, contextlib
import streamlit as st
import main


@st.cache_resource
def load_agent():
    return main.setup_rag_index(rebuild=False)


retriever = load_agent()

st.title("EuroLeague RAG Agent — Security Demo")
st.write("Offensive prompt testing in different defensive phases")

phase = st.sidebar.selectbox("Defensive phase:", [1, 2, 3, 4, 5],
                             format_func=lambda p: f"Phase {p}")

PHASE_INFO = {
    1: "Baseline — no defenses",
    2: "Blacklist + XML guardrails",
    3: "DSAG (Zero-Trust AST) + data filtering",
    4: "Static QCSF rules on query + retrieved context",
    5: "Dynamic QCSF (LLM-as-a-Judge)",
}

st.sidebar.caption(PHASE_INFO[phase])

with st.sidebar.expander("About the phases"):
    for p, desc in PHASE_INFO.items():
        st.write(f"**Phase {p}:** {desc}")

@st.cache_data
def load_test_cases():
    """The same 74-case benchmark suite the harness runs, grouped by category
    family: 'Direct RCE / Reverse Shell' and 'Direct RCE / DoS' are separate
    categories in the file but belong under one heading in the UI, and the
    'Category C:' / 'Category D:' prefixes are dropped from the label."""
    with open("benchmarks/tests.json", encoding="utf-8") as f:
        cases = json.load(f)["test_cases"]

    families = {}
    for case in cases:
        family = case["category"].split(" / ")[0].split(": ", 1)[-1]
        families.setdefault(family, []).append(case)
    return families


FAMILIES = load_test_cases()

st.sidebar.write("Examples:")
family = st.sidebar.selectbox(
    "Category:", list(FAMILIES),
    format_func=lambda f: f"{f} ({len(FAMILIES[f])})")

case = st.sidebar.selectbox(
    "Test case:", FAMILIES[family],
    format_func=lambda c: f"{c['id']} — {c['query'][:40]}"
                          f"{'…' if len(c['query']) > 40 else ''}")

st.sidebar.caption(f"**{case['category']}**")

if st.sidebar.button("Run this example"):
    st.session_state.pending = case["query"]
    st.session_state.pending_expected = case["expected_behaviour"]

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_message(msg):
    """Draw one turn. Shared by the replayed history and the live response so an
    older turn keeps the status and phase it actually ran under, rather than the
    phase currently selected in the sidebar."""
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.write(msg["content"])
            return

        status = msg["status"]
        if status.startswith(("BLOCKED", "SANDBOX")):
            st.success(f"Blocked by defense ({status}) — Phase {msg['phase']}")
        elif status == "ERROR":
            st.warning(f"Execution failed ({status}) — Phase {msg['phase']}")
        else:
            st.info(f"Executed ({status}) — Phase {msg['phase']}")
            st.caption("If the prompt contained an attack, check the response: "
                       "the agent may have ignored it or executed it.")

        st.write(msg["content"])

        # Shown only after the run, so the outcome is seen before the expectation.
        if msg.get("expected"):
            st.caption(f"Expected: {msg['expected']}")

        with st.expander("Execution log"):
            st.code(msg["logs"])


for msg in st.session_state.messages:
    render_message(msg)

# A prompt can arrive two ways: typed into the chat box, or queued by one of the
# sidebar example buttons. chat_input must be called on every rerun or the widget
# disappears, and "pending" is always popped so a queued example can never leak
# into a later turn.
typed = st.chat_input("Type your prompt here...")
pending = st.session_state.pop("pending", None)
expected = st.session_state.pop("pending_expected", None)
prompt = typed or pending
if typed:
    expected = None   # a hand-written prompt has no expected behaviour

if prompt:
    user_msg = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_msg)
    render_message(user_msg)

    # Capture everything main.py prints, so the generated code and the
    # security checks are visible in the UI instead of only in the terminal.
    buf = io.StringIO()
    with st.spinner(f"Running under Phase {phase}..."):
        with contextlib.redirect_stdout(buf):
            output, status = main.ask_agent(
                prompt, phase=phase, retriever_instance=retriever
            )

    assistant_msg = {"role": "assistant", "content": output, "status": status,
                     "logs": buf.getvalue(), "phase": phase, "expected": expected}
    st.session_state.messages.append(assistant_msg)
    render_message(assistant_msg)