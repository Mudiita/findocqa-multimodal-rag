"""FinDocQA web app.

Run:  venv310/bin/streamlit run app.py

Uses main.query() unchanged. The sidebar filter is applied here in the app (it narrows which documents
a question may use); when everything is selected, queries run exactly as in the evaluation.
"""

import asyncio
import csv
import difflib
import io
import re
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st

import main

DISCLAIMER = "Factual information from the documents, not financial advice."
MAX_QUERIES_PER_SESSION = 15
FEEDBACK_FILE = main.ROOT / "feedback.csv"
FEEDBACK_FIELDS = ["timestamp", "session_id", "message_id", "rating", "question", "answer", "citations",
                   "number_check", "response_time_s", "filters"]
CATEGORY_LABELS = {"insurance": "Health insurance", "mutual_fund": "Mutual funds", "fixed_deposit": "Fixed deposits"}
ALL_CATEGORIES = list(CATEGORY_LABELS)
# (tag, question, filter needed by the example or None). None of these needs a filter, so clicking
# one never changes the sidebar; an example with a filter would set the visible widgets first.
EXAMPLES = [
    ("Insurance", "What is the free look period in the Aditya Birla Activ Health policy?", None),
    ("Insurance · comparison", "How do the free look periods of Tata AIG MediCare and Care Health compare?", None),
    ("Mutual fund · chart", "What does the rating profile chart of ICICI Prudential Short Term Fund show?", None),
    ("Mutual fund", "Who manages the HDFC Mid-Cap Opportunities Fund and since when?", None),
    ("Fixed deposit", "What is the 1 year FD rate at HDFC Bank for normal and senior citizens?", None),
    ("Fixed deposit", "What are SBI's FD rates for 7 to 45 days and 46 to 179 days?", None),
]
WITHHELD_MESSAGE = "Could not verify the figures, please check the cited page."
NO_MATCH_HINT = ("No match in the selected documents. Your filter may exclude the relevant document. "
                 "Click Reset filters to search all 30.")
API_ERROR_MESSAGE = ("Sorry, the answer service is not reachable right now (the AI provider returned an error). "
                     "Please try again in a minute.")
CLARIFICATION = re.compile(r"^(Which .* do you mean\?|I can't recommend)", re.I)

st.set_page_config(page_title="FinDocQA", page_icon="📄", layout="wide")

# --- Styling (theme-neutral colours so the dark theme keeps working) ---------------------------------
st.markdown("""
<style>
[data-testid="stMainBlockContainer"] { max-width: 1100px; padding-top: 2.5rem; }
[data-testid="stMainBlockContainer"] [data-testid="stMarkdownContainer"] p,
[data-testid="stMainBlockContainer"] [data-testid="stMarkdownContainer"] li { font-size: 18px; line-height: 1.6; }
h1 { font-size: 2.4rem !important; line-height: 1.2 !important; padding-bottom: 0.2rem !important; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { font-size: 14px !important; opacity: 0.8; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p, [data-testid="stSidebar"] label,
[data-testid="stSidebar"] span, [data-baseweb="tag"] span { font-size: 16px; }
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] p,
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] li { font-size: 16px; }
[data-testid="stExpander"] summary p { font-size: 16px; font-weight: 500; }
[data-testid="stChatMessage"] { padding: 0.9rem 1rem; margin-bottom: 0.4rem; }
.fdq-disclaimer { font-size: 14px; padding: 0.45rem 0.8rem; border-left: 3px solid rgba(100,140,220,0.6);
  background: rgba(128,128,128,0.08); border-radius: 4px; margin: 0.2rem 0 1.4rem 0; }
.fdq-sub { font-size: 18px; opacity: 0.85; margin-bottom: 0.6rem; }
.fdq-hint { font-size: 16px; padding: 0.6rem 0.9rem; border-left: 3px solid rgba(230,160,40,0.8);
  background: rgba(230,160,40,0.10); border-radius: 4px; margin: 0.3rem 0 0.4rem 0; }
.stButton button { height: auto; min-height: 5.4rem; align-items: flex-start; justify-content: flex-start;
  text-align: left; padding: 0.7rem 0.9rem; }
.stButton button div[data-testid="stMarkdownContainer"] p { white-space: normal; text-align: left; font-size: 16px;
  line-height: 1.45; margin: 0; }
.stButton button div[data-testid="stMarkdownContainer"] p strong { font-size: 13px; letter-spacing: 0.03em;
  text-transform: uppercase; opacity: 0.75; font-weight: 600; }
[data-testid="stSidebar"] .stButton button { min-height: 0; justify-content: center; }
[data-testid="stChatMessage"] .stButton button { min-height: 0; width: auto; padding: 0.4rem 0.9rem; }
[data-testid="stChatMessage"] .stButton button p { font-size: 15px !important; }
.fdq-note { font-size: 15px; padding: 0.45rem 0.8rem; border-left: 3px solid rgba(100,140,220,0.6);
  background: rgba(100,140,220,0.08); border-radius: 4px; margin: 0.3rem 0 0.4rem 0; }
</style>
""", unsafe_allow_html=True)


# --- Backend: one event loop thread, queries run one at a time ----------------------------------------

@st.cache_resource(show_spinner="Loading the document index…")
def backend():
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True, name="findocqa-loop").start()
    rag = main.build_rag()
    asyncio.run_coroutine_threadsafe(rag._ensure_lightrag_initialized(), loop).result()

    async def make_lock():
        return asyncio.Lock()

    lock = asyncio.run_coroutine_threadsafe(make_lock(), loop).result()
    return loop, rag, lock


async def _query(question: str, allowed: set[str] | None):
    loop, rag, lock = backend()
    async with lock:
        if allowed is None:
            return await main.query(rag, question)
        # App-level filter: only documents inside the sidebar selection may be used.
        original_find, original_retrieve = main.find_named_documents, main.retrieve_chunks

        def find(q, manifest):
            named = original_find(q, manifest)
            if named:
                return [d for d in named if d in allowed]
            return sorted(allowed) if len(allowed) <= main.MAX_FOCUS_DOCS else []

        async def retrieve(rag_, q, focus, mode):
            chunks = await original_retrieve(rag_, q, focus, mode)
            return [c for c in chunks if Path(c["file_path"]).name in allowed]

        main.find_named_documents, main.retrieve_chunks = find, retrieve
        try:
            return await main.query(rag, question)
        finally:
            main.find_named_documents, main.retrieve_chunks = original_find, original_retrieve


def run_query(question: str, allowed: set[str] | None):
    loop, _, _ = backend()
    return asyncio.run_coroutine_threadsafe(_query(question, allowed), loop).result(timeout=180)


# --- Helpers ---------------------------------------------------------------------------------------

@st.cache_data
def manifest() -> dict[str, dict]:
    return main.load_manifest()


@st.cache_data(max_entries=64)
def page_png(document: str, page: int) -> bytes:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(main.ROOT / manifest()[document]["file_path"]))
    try:
        image = pdf[page - 1].render(scale=1.4).to_pil()
    finally:
        pdf.close()
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def is_api_error(exc: Exception) -> bool:
    import openai

    return isinstance(exc, (openai.APIError, openai.APIConnectionError, openai.RateLimitError,
                            openai.AuthenticationError, TimeoutError)) or "openai" in type(exc).__module__


def clean_answer(text: str) -> tuple[str, str | None]:
    """Strip internal source markers ([S3 p.2] -> [p.2]) and split off the number-guard warning."""
    warning = None
    if "\n\nWarning: could not verify" in text:
        text, warning = text.split("\n\nWarning: ", 1)
        warning = warning[:1].upper() + warning[1:]
    text = re.sub(r"\[\[?S\d+\s+(p\.\s*\d+)\]\]?", r"[\1]", text)
    return text.strip(), warning


def save_feedback(message_id: str):
    msg = next(m for m in st.session_state.messages if m.get("id") == message_id)
    value = st.session_state.get(f"fb_{message_id}")
    if value is None:
        return
    new_file = not FEEDBACK_FILE.exists()
    with open(FEEDBACK_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FEEDBACK_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="seconds"), "session_id": st.session_state.session_id,
            "message_id": message_id, "rating": "up" if value == 1 else "down", "question": msg["question"],
            "answer": msg["content"], "citations": "; ".join(f"{c['document']} p.{c['page']}" for c in msg["citations"]),
            "number_check": msg.get("verification", ""), "response_time_s": f"{msg.get('elapsed', 0):.2f}",
            "filters": msg.get("searching", ""),
        })
    st.toast("Thanks for the feedback!")



# --- Name checks: run before the engine, so a misspelt or unknown name never reaches main.query() --------

NAME_STOPWORDS = {"and", "or", "the", "of", "in", "for", "vs", "versus", "with", "a", "an", "to", "is", "are",
                  "what", "which", "how", "compare", "compared", "comparing", "comparison", "difference",
                  "between", "does", "do", "my", "me", "at", "on", "than", "from", "about", "its", "their", "s"}
COMPARISON = re.compile(r"\b(compare|compared|comparing|comparison|vs|versus|difference between)\b", re.I)
SIDE_SPLIT = {"and", "vs", "versus", "with", "to"}
# Topic words that never name a provider; what is left of a comparison side after removing them is name-like.
TOPIC_WORDS = (set(main.PRODUCT_CATEGORY_WORDS) | {w for hints in main.CATEGORY_HINTS.values() for h in hints
                                                    for w in h.split()} | {
    "rate", "rates", "day", "days", "year", "years", "month", "months", "period", "periods", "ratio", "free", "look",
    "waiting", "expense", "returns", "return", "senior", "citizen", "citizens", "normal", "public", "interest",
    "tenure", "tenures", "room", "rent", "limit", "limits", "premium", "sum", "insured", "manager", "managers",
    "plans", "bank", "banks", "deposits", "option", "options", "direct", "regular", "growth", "exclusions", "ratios",
    "grace", "renewal", "cashless", "riskometer", "risk", "allocation", "fees", "fee", "load", "exit", "minimum",
    "investment", "amount", "terms", "features", "benefits", "maternity", "copay", "co", "pay", "deductible",
    "both", "two", "these", "those", "them", "all", "better", "higher", "lower", "best", "more", "less", "one"})
FUZZY_MULTI, FUZZY_SINGLE = 0.75, 0.8
PRODUCT_FILLER = re.compile(r"\s*\b(plan|fund|policy|scheme)\b", re.I)
GROUP_NOUNS = {"insurance": "plans", "mutual_fund": "funds", "fixed_deposit": "rate sheets"}


def _norm_text(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower().replace("&", " and ")).split())


@st.cache_resource
def dictionary_words() -> set[str]:
    """Ordinary English words: a single typed word like "start" or "data" is not treated as a misspelt name."""
    try:
        return {w.strip().lower() for w in open("/usr/share/dict/words")}
    except OSError:
        return set()


@st.cache_data
def name_index() -> dict[str, dict]:
    """normalised name -> kind, display name, providers, documents and whether it may be a fuzzy target."""
    index: dict[str, dict] = {}

    def add(key, kind, display, rows, fuzzy):
        if not key:
            return
        if index.get(key, {}).get("kind") == "provider" and kind != "provider":
            return  # "hdfc" is the HDFC fund house, not also HDFC Bank and HDFC ERGO
        entry = index.setdefault(key, {"kind": kind, "display": display, "providers": set(), "docs": set(),
                                       "fuzzy": fuzzy})
        if entry["kind"] != kind and kind == "provider":  # a provider name wins over an alias or product
            entry.update(kind=kind, display=display, fuzzy=fuzzy)
        entry["providers"] |= {r["provider"] for r in rows}
        entry["docs"] |= {r["file"] for r in rows}

    rows = [dict(r, file=n) for n, r in manifest().items()]
    by_provider: dict[str, list[dict]] = {}
    for r in rows:
        by_provider.setdefault(r["provider"], []).append(r)
    for provider, prows in by_provider.items():
        key = _norm_text(provider)
        add(key, "provider", provider, prows, len(key) >= 4)
        if " " in key:
            add(key.replace(" ", ""), "provider", provider, prows, False)  # "tataaig", "starhealth"
        for alias in main.PROVIDER_ALIASES.get(provider, []):
            add(_norm_text(alias), "provider", provider, prows, False)
    for provider, prows in by_provider.items():
        first = _norm_text(provider).split()[0]
        if " " in _norm_text(provider) and len(first) >= 4 and first not in main.GENERIC_WORDS:
            add(first, "alias", provider.split()[0], prows, True)  # "Kotak", "Niva", "ICICI"
    for r in rows:
        full = _norm_text(r["product"])
        words = full.split()
        generic = all(w in main.PRODUCT_CATEGORY_WORDS or w == "plan" for w in words)
        for key in {full, _norm_text(PRODUCT_FILLER.sub("", r["product"]))}:
            if len(key) < 4 or (generic and len(key.split()) < 2):
                continue
            add(key, "product", r["product"], [r], not generic)
    return index


def exact_names(question: str) -> list[tuple[int, int, str]]:
    """(start, end, key) for known names in the question, longest first, without overlaps."""
    words = _norm_text(question).split()
    index = name_index()
    used: set[int] = set()
    found = []
    for size in (4, 3, 2, 1):
        for i in range(len(words) - size + 1):
            span = set(range(i, i + size))
            key = " ".join(words[i:i + size])
            if key in index and not span & used:
                found.append((i, i + size, key))
                used |= span
    return sorted(found)


def spelling_suggestions(question: str) -> list[tuple[str, str]]:
    """(text as typed, known name) pairs for phrases close to, but not exactly, a known name."""
    words = _norm_text(question).split()
    index = name_index()
    used = {i for start, end, _ in exact_names(question) for i in range(start, end)}
    dictionary = dictionary_words()
    found = []
    for size in (3, 2, 1):
        candidates = [k for k, e in index.items() if e["fuzzy"] and len(k.split()) == size]
        for i in range(len(words) - size + 1):
            span = set(range(i, i + size))
            gram_words = words[i:i + size]
            if span & used or gram_words[0] in NAME_STOPWORDS or gram_words[-1] in NAME_STOPWORDS:
                continue
            if all(w in TOPIC_WORDS or w in NAME_STOPWORDS or w.isdigit() for w in gram_words):
                continue
            gram = " ".join(gram_words)
            if size == 1 and (gram in dictionary or len(gram) < 4):  # "start" is a word, not a misspelt "Star"
                continue
            match = difflib.get_close_matches(gram, candidates, n=1, cutoff=FUZZY_SINGLE if size == 1 else FUZZY_MULTI)
            if not match or (size == 1 and abs(len(gram) - len(match[0])) > 1):  # "digital" is not "Digit"
                continue
            if size > 1 and any(difflib.SequenceMatcher(None, a, b).ratio() < 0.5
                                for a, b in zip(gram_words, match[0].split())):
                continue  # every word must be close: "cigna health" is not "care health"
            found.append((gram, index[match[0]]["display"]))
            used |= span
    return found


def corrected_question(question: str, suggestions: list[tuple[str, str]]) -> str:
    for typed, name in suggestions:
        pattern = r"\b" + r"\W+".join(map(re.escape, typed.split())) + r"\b"
        question = re.sub(pattern, name, question, count=1, flags=re.I)
    return question


def named_entities(question: str) -> set:
    """Distinct things named: a single-document product, or a provider not already pinned down by its product."""
    index = name_index()
    entries = [index[key] for _, _, key in exact_names(question)]
    products = {next(iter(e["docs"])): e for e in entries if e["kind"] == "product" and len(e["docs"]) == 1}
    covered = {p for e in products.values() for p in e["providers"]}
    providers = {frozenset(e["providers"]) for e in entries if e["kind"] != "product" or len(e["docs"]) > 1}
    return set(products) | {p for p in providers if not p <= covered}


def comparison_sides(question: str) -> list[list[str]]:
    """The things being compared, as word lists ("between X and Y", "X vs Y", "compare X with Y")."""
    words = _norm_text(question).split()
    for marker in ("between", "compare", "compared", "comparing", "comparison"):
        if marker in words and words.index(marker) < len(words) - 1:
            words = words[words.index(marker) + 1:]
            break
    sides, current = [], []
    for w in words:
        if w in SIDE_SPLIT:
            sides.append(current)
            current = []
        else:
            current.append(w)
    sides.append(current)
    return [s for s in sides if s]


def unrecognised_parts(question: str) -> list[str]:
    """Name-like comparison sides that contain no known name."""
    index = name_index()
    parts = []
    for side in comparison_sides(question):
        text = " ".join(side)
        if any(text_key in index for text_key in
               (" ".join(side[i:j]) for i in range(len(side)) for j in range(i + 1, min(i + 4, len(side)) + 1))):
            continue
        leftover = [w for w in side if w not in TOPIC_WORDS and w not in NAME_STOPWORDS and not w.isdigit()]
        if leftover:
            parts.append(" ".join(leftover))
    return parts


def precheck(question: str) -> dict | None:
    """A reply that replaces the engine call when a name is misspelt or unknown; None when the question is fine."""
    suggestions = spelling_suggestions(question)
    if suggestions:
        fixed = corrected_question(question, suggestions)
        label = fixed if fixed.endswith("?") else fixed + "?"
        intro = ("I could not recognise one of the names." if COMPARISON.search(question)
                 else f"I could not recognise \"{suggestions[0][0]}\".")
        return {"content": intro, "suggestion": fixed, "suggestion_label": f"Did you mean: {label}"}
    if COMPARISON.search(question) and len(named_entities(question)) < 2:
        parts = unrecognised_parts(question)
        if parts:
            providers = sorted({r["provider"] for r in manifest().values()}, key=str.lower)
            quoted = " and ".join(f"\"{p}\"" for p in parts)
            return {"content": f"I could not recognise {quoted} as a provider or product in the documents. "
                               f"Available providers: {', '.join(providers)}."}
    return None


def multi_document_hints(question: str, answer: str) -> list[str]:
    """Hints for a named provider with several documents in the asked category, when none is specified."""
    if CLARIFICATION.match(answer) or re.search(r"\bwhich (plan|fund|policy|product|document|one)\b", answer, re.I):
        return []  # the answer already asks which one
    index = name_index()
    padded = f" {_norm_text(question)} "
    squashed = padded.replace(" ", "")
    asked = {c for c, hints in main.CATEGORY_HINTS.items() if any(f" {_norm_text(h)} " in padded for h in hints)}
    hints = []
    for _, _, key in exact_names(question):  # longest names first: "HDFC Bank" is never read as "HDFC"
        entry = index[key]
        if entry["kind"] == "product" or len(entry["providers"]) != 1:
            continue
        provider = next(iter(entry["providers"]))
        groups: dict[str, list[dict]] = {}
        for r in manifest().values():
            if r["provider"] == provider:
                groups.setdefault(r["category"], []).append(r)
        if asked:  # an FD question about "HDFC" is about HDFC Bank, never the HDFC funds
            groups = {c: g for c, g in groups.items() if c in asked}
        for category, rows in groups.items():
            if len(rows) < 2:
                continue
            labels = [PRODUCT_FILLER.sub("", r["product"]).strip() or r["product"] for r in rows]
            if any(f" {_norm_text(label)} " in padded or "".join(_norm_text(label).split()[:2]) in squashed
                   for label in labels):
                continue  # one of its products is already named ("midcap" names "Mid-Cap Opportunities")
            listed = ", ".join(labels[:-1]) + f" and {labels[-1]}"
            hint = (f"{provider} has {len(rows)} {GROUP_NOUNS[category]} in the documents: {listed}. "
                    "Ask about one for a precise answer.")
            if hint not in hints:
                hints.append(hint)
    return hints


def ask_again(question: str):
    st.session_state.pending = question


def reset_filters():
    st.session_state.categories = list(ALL_CATEGORIES)
    st.session_state.providers = []


def apply_example(question: str, example_filter: dict | None):
    if example_filter:  # make the needed filter visible in the sidebar before the question runs
        st.session_state.categories = example_filter.get("categories", list(ALL_CATEGORIES))
        st.session_state.providers = example_filter.get("providers", [])
    st.session_state.pending = question


# --- Session state -----------------------------------------------------------------------------------

st.session_state.setdefault("messages", [])
st.session_state.setdefault("queries_used", 0)
st.session_state.setdefault("session_id", uuid.uuid4().hex[:8])
st.session_state.setdefault("pending", None)
st.session_state.setdefault("categories", list(ALL_CATEGORIES))
st.session_state.setdefault("providers", [])

docs = manifest()
TOTAL = len(docs)

# --- Sidebar: Filters and Session -------------------------------------------------------------------

with st.sidebar:
    st.subheader("Filters")
    categories = st.multiselect("Categories", ALL_CATEGORIES, format_func=CATEGORY_LABELS.get, key="categories",
                                placeholder="Choose at least one category")
    provider_pool = sorted({r["provider"] for r in docs.values() if r["category"] in categories})
    # Keep only providers that still exist under the chosen categories.
    st.session_state.providers = [p for p in st.session_state.providers if p in provider_pool]
    providers = st.multiselect("Providers", provider_pool, key="providers", placeholder="All providers")
    allowed = {n for n, r in docs.items()
               if r["category"] in categories and (not providers or r["provider"] in providers)}
    filter_active = len(allowed) < TOTAL
    if not filter_active:
        searching = f"all {TOTAL} documents"
    else:
        scope = ", ".join(providers) if providers else ", ".join(CATEGORY_LABELS[c] for c in categories) or "nothing"
        searching = f"{scope} ({len(allowed)} document{'s' if len(allowed) != 1 else ''})"
    st.caption(f"Searching: {searching}")
    if not allowed:
        st.warning("No documents match this filter.")
    st.button("Reset filters", on_click=reset_filters, key="reset_filters", width="stretch",
              disabled=not filter_active)

    st.subheader("Session")
    used = st.session_state.queries_used
    st.progress(min(used / MAX_QUERIES_PER_SESSION, 1.0), text=f"{used} / {MAX_QUERIES_PER_SESSION} questions used")
    st.caption(f"{TOTAL} documents: {sum(r['category'] == 'insurance' for r in docs.values())} insurance policies, "
               f"{sum(r['category'] == 'mutual_fund' for r in docs.values())} mutual fund factsheets, "
               f"{sum(r['category'] == 'fixed_deposit' for r in docs.values())} bank FD rate sheets.")

# --- Header ---------------------------------------------------------------------------------------

st.title("📄 FinDocQA")
st.markdown('<div class="fdq-sub">Answers about Indian health insurance policies, mutual fund factsheets and bank '
            'FD rates, with the document and page behind every claim.</div>', unsafe_allow_html=True)
st.markdown(f'<div class="fdq-disclaimer">ℹ️ {DISCLAIMER}</div>', unsafe_allow_html=True)

limit_reached = st.session_state.queries_used >= MAX_QUERIES_PER_SESSION

# --- Examples (2 x 3 grid, tag on its own line, text wraps) ------------------------------------------

st.markdown("**Try an example**")
for start in (0, 3):
    for offset, col in enumerate(st.columns(3, gap="small")):
        i = start + offset
        tag, question, example_filter = EXAMPLES[i]
        with col:
            st.button(f"**{tag}**\n\n{question}", key=f"example_{i}", width="stretch", disabled=limit_reached,
                      on_click=apply_example, args=(question, example_filter))

# --- Chat --------------------------------------------------------------------------------------------


def render_assistant(msg: dict):
    if msg.get("error"):
        st.error(msg["content"])
        return
    if msg.get("precheck"):  # name check reply: the engine was not called
        st.markdown(msg["content"])
        if msg.get("suggestion"):
            st.button(msg["suggestion_label"], key=f"respell_{msg['id']}", on_click=ask_again,
                      args=(msg["suggestion"],), disabled=limit_reached, type="primary", icon="🔁")
        return
    if msg.get("verification") == "withheld":
        st.warning(f"⚠️ {WITHHELD_MESSAGE}")
        if msg.get("searched"):
            st.caption("Documents searched: " + ", ".join(msg["searched"]))
    else:
        st.markdown(msg["content"])
        if msg.get("warning"):
            st.warning(f"⚠️ {msg['warning']}")
    if msg.get("no_match_hint"):
        st.markdown(f'<div class="fdq-hint">🔎 {NO_MATCH_HINT}</div>', unsafe_allow_html=True)
    for hint in msg.get("multi_hints", []):
        st.markdown(f'<div class="fdq-note">📚 {hint}</div>', unsafe_allow_html=True)
    pages: dict[tuple[str, int], list[dict]] = {}
    for c in msg["citations"]:
        pages.setdefault((c["document"], c["page"]), []).append(c)
    meta = f"⏱ {msg['elapsed']:.1f} s"
    if pages:
        meta += f" · {len(pages)} cited page{'s' if len(pages) != 1 else ''}"
    meta += f" · Searching: {msg['searching']}"
    st.caption(meta)
    for (doc, page), claims in pages.items():
        meta_row = docs.get(doc, {})
        title = f"📑 {meta_row.get('provider', doc)} · {meta_row.get('product', '')} — page {page}"
        if not all(c["verified"] for c in claims):
            title += " (page not confirmed)"
        with st.expander(title):
            for c in claims:
                st.markdown(f"- {c['claim']}")
            details = doc + (f" · document date {meta_row['doc_date']}" if meta_row.get("doc_date") else "")
            st.caption(details)
            if page:
                st.image(page_png(doc, page), caption=f"{doc} — page {page}", width=760)


if not st.session_state.messages:
    st.markdown("#### What you can ask")
    st.markdown(
        "- **Insurance terms** — waiting periods, free look period, room rent limits, exclusions in a policy\n"
        "- **Fund facts** — fund managers, expense ratios, holdings, risk levels and charts in a factsheet\n"
        "- **FD rates** — a bank's interest rate for a tenure, senior citizen rates, comparisons between banks"
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_assistant(msg)
            if not msg.get("error") and not msg.get("precheck"):
                st.feedback("thumbs", key=f"fb_{msg['id']}", on_change=save_feedback, args=(msg["id"],))

typed = st.chat_input("Ask about a policy, fund or FD rate…" if not limit_reached
                      else f"Session limit of {MAX_QUERIES_PER_SESSION} questions reached", disabled=limit_reached)
if limit_reached:
    st.info(f"You have used all {MAX_QUERIES_PER_SESSION} questions for this session. "
            "Reload the page to start a new session.")

question = typed or st.session_state.pending
st.session_state.pending = None

if question and not limit_reached:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    msg = {"role": "assistant", "id": uuid.uuid4().hex[:10], "question": question, "citations": [],
           "searching": searching}
    reply = precheck(question)
    with st.chat_message("assistant"):
        started = time.perf_counter()
        try:
            if reply:  # misspelt or unknown name: answer here, without calling the engine or using a question
                msg.update(reply, precheck=True)
                raise StopIteration
            st.session_state.queries_used += 1
            if not allowed:
                raise LookupError("empty filter")
            with st.spinner("Searching documents..."):
                result = run_query(question, allowed if filter_active else None)
            answer, warning = clean_answer(result.answer)
            msg.update(content=answer, warning=warning, verification=result.verification,
                       searched=result.focus_documents,
                       citations=[{"document": c.document, "page": c.page, "claim": c.claim, "verified": c.verified}
                                  for c in result.citations])
            # A filtered search that found nothing citable: point the user at the filter.
            msg["no_match_hint"] = (filter_active and not msg["citations"] and result.verification != "withheld"
                                    and not CLARIFICATION.match(answer))
            msg["multi_hints"] = multi_document_hints(question, answer)
        except StopIteration:
            pass
        except LookupError:
            msg.update(content="No documents match the current filter.", no_match_hint=True, verification="ok")
        except Exception as exc:  # noqa: BLE001 - shown to the user as a friendly message
            msg.update(error=True, content=API_ERROR_MESSAGE if is_api_error(exc)
                       else "Sorry, something went wrong while answering. Please try again.",
                       error_detail=f"{type(exc).__name__}: {exc}")
        msg["elapsed"] = time.perf_counter() - started
        st.session_state.messages.append(msg)
        render_assistant(msg)
        if not msg.get("error") and not msg.get("precheck"):
            st.feedback("thumbs", key=f"fb_{msg['id']}", on_change=save_feedback, args=(msg["id"],))
    if st.session_state.queries_used >= MAX_QUERIES_PER_SESSION:
        st.rerun()  # redraw with the input disabled
