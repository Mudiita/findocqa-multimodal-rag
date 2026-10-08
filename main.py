"""FinDocQA: multimodal RAG over insurance / mutual fund / fixed deposit PDFs.

Usage:
    python main.py ingest --test        # ingest 2 PDFs only, print token usage + cost estimate
    python main.py ingest               # ingest all three folders (skips already-ingested files)
    python main.py reingest documents/fixed_deposits/hdfc_fd.pdf   # replace one file in the index
    python main.py ask "What is the 1-year FD rate at HDFC Bank?"
    python main.py                      # interactive prompt

Citations: RAG-Anything merges all text of a document into one string before
chunking, which drops MinerU's page_idx. We re-insert page markers ("[[page N]]")
into the parsed content list before insertion, so every text chunk carries its
page. Multimodal chunks (tables/images) keep page_idx in chunk metadata.
"""

import argparse
import asyncio
import copy
import html
import csv
import json
import os
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
# RAG-Anything shells out to the `mineru` CLI; make sure the venv's bin dir is on PATH
# even when the venv isn't activated.
os.environ["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"
# MinerU loads 64 pages per processing window by default, which ran an 8 GB Mac out of
# memory on 50+ page PDFs (worker killed). Smaller windows only lower peak memory.
os.environ.setdefault("MINERU_PROCESSING_WINDOW_SIZE", "8")

from lightrag import QueryParam  # noqa: E402
from lightrag.base import DocStatus  # noqa: E402
from lightrag.llm.openai import openai_complete_if_cache, openai_embed  # noqa: E402
from lightrag.utils import EmbeddingFunc, TokenTracker  # noqa: E402
from raganything import RAGAnything, RAGAnythingConfig  # noqa: E402

WORKING_DIR = ROOT / "rag_storage"
PARSER_OUTPUT_DIR = ROOT / "parsed_output"
MANIFEST_PATH = ROOT / "dataset_manifest.csv"
DOC_FOLDERS = [
    ROOT / "documents" / "insurance",
    ROOT / "documents" / "mutual_funds",
    ROOT / "documents" / "fixed_deposits",
]
TEST_FILES = [
    ROOT / "documents" / "insurance" / "starhealth_premium.pdf",
    ROOT / "documents" / "fixed_deposits" / "hdfc_fd.pdf",
]

VISION_MODEL = "gpt-4o"
LLM_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072

# USD per 1M tokens (input, output). Check current OpenAI pricing before relying on these.
PRICES = {
    VISION_MODEL: (2.50, 10.00),
    LLM_MODEL: (0.15, 0.60),
    EMBED_MODEL: (0.13, 0.0),
}

PAGE_MARKER = "[[page {}]]"
PAGE_MARKER_RE = re.compile(r"\[\[page (\d+)\]\]")

trackers = {model: TokenTracker() for model in PRICES}

COST_FILE = WORKING_DIR / "ingest_cost.json"
BACKENDS_FILE = WORKING_DIR / "ingest_backends.json"  # which MinerU backend parsed each file
# "pipeline" needs far less memory than MinerU's default vision (hybrid) backend,
# which failed on 50+ page PDFs on an 8 GB Mac.
DEFAULT_BACKEND = "auto"
VISION_BACKEND_LABEL = "hybrid (vision, MinerU default)"
# "auto" picks per category. Pipeline kept every row of the insurance and FD tables we checked,
# but shifted values between rows in a mutual fund factsheet's holdings table, which the
# vision backend got right. Factsheets are short, so the vision backend fits in memory.
BACKEND_BY_FOLDER = {"insurance": "pipeline", "fixed_deposits": "pipeline", "mutual_funds": "vision"}


def resolve_backend(backend: str, folder_name: str) -> str:
    return BACKEND_BY_FOLDER.get(folder_name, "pipeline") if backend == "auto" else backend


def mineru_kwargs(backend: str) -> dict:
    # MinerU's default backend is the vision (hybrid) one, so "vision" passes no -b flag.
    return {} if backend == "vision" else {"backend": backend}


def backend_label(backend: str) -> str:
    return VISION_BACKEND_LABEL if backend == "vision" else backend
PROGRESS_LOG = ROOT / "ingest_progress.log"


class BudgetExceeded(RuntimeError):
    pass


# limit: USD cap for the full ingestion (None = no cap); spent_before: cost of earlier runs.
budget = {"limit": None, "spent_before": 0.0, "exceeded": False}


def session_cost() -> float:
    total = 0.0
    for model, t in trackers.items():
        price_in, price_out = PRICES[model]
        total += t.prompt_tokens / 1e6 * price_in + t.completion_tokens / 1e6 * price_out
    return total


def _check_budget():
    if budget["limit"] is not None and budget["spent_before"] + session_cost() >= budget["limit"]:
        budget["exceeded"] = True
        raise BudgetExceeded(f"cost cap of ${budget['limit']:.2f} reached")


# ---------------------------------------------------------------------------
# Model functions
# ---------------------------------------------------------------------------


def _api_key() -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        sys.exit("OPENAI_API_KEY is not set. Add it to .env (OPENAI_API_KEY=sk-...).")
    return key


async def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
    _check_budget()
    return await openai_complete_if_cache(
        LLM_MODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages or [],
        api_key=_api_key(),
        token_tracker=trackers[LLM_MODEL],
        **kwargs,
    )


async def vision_model_func(
    prompt, system_prompt=None, history_messages=None, image_data=None, messages=None, **kwargs
):
    _check_budget()
    common = dict(api_key=_api_key(), token_tracker=trackers[VISION_MODEL])
    if messages:
        return await openai_complete_if_cache(
            VISION_MODEL, "", system_prompt=None, history_messages=[], messages=messages, **common, **kwargs
        )
    if image_data:
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                ],
            }
        )
        return await openai_complete_if_cache(
            VISION_MODEL, "", system_prompt=None, history_messages=[], messages=msgs, **common, **kwargs
        )
    return await llm_model_func(prompt, system_prompt, history_messages, **kwargs)


async def _embed(texts, **kwargs):
    _check_budget()
    return await openai_embed.func(
        texts, model=EMBED_MODEL, api_key=_api_key(), token_tracker=trackers[EMBED_MODEL], **kwargs
    )


embedding_func = EmbeddingFunc(embedding_dim=EMBED_DIM, max_token_size=8192, func=_embed)


# ---------------------------------------------------------------------------
# RAG-Anything with page-aware text
# ---------------------------------------------------------------------------


def add_page_markers(content_list):
    """Prefix the first text block of each page with [[page N]] (1-based)."""
    marked = copy.deepcopy(content_list)
    last_page = None
    for item in marked:
        if item.get("type", "text") != "text" or not str(item.get("text", "")).strip():
            continue
        page = int(item.get("page_idx", 0) or 0) + 1
        if page != last_page:
            item["text"] = f"{PAGE_MARKER.format(page)} {item['text']}"
            last_page = page
    return marked


def charts_as_images(content_list):
    """MinerU tags plots as type "chart", which RAG-Anything sends to its text-only generic processor
    (it never looks at the picture). Re-type them as images so the vision model reads the chart."""
    out = []
    for item in content_list:
        if item.get("type") == "chart" and item.get("img_path"):
            converted = {k: v for k, v in item.items() if k not in ("chart_caption", "chart_footnote")}
            converted.update(type="image", image_caption=item.get("chart_caption") or [],
                             image_footnote=item.get("chart_footnote") or [])
            item = converted
        out.append(item)
    return out


class PagedRAGAnything(RAGAnything):
    async def parse_document(self, *args, **kwargs):
        content_list, doc_id = await super().parse_document(*args, **kwargs)
        return add_page_markers(charts_as_images(content_list)), doc_id


def build_rag() -> PagedRAGAnything:
    config = RAGAnythingConfig(
        working_dir=str(WORKING_DIR),
        parser_output_dir=str(PARSER_OUTPUT_DIR),
        parser="mineru",
        parse_method="auto",
        enable_table_processing=True,
        enable_image_processing=True,
        enable_equation_processing=False,
    )
    return PagedRAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def load_manifest() -> dict[str, dict]:
    with open(MANIFEST_PATH, newline="") as f:
        return {Path(row["file_path"]).name: row for row in csv.DictReader(f)}


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


async def ingested_files(rag: RAGAnything) -> set[str]:
    """File names whose text and multimodal content are both fully processed."""
    await rag._ensure_lightrag_initialized()
    docs = await rag.lightrag.doc_status.get_docs_by_status(DocStatus.PROCESSED)
    done = set()
    for doc_id, status in docs.items():
        if await rag.is_document_fully_processed(doc_id):
            done.add(Path(status.file_path).name)
    return done


def cost_report(title: str):
    print(f"\n=== {title} ===")
    total = 0.0
    for model, t in trackers.items():
        price_in, price_out = PRICES[model]
        cost = t.prompt_tokens / 1e6 * price_in + t.completion_tokens / 1e6 * price_out
        total += cost
        print(
            f"{model:24s} calls={t.call_count:5d} in={t.prompt_tokens:9d} "
            f"out={t.completion_tokens:8d}  ${cost:.4f}"
        )
    print(f"{'TOTAL':24s} ${total:.4f}")
    return total


async def _ingest_files(rag: RAGAnything, paths: list[Path]):
    # process_folder_complete works on folders, so stage the files in a temp dir.
    with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
        for p in paths:
            os.symlink(p, Path(tmp) / p.name)
        await rag.process_folder_complete(tmp, output_dir=str(PARSER_OUTPUT_DIR), recursive=False)


async def remove_document(rag: RAGAnything, file_name: str):
    """Delete a file's doc record, chunks, vectors, graph entities and LLM cache."""
    await rag._ensure_lightrag_initialized()
    lightrag = rag.lightrag
    for status in DocStatus:
        for doc_id, doc in (await lightrag.doc_status.get_docs_by_status(status)).items():
            if Path(doc.file_path).name != file_name:
                continue
            result = await lightrag.adelete_by_doc_id(doc_id, delete_llm_cache=True)
            print(f"Deleted {doc_id} ({status.value}): {result.status}")
            if rag.multimodal_status_cache is not None:
                await rag.multimodal_status_cache.delete([doc_id])

    # Entities created by RAG-Anything's multimodal step aren't tracked for deletion,
    # so remove any graph nodes that still come only from this file.
    orphans = [
        node["id"]
        for node in await lightrag.chunk_entity_relation_graph.get_all_nodes()
        if {Path(f).name for f in str(node.get("file_path", "")).split("<SEP>")} == {file_name}
    ]
    for entity in orphans:
        await lightrag.adelete_by_entity(entity)
    if orphans:
        print(f"Removed {len(orphans)} leftover graph entities")


async def reingest(file_path: str, backend: str = DEFAULT_BACKEND):
    path = Path(file_path).resolve()
    if not path.is_file():
        sys.exit(f"File not found: {file_path}")
    rag = build_rag()
    await remove_document(rag, path.name)
    # MinerU reuses <stem>_<hash>/ output folders without clearing them, so old
    # images would pile up next to the new parse. Remove the file's old output first.
    old_output = re.compile(rf"{re.escape(path.stem)}_[0-9a-f]{{8}}")
    for parent in (PARSER_OUTPUT_DIR, PARSER_OUTPUT_DIR / path.parent.name):
        for d in parent.glob(f"{path.stem}_*"):
            if d.is_dir() and old_output.fullmatch(d.name):
                shutil.rmtree(d)
                print(f"Removed old parse output {d.relative_to(ROOT)}")
    backend = resolve_backend(backend, path.parent.name)
    await rag.process_document_complete(
        str(path), output_dir=str(PARSER_OUTPUT_DIR / path.parent.name), **mineru_kwargs(backend)
    )
    if path.name in await ingested_files(rag):
        save_backend(path.name, backend_label(backend), await ingested_files(rag))
    await rag.finalize_storages()
    print(f"Parsed with MinerU backend: {backend_label(backend)}")
    cost_report(f"Reingest token usage ({path.name})")


async def ingest(test: bool = False, max_cost: float | None = 7.0, backend: str = DEFAULT_BACKEND):
    rag = build_rag()
    done = await ingested_files(rag)

    if test:
        pending = [p for p in TEST_FILES if p.name not in done]
        if not pending:
            print("Test files already ingested.")
            return
        await _ingest_files(rag, pending)
    else:
        await rag.finalize_storages()
        await ingest_all(max_cost, backend)
        return

    await rag.finalize_storages()
    cost_report("Ingestion token usage (this run)")
    now_done = await ingested_files(build_rag())
    print(f"Fully ingested documents: {len(now_done)}")


def log_progress(message: str):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {message}"
    print(line, flush=True)
    with open(PROGRESS_LOG, "a") as f:
        f.write(line + "\n")


def load_backends(done: set[str]) -> dict[str, str]:
    backends = json.loads(BACKENDS_FILE.read_text()) if BACKENDS_FILE.exists() else {}
    # Files indexed before backends were recorded were all parsed with MinerU's default.
    for name in done:
        backends.setdefault(name, VISION_BACKEND_LABEL)
    return backends


def save_backend(name: str, backend: str, done: set[str]):
    backends = load_backends(done)
    backends[name] = backend
    BACKENDS_FILE.write_text(json.dumps(backends, indent=1, sort_keys=True))


def _save_cost(total: float):
    COST_FILE.write_text(json.dumps({"spent_usd": round(total, 6)}))


async def ingest_all(max_cost: float | None, backend: str = DEFAULT_BACKEND):
    """Ingest every PDF one file at a time: skips indexed files, logs each file,
    stops at the cost cap, and can be rerun to resume after a stop or crash."""
    spent_before = json.loads(COST_FILE.read_text())["spent_usd"] if COST_FILE.exists() else 0.0
    budget.update(limit=max_cost, spent_before=spent_before, exceeded=False)
    rag = build_rag()
    done = await ingested_files(rag)
    pdfs = [(folder, p) for folder in DOC_FOLDERS for p in sorted(folder.glob("*.pdf"))]
    cap = f"${max_cost:.2f}" if max_cost is not None else "none"
    backends = load_backends(done)
    BACKENDS_FILE.write_text(json.dumps(backends, indent=1, sort_keys=True))
    log_progress(
        f"START {len(pdfs)} PDFs, {len(done)} already indexed, cost so far ${spent_before:.4f}, "
        f"cap {cap}, backend for new files: "
        + (", ".join(f"{k}={v}" for k, v in BACKEND_BY_FOLDER.items()) if backend == "auto" else backend)
    )

    failed, processed = [], []
    try:
        for i, (folder, pdf) in enumerate(pdfs, 1):
            tag = f"[{i:2d}/{len(pdfs)}] {folder.name}/{pdf.name}"
            if pdf.name in done:
                log_progress(f"{tag}  SKIP (already indexed, backend: {backends.get(pdf.name, 'unknown')})")
                continue
            if budget["limit"] is not None and spent_before + session_cost() >= budget["limit"]:
                budget["exceeded"] = True
                break
            # A file left half-done by an earlier stop or crash is cleared and redone.
            await remove_document(rag, pdf.name)
            file_backend = resolve_backend(backend, folder.name)
            log_progress(f"{tag}  START (backend: {backend_label(file_backend)})")
            started, cost_at_start = time.time(), session_cost()
            error = None
            try:
                await rag.process_document_complete(
                    str(pdf), output_dir=str(PARSER_OUTPUT_DIR / folder.name), **mineru_kwargs(file_backend)
                )
            except Exception as exc:  # noqa: BLE001 - record and continue with the next file
                error = f"{type(exc).__name__}: {exc}"
            file_cost = session_cost() - cost_at_start
            _save_cost(spent_before + session_cost())
            took = (
                f"{(time.time() - started) / 60:.1f} min, ${file_cost:.4f}, "
                f"total ${spent_before + session_cost():.4f}, backend: {backend_label(file_backend)}"
            )

            if budget["exceeded"]:
                await remove_document(rag, pdf.name)  # don't leave a half-indexed file behind
                log_progress(f"{tag}  STOPPED by cost cap ({took}); partial data removed, will redo on resume")
                break
            if error is None and pdf.name in await ingested_files(rag):
                processed.append(pdf.name)
                save_backend(pdf.name, backend_label(file_backend), done)
                log_progress(f"{tag}  DONE ({took})")
            else:
                failed.append((pdf.name, error or "document not fully processed"))
                log_progress(f"{tag}  FAILED ({took}): {(error or 'document not fully processed')[:300]}")
    finally:
        _save_cost(spent_before + session_cost())
        await rag.finalize_storages()

    total = spent_before + session_cost()
    remaining = [p.name for _, p in pdfs if p.name not in await ingested_files(build_rag())]
    if budget["exceeded"]:
        log_progress(f"STOPPED: cost cap {cap} reached (total ${total:.4f}). {len(remaining)} files not indexed; "
                     f"rerun `python main.py ingest` with a higher --max-cost to resume.")
    log_progress(f"END processed {len(processed)}, failed {len(failed)}, not indexed {len(remaining)}, "
                 f"cost this run ${session_cost():.4f}, total ${total:.4f}")
    for name, err in failed:
        log_progress(f"  FAILED FILE {name}: {err[:300]}")
    cost_report("Ingestion token usage (this run)")


# ---------------------------------------------------------------------------
# Query with per-claim citations
# ---------------------------------------------------------------------------

ANSWER_SYSTEM_PROMPT = """You answer questions about Indian financial products \
(health insurance, mutual funds, fixed deposits) using ONLY the numbered sources provided.

RULE 1 (overrides everything else): every fact and number must come from the sources. Never use outside or
general knowledge, even for well-known facts (tax rules, other banks' rates, market data). If the sources do not
contain what is asked - for example the bank, fund or policy asked about is not among the sources - say that it
is not available in the documents and return an empty "claims" list. Do not guess or fill in typical values.

Each source is labelled [S<n>] and its text contains page markers like [[page 7]]; \
text after a marker belongs to that page until the next marker.

Return JSON:
{
  "answer": "<concise answer; put a citation like [S2 p.7] right after each factual claim>",
  "claims": [{"claim": "<one factual claim>", "source": <n>, "page": <page number>}]
}
Every factual statement in the answer must appear in "claims" with the source and page it came from.
When a source answers the question with a table row or a list, report every value it gives \
for that item (for example, all rate columns of the matching row), not just the first.
Cite the page marker of the source text you actually used.

How to choose among values that ARE in the sources:
- Fixed deposit rates: if a bank's source has several tables, use its standard retail (callable / premature
  withdrawal allowed) domestic table for deposits below Rs 3 crore unless the question asks about another slab
  or deposit type, and name the table you used, e.g. "(below Rs 3 crore, regular FD)". Give both the general and
  the senior citizen rate when the source gives both. Do not quote other slabs or special deposit types
  (non-callable, bulk, NRE, named schemes) unless asked.
- Attribute a figure to a bank, fund or policy only if it comes from that provider's own source (see each
  source's header). Never use a figure from another provider's document; if the named provider's sources do not
  contain the figure, say it is not available in that document.
- Risk questions about a fund: report the riskometer level (scheme, and benchmark if given) from the RISKOMETER
  source, and the Potential Risk Class if given.
Note the document date when rates or figures may be time-sensitive."""


@dataclass
class Citation:
    claim: str
    document: str
    page: int | None
    provider: str = ""
    product: str = ""
    category: str = ""
    doc_date: str = ""
    verified: bool = False  # page actually occurs in the cited source chunk


@dataclass
class QAResult:
    question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    focus_documents: list[str] = field(default_factory=list)  # documents named in the question
    unverified_numbers: list[str] = field(default_factory=list)  # figures not found in any retrieved source
    computed_numbers: list[str] = field(default_factory=list)  # differences/sums of verified figures
    verification: str = "ok"  # ok | flagged | withheld


def tables_to_text(content: str) -> str:
    """Rewrite HTML tables as one pipe-separated line per row, which the LLM reads more reliably."""

    def convert(match):
        lines = []
        for row in re.findall(r"<tr[^>]*>(.*?)</tr>", match.group(0), re.S):
            cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S)
            cells = [html.unescape(re.sub(r"<[^>]+>", "", c)).strip() for c in cells]
            if any(cells):
                lines.append(" | ".join(cells))
        return "\n" + "\n".join(lines) + "\n"

    return re.sub(r"<table[^>]*>.*?</table>", convert, content, flags=re.S)


async def _resolve_chunk_pages(rag: RAGAnything, chunk: dict) -> tuple[str, set[int]]:
    """Return chunk text guaranteed to start with a page marker, plus the pages it covers."""
    content = chunk["content"]
    stored = (await rag.lightrag.text_chunks.get_by_id(chunk["chunk_id"])) or {}

    if stored.get("is_multimodal"):
        page = int(stored.get("page_idx", 0) or 0) + 1
        return f"{PAGE_MARKER.format(page)} {content}", {page}

    pages = {int(p) for p in PAGE_MARKER_RE.findall(content)}
    # RAG-Anything >= 1.4.2 stores the 0-based page span of each text chunk.
    if stored.get("page_idx") is not None:
        first = int(stored["page_idx"]) + 1
        last = int(stored.get("page_idx_end") if stored.get("page_idx_end") is not None else stored["page_idx"]) + 1
        pages |= set(range(first, last + 1))
        if not content.lstrip().startswith("[[page"):
            # Chunk starts mid-page: label the leading text with the chunk's first page.
            content = f"{PAGE_MARKER.format(first)} {content}"
    return content, pages


# --- Riskometer extraction ----------------------------------------------------
# Mutual fund riskometers are printed as a gauge plus a short caption ("The risk of the
# scheme is moderate"), often inside an image, so the level is frequently missing from the
# parsed text. This one-time step asks the vision model to read it from the rendered page.

RISKOMETER_FILE = WORKING_DIR / "riskometers.json"
RISK_PAGE_HINT = re.compile(r"risk\s*-?\s*o\s*-?\s*meter|principal\s+will\s+be\s+at|risk\s+of\s+the\s+scheme", re.I)
RISKOMETER_PROMPT = """This is a page from an Indian mutual fund factsheet or presentation.
Set has_riskometer=true ONLY if the riskometer gauge graphic itself (a semicircular dial from Low to
Very High with a needle) is visible on this page. Text that merely mentions risk does not count.
If it is visible, read it.
Prefer the printed caption (e.g. "The risk of the scheme is moderate", "principal will be at Very High risk");
use the needle position only if there is no caption. Also read the Potential Risk Class (PRC) matrix if present
(e.g. "B-III" = moderate credit risk, relatively high interest rate risk).
Return JSON only:
{"has_riskometer": true/false,
 "fund_name": "<scheme name printed on the page>",
 "scheme_risk": "<Low | Low to Moderate | Moderate | Moderately High | High | Very High | null>",
 "benchmark_risk": "<same scale or null>",
 "potential_risk_class": "<e.g. B-III, or null>",
 "potential_risk_class_meaning": "<interest rate risk and credit risk in words, or null>",
 "caption": "<exact caption text you read, or null>"}"""


def _candidate_risk_pages(pdf_path: Path) -> list[int]:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(pdf_path))
    if len(pdf) <= 3:  # a factsheet: check every page
        return list(range(1, len(pdf) + 1))
    pages = []
    for i in range(len(pdf)):
        text = pdf[i].get_textpage().get_text_range()
        # Pages with (almost) no text layer are scans or images: only a visual check can tell.
        if len(text.strip()) < 100 or RISK_PAGE_HINT.search(text):
            pages.append(i + 1)
    return pages


def _page_image_b64(pdf_path: Path, page: int) -> str:
    import base64
    import io

    import pypdfium2 as pdfium

    image = pdfium.PdfDocument(str(pdf_path))[page - 1].render(scale=1.8).to_pil().convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode()


def _same_fund(printed_name: str, product: str) -> bool:
    squash = lambda x: re.sub(r"[^a-z0-9]", "", x.lower())
    words = [w for w in re.split(r"[\s/-]+", product) if len(w) >= 4]
    return any(squash(w) in squash(printed_name) for w in words) if words else True


async def extract_riskometers():
    """Read riskometer levels from every mutual fund document and save them with page numbers."""
    manifest = load_manifest()
    results = json.loads(RISKOMETER_FILE.read_text()) if RISKOMETER_FILE.exists() else {}
    for name, row in manifest.items():
        if row.get("category") != "mutual_fund" or name in results:
            continue
        path = ROOT / row["file_path"]
        found = []
        for page in _candidate_risk_pages(path):
            image = _page_image_b64(path, page)
            # Read twice and keep the result only if both reads agree (guards against misreads).
            reads = [
                json.loads(await vision_model_func(
                    RISKOMETER_PROMPT, image_data=image, response_format={"type": "json_object"}, temperature=0
                ))
                for _ in range(2)
            ]
            levels = {(r.get("scheme_risk") or "").lower() for r in reads if r.get("has_riskometer")}
            if len(levels) != 1 or not all(r.get("has_riskometer") for r in reads) or not next(iter(levels)):
                continue
            data = reads[0]
            # Presentations show riskometers of sibling schemes; keep only this document's fund.
            if not _same_fund(data.get("fund_name") or "", row.get("product", "")):
                continue
            found.append({"page": page, **{k: v for k, v in data.items() if k != "has_riskometer"}})
        results[name] = found
        RISKOMETER_FILE.write_text(json.dumps(results, indent=1))
        summary = "; ".join(f"p.{f['page']}: {f['scheme_risk']}" for f in found) or "none found"
        print(f"{name:38s} {summary}", flush=True)
    cost_report("Riskometer extraction cost")


RISK_LEVELS = ["low to moderate", "moderately high", "very high", "moderate", "high", "low"]  # longest first


def _level_from_caption(caption: str | None) -> str | None:
    text = (caption or "").lower()
    match = re.search(r"(?:risk of the (?:scheme|benchmark) is|will be at)\s+(" + "|".join(RISK_LEVELS) + ")", text)
    return match.group(1).title().replace(" To ", " to ") if match else None


def load_riskometers() -> dict[str, list[dict]]:
    data = json.loads(RISKOMETER_FILE.read_text()) if RISKOMETER_FILE.exists() else {}
    for entries in data.values():
        for e in entries:
            # The printed caption is authoritative; needle readings are only a fallback.
            e["scheme_risk"] = _level_from_caption(e.get("caption")) or e["scheme_risk"]
    return data


def riskometer_sources(focus_docs: list[str]) -> list[tuple[str, int, str]]:
    """(document, page, text) sources describing the riskometers of the named funds."""
    sources = []
    for doc, entries in load_riskometers().items():
        if doc not in focus_docs:
            continue
        for e in entries:
            parts = [f"Riskometer for {e.get('fund_name') or doc}: scheme risk level is {e['scheme_risk']}."]
            if e.get("benchmark_risk"):
                parts.append(f"Benchmark risk level is {e['benchmark_risk']}.")
            if e.get("potential_risk_class"):
                parts.append(
                    f"Potential Risk Class: {e['potential_risk_class']}"
                    + (f" ({e['potential_risk_class_meaning']})." if e.get("potential_risk_class_meaning") else ".")
                )
            if e.get("caption"):
                parts.append(f'Printed caption: "{e["caption"]}".')
            sources.append((doc, e["page"], f"{PAGE_MARKER.format(e['page'])} " + " ".join(parts)))
    return sources


RISK_QUESTION = re.compile(r"\brisk|riskometer|risk-o-meter|risky\b", re.I)


# --- Provider-aware retrieval ------------------------------------------------
# Policies share near-identical regulator wording (e.g. the IRDAI pre-existing disease
# definition) and chunk text rarely names the insurer, so hybrid retrieval for
# "Tata AIG's definition" returns other insurers' clauses. When a question names a
# provider/product from the manifest, we add that document's best vector-search chunks.

FOCUS_CHUNKS_PER_DOC = 6
FOCUS_SEARCH_TOP_K = 150  # wide vector search, then keep only the named documents' chunks
MAX_FOCUS_DOCS = 10  # ranked by match strength (not alphabetical); only a safety limit
SHORT_DOC_PAGES = 3

# Words too common to identify a provider on their own ("Care Health", "New India Assurance").
GENERIC_WORDS = {
    "new", "india", "care", "health", "general", "bank", "insurance", "assurance", "of", "the", "and",
    "mahindra", "central", "prudential",
}
CATEGORY_HINTS = {
    "fixed_deposit": ["fd", "fds", "fixed deposit", "fixed deposits", "term deposit", "deposit rate", "deposit rates"],
    "insurance": ["insurance", "insurer", "policy", "policies", "mediclaim", "waiting period", "sum insured",
                  "claim", "claims", "pre existing", "cover", "coverage", "hospitalisation", "hospitalization"],
    "mutual_fund": ["fund", "funds", "factsheet", "factsheets", "fact sheet", "nav", "expense ratio", "portfolio", "holding", "holdings", "aum", "cagr",
                    "sip", "scheme", "benchmark", "fund manager"],
}
PROVIDER_ALIASES = {"Bank of Baroda": ["bob"]}
PRODUCT_CATEGORY_WORDS = {
    "large", "mid", "midcap", "small", "cap", "short", "long", "term", "index", "nifty", "next", "50", "fund",
    "plan", "debt", "hybrid", "equity", "corporate", "bond", "balanced", "advantage", "group", "health", "care",
    "retail", "fd", "deposit", "combined", "complete", "basic", "premium", "plus", "my", "prospectus", "mediclaim",
}


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower().replace("&", " and ")).split())


def _has_phrase(phrase: str, text: str) -> bool:
    return bool(phrase) and f" {phrase} " in f" {text} "


PRODUCT_FILLER_WORDS = {"plan", "fund", "policy", "scheme"}


def find_named_documents(question: str, manifest: dict[str, dict]) -> list[str]:
    """Manifest documents the question names, ranked by match strength.

    Strength: provider + product named (3) > distinctive product alone (2.5) > full provider name (2) >
    partial provider name (1). When a specific product matched, weaker provider-only matches that share its
    provider word are dropped ("HDFC Large Cap" should not pull in HDFC Bank or HDFC ERGO)."""
    q = _norm(question)
    products = {name: _norm(row.get("product", "")) for name, row in manifest.items()}
    product_owners: dict[str, int] = {}
    for p in products.values():
        product_owners[p] = product_owners.get(p, 0) + 1

    strong, weak, product_hit = set(), {}, set()
    for name, row in manifest.items():
        provider = _norm(row.get("provider", ""))
        if _has_phrase(provider, q) or any(_has_phrase(a, q) for a in PROVIDER_ALIASES.get(row["provider"], [])):
            strong.add(name)
        else:
            key = next((w for w in provider.split() if w not in GENERIC_WORDS), None)
            if key and _has_phrase(key, q):
                weak[name] = key
        product = products[name]
        parts = [product] + [_norm(p) for p in row.get("product", "").split("/") if len(p) >= 5]
        # Partial product names: "Star Health Premium" should match the product "Premium plan".
        core = " ".join(w for w in product.split() if w not in PRODUCT_FILLER_WORDS)
        if core and core != product:
            parts.append(core)
        if any(_has_phrase(p, q) for p in parts if p):
            product_hit.add(name)

    # "Tata AIG" names the insurer, so the weak "tata" match on Tata's mutual fund is dropped.
    # Only a more specific name overrides: plain "HDFC" (the fund house) must not drop HDFC Bank.
    specific_keys = {
        w
        for name in strong
        for w in _norm(manifest[name]["provider"]).split()
        if len(_norm(manifest[name]["provider"]).split()) > 1
    }
    weak = {name: k for name, k in weak.items() if k not in specific_keys}

    score: dict[str, float] = {}
    provider_docs = strong | set(weak)
    by_provider: dict[str, set[str]] = {}
    for name in provider_docs:
        by_provider.setdefault(manifest[name]["provider"], set()).add(name)
    for provider, names in by_provider.items():
        # A named product picks its document; a provider without a named product keeps all its documents.
        for n in (names & product_hit) or names:
            score[n] = 3.0 if n in product_hit else (2.0 if n in strong else 1.0)
    # Distinctive product names count even without the provider ("Mid-Cap Opportunities"), but names made
    # only of category words ("Large Cap", "Short Term", "Group Health") need the provider too.
    for n in product_hit:
        if (product_owners[products[n]] == 1 and n not in provider_docs
                and any(w not in PRODUCT_CATEGORY_WORDS for w in products[n].split())):
            score[n] = 2.5

    # Drop weaker provider-only matches when the same provider word matched a specific product.
    product_words = {w for n in score if n in product_hit for w in _norm(manifest[n]["provider"]).split()}
    for n in list(score):
        if n in weak and score[n] <= 1.0 and weak[n] in product_words:
            del score[n]

    # Category words narrow ambiguous providers: "HDFC FD" -> the FD, not HDFC ERGO or HDFC funds.
    hinted = {cat for cat, words in CATEGORY_HINTS.items() if any(_has_phrase(_norm(w), q) for w in words)}
    if hinted:
        # A named provider with no document in the asked-about area ("Axis" + "FD": Axis only has a fund
        # factsheet) is not a match; falling back to its other documents invites made-up answers.
        score = {n: v for n, v in score.items() if manifest[n].get("category") in hinted or n in product_hit}
    ranked = sorted(score, key=lambda n: (-score[n], n))
    return ranked[:MAX_FOCUS_DOCS]


_page_counts: dict[str, int] = {}


def page_count(doc: str) -> int:
    if doc not in _page_counts:
        import pypdfium2 as pdfium

        row = load_manifest().get(doc)
        _page_counts[doc] = len(pdfium.PdfDocument(str(ROOT / row["file_path"]))) if row else 99
    return _page_counts[doc]


async def _all_chunks_of(rag: RAGAnything, doc: str) -> list[dict]:
    for status in (DocStatus.PROCESSED,):
        for _, d in (await rag.lightrag.doc_status.get_docs_by_status(status)).items():
            if Path(d.file_path).name == doc:
                stored = await rag.lightrag.text_chunks.get_by_ids(list(d.chunks_list or []))
                return [{"chunk_id": cid, "content": c["content"], "file_path": c["file_path"]}
                        for cid, c in zip(d.chunks_list, stored) if c]
    return []


async def retrieve_chunks(rag: RAGAnything, question: str, focus_docs: list[str], mode: str) -> list[dict]:
    retrieved = await rag.lightrag.aquery_data(question, param=QueryParam(mode=mode))
    chunks = (retrieved.get("data") or {}).get("chunks") or []
    if not focus_docs:
        return chunks
    wide = await rag.lightrag.aquery_data(
        question, param=QueryParam(mode="naive", chunk_top_k=FOCUS_SEARCH_TOP_K, max_total_tokens=10**6)
    )
    focus = []
    for doc in focus_docs:
        doc_chunks = [c for c in (wide.get("data") or {}).get("chunks") or [] if Path(c["file_path"]).name == doc]
        if page_count(doc) <= SHORT_DOC_PAGES:
            # Short factsheets: include every chunk (best-ranked first) so a figure in a low-ranked table
            # (e.g. a volatility-measures table) is never cut off.
            ranked_ids = {c["chunk_id"] for c in doc_chunks}
            doc_chunks += [c for c in await _all_chunks_of(rag, doc) if c["chunk_id"] not in ranked_ids]
            focus.extend(doc_chunks)
        else:
            focus.extend(doc_chunks[:FOCUS_CHUNKS_PER_DOC])
    seen = {c["chunk_id"] for c in focus}
    # The question names its providers: answer only from their documents, so figures from
    # other documents can't leak into the answer.
    named = set(focus_docs)
    return focus + [c for c in chunks if c["chunk_id"] not in seen and Path(c["file_path"]).name in named]


# --- Number guard ---------------------------------------------------------------
# Final check against made-up figures: every number in the answer must appear somewhere in the
# retrieved source text (or the question). Small integers (<= 31: days, months, counts) match almost
# anything, so they are not checked. Differences/sums of two verified numbers count as computed.

NUMBER_TOKEN = re.compile(r"(?<![\w.])(?:rs\.?|₹)?\s*(\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*(%|l\b|lakhs?|crores?|cr\b)?", re.I)
CITATION_MARK = re.compile(r"\[\[?S\d+[^\]]*\]\]?|\bp\.\s*\d+|\[\[page \d+\]\]", re.I)
NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "fifteen": 15, "eighteen": 18, "twenty": 20, "twenty four": 24, "thirty": 30,
    "thirty six": 36, "forty five": 45, "forty eight": 48, "sixty": 60, "ninety": 90, "hundred": 100,
}
UNVERIFIED_SHARE_TO_WITHHOLD = 0.5


def _numbers(text: str) -> list[tuple[str, float]]:
    text = CITATION_MARK.sub(" ", text)
    out = []
    for m in NUMBER_TOKEN.finditer(text):
        value = float(m.group(1).replace(",", ""))
        out.append((m.group(0).strip(), value))
    return out


SOURCE_NUMBER = re.compile(r"\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?")


def _source_values(text: str) -> set[float]:
    lowered = text.lower()
    for word, value in sorted(NUMBER_WORDS.items(), key=lambda kv: -len(kv[0])):
        lowered = re.sub(rf"\b{word}\b", str(value), lowered.replace("-", " ") if " " in word else lowered)
    # Sources are matched loosely: PDF text often glues digits to words ("NiftyMidcap150Index").
    return {round(float(m.replace(",", "")), 4) for m in SOURCE_NUMBER.findall(lowered)}


def _source_index(claim: dict) -> int:
    """The claim's 1-based source number; 0 when the model left it empty or malformed."""
    try:
        return int(claim.get("source") or 0)
    except (TypeError, ValueError):
        return 0


CITE_MARK = re.compile(r"\[\[?S(\d+)[^\]]*\]\]?")


def _cited_segments(answer: str) -> list[tuple[str, set[int]]]:
    """Split the answer into (text, cited source numbers) pieces: text is attributed to the citation
    marker(s) that directly follow it. Trailing text without a marker gets an empty set."""
    pieces, pos, pending = [], 0, ""
    marks = list(CITE_MARK.finditer(answer))
    i = 0
    while i < len(marks):
        text = answer[pos:marks[i].start()]
        group = {int(marks[i].group(1))}
        j = i
        # Consecutive markers ("[S1 p.2][S7 p.1]") cite the same text.
        while j + 1 < len(marks) and not answer[marks[j].end():marks[j + 1].start()].strip():
            j += 1
            group.add(int(marks[j].group(1)))
        pieces.append((text, group))
        pos, i = marks[j].end(), j + 1
    if answer[pos:].strip():
        pieces.append((answer[pos:], set()))
    return pieces


def check_numbers(answer: str, sources: list[dict], blocks: list[str], question: str, claims: list[dict]):
    """Check every figure against the document(s) it cites.

    Returns (unverified, other_provider, computed, checked_count):
      unverified     - figures found in no retrieved source at all
      other_provider - figures found only in a different provider's document than the one cited
      computed       - differences/sums of verified figures (e.g. a senior-citizen premium)
    """
    doc_text: dict[str, list[str]] = {}
    for src, block in zip(sources, blocks):
        doc_text.setdefault(src["document"], []).append(block)
    doc_values = {d: _source_values("\n".join(t)) for d, t in doc_text.items()}
    all_values = set().union(*doc_values.values()) if doc_values else set()
    question_values = _source_values(question)
    doc_of = lambda n: sources[n - 1]["document"] if 1 <= n <= len(sources) else None
    cited_anywhere = {doc_of(n) for _, g in _cited_segments(answer) for n in g} - {None}
    if not cited_anywhere:  # no inline markers: fall back to the documents of the claims
        cited_anywhere = {doc_of(_source_index(c)) for c in claims} - {None}

    results = []  # (raw, value, status)
    for text, group in _cited_segments(answer):
        docs = ({doc_of(n) for n in group} - {None}) or cited_anywhere
        allowed = set().union(*(doc_values.get(d, set()) for d in docs)) if docs else all_values
        for raw, v in _numbers(text):
            if v.is_integer() and v <= 31 and "%" not in raw:
                continue
            key = round(v, 4)
            if key in allowed or key in question_values:
                results.append((raw, v, "verified"))
            elif key in all_values:
                results.append((raw, v, "other_provider"))
            else:
                results.append((raw, v, "unverified"))
    verified = [v for _, v, st in results if st == "verified"]
    unverified, other, computed = [], [], []
    for raw, v, st in results:
        if st == "verified":
            continue
        if any(abs(abs(a - b) - v) < 1e-6 or abs(a + b - v) < 1e-6 for a in verified for b in verified):
            computed.append(raw)
        elif st == "other_provider":
            other.append(raw)
        else:
            unverified.append(raw)
    return unverified, other, computed, len(results)


# --- Question routing: recommendations and unspecified providers ---------------

ROUTER_PROMPT = """Classify a user question for a document Q&A system over Indian bank FD rate sheets,
health insurance policy documents and mutual fund factsheets from several providers.

Return JSON: {"kind": "...", "category": "fixed_deposit" | "insurance" | "mutual_fund" | null}

kind is one of:
- "recommendation": asks which option is best/better, what to buy or invest in, or what the user should choose.
- "needs_entity": asks for a value that differs by provider or product (a rate, fee, manager, ratio, limit,
  waiting period, benchmark, penalty, minimum amount...) about ONE unspecified thing, written as if a single
  bank/fund/policy were meant ("the rate", "the fund", "this policy"), and names no provider or product.
- "answerable": anything else, including questions that name a provider or product, questions that ask for
  a list or comparison across all providers ("which banks...", "list all policies that...", "highest/lowest
  across..."), general definitions, and off-topic or unavailable information.
category is the product area the question is about, or null if unclear."""


def category_hints(question: str) -> set[str]:
    q = _norm(question)
    return {cat for cat, words in CATEGORY_HINTS.items() if any(_has_phrase(_norm(w), q) for w in words)}


def _options_text(manifest: dict[str, dict], category: str | None, names: list[str] | None = None) -> str:
    rows = [manifest[n] for n in names] if names else [r for r in manifest.values() if r["category"] == category]
    labels = sorted({f"{r['provider']} - {r['product']}" if r["category"] != "fixed_deposit" else r["provider"] for r in rows})
    return "; ".join(labels)


CATEGORY_NAMES = {"fixed_deposit": "bank (fixed deposit)", "insurance": "insurance policy", "mutual_fund": "mutual fund"}
NOT_ADVICE = "This is factual information from the documents, not financial advice."


async def route_question(question: str, focus_docs: list[str], manifest: dict[str, dict]) -> tuple[str, str | None]:
    """Returns (kind, clarification_text). kind: 'answer', 'recommendation' or 'clarify'."""
    hinted = category_hints(question)
    focus_categories = {manifest[d]["category"] for d in focus_docs}
    # A named provider that spans several product areas ("HDFC") with no product or area named.
    if len(focus_categories) > 1 and not hinted and len({manifest[d]["provider"] for d in focus_docs}) <= 2:
        products = "; ".join(f"{manifest[d]['provider']} {CATEGORY_NAMES[manifest[d]['category']]}: {manifest[d]['product']}"
                             for d in focus_docs)
        return "clarify", f"Which product do you mean? The documents cover: {products}. Please name one."
    raw = await llm_model_func(question, system_prompt=ROUTER_PROMPT, response_format={"type": "json_object"}, temperature=0)
    route = json.loads(raw)
    kind, category = route.get("kind"), route.get("category") or next(iter(hinted), None)
    if kind == "recommendation":
        if focus_docs:
            return "recommendation", None
        scope = _options_text(manifest, category) if category else "fixed deposits, health insurance policies and mutual funds"
        return "clarify", (
            "I can't recommend or rank products. I can compare facts (rates, costs, risk levels, waiting periods) "
            f"for specific options you name. The documents cover: {scope}. {NOT_ADVICE}"
        )
    if kind == "needs_entity" and not focus_docs:
        what = CATEGORY_NAMES.get(category, "bank, fund or policy")
        options = _options_text(manifest, category) if category else ""
        return "clarify", f"Which {what} do you mean?" + (f" The documents cover: {options}." if options else "")
    return "answer", None


async def query(rag: RAGAnything, question: str, mode: str = "hybrid") -> QAResult:
    await rag._ensure_lightrag_initialized()
    manifest = load_manifest()

    focus_docs = find_named_documents(question, manifest)
    kind, clarification = await route_question(question, focus_docs, manifest)
    if kind == "clarify":
        return QAResult(question, clarification, focus_documents=focus_docs)

    chunks = await retrieve_chunks(rag, question, focus_docs, mode)
    risk_sources = riskometer_sources(focus_docs) if RISK_QUESTION.search(question) else []
    if not chunks and not risk_sources:
        return QAResult(question, "No relevant content found in the indexed documents.")

    sources, blocks = [], []
    for doc, page, text in risk_sources:
        meta = manifest.get(doc, {})
        sources.append({"document": doc, "pages": {page}, "meta": meta})
        blocks.append(f"[S{len(sources)}] {meta.get('provider', '')} - {meta.get('product', '')} ({doc}) "
                      f"RISKOMETER (read from the page image)\n{text}")
    for n, chunk in enumerate(chunks, start=len(sources) + 1):
        content, pages = await _resolve_chunk_pages(rag, chunk)
        doc = Path(chunk["file_path"]).name
        meta = manifest.get(doc, {})
        content = tables_to_text(content)
        sources.append({"document": doc, "pages": pages, "meta": meta})
        header = f"[S{n}] {meta.get('provider', '')} - {meta.get('product', '')} ({doc}"
        header += f", dated {meta['doc_date']})" if meta.get("doc_date") else ")"
        blocks.append(f"{header}\n{content}")

    system_prompt = ANSWER_SYSTEM_PROMPT
    if kind == "recommendation":
        system_prompt += (
            "\nThe user asks for a recommendation. Do NOT pick, rank or recommend any option. Present the key "
            f"facts for each named option side by side, neutrally, and end the answer with: \"{NOT_ADVICE}\""
        )
    raw = await llm_model_func(
        f"Question: {question}\n\nSources:\n\n" + "\n\n---\n\n".join(blocks),
        system_prompt=system_prompt,
        response_format={"type": "json_object"},
        temperature=0,
    )
    parsed = json.loads(raw)
    answer = parsed.get("answer", "")

    citations = []
    for c in parsed.get("claims", []):
        idx = _source_index(c) - 1
        if not 0 <= idx < len(sources):
            continue
        src = sources[idx]
        page = c.get("page")
        page = int(page) if str(page).isdigit() else None
        if page not in src["pages"] and len(src["pages"]) == 1:
            # A single-page source can only be on that page; fix the model's slip.
            (correct,) = src["pages"]
            answer = answer.replace(f"[S{idx + 1} p.{page}]", f"[S{idx + 1} p.{correct}]")
            page = correct
        meta = src["meta"]
        citations.append(
            Citation(
                claim=c.get("claim", ""),
                document=src["document"],
                page=page,
                provider=meta.get("provider", ""),
                product=meta.get("product", ""),
                category=meta.get("category", ""),
                doc_date=meta.get("doc_date", ""),
                verified=page in src["pages"],
            )
        )
    unverified, other, computed, checked = check_numbers(
        answer, sources, blocks, question, parsed.get("claims", [])
    )
    problems = unverified + other
    verification = "ok"
    if problems and len(problems) / checked >= UNVERIFIED_SHARE_TO_WITHHOLD:
        verification = "withheld"
        detail = []
        if unverified:
            detail.append(f"{', '.join(unverified)} do not appear in the retrieved sources")
        if other:
            detail.append(f"{', '.join(other)} appear only in a different provider's document than the one cited")
        answer = (
            f"I could not verify this answer against the documents: {'; '.join(detail)}, "
            "so I am not showing it. The information may not be in the documents."
        )
        citations = []
    elif problems:
        verification = "flagged"
        answer += f"\n\nWarning: could not verify these figures in the cited documents: {', '.join(problems)}."
    return QAResult(
        question, answer, citations, focus_documents=focus_docs,
        unverified_numbers=problems, computed_numbers=computed, verification=verification,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def print_result(result: QAResult):
    print(f"\nQ: {result.question}\n\nA: {result.answer}\n")
    if result.focus_documents:
        print(f"(Named in question, searched directly: {', '.join(result.focus_documents)})")
    if result.citations:
        print("Sources:")
        for c in result.citations:
            flag = "" if c.verified else "  (page not confirmed in retrieved chunk)"
            date = f", {c.doc_date}" if c.doc_date else ""
            print(f"  - {c.document} p.{c.page} [{c.provider} / {c.category}{date}]{flag}")
            print(f"      {c.claim}")


async def ask(questions: list[str]):
    rag = build_rag()
    for q in questions:
        print_result(await query(rag, q))
    cost_report("Query token usage")
    await rag.finalize_storages()


async def interactive():
    rag = build_rag()
    print("FinDocQA - ask a question (empty line to quit)")
    while (q := input("\n> ").strip()):
        print_result(await query(rag, q))
    await rag.finalize_storages()


def main():
    parser = argparse.ArgumentParser(description="FinDocQA")
    sub = parser.add_subparsers(dest="cmd")
    p_ingest = sub.add_parser("ingest", help="parse and index the PDFs")
    p_ingest.add_argument("--test", action="store_true", help="ingest only 2 PDFs as a test")
    p_ingest.add_argument("--max-cost", type=float, default=7.0, help="stop when ingestion cost reaches this many USD")
    p_ingest.add_argument("--backend", default=DEFAULT_BACKEND, help="MinerU backend for new files: auto (per category, default), pipeline or vision")
    p_reingest = sub.add_parser("reingest", help="remove a file from the index and ingest it again")
    p_reingest.add_argument("file")
    p_reingest.add_argument("--backend", default=DEFAULT_BACKEND, help="MinerU backend: auto (per category, default), pipeline or vision")
    sub.add_parser("extract-riskometers", help="read mutual fund riskometer levels from page images (one-time)")
    p_ask = sub.add_parser("ask", help="ask one or more questions")
    p_ask.add_argument("question", nargs="+")
    args = parser.parse_args()

    _api_key()
    if args.cmd == "ingest":
        asyncio.run(ingest(test=args.test, max_cost=args.max_cost, backend=args.backend))
    elif args.cmd == "reingest":
        asyncio.run(reingest(args.file, backend=args.backend))
    elif args.cmd == "extract-riskometers":
        asyncio.run(extract_riskometers())
    elif args.cmd == "ask":
        asyncio.run(ask([" ".join(args.question)]))
    else:
        asyncio.run(interactive())


if __name__ == "__main__":
    main()
