"""Benchmark 5 câu hỏi trên corpus thư viện HUIT với từng chiến lược chunking.

    python bench.py                      # cả 3 chiến lược + A/B filter, ghi ket_qua_benchmark.txt
    python bench.py --strategy heading   # một chiến lược, chỉ in ra màn hình
    python bench.py --no-llm             # bỏ câu trả lời của agent (chạy nhanh)

Embedding theo EMBEDDING_PROVIDER trong .env (local / openai / gemini; mặc định mock).
LLM của agent: Ollama, model OLLAMA_MODEL (mặc định qwen2.5:3b) tại OLLAMA_HOST.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request
from datetime import date
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

from src import (
    EMBEDDING_PROVIDER_ENV,
    ChunkingStrategyComparator,
    Document,
    EmbeddingStore,
    FixedSizeChunker,
    GeminiEmbedder,
    HeadingChunker,
    KnowledgeBaseAgent,
    LocalEmbedder,
    OpenAIEmbedder,
    RecursiveChunker,
    _mock_embed,
)

DATA_DIR = Path("data/thu-vien-huit")
OUTPUT_FILE = Path("ket_qua_benchmark.txt")
CHUNK_SIZE = 500
TOP_K = 3
BASELINE_DOCS = ["quy-dinh-chung", "luu-hanh-tai-lieu", "huong-dan-su-dung-thu-vien"]

# Dòng chọn chunker: thứ duy nhất khác nhau giữa các lần chạy, mọi thứ khác giữ nguyên để so sánh công bằng.
STRATEGIES = {
    "fixed": lambda: FixedSizeChunker(chunk_size=CHUNK_SIZE, overlap=50),
    "recursive": lambda: RecursiveChunker(chunk_size=CHUNK_SIZE),
    "heading": lambda: HeadingChunker(chunk_size=CHUNK_SIZE, glue_lead_in=False),
    # Bản tinh chỉnh sau phân tích lỗi Q4: giữ câu dẫn kết thúc bằng ":" đi cùng danh sách/bảng phía sau.
    "heading_v2": lambda: HeadingChunker(chunk_size=CHUNK_SIZE, glue_lead_in=True),
}

# evidence: chuỗi (regex) phải có trong chunk thì chunk đó mới thực sự trả lời được câu hỏi.
# answer_check / answer_reject: kiểm tự động câu trả lời của agent — vẫn cần đọc lại bằng mắt.
QUERIES = [
    {
        "query": "Tôi được mượn tối đa bao nhiêu tài liệu về nhà, trong bao nhiêu ngày?",
        "gold_answer": "3 tài liệu, 10 ngày (sinh viên, học viên)",
        "gold_docs": ["quy-dinh-muon-tra-sinh-vien", "luu-hanh-tai-lieu"],
        "evidence": r"Sinh viên, học viên \| 3 \| 10",
        "answer_check": r"(?s)(?=.*\b3\b)(?=.*10 ngày)",
        "answer_reject": r"180|giảng viên|viên chức|cán bộ",
        "filter": {"audience": "student"},
    },
    {
        "query": "Làm thẻ thư viện mới mất bao nhiêu tiền?",
        "gold_answer": "100.000 đ/thẻ (cấp lại 50.000đ/thẻ, gia hạn 50.000 đồng/năm)",
        "gold_docs": ["huong-dan-su-dung-thu-vien"],
        "evidence": r"100\.000 ?đ/thẻ",
        "answer_check": r"100[.,]?000",
    },
    {
        "query": "Trả sách mượn liên thư viện trễ hạn bị phạt bao nhiêu?",
        "gold_answer": "5.000đ/tài liệu/ngày",
        "gold_docs": ["muon-lien-thu-vien"],
        "evidence": r"Phí trễ hạn: 5\.000đ",
        "answer_check": r"5[.,]?000",
    },
    {
        "query": "Làm mất sách tiếng Việt cũ, không còn bán trên thị trường thì phải đền thế nào?",
        "gold_answer": "Đền tiền gấp 5 lần giá bìa (ngoại văn gấp 3 lần) + phí xử lý kỹ thuật; hạn 30 ngày",
        "gold_docs": ["quy-dinh-chung"],
        "evidence": r"gấp 5 lần giá tiền ghi trên bìa",
        "answer_check": r"gấp 5|5 lần|năm lần",
    },
    {
        "query": "Đặt phòng học nhóm thế nào và được dùng bao lâu?",
        "gold_answer": "Đăng ký tại Quầy thông tin hoặc trực tuyến mục “ĐẶT PHÒNG”; 2 giờ/lượt, gia hạn khi không có người chờ; trễ >15 phút bị hủy",
        "gold_docs": ["su-dung-phong-hoc-nhom", "quy-dinh-chung"],
        "evidence": r"02 giờ/lượt|120 phút/lượt",
        "answer_check": r"0?2 giờ|120 phút|hai giờ",
    },
]

# A/B cho câu cần filter: không lọc / lọc cứng student / lọc student + tài liệu dùng chung.
AB_QUERY = 0
AB_ARMS = [
    ("không filter", None),
    ("audience=student", {"audience": "student"}),
    ("audience∈{student,all}", {"audience": ["student", "all"]}),
]


class Report:
    """Print every line and keep it, so the full run can be written to a file."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, line: str = "") -> None:
        print(line)
        self.lines.append(line)


def load_corpus(data_dir: Path) -> list[tuple[str, dict, str]]:
    """Return (doc_id, frontmatter metadata, body) for every .md file."""
    corpus = []
    for path in sorted(data_dir.glob("*.md")):
        match = re.match(r"^---\n(.*?)\n---\n(.*)$", path.read_text(encoding="utf-8"), re.S)
        if not match:
            raise ValueError(f"{path} has no frontmatter")
        metadata = {key: value.strip().strip('"') for key, value in re.findall(r"^(\w+):\s*(.*)$", match.group(1), re.M)}
        corpus.append((path.stem, metadata, match.group(2).strip()))
    return corpus


def make_embedder():
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    backends = {"local": LocalEmbedder, "openai": OpenAIEmbedder, "gemini": GeminiEmbedder}
    if provider in backends:
        try:
            return backends[provider]()
        except Exception as error:  # missing package or key: say so loudly, mock scores are noise
            print(f"!! Không khởi tạo được embedder '{provider}' ({error}) — quay về MOCK, số liệu không có ý nghĩa ngữ nghĩa")
    return _mock_embed


def make_ollama_llm(model: str, host: str):
    def call(prompt: str) -> str:
        payload = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0}}
        request = urllib.request.Request(
            f"{host}/api/generate", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.loads(response.read())["response"].strip()

    return call


def ollama_ready(model: str, host: str) -> bool:
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=3) as response:
            return any(m["name"] == model for m in json.loads(response.read())["models"])
    except Exception:
        return False


def build_store(name: str, corpus: list[tuple[str, dict, str]], embed) -> tuple[EmbeddingStore, list[Document]]:
    """Chunk outside the store; every chunk is its own Document carrying the file's metadata."""
    chunker = STRATEGIES[name]()
    docs = [
        Document(id=f"{doc_id}#{i}", content=chunk, metadata={**metadata, "doc_id": doc_id, "chunk_index": i})
        for doc_id, metadata, body in corpus
        for i, chunk in enumerate(chunker.chunk(body))
    ]
    store = EmbeddingStore(collection_name=f"bench_{name}", embedding_fn=embed)
    store.add_documents(docs)
    return store, docs


def first_rank(results: list[dict], predicate) -> int | None:
    return next((rank for rank, result in enumerate(results, start=1) if predicate(result)), None)


def check_answer(query: dict, answer: str | None) -> tuple[bool | None, str]:
    if answer is None:
        return None, "không chạy LLM"
    if query.get("answer_reject") and re.search(query["answer_reject"], answer):
        return False, "lẫn thông tin đối tượng khác"
    if re.search(query["answer_check"], answer):
        return True, "đúng"
    return False, "thiếu/sai đáp án"


def score(evidence_rank: int | None, answer_ok: bool | None) -> int:
    """SCORING.md: 2 = chunk trả lời được ở top-1 và agent đúng; 1 = có trong top-3 nhưng chưa đủ; 0 = không có."""
    if evidence_rank is None:
        return 0
    return 2 if evidence_rank == 1 and answer_ok is not False else 1


def preview(text: str, width: int = 110) -> str:
    flat = " ↵ ".join(line.strip() for line in text.splitlines() if line.strip())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def run_query(store, agent, query: dict, metadata_filter, report: Report) -> dict:
    results = store.search_with_filter(query["query"], top_k=TOP_K, metadata_filter=metadata_filter)
    for rank, result in enumerate(results, start=1):
        meta = result["metadata"]
        mark = "✓" if re.search(query["evidence"], result["content"]) else " "
        report(f"    #{rank} {result['score']:.3f} {mark} {result['id']:<34} [{meta.get('audience')}] {preview(result['content'])}")
    doc_rank = first_rank(results, lambda r: r["metadata"]["doc_id"] in query["gold_docs"])
    evidence_rank = first_rank(results, lambda r: re.search(query["evidence"], r["content"]))
    answer = agent.answer_with_filter(query["query"], top_k=TOP_K, metadata_filter=metadata_filter) if agent else None
    answer_ok, verdict = check_answer(query, answer)
    return {
        "doc_rank": doc_rank,
        "evidence_rank": evidence_rank,
        "naive": score(doc_rank, None),
        "strict": score(evidence_rank, answer_ok),
        "answer": answer,
        "verdict": verdict,
        "top_ids": [r["id"] for r in results],
    }


def report_baseline(corpus, report: Report) -> None:
    report(f"## Baseline — ChunkingStrategyComparator().compare(body, chunk_size={CHUNK_SIZE}), đã bỏ frontmatter")
    bodies = {doc_id: body for doc_id, _, body in corpus}
    for doc_id in BASELINE_DOCS:
        report(f"  {doc_id} ({len(bodies[doc_id])} ký tự)")
        for name, stats in ChunkingStrategyComparator().compare(bodies[doc_id], chunk_size=CHUNK_SIZE).items():
            report(
                f"    {name:<13} count={stats['count']:3}  avg={stats['avg_length']:7.1f}  "
                f"min={stats['min_length']:4}  max={stats['max_length']:4}"
            )
    report()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--strategy", choices=[*STRATEGIES, "all"], default="all")
    parser.add_argument("--no-llm", action="store_true", help="bỏ qua câu trả lời của agent")
    args = parser.parse_args()

    load_dotenv(override=False)
    report = Report()
    corpus = load_corpus(DATA_DIR)
    embedder = make_embedder()
    embed = lru_cache(maxsize=None)(embedder)  # queries repeat across strategies and A/B arms

    model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    use_llm = not args.no_llm and ollama_ready(model, host)
    if not args.no_llm and not use_llm:
        print(f"!! Ollama chưa sẵn sàng hoặc thiếu model {model} — chạy `ollama serve` và `ollama pull {model}`; tạm bỏ qua LLM")
    llm = make_ollama_llm(model, host) if use_llm else None

    strategies = list(STRATEGIES) if args.strategy == "all" else [args.strategy]
    report(f"# Kết quả benchmark — {date.today().isoformat()}")
    report(f"Corpus: {DATA_DIR} ({len(corpus)} tài liệu) | chunk_size={CHUNK_SIZE} | top_k={TOP_K}")
    report(f"Embedding: {getattr(embedder, '_backend_name', type(embedder).__name__)} | LLM: {model if llm else 'không dùng'}")
    report("Chấm 2 mức: naive = doc_id gold có trong top-3; strict = chunk chứa chuỗi đáp án (✓) + agent trả lời đúng")
    report()
    if args.strategy == "all":
        report_baseline(corpus, report)

    totals: dict[str, dict] = {}
    stores = {}
    for name in strategies:
        store, docs = build_store(name, corpus, embed)
        stores[name] = store
        agent = KnowledgeBaseAgent(store=store, llm_fn=llm) if llm else None
        lengths = [len(doc.content) for doc in docs]
        report(f"## Chiến lược: {name} — {STRATEGIES[name]().__class__.__name__}")
        report(f"Đã nạp {len(docs)} chunk | avg={sum(lengths) / len(lengths):.0f} min={min(lengths)} max={max(lengths)} ký tự")
        rows = []
        for number, query in enumerate(QUERIES, start=1):
            metadata_filter = query.get("filter")
            report(f"  Q{number}. {query['query']}" + (f"   [filter {metadata_filter}]" if metadata_filter else ""))
            report(f"      gold: {query['gold_answer']}")
            row = run_query(store, agent, query, metadata_filter, report)
            report(
                f"      → doc rank={row['doc_rank']} (naive {row['naive']}đ) | evidence rank={row['evidence_rank']} "
                f"| agent: {row['verdict']} → {row['strict']}đ"
            )
            if row["answer"] is not None:
                report(f"      agent: {' '.join(row['answer'].split())}")
            rows.append(row)
        totals[name] = {"naive": sum(r["naive"] for r in rows), "strict": sum(r["strict"] for r in rows), "rows": rows}
        report(f"  Tổng {name}: naive {totals[name]['naive']}/10 | strict {totals[name]['strict']}/10")
        report()

    query = QUERIES[AB_QUERY]
    report(f"## A/B metadata filter — Q{AB_QUERY + 1}: {query['query']}")
    for name in strategies:
        agent = KnowledgeBaseAgent(store=stores[name], llm_fn=llm) if llm else None
        for label, metadata_filter in AB_ARMS:
            report(f"  [{name}] {label}")
            row = run_query(stores[name], agent, query, metadata_filter, report)
            report(f"      → evidence rank={row['evidence_rank']} | agent: {row['verdict']} → {row['strict']}đ")
            if row["answer"] is not None:
                report(f"      agent: {' '.join(row['answer'].split())}")
    report()

    report("## Tổng hợp (strict / naive)")
    report("  " + f"{'chiến lược':<11}" + "".join(f"  Q{i:<3}" for i in range(1, len(QUERIES) + 1)) + "  strict  naive")
    for name, total in totals.items():
        cells = "".join(f"  {r['strict']}/{r['naive']} " for r in total["rows"])
        report(f"  {name:<11}{cells}  {total['strict']:>4}/10  {total['naive']:>3}/10")

    if args.strategy == "all":
        OUTPUT_FILE.write_text("\n".join(report.lines) + "\n", encoding="utf-8")
        print(f"\nĐã ghi {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
