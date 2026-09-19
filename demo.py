"""Demo trực tiếp: hỏi một câu trên corpus HUIT, in top-3 và câu trả lời của agent.

    python demo.py "Tôi được mượn tối đa bao nhiêu tài liệu về nhà, trong bao nhiêu ngày?" --audience student
    python demo.py "Trả sách mượn liên thư viện trễ hạn bị phạt bao nhiêu?" --strategy fixed

Luôn chạy nhánh không filter; có --audience thì chạy thêm nhánh có filter để so sánh.
Dùng lại cấu hình của bench.py (corpus, chunk_size, embedder, Ollama).
"""

from __future__ import annotations

import argparse
import os
from functools import lru_cache

from dotenv import load_dotenv

import bench
from src import KnowledgeBaseAgent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("--strategy", choices=list(bench.STRATEGIES), default="heading_v2")
    parser.add_argument("--audience", help="thêm nhánh metadata_filter={'audience': ...}, ví dụ student")
    args = parser.parse_args()

    load_dotenv(override=False)
    embedder = bench.make_embedder()
    store, docs = bench.build_store(args.strategy, bench.load_corpus(bench.DATA_DIR), lru_cache(maxsize=None)(embedder))
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    llm = bench.make_ollama_llm(model, host) if bench.ollama_ready(model, host) else None
    agent = KnowledgeBaseAgent(store=store, llm_fn=llm) if llm else None

    print(f"Chiến lược {args.strategy}: {len(docs)} chunk | embedding: {getattr(embedder, '_backend_name', '?')} "
          f"| LLM: {model if llm else 'không có (chạy ollama serve)'}")
    print(f"Câu hỏi: {args.question}")
    arms = [("không filter", None)]
    if args.audience:
        arms.append((f"audience={args.audience}", {"audience": args.audience}))
    for label, metadata_filter in arms:
        print(f"\n[{label}]")
        for rank, result in enumerate(store.search_with_filter(args.question, bench.TOP_K, metadata_filter), start=1):
            print(f"  #{rank} {result['score']:.3f} {result['id']:<34} [{result['metadata'].get('audience')}] "
                  f"{bench.preview(result['content'], 90)}")
        if agent:
            answer = agent.answer_with_filter(args.question, top_k=bench.TOP_K, metadata_filter=metadata_filter)
            print(f"  agent: {' '.join(answer.split())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
