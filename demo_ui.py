"""Giao diện web cho demo: hỏi một câu, xem top-3 và câu trả lời của agent, so sánh filter hoặc chiến lược.

    python demo_ui.py                  # nạp corpus + embedder rồi mở http://127.0.0.1:8000
    python demo_ui.py --port 8080 --no-open

Chỉ dùng thư viện chuẩn (http.server), chạy trên máy; tái dùng cấu hình và helper của bench.py.
Câu hỏi trùng một benchmark query thì đánh dấu chunk chứa đáp án và chấm câu trả lời như bench.py.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
import webbrowser
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from dotenv import load_dotenv

import bench
from src import KnowledgeBaseAgent

PAGE = Path(__file__).parent / "ui" / "index.html"
MAX_QUESTION = 500


class BadRequest(Exception):
    status = 400


class LLMUnavailable(Exception):
    status = 503


class Demo:
    """Built once at startup: one store per strategy over the same corpus and embedder."""

    def __init__(self) -> None:
        self.embedder = bench.make_embedder()
        cached = lru_cache(maxsize=None)(self.embedder)
        lock = threading.Lock()  # one model shared by all request threads

        def embed(text: str) -> list[float]:
            with lock:
                return cached(text)

        self.corpus = bench.load_corpus(bench.DATA_DIR)
        self.stores = {name: bench.build_store(name, self.corpus, embed)[0] for name in bench.STRATEGIES}
        self.model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
        self.host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.llm = bench.make_ollama_llm(self.model, self.host)

    def info(self) -> dict:
        return {
            "embedder": getattr(self.embedder, "_backend_name", type(self.embedder).__name__),
            "mock": self.embedder is bench._mock_embed,
            "llm": {"model": self.model, "ready": bench.ollama_ready(self.model, self.host)},
            "documents": len(self.corpus),
            "chunk_size": bench.CHUNK_SIZE,
            "top_k": bench.TOP_K,
            "strategies": {name: store.get_collection_size() for name, store in self.stores.items()},
            "presets": [
                {"query": q["query"], "gold_answer": q["gold_answer"], "filter": q.get("filter")} for q in bench.QUERIES
            ],
        }

    def search(self, body: dict) -> dict:
        question, store, metadata_filter = self._parse(body)
        # Rank every candidate, not just top-k, so a missed answer chunk can still be located.
        ranked = store.search_with_filter(question, top_k=store.get_collection_size(), metadata_filter=metadata_filter)
        query = preset(question)
        has_evidence = (lambda r: bool(re.search(query["evidence"], r["content"]))) if query else None
        response = {
            "candidates": len(ranked),
            "results": [
                {
                    "rank": rank,
                    "id": r["id"],
                    "score": round(r["score"], 4),
                    "title": r["metadata"].get("title"),
                    "audience": r["metadata"].get("audience"),
                    "content": r["content"],
                    "evidence": has_evidence(r) if query else None,
                }
                for rank, r in enumerate(ranked[: bench.TOP_K], start=1)
            ],
        }
        if query:
            rank = bench.first_rank(ranked, has_evidence)
            response["evidence"] = (
                {"rank": rank, "score": round(ranked[rank - 1]["score"], 4), "id": ranked[rank - 1]["id"]} if rank else None
            )
        return response

    def answer(self, body: dict) -> dict:
        question, store, metadata_filter = self._parse(body)
        if not bench.ollama_ready(self.model, self.host):
            raise LLMUnavailable(f"Ollama chưa chạy hoặc thiếu model {self.model}: chạy `ollama serve`")
        agent = KnowledgeBaseAgent(store=store, llm_fn=self.llm)
        start = time.perf_counter()
        text = agent.answer_with_filter(question, top_k=bench.TOP_K, metadata_filter=metadata_filter)
        response = {"answer": text, "seconds": round(time.perf_counter() - start, 1)}
        query = preset(question)
        if query:
            ok, label = bench.check_answer(query, text)
            response["verdict"] = {"ok": ok, "label": label}
        return response

    def _parse(self, body: dict):
        question = str(body.get("question", "")).strip()
        if not question or len(question) > MAX_QUESTION:
            raise BadRequest(f"Câu hỏi phải có 1–{MAX_QUESTION} ký tự")
        strategy = body.get("strategy")
        if strategy not in self.stores:
            raise BadRequest(f"Chiến lược không hợp lệ: {strategy!r}")
        metadata_filter = body.get("filter") or None
        if metadata_filter is not None and not (
            isinstance(metadata_filter, dict)
            and all(
                isinstance(value, str) or (isinstance(value, list) and all(isinstance(v, str) for v in value))
                for value in metadata_filter.values()
            )
        ):
            raise BadRequest("filter phải có dạng {\"audience\": \"student\"} hoặc {\"audience\": [\"student\", \"all\"]}")
        return question, self.stores[strategy], metadata_filter


def preset(question: str) -> dict | None:
    return next((q for q in bench.QUERIES if q["query"] == question), None)


def make_handler(demo: Demo):
    routes = {"/api/search": demo.search, "/api/answer": demo.answer}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/":
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")  # read per request: edit HTML without restart
            elif self.path == "/api/info":
                self._json(200, demo.info())
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            handler = routes.get(self.path)
            if handler is None:
                self._json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(body, dict):
                    raise BadRequest("body phải là JSON object")
                self._json(200, handler(body))
            except json.JSONDecodeError:
                self._json(400, {"error": "body không phải JSON"})
            except (BadRequest, LLMUnavailable) as error:
                self._json(error.status, {"error": str(error)})
            except Exception as error:  # e.g. Ollama timeout mid-answer: show it in the page, keep serving
                self._json(502, {"error": f"{type(error).__name__}: {error}"})

        def _json(self, status: int, data: dict) -> None:
            self._send(status, json.dumps(data, ensure_ascii=False).encode(), "application/json; charset=utf-8")

        def _send(self, status: int, payload: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args) -> None:
            if self.path != "/api/info":
                print(f"  {self.command} {self.path} {args[1] if len(args) > 1 else ''}")

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-open", action="store_true", help="không tự mở trình duyệt")
    args = parser.parse_args()

    load_dotenv(override=False)
    print(f"Đang nạp embedder và chunk corpus cho {len(bench.STRATEGIES)} chiến lược…")
    start = time.perf_counter()
    demo = Demo()
    sizes = ", ".join(f"{name} {store.get_collection_size()}" for name, store in demo.stores.items())
    print(f"Sẵn sàng sau {time.perf_counter() - start:.0f}s | chunk: {sizes}")
    if not bench.ollama_ready(demo.model, demo.host):
        print(f"!! Ollama chưa sẵn sàng ({demo.model}) — vẫn xem được retrieval; chạy `ollama serve` để có câu trả lời")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(demo))
    url = f"http://127.0.0.1:{args.port}"
    print(f"Mở {url} — Ctrl+C để dừng")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
