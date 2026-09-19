from typing import Callable

from .store import EmbeddingStore

NO_CONTEXT_ANSWER = "Không tìm thấy thông tin liên quan trong cơ sở tri thức."

PROMPT_TEMPLATE = """Bạn là trợ lý trả lời câu hỏi dựa trên tài liệu được cung cấp.

Quy tắc:
- Chỉ dùng thông tin trong phần NGỮ CẢNH bên dưới, không dùng kiến thức bên ngoài.
- Sau mỗi ý, ghi số nguồn đã dùng, ví dụ [1] hoặc [2][3].
- Nếu ngữ cảnh không chứa câu trả lời, trả lời đúng một câu: "Không tìm thấy thông tin trong tài liệu."

NGỮ CẢNH:
{context}

CÂU HỎI: {question}

TRẢ LỜI:"""


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        chunks = self.store.search(question, top_k=top_k)
        return self._generate(question, chunks)

    def answer_with_filter(self, question: str, top_k: int = 3, metadata_filter: dict | None = None) -> str:
        """Same as answer(), but retrieves through search_with_filter()."""
        chunks = self.store.search_with_filter(question, top_k=top_k, metadata_filter=metadata_filter)
        return self._generate(question, chunks)

    def build_prompt(self, question: str, chunks: list[dict]) -> str:
        # Number each chunk and name its source so every claim in the answer traces back
        # to a specific chunk and file.
        context = "\n\n".join(
            f"[{index}] (nguồn: {chunk.get('id') or chunk['metadata'].get('doc_id')})\n{chunk['content'].strip()}"
            for index, chunk in enumerate(chunks, start=1)
        )
        return PROMPT_TEMPLATE.format(context=context, question=question.strip())

    def _generate(self, question: str, chunks: list[dict]) -> str:
        # Empty store or nothing passed the filter: answer directly instead of paying for an LLM call.
        if not chunks:
            return NO_CONTEXT_ANSWER
        return self.llm_fn(self.build_prompt(question, chunks))
