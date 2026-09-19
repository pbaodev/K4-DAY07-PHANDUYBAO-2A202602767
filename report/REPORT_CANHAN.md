# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Phan Duy Bao — 2A202602767
**Nhóm:** Làm cá nhân (1 thành viên)
**Ngày:** 2026-09-19

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Hai vector embedding gần như cùng hướng, tức là mô hình coi hai đoạn văn cùng nói về một ý, bất kể độ dài hay cách dùng từ. Cosine gần 1 là rất giống nghĩa, gần 0 là không liên quan.

**Ví dụ có độ tương tự CAO:**
- Câu A: "Sinh viên được mượn sách tối đa 10 ngày."
- Câu B: "Thời hạn mượn tài liệu của người học là mười ngày."
- Tại sao tương đồng: cùng một quy định nhưng gần như **không trùng từ khoá** (sinh viên / người học, sách / tài liệu, 10 / mười). Đo bằng embedder đa ngữ MiniLM được **0.856**, cho thấy embedding khớp theo nghĩa chứ không khớp theo mặt chữ.

**Ví dụ có độ tương tự THẤP:**
- Câu A: "Sinh viên được mượn sách tối đa 10 ngày."
- Câu B: "Học phí học kỳ hè được đóng qua ngân hàng."
- Tại sao khác: cùng bối cảnh đại học nhưng khác chủ đề (mượn sách so với đóng học phí). Đo được **0.405**.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Cosine chỉ đo *hướng* của vector, tức là nội dung ngữ nghĩa, và bỏ qua *độ lớn*, vốn dễ bị ảnh hưởng bởi độ dài văn bản hay cách mô hình scale vector. Giá trị nằm trong khoảng [-1, 1] nên dễ diễn giải và dễ đặt ngưỡng. Với vector đã chuẩn hoá (‖v‖ = 1), ta có ‖a − b‖² = 2 − 2·cos(a, b), nên hai thước đo xếp hạng giống nhau. Cosine vẫn là lựa chọn an toàn khi không chắc backend có chuẩn hoá vector hay không.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:* ⌈(10 000 − 50) / (500 − 50)⌉ = ⌈9 950 / 450⌉ = ⌈22.11⌉ = 23
> *Kiểm lại bằng code:* `len(FixedSizeChunker(chunk_size=500, overlap=50).chunk("a"*10000))` → **23**
> *Đáp án:* **23 chunks**

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> ⌈(10 000 − 100) / 400⌉ = ⌈24.75⌉ = **25 chunks** (kiểm bằng code cũng ra 25): bước nhảy giảm từ 450 xuống 400 nên cần thêm 2 chunk. Overlap lớn hơn giúp một câu hay điều khoản nằm ở ranh giới hai chunk vẫn có mặt trọn vẹn trong ít nhất một chunk, tức là thêm một cơ hội lọt top-k. Cái giá là nhiều chunk hơn (tốn embedding, bộ nhớ) và top-k dễ trả về các chunk gần trùng nhau.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Tách bằng `re.split(r"(?<=[.!?])\s+", text)`. Lookbehind `(?<=…)` cắt ở khoảng trắng **nằm sau** dấu câu, nên dấu chấm vẫn ở lại cuối câu chứ không bị nuốt. Sau đó strip từng câu, bỏ câu rỗng, rồi gom `max_sentences_per_chunk` câu thành một chunk. Text rỗng hoặc chỉ có khoảng trắng thì trả `[]`.
>
> Edge case (tôi đã chạy thử): số thập phân và số tiền như `2.5`, `2.000đ` **không** bị cắt sai, vì sau dấu chấm không có khoảng trắng. Chữ viết tắt thì **bị cắt sai**: `TS. Nguyễn` bị tách sau `TS.`, `v.v. được` bị tách sau `v.v.`. Ngoài ra, văn bản quy định dạng danh sách hoặc bảng không có dấu câu, nên một "câu" có thể dài tới 1 500 ký tự.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Có ba base case: (1) mảnh đã vừa `chunk_size` thì trả về luôn; (2) hết separator, hoặc gặp separator `""`, thì cắt cứng theo `chunk_size` (xử lý được `separators=[]`); (3) separator hiện tại không có trong text thì thử separator nhỏ hơn.
>
> Thuật toán chạy **hai chiều**. Chiều xuống: mảnh nào vẫn quá dài thì đệ quy với các separator còn lại. Chiều lên: các mảnh nhỏ liền kề được gom vào buffer cho tới sát `chunk_size`. Separator được giữ ở cuối mỗi mảnh, nên nối các chunk lại vẫn ra đúng văn bản gốc, không mất chữ nào.
>
> Lỗi tôi tìm ra và đã sửa: bản đầu xả buffer ra trước khi đệ quy, nên một heading ngắn như `## Mục tiêu` bị thành chunk riêng 11 ký tự. Bản sửa đệ quy trên `buffer + mảnh dài`, để heading dính vào mảnh con đầu tiên, và giữ mảnh con cuối làm buffer để còn gom tiếp. Trên `rag_system_design.md`, số chunk giảm từ 18 xuống 10 và chunk ngắn nhất tăng từ 11 lên 195 ký tự.

**`compute_similarity` / `ChunkingStrategyComparator`:**
> `compute_similarity` dùng lại `_dot` cho tích vô hướng và cho hai chuẩn ‖a‖, ‖b‖; nếu một chuẩn bằng 0 thì trả `0.0` thay vì để `ZeroDivisionError`. Comparator chạy `FixedSizeChunker(overlap=0)`, `SentenceChunker(3)` và `RecursiveChunker` trên cùng text, trả về đúng 3 key `fixed_size`, `by_sentences`, `recursive`. Mỗi key có `count`, `avg_length` (bằng 0 khi không có chunk nào, để không chia cho 0), `chunks`, và thêm `min_length` / `max_length`. Tôi chọn `overlap=0` để cả 3 chiến lược đều ra chunk không chồng nhau, khi đó `count` mới so sánh công bằng được.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> Store chỉ chạy in-memory. Tôi **bỏ nhánh ChromaDB** vì code mẫu gán `_use_chroma = True` *trước khi* tạo client, nên máy nào có cài `chromadb` sẽ khiến mọi method rẽ vào nhánh chưa cài đặt.
>
> `_make_record` **copy** metadata, để store không giữ chung object với code gọi, và `setdefault("doc_id", doc.id)`. Nhờ vậy chunk `"file#3"` do tôi tạo vẫn giữ `doc_id` của file gốc. Mỗi record gồm `id`, `content`, `metadata`, `embedding`; `add_documents` không tự chunk (1 Document = 1 record).
>
> `search` gọi helper `_search_records`: embed câu hỏi, tính **cosine đầy đủ** với từng record, sắp giảm dần và lấy `top_k`; kết quả bỏ trường `embedding` cho gọn. Tôi dùng cosine thay vì dot product trần: với vector đã chuẩn hoá thì kết quả như nhau, còn nếu backend trả vector chưa chuẩn hoá thì cosine vẫn đúng.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> **Lọc trước rồi mới xếp hạng.** Nếu lấy top-k rồi mới lọc, k slot có thể đã bị tài liệu sai đối tượng chiếm hết và kết quả còn 0, dù store vẫn có chunk hợp lệ. `search` và `search_with_filter` đi chung một đường code (`_search_records`), chỉ khác tập ứng viên đầu vào, nên không có filter thì kết quả giống hệt `search`. Tôi mở rộng thêm: giá trị filter là list thì khớp bất kỳ phần tử nào, ví dụ `{"audience": ["student", "all"]}`, để đo thêm nhánh A/B thứ ba.
>
> `delete_document` giữ lại các record có `metadata["doc_id"] != doc_id`, và trả `True` nếu kích thước store giảm. Như vậy xoá một file là xoá mọi chunk của nó.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Ba bước: `store.search` → `build_prompt` → `llm_fn`. Prompt đánh số từng chunk `[1] (nguồn: quy-dinh-muon-tra-sinh-vien#3)`, nên mỗi ý trong câu trả lời truy vết được về đúng chunk và đúng file.
>
> Prompt có 3 quy tắc: chỉ dùng NGỮ CẢNH; ghi số nguồn sau mỗi ý; không có thông tin thì trả lời đúng câu "Không tìm thấy thông tin trong tài liệu."
>
> Khi không có chunk nào (store rỗng hoặc filter không khớp), agent trả thông báo ngay mà **không gọi LLM**. Tôi thêm `answer_with_filter` (dùng `search_with_filter`) song song với `answer`, giữ nguyên chữ ký `answer` theo yêu cầu. Khi thử với `qwen2.5:3b`, agent trả lời đúng kèm `[1]`, và nói "Không tìm thấy…" khi ngữ cảnh không chứa đáp án. Tuy vậy, ở benchmark Q5 model vẫn bịa một vế (xem mục 5).

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
============================= test session starts ==============================
platform darwin -- Python 3.11.15, pytest-9.1.1, pluggy-1.6.0 -- /Users/baobun/AI20K/LAB-AI20K/K4-DAY07-PHANDUYBAO-2A202602767/.venv/bin/python
cachedir: .pytest_cache
rootdir: /Users/baobun/AI20K/LAB-AI20K/K4-DAY07-PHANDUYBAO-2A202602767
plugins: anyio-4.15.1
collecting ... collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================== 42 passed in 0.02s ==============================
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

Embedder: `paraphrase-multilingual-MiniLM-L12-v2` (local). Cột **Dự đoán** được ghi **trước khi** chạy `compute_similarity`.

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Tôi muốn gia hạn sách đang mượn. | Làm sao để kéo dài thời gian mượn tài liệu? | cao / thấp | | |
| 2 | Thư viện mở cửa lúc 7 giờ sáng. | The library opens at 7 a.m. | cao / thấp | | |
| 3 | Trả sách trễ hạn bị phạt 1.000 đồng mỗi ngày. | Trả sách trễ hạn không bị phạt. | cao / thấp | | |
| 4 | Làm thẻ thư viện mới mất 100.000 đồng. | Phí gửi xe máy trong trường là 5.000 đồng. | cao / thấp | | |
| 5 | Phòng học nhóm nằm ở tầng 3. | Hôm nay trời mưa rất to. | cao / thấp | | |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> *(điền sau khi chạy)*

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chiến lược của tôi: **HeadingChunker v2** (`chunk_size=500`, `glue_lead_in=True`) — chunk theo tiêu đề/mục, gắn breadcrumb, giữ câu dẫn đi cùng danh sách/bảng. Chạy trên mã nguồn `src/` của tôi bằng `python bench.py`: 103 chunk, embedder MiniLM đa ngữ, agent `qwen2.5:3b`, `top_k=3`. Dùng cùng 5 câu hỏi với `REPORT_NHOM.md`; Q1 chạy với `metadata_filter={"audience": "student"}`.

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Mượn tối đa bao nhiêu tài liệu, bao nhiêu ngày? | `quy-dinh-muon-tra-sinh-vien#3` — "…Sinh viên, học viên > 5. > a. Mượn tài liệu": câu dẫn + bảng "Sinh viên, học viên \| 3 \| 10" | 0.676 | Có (#1) → 2đ | "Được mượn tối đa 3 tài liệu… thời hạn 10 ngày." [1] — đúng |
| 2 | Làm thẻ thư viện mới mất bao nhiêu? | `huong-dan-su-dung-thu-vien#9` — "> 2. Đăng ký làm thẻ": bảng Lệ phí Thẻ 100.000 đ/thẻ | 0.776 | Có (#1) → 2đ | "Thẻ cấp mới mất 100,000 đ." [1] — đúng |
| 3 | Phạt trễ hạn mượn liên thư viện? | `luu-hanh-tai-lieu#10` — "> 2. Dịch vụ cho mượn về > Lưu ý": phạt **1.000đ**/tài liệu/ngày (mượn thường) | 0.824 | **Không** — chunk đúng (5.000đ) không có trong top-3 → 0đ | "Bị phạt 1.000 đồng/1 tài liệu/ngày." [1] — **sai**, trả lời trung thành với chunk gây nhiễu |
| 4 | Làm mất sách tiếng Việt cũ, không còn bán thì đền thế nào? | `quy-dinh-chung#21` — "> 7. Đền bù làm mất > b.": 2 gạch đầu dòng "sách đền bìa cứng / ấn bản mới hơn", thuộc trường hợp tài liệu **còn** lưu hành; câu dẫn của chúng lại nằm ở `#20` | 0.458 | Có, nhưng ở #3 → 1đ | "Đền tiền gấp 5 lần giá tiền ghi trên bìa." [3] — đúng |
| 5 | Đặt phòng học nhóm thế nào, dùng bao lâu? | `su-dung-phong-hoc-nhom#0` — "> Phòng học nhóm": 04 phòng, tầng 3, 02 giờ/lượt | 0.712 | Có một vế (#1) → 1đ | "2 giờ/lượt, gia hạn nếu không có người chờ" — đúng; vế đặt phòng **bịa** ("liên hệ bộ phận quản lý tài nguyên học tập") |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 4 / 5 (điểm theo SCORING.md: 6 / 10)

So với bản chưa tinh chỉnh (heading v1), v2 sửa được Q4: chunk chứa "gấp 5 lần" lên từ #7 lên #3, agent từ trả lời sai chuyển sang trả lời đúng. Tổng điểm tăng từ 5 lên 6 trên 10. Bằng chứng A/B cho Q1 (v2): không filter thì chunk đúng ở #3; filter `student` thì ở #1. Cả hai nhánh agent đều trả lời đúng; đây là chiến lược duy nhất đạt được điều này.

Giới hạn còn lại của v2 (thấy ở top-1 của Q4): khi khối "câu dẫn + danh sách" vẫn dài hơn budget, bước dự phòng `RecursiveChunker` lại tách câu dẫn ra. Ví dụ "…Thư viện xem xét chấp nhận đền bù:" nằm ở `#20`, còn danh sách của nó ở `#21`. Bước tiếp theo sẽ là gắn câu dẫn vào đầu từng mảnh con, giống cách đang làm với breadcrumb.

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> *(điền sau buổi demo)*

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | / 5 |
| Hướng tiếp cận của tôi (My Approach) | / 10 |
| Hoàn thiện code (Core Implementation — tests) | / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | / 5 |
| Kết quả truy xuất của tôi (Competition Results) | / 10 |
| **Tổng phần cá nhân** | **/ 60** |
