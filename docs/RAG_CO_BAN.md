# RAG cơ bản cho Lab 07

**Mục đích:** đọc xong tài liệu này, bạn làm được ba việc:
- hiểu hướng dẫn lab;
- giải thích từng bước code trong `src/`;
- đọc kết quả benchmark và trả lời khi bị phản biện.

**Cần biết trước:** Python cơ bản, và đã từng gọi LLM (Lab 01). Phần toán chỉ cần phép nhân vector.

**Ví dụ:** lấy từ chính bài lab: dữ liệu thư viện HUIT, embedder MiniLM, LLM `qwen2.5:3b`. Mọi con số đều đo thật.

## Đọc theo mức độ

| Mức | Phần | Đọc xong làm được |
|---|---|---|
| 1 · Nền tảng | §1–§2 | Giải thích RAG là gì, vì sao cần, gồm những bước nào |
| 2 · Làm lab | §3–§6 | Hiểu và sửa code chunking, store, agent |
| 3 · Đánh giá | §7–§8 | Chấm kết quả, xác định lỗi ở tầng nào, biết hướng nâng cấp |

Đối chiếu với checkpoint trong hướng dẫn lab:

| Checkpoint | Đọc phần |
|---|---|
| CP2 (metadata) | §5 |
| CP3 (chunking, cosine) | §3, §4 |
| CP4 (store, agent) | §5, §6 |
| CP5–CP6 (benchmark, A/B, phân tích lỗi) | §7 |
| Hỏi đáp khi demo | §8 |

---

## 1. RAG giải quyết vấn đề gì · Mức 1

LLM chỉ biết những gì có trong dữ liệu huấn luyện. Quy định nội bộ của thư viện HUIT không nằm trong đó.

Hỏi `qwen2.5:3b` mà không kèm tài liệu nào:

> **Hỏi:** Thư viện HUIT cho sinh viên mượn mấy cuốn sách về nhà và trong bao nhiêu ngày? Trả lời ngắn gọn.
> **Đáp:** Thư viện HUIT không cung cấp thông tin cụ thể về số lượng sách mà sinh viên có thể mượn cũng như thời hạn mượn. Để biết chính xác, bạn nên liên hệ trực tiếp với thư viện…

Model này từ chối. Model khác, hoặc cùng model với câu hỏi khác, có thể bịa ra một con số nghe rất hợp lý. Dù từ chối hay bịa, người hỏi vẫn không có được đáp án.

**RAG (Retrieval-Augmented Generation):** trước khi hỏi LLM, hệ thống *tìm* các đoạn tài liệu liên quan rồi *đưa kèm* chúng vào prompt. Cùng câu hỏi đó, chạy qua pipeline của lab (có lọc theo sinh viên):

> Theo quy định [1], bạn được phép mượn tối đa 3 tài liệu về nhà. Thời hạn mượn tài liệu về nhà là 10 ngày.

Có thể hình dung RAG như thi mở sách:
- LLM là thí sinh giỏi diễn đạt.
- Retrieval là việc lật đúng trang sách.
- Lật sai trang thì thí sinh giỏi đến đâu cũng trả lời sai (xem Q3 ở §6).

Khi nào nên dùng RAG:

| Nhu cầu | Chọn |
|---|---|
| Kiến thức riêng, hay thay đổi, cần trích nguồn (quy định, tài liệu nội bộ) | RAG |
| Đổi giọng văn, định dạng hoặc kỹ năng của model | Fine-tuning |
| Kiến thức phổ thông mà model đã biết | Hỏi thẳng LLM |

## 2. Pipeline gồm hai giai đoạn · Mức 1

```
NẠP (chạy một lần, khi có tài liệu)
  tài liệu .md ──► cắt thành chunk ──► embed từng chunk ──► lưu vào store
                                                               │
HỎI (chạy cho mỗi câu hỏi)                                     ▼
  câu hỏi ──► embed ──► tìm top-k chunk gần nhất ◄──────────── store
                              │
                              ▼
            ghép prompt (chunk + câu hỏi) ──► LLM ──► câu trả lời kèm [n]
```

Ba chữ cái trong tên RAG là ba bước của giai đoạn HỎI:

| Bước | Làm gì | Trong lab |
|---|---|---|
| **R**etrieve | Tìm top-k chunk giống câu hỏi nhất | `EmbeddingStore.search_with_filter`, [src/store.py](../src/store.py) |
| **A**ugment | Ghép các chunk vào prompt | `KnowledgeBaseAgent.build_prompt`, [src/agent.py](../src/agent.py) |
| **G**enerate | LLM viết câu trả lời | `llm_fn` = `qwen2.5:3b` qua Ollama, [bench.py](../bench.py) |

Giai đoạn NẠP phục vụ bước R. Chất lượng của nó phụ thuộc vào hai thứ: embedding (§3) và cách cắt chunk (§4).

---

## 3. Embedding và cosine similarity · Mức 2

**Embedding** biến một đoạn văn thành một vector số có độ dài cố định. Lab dùng `paraphrase-multilingual-MiniLM-L12-v2`: đoạn văn nào cũng thành 384 số. Mô hình này được huấn luyện để **câu cùng nghĩa cho ra vector cùng hướng**, kể cả khi hai câu không có từ nào chung.

**Cosine similarity** đo mức độ cùng hướng của hai vector:

```
cos(a, b) = (a · b) / (‖a‖ × ‖b‖)
```

- `a · b` (tích vô hướng) là tổng các tích từng cặp phần tử.
- `‖a‖ = √(a · a)` là độ dài của vector.
- Kết quả gần 1: cùng hướng, tức cùng nghĩa. Gần 0: vuông góc, tức không liên quan.

Tính tay với vector 2 chiều:

| | a = (1, 2), b = (2, 3) | a = (1, 2), c = (2, −1) |
|---|---|---|
| Tích vô hướng | 1·2 + 2·3 = 8 | 1·2 + 2·(−1) = 0 |
| Tích độ dài | √5 × √13 ≈ 8.062 | √5 × √5 = 5 |
| Cosine | **0.992**: gần cùng hướng | **0**: không liên quan |

Code tương ứng trong [src/chunking.py](../src/chunking.py):

```python
def compute_similarity(vec_a, vec_b):
    norm_a = math.sqrt(_dot(vec_a, vec_a))
    norm_b = math.sqrt(_dot(vec_b, vec_b))
    if norm_a == 0 or norm_b == 0:      # vector toàn số 0: tránh chia cho 0
        return 0.0
    return _dot(vec_a, vec_b) / (norm_a * norm_b)
```

Đo bằng embedder của lab:

| Câu A | Câu B | Cosine |
|---|---|---|
| Sinh viên được mượn sách tối đa 10 ngày. | Thời hạn mượn tài liệu của người học là mười ngày. | 0.856 |
| Sinh viên được mượn sách tối đa 10 ngày. | Học phí học kỳ hè được đóng qua ngân hàng. | 0.405 |

Cặp thứ nhất gần như không có từ chung mà điểm vẫn cao. Nghĩa là embedding so khớp theo **nghĩa** chứ không theo mặt chữ, và đó là lợi thế của nó so với tìm kiếm theo từ khóa.

Khi đọc score, nhớ hai điều sau:
- **Chỉ so sánh score trong cùng một câu hỏi.** Với heading_v2, top-3 của Q1 nằm trong khoảng 0.68–0.70, còn top-3 của Q3 là 0.78–0.82. Vì vậy không có ngưỡng chung kiểu "trên 0.7 là đúng".
- **Score cao không có nghĩa là trả lời được câu hỏi.** Cosine chỉ đo mức *giống chủ đề*. Ở Q3, chunk xếp đầu đạt 0.824 nhưng nói về tiền phạt của mượn thường, không phải của mượn liên thư viện.

## 4. Chunking · Mức 2

**Vì sao phải cắt tài liệu:**
- Nếu cả tài liệu 10.000 ký tự chỉ có một vector, vector đó là "trung bình" của mọi chủ đề trong tài liệu, nên không thật sự giống câu hỏi nào.
- Prompt có giới hạn độ dài. Đưa nguyên tài liệu vào vừa tốn chi phí vừa làm LLM lạc hướng.

Vì vậy tài liệu được cắt thành các đoạn vài trăm ký tự gọi là **chunk**, mỗi chunk có một vector riêng. Lab dùng `chunk_size = 500` ký tự.

| Chunk quá nhỏ | Chunk quá lớn |
|---|---|
| Mất ngữ cảnh: dòng bảng `3 \| 10` đứng một mình thì không ai biết đó là số gì | Nhiều ý trộn lẫn, vector bị loãng, top-k bị chiếm chỗ bởi chữ thừa |

**Các chunker trong lab:**

| Chunker | Cắt ở đâu | Điểm yếu gặp trong lab |
|---|---|---|
| `FixedSizeChunker` | Cứ 500 ký tự cắt một lần, hai chunk liền nhau dùng chung 50 ký tự (overlap) | Cắt giữa chữ: có chunk bắt đầu bằng "iệu định mượn…" |
| `SentenceChunker` | Gom N câu, tách câu ở `. ! ?` | Văn bản dạng danh sách không có dấu câu, nên một "câu" dài tới 1.500 ký tự |
| `RecursiveChunker` | Thử lần lượt các dấu tách `\n\n` → `\n` → `. ` → ` `, rồi gom các mảnh nhỏ lại cho gần đủ 500 | Tách câu dẫn khỏi bảng (ví dụ ngay dưới) |
| `HeadingChunker` (tự viết) | Cắt theo tiêu đề Markdown, gắn đường dẫn tiêu đề (breadcrumb) vào đầu mỗi chunk | Bản v1 vẫn tách câu dẫn khỏi danh sách (Q4). Bản v2 sửa bằng quy tắc: câu kết thúc bằng `:` đi cùng khối ngay sau nó |

Về overlap: 50 ký tự cuối của chunk trước được lặp lại ở đầu chunk sau, để câu nằm đúng chỗ cắt vẫn còn trọn trong ít nhất một chunk. Số chunk tính theo công thức ⌈(độ dài − overlap) / (size − overlap)⌉. Ví dụ tài liệu 10.000 ký tự, size 500, overlap 50 cho ra ⌈9.950 / 450⌉ = 23 chunk.

**Ví dụ quan trọng nhất của lab: chỗ cắt quan trọng hơn kích thước chunk.**

Đoạn gốc trong `quy-dinh-muon-tra-sinh-vien.md`:

```
- … Số lượng đầu tên tài liệu và thời hạn mượn tài liệu về nhà được quy định theo từng đối tượng như sau:

| Đối tượng | Số lượng tài liệu | Số ngày | Gia hạn — Số lần | Gia hạn — Số ngày/lần |
|---|---|---|---|---|
| Sinh viên, học viên | 3 | 10 | 1 | 10 |
```

Với Q1 "Tôi được mượn tối đa bao nhiêu tài liệu về nhà, trong bao nhiêu ngày?":

| Chunk chứa dòng đáp án | Cosine với Q1 | Hạng (không filter) |
|---|---|---|
| Recursive: bảng + đoạn phía sau, **thiếu câu dẫn** | 0.443 | #18 / 77, agent không nhìn thấy |
| Heading v2: breadcrumb + **câu dẫn + bảng** | 0.676 | #3 / 103, agent trả lời đúng |

Đo riêng từng phần: chỉ riêng bảng được 0.372; câu dẫn + bảng được 0.672. Bảng chỉ toàn số, không có chữ nào giống câu hỏi. Chính câu dẫn mới mang các cụm "mượn tài liệu về nhà" và "thời hạn". **Bài học: giữ câu dẫn đi cùng dữ liệu mà nó dẫn.**

## 5. Vector store và metadata filter · Mức 2

**Store** lưu mỗi chunk thành một bản ghi:

```python
{
    "id": "quy-dinh-muon-tra-sinh-vien#3",
    "content": "Quy định mượn/trả tài liệu — Sinh viên, học viên > … như sau: | Sinh viên, học viên | 3 | 10 | …",
    "metadata": {"doc_id": "quy-dinh-muon-tra-sinh-vien", "audience": "student", "category": "muon-tra"},
    "embedding": [0.012, -0.087, ...],   # 384 số
}
```

**Tìm kiếm** gồm bốn bước: embed câu hỏi, tính cosine với mọi bản ghi, sắp xếp giảm dần, lấy k bản ghi đầu (lab dùng k = 3). Cách duyệt hết mọi bản ghi như vậy gọi là brute force: đủ nhanh với khoảng 100 chunk, còn với hàng triệu chunk thì cần index riêng (§8).

**Metadata** là thông tin *về* tài liệu. Trong lab, nó được khai báo ở frontmatter đầu mỗi file:

```yaml
---
doc_id: quy-dinh-muon-tra-sinh-vien
audience: student
category: muon-tra
source_url: https://thuvien.huit.edu.vn/Page/quy-dinh-su-dung-thu-vien
retrieved_at: 2026-09-19
---
```

**Metadata filter** giới hạn việc tìm kiếm trong các chunk thỏa điều kiện, ví dụ `{"audience": "student"}`. Lab **lọc trước rồi mới xếp hạng**. Nếu làm ngược lại (lấy top-3 rồi mới lọc) thì kết quả có thể còn ít hơn k. Ví dụ ở Q1 với chiến lược heading, top-3 chỉ có một chunk `student`, nên lọc sau chỉ còn 1 chunk; lọc trước thì vẫn đủ 3 chunk `student`.

A/B ở Q1 (heading):

| | Không filter | `audience = student` |
|---|---|---|
| Top-1 | `muon-lien-thu-vien#4` (2 tài liệu / 20 ngày, quy định mượn liên thư viện) | Chunk đúng (3 tài liệu / 10 ngày) |
| Agent | Lẫn "180 ngày" của giảng viên ✗ | "3 tài liệu, 10 ngày" ✓ |

**Cái giá của filter:**
- Lọc cứng theo `student` sẽ loại luôn các tài liệu `audience: all`, mà đáp án Q2, Q3, Q5 lại nằm trong đó. Vì vậy filter phải bật theo ý định của câu hỏi (hoặc theo tài khoản người dùng), không bật mặc định.
- Filter chỉ có tác dụng khi metadata được gán đúng theo chiều cần lọc. Lab phải tách trang quy định gốc thành 3 file theo đối tượng thì mới lọc được theo audience.

## 6. Prompt và LLM · Mức 2

Bước Augment ghép các chunk vào một khung prompt cố định. Dưới đây là bản rút gọn từ [src/agent.py](../src/agent.py):

```
Quy tắc:
- Chỉ dùng thông tin trong phần NGỮ CẢNH bên dưới, không dùng kiến thức bên ngoài.
- Sau mỗi ý, ghi số nguồn đã dùng, ví dụ [1] hoặc [2][3].
- Nếu ngữ cảnh không chứa câu trả lời, trả lời đúng một câu: "Không tìm thấy thông tin trong tài liệu."

NGỮ CẢNH:
[1] (nguồn: quy-dinh-muon-tra-sinh-vien#3)
Quy định mượn/trả tài liệu — Sinh viên, học viên > … | Sinh viên, học viên | 3 | 10 | 1 | 10 |

[2] (nguồn: …)

CÂU HỎI: Tôi được mượn tối đa bao nhiêu tài liệu về nhà, trong bao nhiêu ngày?
TRẢ LỜI:
```

Mỗi thành phần trong prompt và code đều có lý do:

| Thành phần | Ngăn điều gì |
|---|---|
| "Chỉ dùng NGỮ CẢNH" | Model bịa từ kiến thức chung (hallucination) |
| Đánh số `[n]` kèm tên nguồn | Câu trả lời không kiểm chứng được |
| Câu trả lời cố định khi không tìm thấy | Model trả lời đại khi thiếu thông tin |
| Không có chunk nào thì trả lời luôn, không gọi LLM | Tốn một lần gọi LLM vô ích |
| `temperature = 0` | Mỗi lần chạy ra một kiểu, benchmark không lặp lại được |

Prompt tốt vẫn không đảm bảo câu trả lời đúng:
- **Q3:** agent trả lời "1.000 đồng/1 tài liệu/ngày [1]". Con số này *có thật* trong chunk [1], nhưng chunk đó nói về mượn thường. Chunk chứa đáp án đúng (5.000đ, mượn liên thư viện) xếp #4, nằm ngoài top-3. Câu trả lời trung thành với ngữ cảnh nhưng ngữ cảnh lại sai: **bám nguồn (grounded) chưa chắc đã đúng**.
- **Q5:** câu hỏi có hai vế, "đặt phòng thế nào" và "dùng bao lâu". Top-3 chỉ có thông tin về thời lượng, nên model 3B tự thêm "liên hệ bộ phận quản lý tài nguyên học tập". Câu này không có trong tài liệu, dù prompt đã cấm bịa.

---

## 7. Đánh giá và tìm lỗi · Mức 3

Đánh giá RAG phải tách thành hai tầng:
- **Retrieval:** có lấy được đúng chunk không?
- **Generation:** có trả lời đúng dựa trên chunk đó không?

**Hai cách chấm retrieval:**

| Cách chấm | Kiểm tra gì | Điểm trong lab (4 chiến lược) |
|---|---|---|
| Theo `doc_id` | File đúng có nằm trong top-3 không | 8/10 cho **cả 4** chiến lược |
| Theo nội dung | Chunk trong top-3 có *chứa câu đáp án* không, và agent có trả lời đúng không | 4–6/10 |

Ví dụ Q1 với recursive: top-3 có chunk thuộc file `quy-dinh-muon-tra-sinh-vien`, nên chấm theo `doc_id` được 2 điểm. Nhưng không chunk nào chứa dòng bảng hạn mức, nên chấm theo nội dung chỉ được 0. Đúng file chưa chắc đã đúng đoạn.

Thang điểm của lab ([SCORING.md](SCORING.md)): **2** = chunk chứa đáp án ở top-1 và agent trả lời đúng; **1** = chunk đó có trong top-3 nhưng câu trả lời chưa đủ; **0** = không có trong top-3.

**Tìm lỗi theo tầng.** Khi câu trả lời sai, lần lượt kiểm tra:

| # | Kiểm tra | Nếu "không", lỗi nằm ở | Ví dụ trong lab |
|---|---|---|---|
| 1 | Có chunk nào chứa trọn đáp án không? | Dữ liệu / chunking | Q1 với recursive: bảng bị tách khỏi câu dẫn |
| 2 | Chunk đó có lọt vào top-k không? | Retrieval | Q3: chunk đúng ở #4. Q5 (heading): chunk thủ tục ở #5 |
| 3 | Đã có chunk đúng, LLM có dùng đúng không? | Generation | Q1 không filter (heading): chunk đúng ở #2, nhưng agent lấy thêm "180 ngày" từ chunk #3 |

Sửa đúng tầng bị lỗi:
- Tầng 1: đổi cách chunk.
- Tầng 2: dùng filter, hybrid search, rerank hoặc tăng k.
- Tầng 3: sửa prompt, dùng model lớn hơn, hoặc bớt chunk gây nhiễu.

**Cẩn thận với máy chấm tự động:**
- Regex chấm Q5 từng cho 2 điểm một câu trả lời có phần bịa. Luôn đọc lại câu trả lời bằng mắt.
- 5 câu hỏi là quá ít để kết luận chiến lược nào tốt hơn theo nghĩa thống kê.

## 8. Giới hạn và hướng nâng cấp · Mức 3

Pipeline của lab là **naive RAG**: tìm một lần, đọc, rồi trả lời. Các kỹ thuật advanced RAG dưới đây nhắm đúng vào những lỗi lab đã gặp:

| Kỹ thuật | Làm gì | Lỗi trong lab nó xử lý |
|---|---|---|
| Hybrid search (BM25 + vector) | Cộng điểm khớp từ khóa với điểm ngữ nghĩa | Q3: cụm "trễ hạn bị phạt" lấn át từ khóa "liên thư viện" |
| Rerank (cross-encoder) | Lấy top-20 rồi chấm lại từng cặp (câu hỏi, chunk) bằng model chính xác hơn | Q3, Q5: chunk đúng nằm ở #4, #5 |
| Tách câu hỏi (query decomposition) | Tách câu hỏi nhiều vế thành nhiều truy vấn con | Q5: "đặt thế nào" + "dùng bao lâu" |
| MMR | Chọn top-k vừa liên quan vừa khác nhau | Q5: 3 chunk của cùng một file chiếm hết top-3 |
| Chọn filter theo ý định | Tự suy ra audience từ câu hỏi hoặc từ tài khoản người dùng | Trong bench, filter đang được gắn sẵn cho từng câu hỏi |
| Ngưỡng điểm / tự kiểm | Score thấp, hoặc câu trả lời không khớp nguồn, thì trả lời "không biết" | Agent luôn nhận đủ 3 chunk nên gần như không bao giờ từ chối |
| Vector DB + index ANN | Lưu vector xuống đĩa, tìm kiếm gần đúng nhưng nhanh | Store in-memory phải embed lại toàn bộ mỗi lần chạy |

---

## Tự kiểm tra

<details>
<summary>1. Vì sao không embed nguyên cả tài liệu thành một vector?</summary>

Vì vector của cả tài liệu là "trung bình" của nhiều chủ đề, nên không thật sự giống câu hỏi nào. Ngoài ra prompt cũng không chứa nổi cả tài liệu. (§4)
</details>

<details>
<summary>2. Hai chunk có cosine 0.70 và 0.68 với câu hỏi. Chunk 0.70 có chắc trả lời tốt hơn không?</summary>

Không. Cosine đo mức giống chủ đề, không đo việc chunk có chứa đáp án hay không. Ở Q3, chunk 0.824 sai còn chunk 0.775 mới đúng. (§3)
</details>

<details>
<summary>3. Bật filter <code>audience=student</code> cho mọi câu hỏi thì có hại gì?</summary>

Filter sẽ loại mất các tài liệu `audience: all`, trong khi đó là nơi chứa đáp án Q2, Q3, Q5. (§5)
</details>

<details>
<summary>4. Agent trả lời sai nhưng vẫn trích [1]. Lỗi ở tầng nào, kiểm tra thế nào?</summary>

Mở chunk [1] ra xem. Nếu con số sai có trong chunk thì lỗi ở retrieval (lấy nhầm đoạn), như Q3. Nếu chunk không có con số đó thì lỗi ở generation (model bịa). (§6, §7)
</details>

<details>
<summary>5. Chấm theo doc_id được 8/10. Như vậy đã đủ để nói hệ thống tốt chưa?</summary>

Chưa. Đúng file chưa chắc đã đúng đoạn: chấm theo nội dung, lab chỉ còn 4–6/10. (§7)
</details>

## Tra nhanh thuật ngữ

| Thuật ngữ | Nghĩa |
|---|---|
| Chunk, overlap | §4 |
| Embedding, cosine similarity | §3 |
| Top-k, metadata, filter | §5 |
| Grounding, hallucination | §6 |
| Corpus | Toàn bộ tài liệu nạp vào hệ thống (lab: 9 file của thư viện HUIT) |
| Gold answer | Đáp án chuẩn do người soạn benchmark xác định trước |
| Provenance | Nguồn gốc của tài liệu: URL, ngày lấy, phiên bản |
| Benchmark | Bộ câu hỏi cố định kèm gold answer, dùng để so sánh các chiến lược một cách công bằng |
