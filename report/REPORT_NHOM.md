# Báo Cáo Nhóm — Lab 7: Embedding & Vector Store

**Nhóm:** Làm cá nhân (1 thành viên — đã được giảng viên đồng ý)
**Thành viên:** Phan Duy Bao — 2A202602767
**Ngày:** 2026-09-19

> **Nộp 1 bản / nhóm.** Phần cá nhân (hướng tiếp cận, kết quả riêng, dự đoán…) mỗi thành viên nộp riêng trong `REPORT_CANHAN.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần nhóm: 40** = Lựa chọn tài liệu (10) + Thiết kế chiến lược (15) + Chất lượng truy xuất (10) + Thuyết trình (5).

> **Ghi chú làm cá nhân:** không có thành viên khác để chia chiến lược, nên tôi tự chạy **4 chiến lược** trên cùng bộ tài liệu, cùng 5 câu hỏi, cùng embedder và cùng `chunk_size=500`. Chỉ đúng một dòng chọn chunker trong `bench.py` thay đổi giữa các lần chạy. Các khối "Thành viên 1/2/3" trong template được thay bằng "Chiến lược 1/2/3/4". Toàn bộ số liệu bên dưới lấy từ `ket_qua_benchmark.txt`.

**Cấu hình đo:** embedding `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (chạy local); LLM của agent là `qwen2.5:3b` chạy qua Ollama, `temperature=0`; `top_k=3`.

---

## 1. Lựa chọn tài liệu (Document Set Quality) — Nhóm (10 điểm)

### Chủ đề (Domain) & Lý Do Chọn

**Chủ đề:** Dịch vụ và quy định sử dụng thư viện đại học — Thư viện Trường ĐH Công Thương TP.HCM (HUIT).

**Tại sao nhóm chọn chủ đề này?**
> Quy định thư viện có số liệu cụ thể (số tài liệu, số ngày, mức phí, mức đền bù), nên gold answer kiểm chứng được mà không phải suy đoán. Hạn mức mượn của HUIT **khác nhau theo đối tượng** (sinh viên 10 ngày, giảng viên 180 ngày), nhờ vậy câu hỏi cần `metadata_filter={"audience": "student"}` có ý nghĩa thật. Văn bản quy định chia mục đánh số, hợp để thử chunk theo heading.
>
> Tôi chọn nguồn sau khi kiểm tra khả năng crawl của 8 thư viện. HCMUS và UT bị `robots.txt` chặn, một số trường lỗi chứng chỉ SSL. HUIT không có `robots.txt` (HTTP 404), nên theo RFC 9309 không có giới hạn truy cập tự động. Trang của HUIT là HTML tĩnh. Crawler vẫn giãn cách ít nhất 1 giây giữa các request và chỉ lấy 13 trang ứng viên.

### Danh sách tài liệu (Data Inventory)

Toàn bộ lấy ngày **2026-09-19**. Trang nguồn không nêu số hiệu hay ngày hiệu lực, nên `document_version = not-stated`. Số ký tự tính trên phần thân (đã bỏ frontmatter). Mọi file đều có `department: library` và `language: vi`.

| # | Tên tài liệu (`doc_id`) | Nguồn (Source URL) | Ngày lấy / Phiên bản | Số ký tự | Metadata đã gán |
|---|--------------|------------|--------------------|----------|-----------------|
| 1 | `quy-dinh-chung` | https://thuvien.huit.edu.vn/Page/quy-dinh-su-dung-thu-vien (mục 1, 2, 4, 6–10) | 2026-09-19 / not-stated | 9 843 | audience=all, category=noi-quy |
| 2 | `quy-dinh-muon-tra-sinh-vien` | cùng trang trên (mục 3, 5, 11a — phần cho sinh viên) | 2026-09-19 / not-stated | 2 606 | audience=**student**, category=muon-tra |
| 3 | `quy-dinh-muon-tra-giang-vien` | cùng trang trên (mục 3, 5, 11a — phần cho giảng viên) | 2026-09-19 / not-stated | 2 140 | audience=**faculty**, category=muon-tra |
| 4 | `luu-hanh-tai-lieu` | https://thuvien.huit.edu.vn/Page/luu-hanh-tai-lieu | 2026-09-19 / not-stated | 3 422 | audience=all, category=dich-vu |
| 5 | `muon-lien-thu-vien` | https://thuvien.huit.edu.vn/Page/muon-lien-thu-vien | 2026-09-19 / not-stated | 2 421 | audience=all, category=dich-vu |
| 6 | `huong-dan-su-dung-thu-vien` | https://thuvien.huit.edu.vn/Page/huong-dan-su-dung-thu-vien | 2026-09-19 / not-stated | 5 388 | audience=all, category=huong-dan |
| 7 | `su-dung-phong-hoc-nhom` | https://thuvien.huit.edu.vn/Page/su-dung-phong-hoc-nhom | 2026-09-19 / not-stated | 2 001 | audience=all, category=dich-vu |
| 8 | `muon-tra-sach-tu-dong` | https://thuvien.huit.edu.vn/Page/muon-tra-sach-tu-dong | 2026-09-19 / not-stated | 1 135 | audience=student, category=huong-dan |
| 9 | `nhan-tra-tai-san-that-lac` | https://thuvien.huit.edu.vn/Page/nhan-tra-tai-san-that-lac | 2026-09-19 / not-stated | 527 | audience=all, category=dich-vu |

Phân bố `audience`: all ×6, student ×2, faculty ×1. Kết quả script kiểm tra CP2: 9/9 file OK, `sources.csv` khớp 1-1.

**Danh sách kiểm tra quản trị dữ liệu (Data governance checklist):**
- [x] Tập tài liệu (Corpus) chỉ chứa nguồn công khai/được phép dùng và không chứa dữ liệu cá nhân, thông tin đăng nhập hoặc tài liệu nội bộ. Email và hotline xuất hiện trong tài liệu là kênh liên hệ công khai của thư viện, không phải thông tin cá nhân.
- [x] Mỗi tài liệu có `source_url`, `retrieved_at`, `document_version` (hoặc ngày hiệu lực) trong metadata.

**Quy trình làm sạch và kiểm chứng:**
1. Crawl bằng `scripts/fetch_public_pages.py`. Bản thô chứa khoảng 70 dòng menu trước nội dung chính, vì menu của HUIT không nằm trong thẻ `<nav>`.
2. Bỏ những dòng xuất hiện ở ít nhất 10/13 trang (menu và footer dùng chung), sau đó viết lại thành Markdown: tiêu đề mục, danh sách, bảng.
3. Kiểm chứng tự động: đối chiếu **từng dòng và từng ô bảng** của file sạch với bản crawl thô. Mọi nội dung trùng nguyên văn; chỉ tiêu đề và header bảng hai tầng là do tôi đặt khi định dạng lại.
4. Chuẩn hoá Unicode về NFC. Một phần trang gốc dùng dấu tổ hợp (NFD); nếu không chuẩn hoá, cùng một chữ sẽ ra token khác nhau khi embed.
5. **Tách trang quy định theo `audience`** ở mức câu và dòng bảng, không sửa chữ nào. Ví dụ, mục 3d gồm hai câu: câu "chỉ áp dụng đối với… sinh viên" vào file sinh viên, câu "không áp dụng… đối với giảng viên" vào file giảng viên.

**Trang bị loại và mâu thuẫn trong nguồn** (đều là vấn đề thật của trang gốc):

| Trang | Quyết định | Lý do |
|---|---|---|
| `lich-phuc-vu`, `vi-tri-thoi-gian-hoat-dong` | Loại | Cả hai thiếu giờ thứ 2 đến thứ 6 (nhiều khả năng nằm trong ảnh), và ghi giờ thứ 7 mâu thuẫn nhau: 8:00–16:30 so với 7:00–20:00 |
| `cung-cap-khong-gian-tien-ich` | Loại | Ghi "6 quyển / 3 tuần, gia hạn 7 ngày × 3 lần", trái với quy định chính thức (3 tài liệu / 10 ngày) |
| `dich-vu-tham-khao`, `yeu-cau-bo-sung-tai-lieu`, `phat-hanh-giao-trinh` | Loại | Quá mỏng (dưới 500 ký tự) hoặc chỉ là danh sách khoa |
| `luu-hanh-tai-lieu` vs `quy-dinh` | Giữ cả hai, **không** dùng làm gold | Số lần gia hạn của sinh viên: 2 lần (lưu hành) so với 1 lần (quy định) |
| `su-dung-phong-hoc-nhom` vs `quy-dinh` §9b | Giữ, **không** dùng làm gold | Thời điểm nhận kết quả đặt phòng: khung 9g–10g / 14g–15g so với "sau 15 phút" |
| `quy-dinh` §11b | Ghi nhận | Trang gốc dừng ở tiêu đề "b. Xử lý các hành vi vi phạm", không có nội dung |

### Cấu trúc Metadata (Metadata Schema)

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho truy xuất (retrieval)? |
|----------------|------|---------------|-------------------------------|
| `doc_id` | string (= tên file) | `quy-dinh-muon-tra-sinh-vien` | Khoá để `delete_document()` xoá mọi chunk của một file; chunk `file#3` truy vết về đúng file gốc |
| `audience` | enum `student` / `faculty` / `all` | `student` | Lọc theo đối tượng: câu Q1 chỉ trả lời đúng khi lọc `student` (xem mục 3) |
| `category` | enum `noi-quy` / `muon-tra` / `dich-vu` / `huong-dan` | `muon-tra` | Nhóm theo loại văn bản; là ứng viên để lọc hoặc định tuyến câu hỏi (xem đề xuất cho Q3) |
| `department` | string | `library` | Dùng khi mở rộng corpus sang phòng ban khác (đào tạo, tài chính…) |
| `language` | string | `vi` | Lọc ngôn ngữ nếu thêm tài liệu tiếng Anh |
| `source_url`, `retrieved_at`, `document_version` | string / date | `…/luu-hanh-tai-lieu`, `2026-09-19`, `not-stated` | Truy vết và kiểm tra độ mới; hữu ích khi hai trang mâu thuẫn (xem bảng trên) |
| `source_sections` | string (chỉ file tách) | `3, 5, 11a (phần áp dụng cho sinh viên…)` | Truy vết file tách về đúng mục trong trang gốc |
| `chunk_index` | int (do `bench.py` thêm) | `3` | Vị trí chunk trong file, phục vụ debug và trích dẫn |

---

## 2. Thiết kế chiến lược (Strategy Design) — Nhóm (15 điểm)

### Phân tích đường cơ sở (Baseline Analysis)

`ChunkingStrategyComparator().compare(body, chunk_size=500)` trên 3 tài liệu (đã bỏ frontmatter):

| Tài liệu | Chiến lược (Strategy) | Số lượng Chunk | Độ dài trung bình | Giữ được ngữ cảnh không? |
|-----------|----------|-------------|------------|-------------------|
| `quy-dinh-chung` (9 843) | FixedSizeChunker (`fixed_size`) | 20 | 492.1 | Không — cắt ngang từ và câu (trong benchmark có chunk mở đầu bằng "ào hệ thống…") |
| | SentenceChunker (`by_sentences`) | 29 | 337.9 (93–990) | Một phần — danh sách và bảng không có dấu chấm nên bị gộp thành "câu" dài 990 ký tự |
| | RecursiveChunker (`recursive`) | 24 | 408.4 | Phần lớn — cắt theo đoạn và dòng, nhưng không biết chunk thuộc mục nào |
| `luu-hanh-tai-lieu` (3 422) | FixedSizeChunker (`fixed_size`) | 7 | 488.9 | Không — bảng hạn mức bị cắt giữa hàng |
| | SentenceChunker (`by_sentences`) | 5 | 682.2 (409–1245) | Kém — chunk dài tới 1 245 ký tự, vượt xa `chunk_size` |
| | RecursiveChunker (`recursive`) | 10 | 340.4 | Khá — nhưng bảng bị tách khỏi câu dẫn "…theo từng đối tượng như sau:" |
| `huong-dan-su-dung-thu-vien` (5 388) | FixedSizeChunker (`fixed_size`) | 11 | 489.8 | Không |
| | SentenceChunker (`by_sentences`) | 12 | 447.1 (91–1532) | Kém — bảng tầng/tiện ích gộp thành một "câu" 1 532 ký tự |
| | RecursiveChunker (`recursive`) | 14 | 383.2 | Khá |

Nhận xét: `SentenceChunker` không hợp với văn bản quy định. Loại văn bản này chủ yếu gồm danh sách và bảng, không có dấu câu kết thúc, nên độ dài chunk mất kiểm soát. Vì vậy tôi không đưa nó vào benchmark.

### Chiến lược của từng thành viên

> Làm cá nhân nên mỗi khối là một chiến lược. Cả 4 chiến lược đều dùng `chunk_size=500`.

**Chiến lược 1 — FixedSizeChunker có overlap**
- **Loại chiến lược:** FixedSize, `chunk_size=500, overlap=50`
- **Mô tả & lý do chọn:** Đường cơ sở đơn giản nhất: cửa sổ trượt 500 ký tự, chồng nhau 50 ký tự. Phần overlap cho thông tin nằm ở ranh giới hai chunk thêm một cơ hội lọt top-k. Chiến lược này không dùng cấu trúc văn bản.

**Chiến lược 2 — RecursiveChunker**
- **Loại chiến lược:** Recursive, separator `["\n\n", "\n", ". ", " ", ""]`, `chunk_size=500`
- **Mô tả & lý do chọn:** Cắt theo ranh giới lớn trước (đoạn văn), rồi mới tới ranh giới nhỏ (dòng, câu, từ), sau đó gom các mảnh nhỏ liền kề cho tới sát `chunk_size`. Nhờ vậy chunk không cắt ngang câu, và sẽ tốt hơn fixed nếu nội dung trả lời nằm gọn trong một đoạn.

**Chiến lược 3 — HeadingChunker (custom, chunk theo tiêu đề/mục — bắt buộc của L3A)**
- **Loại chiến lược:** custom
- **Mô tả & lý do chọn:** Văn bản quy định được người soạn chia sẵn thành mục (`## 5. Quy định mượn/trả` → `### a. Mượn tài liệu`), mỗi mục là một đơn vị ngữ nghĩa trọn vẹn. Chunker tách trước mỗi dòng heading, mỗi mục thành một chunk. **Mọi chunk bắt đầu bằng breadcrumb** (ví dụ `Quy định mượn/trả tài liệu — Sinh viên, học viên > 5. Quy định mượn/trả tài liệu > a. Mượn tài liệu của thư viện`), nên dù bị cắt nhỏ, chunk vẫn mang theo "thuộc mục nào, áp dụng cho ai". Mục dài quá ngưỡng thì chia tiếp bằng `RecursiveChunker`, và breadcrumb được gắn lại vào từng mảnh con.

**Chiến lược 4 — HeadingChunker v2 (tinh chỉnh sau phân tích lỗi Q4)**
- **Loại chiến lược:** custom, `HeadingChunker(glue_lead_in=True)`
- **Mô tả & lý do chọn:** Phân tích lỗi Q4 (mục 4) cho thấy bước gom của recursive là **gom tham lam theo chiều xuôi**. Câu dẫn "Đối với tài liệu mua, tặng đã cũ không còn lưu hành trên thị trường:" bị gắn vào danh sách *phía trước*, nên mảnh chứa "gấp 5 lần" mất chủ ngữ. Bản v2 giữ mọi đoạn kết thúc bằng `:` đi cùng khối ngay sau nó, rồi mới gom.
- **Code snippet:**
```python
class HeadingChunker:
    HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")

    def chunk(self, text: str) -> list[str]:
        sections, path, body = [], [], []

        def flush():
            content = "\n".join(body).strip()
            if content:  # heading không có thân riêng: tiêu đề của nó sống tiếp trong breadcrumb của mục con
                sections.append((" > ".join(title for _, title in path), content))
            body.clear()

        for line in text.splitlines():
            match = self.HEADING_PATTERN.match(line)
            if match and len(match.group(1)) <= self.max_level:
                flush()
                level = len(match.group(1))
                path[:] = [(l, t) for l, t in path if l < level] + [(level, match.group(2))]
            else:
                body.append(line)
        flush()

        chunks = []
        for heading_path, content in sections:
            prefix = f"{heading_path}\n" if heading_path else ""
            budget = max(self.chunk_size - len(prefix), self.chunk_size // 2)
            if len(content) <= budget:
                pieces = [content]
            elif self.glue_lead_in:                      # v2
                pieces = self._split_keeping_lead_ins(content, budget)
            else:                                        # v1
                pieces = RecursiveChunker(chunk_size=budget).chunk(content)
            chunks.extend(prefix + piece for piece in pieces)
        return chunks

    @staticmethod
    def _split_keeping_lead_ins(content, budget):
        blocks = []
        for block in (b.strip() for b in content.split("\n\n")):
            if not block:
                continue
            if blocks and blocks[-1].endswith(":"):     # câu dẫn đi cùng khối phía sau
                blocks[-1] = f"{blocks[-1]}\n\n{block}"
            else:
                blocks.append(block)
        # ... gom các block lên tới budget như RecursiveChunker; block quá dài thì RecursiveChunker
```

### So Sánh Giữa Các Thành Viên

Điểm truy xuất chấm theo `docs/SCORING.md`: 2đ nếu chunk chứa đáp án ở top-1 và agent trả lời đúng, 1đ nếu chunk có trong top-3 nhưng chưa đủ, 0đ nếu không có. Cột "chấm theo doc_id" là cách chấm ngây thơ, chỉ để đối chiếu.

| Thành viên | Chiến lược (Strategy) | Điểm truy xuất (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Chiến lược 1 | Fixed 500/50 | **6** (chấm theo doc_id: 8) — 69 chunk | Overlap cho thông tin ở ranh giới thêm cơ hội; ở Q1, cửa sổ tình cờ chứa cả câu dẫn lẫn bảng | Cắt ngang từ và câu, chunk khó đọc và không biết thuộc mục nào; thắng Q1 là do vị trí cửa sổ, đổi `chunk_size` là có thể mất |
| Chiến lược 2 | Recursive 500 | **4** (8) — 77 chunk | Chunk đọc được, không cắt ngang câu | Bảng là một đoạn riêng nên bị tách khỏi câu dẫn; chunk bảng chỉ còn `\|` và con số, xếp **#7** ở Q1 |
| Chiến lược 3 | Heading v1 | **5** (8) — 99 chunk | Breadcrumb cho ngữ cảnh mục và đối tượng; Q1 lên #1 với score cao nhất (0.676) | Mục dài bị chia nhỏ khiến mảnh con mất câu dẫn: Q4 rơi xuống **#7** |
| Chiến lược 4 | Heading v2 | **6** (8) — 103 chunk | Sửa lỗi câu dẫn: Q4 từ #7 lên #3, agent trả lời đúng; **chiến lược duy nhất trả lời đúng Q1 kể cả khi không filter**; đưa chunk thủ tục của Q5 lên gần nhất (#5, các chiến lược khác #9–#10) | Q3 vẫn hỏng; nhiều chunk nhất; breadcrumb dài chiếm tới khoảng 1/4 dung lượng chunk; các mục cùng tài liệu có thể chiếm hết top-3 (Q5); khối "câu dẫn + danh sách" dài hơn budget vẫn bị tách (câu dẫn của `#21` nằm ở `#20`) |

**Chiến lược nào tốt nhất cho chủ đề này? Tại sao?**
> **Heading v2.** Trên 5 câu, v2 và fixed hoà nhau 6/10, nhưng v2 thắng ở những điểm quan trọng với văn bản quy định. Thứ nhất, v2 là chiến lược duy nhất trả lời đúng Q1 ở cả ba nhánh A/B, vì breadcrumb "— Sinh viên, học viên > 5. Mượn tài liệu" gắn sẵn đối tượng vào chunk. Thứ hai, mỗi chunk truy vết được về đúng mục trong quy định. Thứ ba, hành vi của v2 bám theo cấu trúc do người soạn đặt ra, còn điểm của fixed ở Q1 phụ thuộc vào việc cửa sổ 500 ký tự tình cờ bao trọn câu dẫn và bảng. Cũng cần nói thẳng: 5 câu hỏi là quá ít để kết luận có ý nghĩa thống kê. Chênh lệch giữa các chiến lược chỉ nằm ở 2 câu (Q1, Q4), và **đổi ranh giới chunk tác động lên kết quả nhiều hơn đổi kích thước chunk.**

---

## 3. Câu hỏi đánh giá & Chất lượng truy xuất (Retrieval Quality) — Nhóm (10 điểm)

### Câu hỏi đánh giá & Câu trả lời chuẩn (nhóm thống nhất)

| # | Câu hỏi (Query) | Câu trả lời chuẩn (Gold Answer) | Chunk nào chứa thông tin? |
|---|-------|-------------------------------|--------------------------|
| 1 | Tôi được mượn tối đa bao nhiêu tài liệu về nhà, trong bao nhiêu ngày? **(chạy với `metadata_filter={"audience": "student"}`)** | 3 tài liệu, 10 ngày (sinh viên, học viên) | `quy-dinh-muon-tra-sinh-vien` §5a (bảng hạn mức); `luu-hanh-tai-lieu` §2 cũng có dòng này |
| 2 | Làm thẻ thư viện mới mất bao nhiêu tiền? | 100.000 đ/thẻ (cấp lại 50.000đ/thẻ, gia hạn 50.000 đồng/năm) | `huong-dan-su-dung-thu-vien` §2 — Lệ phí Thẻ |
| 3 | Trả sách mượn liên thư viện trễ hạn bị phạt bao nhiêu? | 5.000đ/tài liệu/ngày | `muon-lien-thu-vien` — Quy định. Tài liệu gây nhiễu: `luu-hanh-tai-lieu` ghi 1.000đ cho mượn thường |
| 4 | Làm mất sách tiếng Việt cũ, không còn bán trên thị trường thì phải đền thế nào? | Đền tiền gấp 5 lần giá bìa (ngoại văn gấp 3 lần), cộng phí xử lý kỹ thuật; hạn 30 ngày | `quy-dinh-chung` §7b–7e |
| 5 | Đặt phòng học nhóm thế nào và được dùng bao lâu? | Đăng ký tại Quầy thông tin hoặc trực tuyến mục "ĐẶT PHÒNG"; 2 giờ/lượt, gia hạn khi không có người chờ; đến trễ quá 15 phút bị hủy | `su-dung-phong-hoc-nhom`; `quy-dinh-chung` §9 |

Các câu hỏi đa dạng về dạng hỏi: tra theo đối tượng (1), tra số liệu (2), phân biệt tài liệu gây nhiễu (3), hỏi điều kiện (4), hỏi quy trình (5). Mọi gold answer chỉ dựa trên những điểm **mà các trang nguồn ghi nhất quán với nhau**; các chỗ mâu thuẫn trong bảng ở mục 1 đều được tránh.

**Cách chấm hai mức** (`bench.py`): mỗi câu khai báo một chuỗi đặc trưng (regex) phải có trong chunk thì chunk đó mới thực sự trả lời được câu hỏi, ví dụ `Sinh viên, học viên \| 3 \| 10` hay `gấp 5 lần giá tiền ghi trên bìa`. Câu trả lời của agent được kiểm bằng regex và **đọc lại bằng mắt**. Nhờ đọc lại, tôi bắt được hai lỗi của chính máy chấm:
- **Q1 không filter:** agent trả lời "từ 1 đến 10 ngày", lẫn cả hai đối tượng, nhưng vẫn khớp "3" và "10 ngày". Sửa: loại mọi câu trả lời nhắc tới giảng viên hay viên chức.
- **Q5:** ban đầu chỉ kiểm vế thời lượng ("2 giờ"), nên cả 4 chiến lược đều được 2đ, trong khi agent **bịa** vế "đặt thế nào". Sửa: phải nêu đúng kênh đặt phòng (Quầy thông tin hoặc mục "ĐẶT PHÒNG"). Điểm Q5 giảm từ 2 xuống 1 ở cả 4 chiến lược, đúng với quy tắc "câu trả lời thiếu chi tiết" của SCORING.md.

### Tổng hợp chất lượng truy xuất của nhóm

Điểm từng câu (chấm theo nội dung / chấm theo doc_id):

| Chiến lược | Q1 | Q2 | Q3 | Q4 | Q5 | Chấm theo nội dung | Chấm theo doc_id |
|---|---|---|---|---|---|---|---|
| fixed | 2/2 | 2/2 | 0/0 | 1/2 | 1/2 | **6/10** | 8/10 |
| recursive | 0/2 | 2/2 | 0/0 | 1/2 | 1/2 | **4/10** | 8/10 |
| heading | 2/2 | 2/2 | 0/0 | 0/2 | 1/2 | **5/10** | 8/10 |
| heading_v2 | 2/2 | 2/2 | 0/0 | 1/2 | 1/2 | **6/10** | 8/10 |

| # | Câu hỏi | Chiến lược tốt nhất cho câu này | Có chunk liên quan trong top-3? | Ghi chú |
|---|---------|-------------------------------|-------------------------------|---------|
| 1 | Mượn tối đa bao nhiêu, bao lâu? | fixed, heading, heading_v2 (2đ) | Có (3/4 chiến lược, đều ở #1) | recursive: chunk bảng xếp #7 vì bị tách khỏi câu dẫn |
| 2 | Phí làm thẻ mới | Cả 4 (2đ) | Có, đều ở #1 | Từ khoá "thẻ" và "lệ phí" khớp trực tiếp |
| 3 | Phạt trễ hạn mượn liên thư viện | Không chiến lược nào | **Không** (0/4) | Chunk "phạt 1.000đ" của mượn thường thắng; chunk đúng xếp #4 (0.775 so với 0.837). Xem failure case |
| 4 | Đền sách cũ không còn bán | fixed, recursive, heading_v2 (1đ) | Có nhưng không ở top-1 | Top-1 là đoạn "tài liệu **còn** lưu hành": cùng mục 7b nhưng sai trường hợp |
| 5 | Đặt phòng học nhóm | Cả 4 (1đ); heading/heading_v2 gần nhất | Có một vế: "2 giờ/lượt" ở #1 | Chunk chứa bước đăng ký nằm ngoài top-3 (#5 ở heading, #9–#10 ở fixed/recursive); agent **bịa** "liên hệ bộ phận quản lý tài nguyên học tập". Xem failure case 3 |

**Lọc bằng metadata có giúp ích không? Ở câu hỏi nào?**
> **Có, và giúp rất rõ ở Q1.** Tôi chạy Q1 với 3 nhánh trên cả 4 chiến lược (ô ghi thứ hạng của chunk chứa đáp án và kết quả của agent):
>
> | Chiến lược | Không filter | `audience=student` | `audience ∈ {student, all}` |
> |---|---|---|---|
> | fixed | #3, lẫn giảng viên ("1–10 ngày") | **#1, đúng** | #2, đúng |
> | recursive | không có, sai ("2 tài liệu, 20 ngày") | không có, sai | không có, sai ("2 tài liệu, 20 ngày") |
> | heading | #2, lẫn giảng viên ("180 ngày") | **#1, đúng** | #2, sai ("2 tài liệu, 20 ngày") |
> | heading_v2 | #3, đúng | **#1, đúng** | #3, đúng |
>
> Khi không filter, top-1 luôn là chunk *không* dành cho sinh viên: quy định mượn liên thư viện ("2 tài liệu/1 lần mượn; thời hạn 20 ngày") hoặc hạn mức giảng viên, và agent trả lời theo đúng chunk đó. Lọc cứng `student` đưa chunk đúng lên #1 ở 3/4 chiến lược. Recursive vẫn hỏng, nhưng lỗi nằm ở cách chunk chứ không phải ở filter. Lọc mềm `{student, all}` kéo các tài liệu dùng chung quay lại chiếm top, nên kém hơn lọc cứng.
>
> Đánh đổi: lọc cứng `student` loại hết 6 tài liệu `all`. Trong khi đó đáp án của Q2, Q3, Q5 lại nằm trong tài liệu `all`, nên nếu áp filter cho mọi câu thì mất luôn các câu đó. Filter phải bật **theo ý định của câu hỏi**, không bật mặc định. Filter cũng chỉ có tác dụng vì tài liệu đã được **tách theo `audience` ngay ở tầng dữ liệu**: nếu giữ trang quy định là một file `audience: all` thì filter không lọc được gì.

---

## 4. Thuyết trình (Demo) & Bài học nhóm — Nhóm (5 điểm)

**Những phân tích (insights) hay nhất nhóm sẽ trình bày:**
> 1. **Chấm theo `doc_id` thổi phồng kết quả:** cả 4 chiến lược đều được 8/10 nếu chỉ chấm theo tài liệu, nhưng chấm theo nội dung và đọc câu trả lời chỉ còn 4–6/10. Q4 ở heading v1 là ví dụ rõ nhất: đúng tài liệu ở #1 (2đ nếu chấm theo tài liệu), nhưng không chunk nào trong top-3 chứa đáp án (0đ).
> 2. **Metadata chỉ có tác dụng khi dữ liệu được tách đúng chiều lọc.** A/B ở Q1 cho thấy lọc cứng `student` biến câu trả lời "2 tài liệu, 20 ngày" (sai) thành "3 tài liệu, 10 ngày" (đúng).
> 3. **Tinh chỉnh chunker dựa trên phân tích lỗi:** một lỗi nhỏ trong bước gom (câu dẫn bị gắn vào danh sách phía trước) đẩy đáp án Q4 xuống #7. Sửa bằng `glue_lead_in` đưa nó lên #3, và agent chuyển từ trả lời sai sang trả lời đúng.

**Phân tích lỗi (Failure Analysis):**

*Failure case 1 — Q3 "Trả sách mượn liên thư viện trễ hạn bị phạt bao nhiêu?" (hỏng ở cả 4 chiến lược)*
- **Hỏng ở đâu:** top-3 toàn là chunk "xử lý quá hạn" của mượn thường. Ở heading v1, chunk `muon-lien-thu-vien` chứa "Phí trễ hạn: 5.000đ" xếp **#4** (0.775, so với top-1 là 0.837). Agent trả lời **tự tin nhưng sai**: "bị phạt 1.000 đồng/1 tài liệu/ngày [1]", kèm trích dẫn.
- **Vì sao:**
  - Cosine đo *độ giống chủ đề*, không đo *mật độ thông tin trả lời được*. Cụm "trễ hạn bị phạt" khớp mạnh với 3–4 chunk về xử lý quá hạn, trong khi chunk mượn liên thư viện gồm 7 quy định khác nhau nên tín hiệu "phạt" bị pha loãng.
  - Model embedding nhỏ (MiniLM, 384 chiều) không coi cụm "liên thư viện" là yếu tố quyết định.
  - Agent trung thành với ngữ cảnh: grounding tốt không có nghĩa là đúng, lỗi retrieval truyền thẳng sang câu trả lời.
- **Đề xuất sửa:**
  - (a) Hybrid search: BM25 và vector kết hợp, để từ khoá hiếm "liên thư viện" được tính điểm.
  - (b) Thêm metadata `service` (`muon-thuong` / `lien-thu-vien`) và định tuyến câu hỏi chứa "liên thư viện" sang filter tương ứng.
  - (c) Lấy top-10 rồi rerank bằng cross-encoder.
  - (d) Tách mỗi gạch đầu dòng quy định thành chunk nhỏ hơn, để chunk "Phí trễ hạn: 5.000đ" không bị pha loãng.

*Failure case 2 — Q1 với RecursiveChunker (hỏng cả khi đã filter)*
- **Hỏng ở đâu:** chunk chứa bảng hạn mức xếp **#7/10** (0.443) dù đã lọc `student`. Agent trả lời "không nêu rõ số ngày".
- **Vì sao:** trong Markdown, bảng là một đoạn riêng (ngăn bởi dòng trống), nên recursive cắt ở `\n\n` và tách bảng khỏi câu dẫn "Số lượng… và thời hạn mượn tài liệu về nhà được quy định theo từng đối tượng như sau:". Chunk bảng chỉ còn `| Đối tượng | Số lượng tài liệu | Số ngày |` và các con số, gần như không mang ngữ nghĩa để khớp với câu hỏi.
- **Đề xuất sửa:** giữ câu dẫn đi cùng bảng (chính là `glue_lead_in` của heading v2); hoặc khi chunk, chuyển mỗi dòng bảng thành câu ("Sinh viên, học viên: mượn 3 tài liệu, 10 ngày") trong khi file nguồn giữ nguyên.

*Failure case 3 — Q5 "Đặt phòng học nhóm thế nào và được dùng bao lâu?" (câu hỏi hai vế; agent bịa một vế)*
- **Hỏng ở đâu:** top-3 chỉ chứa phần mô tả phòng (số phòng, chỗ ngồi, "02 giờ/lượt"). Chunk chứa bước đăng ký ("Bước 1: Đăng ký… tại Quầy thông tin… chọn “ĐẶT PHÒNG”") xếp **#5** ở heading và **#9–#10** ở fixed/recursive. Agent trả lời đúng thời lượng nhưng **tự bịa** cách đặt: "liên hệ với bộ phận quản lý tài nguyên học tập", dù prompt yêu cầu chỉ dùng ngữ cảnh.
- **Vì sao:**
  - Một vector không biểu diễn đủ câu hỏi hai vế: cụm "phòng học nhóm" lấn át, nên các chunk mô tả phòng thắng chunk thủ tục.
  - Với heading chunker, 3 mục "Phòng học nhóm / Phòng thuyết trình / Phòng hội thảo" của **cùng một tài liệu** chiếm cả 3 slot, đúng hiện tượng "chunker theo heading lấy trọn top-3 từ tài liệu gold mà không chunk nào chứa trọn đáp án".
  - Model 3B không tuân thủ triệt để quy tắc chống bịa.
- **Đề xuất sửa:**
  - (a) Tách câu hỏi nhiều vế thành các truy vấn con rồi gộp ngữ cảnh.
  - (b) Đa dạng hoá top-k (MMR), không để một tài liệu chiếm hết các slot.
  - (c) Kiểm hậu kỳ: mỗi ý trong câu trả lời phải trỏ tới một chunk có chứa nội dung đó; ý nào không có nguồn thì loại. Hoặc dùng LLM lớn hơn.

**Bài học rút ra khi so sánh trong nhóm:**
> Cùng dữ liệu, cùng embedder, chỉ đổi chunker mà Q1 dao động từ 0 đến 2 điểm, còn chunk đáp án của Q4 dao động từ #2 (fixed, recursive) đến #7 (heading v1). **Ranh giới chunk quan trọng hơn kích thước chunk**: cả 4 chiến lược đều quanh 500 ký tự, nhưng chiến lược nào để câu dẫn đi cùng dữ liệu nó dẫn thì thắng. Tôi cũng thấy phải chấm ở mức nội dung và đọc câu trả lời bằng mắt. Nếu chỉ nhìn `doc_id`, tôi đã kết luận sai rằng 4 chiến lược ngang nhau; còn nếu chỉ tin regex, tôi đã chấm 2đ cho một câu trả lời bịa ở Q5.

**Nếu làm lại, nhóm sẽ thay đổi gì trong chiến lược dữ liệu (data strategy)?**
> 1. Thêm metadata `service` (mượn thường / liên thư viện / phòng học) ngay khi làm sạch để có thể lọc hoặc định tuyến Q3.
> 2. Chuyển bảng thành câu ở tầng chunk, vì bảng Markdown gần như vô nghĩa với embedding.
> 3. Viết nhiều hơn 5 câu hỏi (khoảng 15–20), để chênh lệch giữa các chiến lược có ý nghĩa; so thêm một embedder đa ngữ mạnh hơn MiniLM.
> 4. Ghi rõ trong metadata khi hai trang nguồn mâu thuẫn (ví dụ trường `conflicts_with`), để agent cảnh báo người dùng thay vì chọn đại một con số.

---

## Tự Đánh Giá (Phần Nhóm)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Lựa chọn tài liệu (Document Set Quality) | / 10 |
| Thiết kế chiến lược (Strategy Design) | / 15 |
| Chất lượng truy xuất (Retrieval Quality) | / 10 |
| Thuyết trình (Demo) | / 5 |
| **Tổng phần nhóm** | **/ 40** |
