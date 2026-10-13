# CS-01 — Kiểm tra khả năng can thiệp ASR: token ngữ cảnh, giới hạn độ dài, điểm giải mã

**Đề tài:** Nghiên cứu phương pháp ưu tiên thuật ngữ theo ngữ cảnh cho nhận dạng lời nói chuyển mã Việt–Anh (code-switching)  
**Mốc công việc:** CS-01 · P1 · Hạn gốc N2  
**Ngày lập báo cáo:** 13/10/2026  
**Trạng thái:** Hoàn thành khảo sát khả thi **trên Windows/CTranslate2**; **chưa nghiệm thu được nội dung “trên thiết bị/QNN”**.

## 1. Mục tiêu và điều kiện bàn giao

Khảo sát liệu pipeline ASR có thể (i) nhận đầu vào ngữ cảnh chứa thuật ngữ, (ii) xử lý danh sách ngữ cảnh ở các độ dài khác nhau và (iii) cho phép quan sát hoặc thay đổi điểm giải mã/token. Bàn giao một thử nghiệm khả thi và bảng xác định rõ tính năng **đã thử**, **có trong giao diện API**, **chưa chứng minh** hoặc **ngoài phạm vi**. Tuyệt đối không suy luận khả năng của QNN từ kết quả Python.

## 2. Môi trường và dữ liệu

| Thành phần | Cấu hình/ghi nhận |
|---|---|
| Máy thử nghiệm | Windows, PowerShell; thư mục dự án `G:\ASR` |
| Python | 3.10.11, môi trường ảo `.venv` |
| Runtime thực tế | `faster-whisper` sử dụng **CTranslate2 4.8.2** |
| Model | Whisper `small` |
| Thiết bị tính toán | CPU, `compute_type="int8"` theo script thí nghiệm |
| Audio | 50 file từ ViMedCSS được xuất ra `data/audio/` |
| Metadata | `data/metadata.csv`: 50 dòng; các cột `audio`, `text`, `terms`, `duration_seconds`, `topic` |
| Từ vựng context | 12 thuật ngữ duy nhất trong metadata 50 dòng |
| Quy mô T2 | 20 audio × 4 điều kiện = **80 lượt nhận dạng**, 0 lỗi xử lý được báo cáo |
| Qualcomm/QNN | Không có thiết bị Qualcomm và không chạy QNN |

**Lưu ý dữ liệu:** ViMedCSS có thuật ngữ y khoa và nhiều phát ngôn tiếng Việt xen thuật ngữ tiếng Anh. Đây là tập thử nhanh cho tính khả thi, chưa đủ để đại diện mọi kiểu chuyển mã Việt–Anh. Một số thuật ngữ metadata dùng chính tả khác transcript (`testosteron`/`testosterone`); cần quy tắc chuẩn hóa riêng khi đo độ đúng thuật ngữ.

## 3. Thiết kế thực nghiệm

### 3.1. T0/T1 — Baseline và ngữ cảnh theo từng mẫu

- **T0:** giải mã audio không truyền `initial_prompt`.
- **T1:** truyền danh sách `terms` từ metadata của **chính audio đang nhận dạng**, ghép thành `initial_prompt`.
- Các cấu hình trong script ban đầu giữ model Whisper small, ngôn ngữ `vi`, `beam_size=5`, không sử dụng VAD và không lấy ngữ cảnh từ các đoạn trước.
- Thí nghiệm là **oracle context**: dùng danh sách thuật ngữ đúng của mẫu để kiểm tra cơ chế, **chưa phải** thuật toán tự chọn thuật ngữ theo ngữ cảnh tại thời điểm sử dụng.

**Quan sát thủ công từ kết quả người thực hiện cung cấp:** Trong một số câu, prompt giúp đầu ra hiển thị `reductase`, `methionin`, `estrogen` thay cho biến thể sai; những câu khác ít thay đổi hoặc tệ hơn, đôi khi sinh văn bản không liên quan. Chưa có thống kê định lượng Term Recall cho toàn bộ T0/T1, nên không báo cáo mức tăng phần trăm.

### 3.2. T2 — Thử các mức độ dài danh sách ngữ cảnh

Mỗi audio trong tập con 20 file chạy ở bốn cấu hình. Thuật ngữ liên quan đến mẫu được ưu tiên đặt đầu danh sách; nếu thiếu thì bổ sung từ **từ vựng toàn bộ metadata**, tạo ra cả thuật ngữ gây nhiễu. Vì vậy thí nghiệm thay đổi **số lượng và thành phần** context cùng lúc, không cô lập được riêng hiệu ứng độ dài prompt.

| Điều kiện | Số thuật ngữ đặt mục tiêu | Số thuật ngữ dùng thực tế | Số lượt |
|---|---:|---:|---:|
| `T0_baseline` | 0 | 0 | 20 |
| `T2_3_terms` | 3 | 3 | 20 |
| `T2_10_terms` | 10 | 10 | 20 |
| `T2_30_terms` | 30 | **12** | 20 |

**Giải thích:** Cấu hình tên `T2_30_terms` **không** đạt 30 thuật ngữ vì metadata chỉ có 12 thuật ngữ duy nhất. Kết quả này phải được hiểu là điều kiện **12 thuật ngữ thực tế**, không phải “giới hạn 12 thuật ngữ của model”. Chưa kiểm tra giới hạn tối đa của token prompt; số thuật ngữ và số **token tokenizer** là hai đại lượng khác nhau.

### 3.3. T3 — Khảo sát API điểm giải mã

Đã kiểm tra docstring của `ctranslate2.models.Whisper.generate` trong CTranslate2 **4.8.2**. Giao diện thể hiện:

- `return_scores: bool = False`: có tham số yêu cầu kết quả mang điểm số.
- `return_logits_vocab: bool = False`: theo docstring, có tham số yêu cầu trả về **log probabilities** của vocabulary; tên tham số không đủ để khẳng định trả logits thô.
- `suppress_tokens`, `repetition_penalty`, `length_penalty`, `no_repeat_ngram_size`: các cách ràng buộc/điều chỉnh hành vi giải mã có sẵn.
- **Không thấy tham số callback/custom logits processor** trong chữ ký `generate` được công bố qua docstring đã kiểm tra.

**Mức độ bằng chứng:** Đã xác nhận **sự hiện diện của các tham số API**, **chưa chạy** thí nghiệm bật `return_scores`/`return_logits_vocab` để xác nhận nội dung kết quả ở runtime, và **chưa thực hiện** thí nghiệm cộng bias vào điểm token *trước khi chọn token*. Không có cơ sở kết luận “CTranslate2 tuyệt đối không thể tùy biến”; chỉ có thể nói **chưa thấy điểm móc tùy chỉnh token score qua API Python công khai được khảo sát**.

## 4. Kết quả định lượng T2

Nguồn số liệu: bản in `results_cs01/t2_summary.csv` người thực hiện cung cấp. WER/CER là tỷ lệ lỗi so với transcript trong `metadata.csv`; số liệu cần được diễn giải thận trọng vì chưa kiểm định chất lượng tham chiếu độc lập.

| Điều kiện | WER ↓ | CER ↓ | Thời gian trung bình/audio ↓ |
|---|---:|---:|---:|
| T0: 0 thuật ngữ | **35,19%** | **23,50%** | 3,838 s |
| T2: 3 thuật ngữ | 36,35% | 24,55% | 3,693 s |
| T2: 10 thuật ngữ | 41,35% | 29,05% | 3,949 s |
| T2: 12 thuật ngữ thực tế (nhãn script: `T2_30_terms`) | 40,19% | 39,20% | 4,359 s |

So với T0, WER ở cấu hình 3/10/12 thuật ngữ tăng xấp xỉ **1,15 / 6,15 / 5,00 điểm phần trăm**. Kết quả T2 trên 20 file **không cho thấy prompt dài hơn cải thiện chất lượng chung**. Một số từ khóa được nhận đúng hơn, nhưng điều đó chưa đồng nghĩa với WER toàn câu giảm. Thời gian thay đổi không đơn điệu; chưa đo lặp để xác định mức biến động do môi trường chạy.

**Không suy ra:** (a) ngữ cảnh luôn làm ASR kém hơn, (b) 12 là giới hạn context của model, (c) hiệu quả chọn thuật ngữ tự động, (d) mức cải thiện Term Recall, (e) khả năng/chất lượng của QNN.

## 5. Ma trận khả năng can thiệp

| Khả năng | Bằng chứng hiện tại | Kết luận đúng phạm vi |
|---|---|---|
| Context dạng văn bản `initial_prompt` | T0/T1/T2 chạy và transcript thay đổi | **Đã thử trên Windows/CTranslate2**; tác động có thật nhưng không ổn định |
| `hotwords` | Xuất hiện trong chữ ký `WhisperModel.transcribe` | **Có trong API**; chưa chạy thử độc lập |
| Danh sách 3, 10, 12 thuật ngữ | 20 mẫu/cấu hình, có đầu ra và WER/CER | **Đã thử**; hiệu quả không tăng đều |
| Danh sách **30** thuật ngữ | Thực tế mới có 12 từ | **Chưa thử** |
| Giới hạn **token đầu vào/prompt** | Chưa đo số token tokenizer, ngưỡng cắt và lỗi quá dài | **Chưa xác định** |
| `max_new_tokens`/`max_length` | Có trong API tương ứng | Là giới hạn **đầu ra**, không thay thế đo giới hạn context đầu vào |
| Điểm giải mã đầu ra | `return_scores`, `return_logits_vocab` xuất hiện trong docstring | **API có tham số**, chưa thực thi bài kiểm chứng giá trị trả về |
| Tùy chỉnh token score trước khi chọn token | Không thấy hook trong giao diện đã kiểm tra | **Chưa xác minh hỗ trợ qua API công khai**; chưa thực hiện bias trực tiếp |
| ASR trên Qualcomm/QNN | Không có môi trường chạy | **Ngoài phạm vi thử nghiệm**; tuyệt đối không suy diễn từ Python |

## 6. Vướng mắc và cách xử lý

| Vướng mắc | Tác động | Đã xử lý / bước cần bổ sung |
|---|---|---|
| Chưa có Python 3.10 ở giai đoạn đầu | Không tạo được `.venv` | Đã cấu hình môi trường Python 3.10.11 |
| Tải ViMedCSS ban đầu ra **0 audio** | Không có dữ liệu để chạy | Phát hiện `audio` là dict với `bytes`, `path`, không phải `array`; sửa đọc `bytes` bằng `soundfile` |
| Tải từ Hugging Face có thể chờ lâu/cảnh báo chưa đăng nhập | Chậm hoặc khó theo dõi tiến độ | Sử dụng tải streaming; đã xuất thành công 50 audio; có thể cấu hình token HF khi cần |
| Metadata chỉ có **12** thuật ngữ duy nhất | Không thể thử đúng 30 thuật ngữ | Ghi nhận cấu hình là **12**; muốn thử 30 phải bổ sung từ vựng và chạy lại |
| Prompt bổ sung thuật ngữ gây nhiễu | Khó tách hiệu ứng **độ dài** và **mức liên quan** | Nên thiết kế hai nhánh: danh sách liên quan và danh sách trộn nhiễu với số lượng bằng nhau |
| Prompt oracle lấy từ chính nhãn audio | Có lợi thế thông tin không có sẵn trong triển khai thật | Đánh dấu rõ là thí nghiệm năng lực, không phải hệ thống ưu tiên thuật ngữ hoàn chỉnh |
| WER/CER khá cao, một số transcript bị cắt/sai từ | Rủi ro hiểu nhầm chất lượng toàn câu | Nghe kiểm tra mẫu, chuẩn hóa chính tả; bổ sung Term Recall/Term Accuracy có quy tắc so khớp rõ ràng |
| `inspect.signature()` không đọc được native method | Không tự xuất chữ ký hàm | Đọc docstring `Whisper.generate` trực tiếp, đã có thông tin tham số |
| Không có callback token-bias trong API đã khảo sát | Không thể chứng minh bias điểm trực tiếp bằng script cấp cao | Cần khảo sát API thấp hơn, sửa decoder hoặc runtime hỗ trợ logits processor để làm T3 can thiệp thực sự |
| Không có thiết bị Qualcomm/QNN | Không đáp ứng đầy đủ chữ **“trên thiết bị”** nếu yêu cầu thiết bị Qualcomm | Công bố giới hạn phạm vi; chỉ chốt PoC Windows, chưa xác nhận QNN |

## 7. Kết luận và quyết định nghiệm thu

**Đạt đối với PoC Windows:** Đã dựng pipeline ASR với ViMedCSS; chạy baseline, truyền context qua `initial_prompt`; đo 20 audio trên các cấu hình 0, 3, 10, **12** thuật ngữ; khảo sát giao diện giải mã CTranslate2 4.8.2 và định vị rõ các điểm chưa chứng minh.

**Chưa đạt nếu tiêu chí bắt buộc là kiểm thử thiết bị/QNN hoặc trực tiếp cộng điểm token:** Không có phép chạy trên Qualcomm/QNN; chưa có phép thay logits trước chọn token; chưa đo trần token của prompt. Do đó, trạng thái khuyến nghị trên bảng theo dõi là:

> **CS-01 — Hoàn thành đánh giá khả thi trên Windows (có giới hạn); chưa xác minh QNN và can thiệp trực tiếp token score.**

### Đề xuất hành động sau CS-01

1. Nếu chỉ cần **bàn giao PoC/khảo sát API**, khóa kết quả hiện tại, gắn `report.md` và CSV cùng nhãn phạm vi Windows; không cần chạy lại ngay.
2. Nếu cần xác minh **giới hạn độ dài**, bổ sung tối thiểu 30 thuật ngữ khác nhau; đo cả **số token thực tế** sau tokenizer và kiểm tra prompt có bị cắt hay không; cố định danh sách nhiễu để so sánh công bằng.
3. Nếu cần đánh giá mục tiêu nghiên cứu **ưu tiên thuật ngữ**, thêm Term Recall/Precision theo quy tắc chuẩn hóa biến thể chính tả, cùng số ca prompt làm mất thuật ngữ đúng hoặc sinh thuật ngữ không có trong audio.
4. Nếu bắt buộc phải có **score bias trực tiếp**, tách thành việc kỹ thuật riêng để khảo sát hoặc thay decoder; không gọi `initial_prompt` là token-score bias.
5. Chỉ cập nhật mục QNN khi có bản model/runtime và kết quả thực nghiệm **trên QNN thật**.

## 8. Tệp minh chứng và khả năng tái lập

Các đường dẫn dưới đây là đường dẫn **trên máy Windows của người thực hiện**, không phải tệp đã được đính kèm vào báo cáo này:

```text
G:\ASR\
├── data\
│   ├── audio\                 # 50 WAV
│   └── metadata.csv
├── cs01_experiment.py         # T0/T1 (nếu lưu tên như hướng dẫn)
├── cs01_t2_t3.py             # T2 + khảo sát API T3
└── results_cs01\
    ├── predictions.csv        # T0/T1 (nếu có)
    ├── t2_predictions.csv    # 80 dòng T2
    ├── t2_summary.csv        # tổng hợp T2
    └── t3_api_inspection.txt # docstring/API T3
```

Lệnh tái chạy T2 (sau khi kích hoạt `.venv` tại `G:\ASR`):

```powershell
python cs01_t2_t3.py --limit 20
```

**Lưu ý tái lập:** Script T2 ban đầu dùng từ vựng suy ra từ `metadata.csv`; kết quả sẽ thay đổi nếu metadata, thứ tự dòng, model tải về hoặc môi trường tính toán thay đổi. Nên lưu phiên bản script và CSV gốc cùng báo cáo, và bổ sung thông số phiên bản `faster-whisper`, checksum dữ liệu nếu đây là hồ sơ nghiệm thu chính thức.
