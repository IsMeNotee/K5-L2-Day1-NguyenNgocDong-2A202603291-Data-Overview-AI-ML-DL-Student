# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`= 468 ; `class_name`="cab" ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`="ImageNet-1K"
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
Record này nói về cả bức ảnh. Vì đây là tác vụ phân loại ảnh (image_classification), gán một nhãn chung cho toàn bộ khung hình và trong JSON không có tọa độ bbox hay polygon cho từng vật thể riêng.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
Danh sách tên lớp đi kèm với taxonomy/checkpoint (ở đây là ImageNet-1K), không phải do mô hình tự nghĩ ra.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
Cần ghi cả 3 để truy vết chính xác ngữ nghĩa nhãn và tránh nhầm lẫn giữa các bộ dữ liệu khác nhau.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
Guideline cần quy định rõ tiêu chí ưu tiên chọn chủ đề chính (ví dụ: chọn vật chiếm diện tích lớn nhất hoặc nằm ở trung tâm ảnh).
6. Vì sao không được lấy `score` của model làm ground truth?
Vì score chỉ là mức độ tự tin (xác suất) toán học của mô hình dự đoán (có thể đoán sai), còn ground truth phải là sự thật do con người kiểm chứng theo guideline.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`="person" ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
Tính từ gốc (0, 0) ở góc trên-trái ảnh, góc trên-trái của hộp này nằm tại pixel (385.33, 69.24), góc dưới-phải nằm tại pixel (498.92, 348.92), chiều rộng hộp là 113.58 pixel và chiều cao là 279.68 pixel.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
Ở ngưỡng 0.20 có 17 prediction. Ở ngưỡng 0.60 có 6 prediction.
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
Khi hạ ngưỡng, mô hình sẽ sinh ra rất nhiều hộp rác (dự đoán sai - false positive) và hộp trùng lặp. Người reviewer sẽ phải tốn thêm thời gian rà soát để xóa bỏ các hộp sai này và chỉnh lại viền các hộp không chuẩn.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
Quy tắc: Hộp giới hạn (bounding box) phải bao sát mép toàn bộ phần nhìn thấy được của vật thể (cách rìa tối đa 1-2 pixel), không cắt lẹm vào thân vật thể và không chứa quá nhiều phần nền (background) dư thừa xung quanh.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
Trên ảnh có một số cái bát (bowl) trên kệ bị vật khác che khuất một nửa. Tôi sẽ đưa vào quy tắc trong guideline (vd: "Vật bị che khuất dưới 70% thì vẫn gán nhãn phần nhìn thấy"). Chỉ khi vật bị che quá nhiều, mờ nhòe đến mức không thể nhận dạng chắc chắn đó là cái bát hay cái cốc thì mới cần escalation (hỏi Lab Coach) để không tự ý đoán mò làm sai dữ liệu.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`="kitchen-001" ; `class_name`="person" ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3-5 điểm đầu của `polygon_xy`=[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0]]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
Trong khi box chỉ là một khung hình chữ nhật chứa cả phần nền thừa ở các góc, polygon (đa giác) cung cấp một chuỗi các điểm tọa độ bám sát theo đường viền thực tế của vật thể. Điều này giúp biểu diễn chính xác hình dáng hình học, độ cong và diện tích thực của đối tượng, tách biệt hoàn toàn vật thể khỏi background.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
`instance_id` (như `kitchen-001`, `kitchen-002`) dùng để phân biệt từng cá thể đối tượng độc lập trong phạm vi một bức ảnh (ví dụ 3 cái cốc thì có 3 ID riêng). Nó **không phải class ID** vì không đại diện cho danh mục/nhãn lớp (như "cốc" hay "người"). Nó **không phải tracking ID** vì nó chỉ đánh số thứ tự cục bộ trong một ảnh tĩnh, chứ không dùng để theo dõi cùng một vật thể di chuyển qua nhiều khung hình liền kề (video).
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
Quy tắc: Các đỉnh của đa giác (polygon) phải được đặt bám sát mép ranh giới thực tế của vật thể. Phải đặt nhiều điểm chốt hơn ở các đoạn viền bo cong hoặc gấp khúc để ôm sát viền; tuyệt đối không để mặt nạ (mask) tràn ra nền xung quanh và không để các cạnh của đa giác tự cắt chéo nhau.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
Trên ảnh `kitchen`, tại khu vực các đồ vật (chai, lọ, bát) đặt sát nhau trên mặt bàn, đường biên viền giữa chúng bị chồng lấp và tối màu. Nếu ranh giới vẫn có thể phân biệt được, tôi sẽ áp dụng **guideline** (vẽ mask tách riêng biệt từng vật, chỉ bao lấy phần diện tích nhìn thấy của từng đối tượng). Nếu vùng biên tối hoặc mờ nhòe đến mức không thể xác định vạch ranh giới thực sự nằm ở pixel nào, tôi sẽ dùng **escalation** (hỏi Lab Coach) để thống nhất quy tắc xử lý, tránh tự ý vẽ bừa làm sai lệch nhãn huấn luyện.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | 1 nhãn lớp cho toàn bộ ảnh| Ảnh `traffic` nhiều vật nhưng bị ép gán 1 lớp duy nhất | Chọn 1 lớp theo quy tắc ưu tiên chủ đề | Kiểm tra nhãn có đúng ưu tiên không |
| Phát hiện vật thể | 1 nhãn lớp + 1 hộp chữ nhật (box) cho mỗi vật | Ảnh `kitchen` model bỏ sót vật nhỏ hoặc vật bị che khuất | Vẽ box bao sát mép phần nhìn thấy của vật | Tìm lỗi sót box, thừa box rác hoặc lẹm biên |
| Instance segmentation | 1 ID riêng + 1 nhãn lớp + 1 đa giác (polygon) | Ảnh `kitchen` vùng viền các bát/lọ chạm nhau bị lem mép | Vẽ polygon bám sát viền thực của từng vật riêng biệt | Kiểm tra lỗi polygon lem ra nền hoặc dính viền nhau |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
Chỉ sử dụng ba bức ảnh COCO công khai có sẵn trong bài lab, tuyệt đối không tải lên ảnh cá nhân hay dữ liệu nội bộ và không ghi thông tin cá nhân vào báo cáo.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Tôi sẽ lập tức dừng thao tác, không lưu lại dữ liệu đó và báo ngay cho Lab Coach / Giảng viên phụ trách lớp.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
