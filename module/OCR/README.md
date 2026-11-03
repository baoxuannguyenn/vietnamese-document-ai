# Module OCR (Optical Character Recognition)

Module đảm nhiệm chức năng **Nhận diện ký tự quang học** - chuyển đổi hình ảnh tài liệu thành văn bản số hóa tiếng Việt.

---

## 🎯 Module này dùng để làm gì?

Tài liệu hành chính thường được lưu dưới dạng file PDF scan hoặc ảnh chụp (văn bản giấy quét qua máy scan). Máy tính không thể trực tiếp đọc, tìm kiếm hoặc sao chép văn bản từ những hình ảnh này.

**Module OCR sinh ra để giải quyết vấn đề đó:**
Nó "nhìn" vào các trang tài liệu, tìm xem chữ nằm ở đâu trên trang giấy, đọc từng chữ tiếng Việt (kể cả các dấu thanh phức tạp) và sắp xếp chúng lại theo đúng thứ tự đọc tự nhiên của con người (từ trên xuống dưới, từ trái sang phải).

---

## 🔄 Quy trình hoạt động của Module

Module OCR hoạt động theo 3 bước tuần tự:

1. **Phát hiện vị trí chữ (Word Detection - YOLO):**
   * Quét toàn bộ trang tài liệu và xác định tọa độ khung chữ nhật (bounding box) của từng từ riêng lẻ, tránh nhận diện nhầm các vết bẩn, dấu mộc hay khoảng trắng.

2. **Nhận diện nội dung chữ (Text Recognition - VietOCR):**
   * Cắt (crop) từng từ vừa tìm được và đưa vào mô hình nhận diện để chuyển thành chữ viết tiếng Việt có dấu chuẩn xác.

3. **Sắp xếp thứ tự đọc (Box Sorter):**
   * Tự động tính toán tọa độ để gom các từ nằm cùng một hàng ngang thành một dòng, và nối các dòng từ đầu trang đến cuối trang theo đúng cấu trúc đoạn văn bản hành chính.

---

## 📥 Đầu vào & 📤 Đầu ra của Module

* **Đầu vào (Input):** File PDF (một hoặc nhiều trang), file ảnh (`.jpg`, `.png`).
* **Đầu ra (Output):** Toàn văn nội dung tài liệu dạng chuỗi ký tự (text), danh sách các dòng văn bản và tọa độ chi tiết của từng từ (dưới dạng file `.json`) để làm nguyên liệu cho Module IE xử lý tiếp theo.
