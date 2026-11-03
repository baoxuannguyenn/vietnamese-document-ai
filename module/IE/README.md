# Module IE (Information Extraction)

Module đảm nhiệm chức năng **Trích xuất thông tin hành chính có cấu trúc** từ nội dung văn bản pháp luật Việt Nam.

---

## 🎯 Module này dùng để làm gì?

Sau khi qua Module OCR, ta chỉ mới có một đoạn văn bản thô dài hàng nghìn chữ. Để đưa vào cơ sở dữ liệu hoặc hệ thống quản lý, ta không thể lưu cả đống văn bản dài mà cần phải biết chính xác: *Đây là Nghị định hay Quyết định? Số hiệu bao nhiêu? Ai ký? Ngày nào ban hành?*

**Module IE sinh ra để giải quyết việc này:**
Nó đóng vai trò như một "trợ lý thông minh" đọc hiểu văn bản, tự động tìm kiếm, chọn lọc và bóc tách đúng **7 thông tin cốt lõi** của văn bản hành chính theo quy chuẩn nhà nước.

---

## 📋 7 Trường thông tin được bóc tách

Module tự động phân tích và trích xuất đúng 7 mục sau:

1. **Loại văn bản (`loai_van_ban`):** Phân loại xem đây là Nghị định, Quyết định, Thông tư, hay Văn bản hợp nhất...
2. **Số, ký hiệu (`so_ky_hieu`):** Mã định danh chính thức của văn bản (ví dụ: `383/2026/NĐ-CP`, `50/2026/TT-BTC`), tuyệt đối không nhầm lẫn với số Công báo.
3. **Cơ quan ban hành (`co_quan_ban_hanh`):** Tên cơ quan ban hành (ví dụ: `CHÍNH PHỦ`, `BỘ TƯ PHÁP`, `BỘ TÀI CHÍNH`...).
4. **Ngày ban hành (`ngay_ban_hanh`):** Ngày ký ban hành văn bản theo định dạng chuẩn ngày/tháng/năm (`DD/MM/YYYY`).
5. **Trích yếu nội dung (`trich_yeu`):** Tóm tắt ngắn gọn nội dung quy định của văn bản.
6. **Người ký (`nguoi_ky`):** Họ tên của người có thẩm quyền ký ở khối chữ ký cuối văn bản (đã loại bỏ chức danh, học vị).
7. **Thông tin Công báo (`cong_bao`):** Số và ngày phát hành của Công báo (nếu có đăng tải trên Công báo nước CHXHCN Việt Nam).

---

## 📥 Đầu vào & 📤 Đầu ra của Module

* **Đầu vào (Input):** Nội dung văn bản thô số hóa (được chuyển giao từ Module OCR).
* **Đầu ra (Output):** Một đối tượng JSON duy nhất chứa đầy đủ 7 trường thông tin đã được chuẩn hóa, sẵn sàng lưu thẳng vào cơ sở dữ liệu hoặc hiển thị lên giao diện phần mềm.
