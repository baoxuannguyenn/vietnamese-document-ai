#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script chuẩn bị dữ liệu Fine-tune cho Qwen:
1. Lấy text Trang 1 (Đầu văn bản).
2. Định vị trang kết thúc văn bản chính (bỏ qua phụ lục) và lấy nguyên văn cụm text thô của khối chữ ký.
3. Ghép nối thành 1 chuỗi ngữ cảnh duy nhất (Context) đưa vào Prompt để Qwen tự suy luận.
4. Lưu vào dataset_for_qwen/train.jsonl theo chuẩn ChatML.
"""

import os
import sys
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# Các từ khóa chức danh/thẩm quyền ký để định vị khối ký
# Các mẫu dòng chức danh chuẩn trong khối chữ ký của Văn bản Hành chính Việt Nam
SIG_PATTERNS = [
    r"^TM\.\s*(CHÍNH PHỦ|ỦY BAN|HỘI ĐỒNG|QUỐC HỘI)",
    r"^KT\.\s*(THỦ TƯỚNG|BỘ TRƯỞNG|CHỦ TỊCH|THỐNG ĐỐC|CHỦ NHIỆM|TỔNG KIỂM TOÁN|TỔNG CỤC TRƯỞNG)",
    r"^(THỦ TƯỚNG|PHÓ THỦ TƯỚNG)\s*$",
    r"^(BỘ TRƯỞNG|THỨ TRƯỞNG)\s*$",
    r"^(THỐNG ĐỐC|PHÓ THỐNG ĐỐC)\s*$",
    r"^(CHỦ TỊCH|PHÓ CHỦ TỊCH)\s*$",
    r"^(CHỦ NHIỆM|PHÓ CHỦ NHIỆM)\s*$",
    r"^(TỔNG KIỂM TOÁN NHÀ NƯỚC|PHÓ TỔNG KIỂM TOÁN NHÀ NƯỚC)\s*$",
    r"^TL\.\s*(THỦ TƯỚNG|BỘ TRƯỞNG|CHỦ TỊCH|CHỦ NHIỆM)",
    r"^TUQ\.\s*(THỦ TƯỚNG|BỘ TRƯỞNG|CHỦ TỊCH)",
    r"XÁC THỰC VĂN BẢN HỢP NHẤT"
]


def is_signature_line(line: str, is_first_page: bool = False, line_idx: int = 0) -> bool:
    line = line.strip()
    if not line:
        return False
    # Ở trang 1, nếu nằm ở phần đầu văn bản (< dòng 15) thường là căn cứ hoặc tên cơ quan ban hành, bỏ qua
    if is_first_page and line_idx < 15:
        return False
    
    # Loại trừ các câu trích dẫn: "của Bộ...", "theo...", "kèm theo...", "quy định..."
    if re.search(r"\b(của|theo|về|gửi|kính gửi|đề nghị|quy định|hướng dẫn|căn cứ)\b", line, re.IGNORECASE):
        return False
    
    for pat in SIG_PATTERNS:
        if re.search(pat, line, re.IGNORECASE):
            return True
    return False


def get_raw_signature_block(page_results: List[Dict[str, Any]], page_lines: List[str], page_width: float, is_first_page: bool = False) -> Optional[str]:
    """
    Lấy NGUYÊN VĂN (RAW TEXT) của khối chữ ký / khối xác thực:
    - Nếu là VBHN: Tìm dòng 'XÁC THỰC VĂN BẢN HỢP NHẤT' hoặc '/VBHN', bốc từ tên cơ quan xác thực (lùi 1-2 dòng) cho tới hết trang.
    - Văn bản thông thường: Tìm dòng chức danh và bốc từ dòng trước đó cho tới hết trang.
    """
    if page_lines:
        # 1. Ưu tiên tìm vị trí bắt đầu khối VBHN nếu có
        vbhn_idx = None
        for i, line in enumerate(page_lines):
            if re.search(r"XÁC THỰC VĂN BẢN HỢP NHẤT|/VBHN", line, re.IGNORECASE):
                vbhn_idx = i
                break

        if vbhn_idx is not None:
            # Lấy lùi lại 1-2 dòng để chứa tên cơ quan xác thực (ví dụ: NGÂN HÀNG NHÀ NƯỚC VIỆT NAM)
            start_i = max(0, vbhn_idx - 2)
            # Không lùi quá xa dính vào dòng CÔNG BÁO hoặc câu văn điều khoản dài
            while start_i < vbhn_idx and (
                re.search(r"^công báo", page_lines[start_i], re.IGNORECASE)
                or len(page_lines[start_i].split()) > 15
            ):
                start_i += 1
            block = page_lines[start_i:]
            return "\n".join([l.strip() for l in block if l.strip()])

        # 2. Nếu là văn bản thông thường: tìm dòng chức danh
        for i, line in enumerate(page_lines):
            if is_signature_line(line, is_first_page=is_first_page, line_idx=i):
                block = page_lines[max(0, i - 1):]
                return "\n".join([l.strip() for l in block if l.strip()])

    # Quét theo bounding box nếu có
    if page_results:
        sorted_boxes = sorted(page_results, key=lambda x: x.get("box", [0, 0, 0, 0])[1])
        for i, item in enumerate(sorted_boxes):
            text = item.get("text", "").strip()
            if is_signature_line(text, is_first_page=is_first_page, line_idx=i) or re.search(r"XÁC THỰC VĂN BẢN HỢP NHẤT", text, re.IGNORECASE):
                start_i = max(0, i - 1)
                block = sorted_boxes[start_i:]
                return "\n".join([b.get("text", "").strip() for b in block if b.get("text", "").strip()])

    return None


def clean_signature_block(sig_text: str) -> str:
    """Loại bỏ các dòng chữ ký số phát hành điện tử và dòng header Công báo ở khối chữ ký."""
    if not sig_text:
        return ""
    lines = [l.strip() for l in sig_text.split("\n") if l.strip()]
    filtered = []
    for line in lines:
        lower = line.lower()
        if any(lower.startswith(p) for p in ["ký bởi:", "cơ quan:", "ngày ký:"]):
            continue
        # Bỏ header Công Báo nếu lọt vào khối chữ ký cuối văn bản (tránh model bốc nhầm số công báo làm số hiệu)
        if re.search(r"^công báo[\s\/]", lower):
            continue
        filtered.append(line)
    return "\n".join(filtered)


def clean_page_1_header(raw_content: str, max_lines: int = 30) -> str:
    """
    Làm sạch và chỉ giữ lại phần Header + Tiêu đề / Trích yếu của Trang 1:
    1. Bỏ qua các dòng chữ ký số điện tử trên đầu trang: 'Ký bởi:', 'Cơ quan:', 'Ngày ký:'.
       - ĐẶC BIỆT: Nếu dòng chứa chữ ký số có dính kèm header 'CÔNG BÁO/Số...' (do OCR quét ngang),
         tự động bóc tách riêng phần 'CÔNG BÁO/Số...' giữ lại.
    2. Cắt dừng lại trước phần Căn cứ pháp lý ('Căn cứ...'), Điều khoản ('Điều 1...'),
       hoặc phần liệt kê luật sửa đổi của VBHN ('được sửa đổi, bổ sung bởi:', '1. Thông tư số...').
    3. Giới hạn tối đa max_lines dòng đầu.
    """
    raw_lines = [l.strip() for l in raw_content.split("\n") if l.strip()]
    filtered_lines = []

    # 1. Bỏ các dòng chữ ký số đầu trang, nhưng cứu lấy phần 'CÔNG BÁO/Số...' nếu bị dính cùng dòng
    for line in raw_lines:
        lower = line.lower()
        if any(lower.startswith(prefix) for prefix in ["ký bởi:", "cơ quan:", "ngày ký:"]):
            cb_match = re.search(r'(CÔNG BÁO\s*[\/\:]?\s*Số\s*\d+.*)', line, re.IGNORECASE)
            if cb_match:
                filtered_lines.append(cb_match.group(1).strip())
            continue
        filtered_lines.append(line)

    # 2. Thu thập tiêu đề và dừng lại khi chạm đến phần căn cứ pháp lý, điều khoản hoặc danh sách luật sửa đổi của VBHN
    result_lines = []
    for i, line in enumerate(filtered_lines):
        # Sau khi đã có ít nhất 4 dòng header (cơ quan, số ký hiệu, tiêu ngữ, ngày tháng, loại vb)
        if i >= 4 and (
            re.match(r"^([cngv]?ăn\s+cứ|cứ\s+(luật|nghị|quyết|thông|hiến)|điều\s+1[\.\:\s]|quyết nghị[\:\s]|quyết định[\:\s])", line, re.IGNORECASE)
            or re.search(r"được sửa đổi[\,\s]+bổ sung bởi", line, re.IGNORECASE)
            or re.match(r"^\d+\.\s*(thông tư|luật|nghị định|quyết định)", line, re.IGNORECASE)
            or (i >= 6 and re.match(r"^(thông tư|luật|nghị định|quyết định)\s+số\s+\d+.*ngày", line, re.IGNORECASE))
        ):
            break
        result_lines.append(line)
        if len(result_lines) >= max_lines:
            break

    return "\n".join(result_lines)


def build_qwen_input_context(ocr_json_path: str) -> Dict[str, Any]:
    """
    Đọc file OCR và ghép nối:
    [Nội dung Đầu Trang 1 đã lọc nhiễu] + [Nội dung Khối ký / Xác thực VBHN]
    """
    with open(ocr_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    file_name = data.get("file_name", "")
    content_list = data.get("content", [])
    pages_list = data.get("pages", [])
    total_pages = data.get("total_pages", len(content_list))

    if total_pages == 0:
        return {"error": "Tài liệu rỗng"}

    content_map = {item.get("page"): item for item in content_list}
    pages_map = {item.get("page"): item for item in pages_list}

    # 1. Text Trang 1: Lọc bỏ chữ ký số điện tử và chỉ lấy phần Header + Trích yếu (dừng trước Căn cứ...)
    page_1_data = content_map.get(1, {})
    raw_p1 = page_1_data.get("content", "").strip()
    page_1_clean_text = clean_page_1_header(raw_p1, max_lines=30)

    # 2. Định vị trang chữ ký / xác thực chuẩn xác
    matched_pages = []
    for item in content_list:
        p = item.get("page", 1)
        lines = [l.strip() for l in item.get("content", "").split("\n") if l.strip()]
        for i, l in enumerate(lines):
            if is_signature_line(l, is_first_page=(p == 1), line_idx=i):
                matched_pages.append((p, l))
                break

    found_page = total_pages
    if matched_pages:
        # Mặc định lấy trang đầu tiên khớp (văn bản chính trước phụ lục)
        found_page = matched_pages[0][0]
        # Nếu có trang xác thực VBHN -> ưu tiên trang VBHN
        for p, l in matched_pages:
            if "XÁC THỰC" in l.upper() or "VBHN" in l.upper():
                found_page = p
                break

    # 3. Lấy text khối chữ ký từ found_page và lọc bỏ chữ ký số
    p_content = content_map.get(found_page, {})
    p_page = pages_map.get(found_page, {})
    results = p_page.get("results", [])
    lines = p_content.get("lines", [])
    img_size = p_page.get("image_size") or {}
    page_w = img_size.get("width") or 1800

    raw_signature_text = get_raw_signature_block(results, lines, page_width=page_w, is_first_page=(found_page == 1))
    if not raw_signature_text and lines:
        raw_signature_text = "\n".join(lines[-15:])
    clean_signature_text = clean_signature_block(raw_signature_text)

    # 4. Ghép nối thành 1 Context Text tinh gọn
    context_text = f"""[PHẦN ĐẦU VĂN BẢN (TRANG 1)]:
{page_1_clean_text}

[KHỐI CHỮ KÝ / XÁC THỰC VĂN BẢN (TRANG {found_page if found_page else total_pages})]:
{clean_signature_text if clean_signature_text else "Không có khối chữ ký"}"""

    # 5. Prompt tinh gọn tối đa, tiết kiệm token, đồng bộ 100% với nhãn Ground Truth
    full_prompt = f"""Trích xuất thông tin hành chính từ văn bản dưới đây thành duy nhất 1 JSON gồm 7 trường: loai_van_ban, so_ky_hieu, co_quan_ban_hanh, ngay_ban_hanh, trich_yeu, nguoi_ky, cong_bao.

Quy tắc:
- loai_van_ban: viết hoa chữ cái đầu (Thông tư, Nghị định, Quyết định...). Nếu có "/VBHN-" thì là "Văn bản hợp nhất".
- so_ky_hieu: BẮT BUỘC là mã có dấu gạch chéo "/" sau chữ "Số:" (VD: 47/2026/TT-BCA, 50/2026/TT-BTC, 840/QĐ-TTg). TUYỆT ĐỐI KHÔNG lấy số đơn lẻ từ "CÔNG BÁO/Số...".
- co_quan_ban_hanh: tên cơ quan in hoa, bỏ Quốc hiệu và Tiêu ngữ; VBHN-VPQH là "VĂN PHÒNG QUỐC HỘI".
- ngay_ban_hanh: BẮT BUỘC lấy từ dòng địa danh (VD: "Hà Nội, ngày 12 tháng 5 năm 2026" -> "12/05/2026"). TUYỆT ĐỐI KHÔNG lấy ngày của Công Báo.
- trich_yeu: tóm tắt nội dung chính ngay dưới loại văn bản.
- nguoi_ky: chỉ họ tên người ký ở khối chữ ký, bỏ chức danh/quân hàm.
- cong_bao: từ dòng "CÔNG BÁO/Số [X]/Ngày [DD-MM-YYYY]" trên đầu trang, chuyển thành định dạng chuẩn "Công báo số [X] ngày YYYY-MM-DD" (ví dụ "CÔNG BÁO/Số 288/Ngày 29-05-2026 60" -> "Công báo số 288 ngày 2026-05-29"). Nếu không có thì để null.

NGỮ CẢNH VĂN BẢN:
\"\"\"
{context_text}
\"\"\""""

    return {
        "file_name": file_name,
        "total_pages": total_pages,
        "signature_page": found_page,
        "raw_signature_text": clean_signature_text,
        "context_text": context_text,
        "full_prompt": full_prompt
    }


def save_dataset_for_qwen(info: Dict[str, Any], dataset_dir: str = "dataset_for_qwen", metadata_csv_path: Optional[str] = None):
    """Lưu dữ liệu vào train.jsonl theo chuẩn ChatML"""
    out_dir = Path(dataset_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    file_name = info.get("file_name", "")
    
    # Chuẩn bị nhãn Ground Truth (assistant output) để fine-tune
    assistant_output = {
        "loai_van_ban": None,
        "so_ky_hieu": None,
        "co_quan_ban_hanh": None,
        "ngay_ban_hanh": None,
        "trich_yeu": None,
        "nguoi_ky": None,
        "cong_bao": None
    }
    
    # Đối chiếu metadata.csv nếu có
    if metadata_csv_path and os.path.exists(metadata_csv_path):
        try:
            import urllib.parse
            import pandas as pd
            df = pd.read_csv(metadata_csv_path)

            def normalize_fname(name: str) -> str:
                name = urllib.parse.unquote(str(name)).lower()
                name = re.sub(r"\s*\(\d+\)", "", name)  # loại bỏ duplicate suffix như " (2)"
                return Path(name).name.strip()

            target_norm = normalize_fname(file_name)
            df["norm_fname"] = df["file_name"].apply(normalize_fname)
            matched = df[df["norm_fname"] == target_norm]

            # Nếu chưa khớp theo tên file chuẩn hóa, thử tìm kiếm chứa substring
            if matched.empty and target_norm:
                matched = df[df["norm_fname"].str.contains(re.escape(Path(target_norm).stem), na=False)]

            if not matched.empty:
                row = matched.iloc[0]
                assistant_output["loai_van_ban"] = str(row.get("Loại văn bản", "")).strip() or None
                assistant_output["so_ky_hieu"] = str(row.get("Số, ký hiệu", "")).strip() or None
                assistant_output["co_quan_ban_hanh"] = str(row.get("Cơ quan ban hành", "")).strip() or None
                assistant_output["ngay_ban_hanh"] = str(row.get("Ngày ban hành", "")).strip() or None
                assistant_output["trich_yeu"] = str(row.get("Trích yếu", "")).strip() or None
                if pd.notna(row.get("Người ký")) and str(row.get("Người ký")).strip():
                    assistant_output["nguoi_ky"] = str(row.get("Người ký")).strip()
                if pd.notna(row.get("Công báo")) and str(row.get("Công báo")).strip():
                    assistant_output["cong_bao"] = str(row.get("Công báo")).strip()
        except Exception as e:
            print(f"[!] Lỗi đọc metadata.csv: {e}")

    # Tạo mẫu ChatML cho Qwen
    chat_sample = {
        "messages": [
            {
                "role": "system",
                "content": "Bạn là chuyên gia trích xuất thông tin hành chính từ văn bản quy phạm pháp luật Việt Nam. Hãy đọc kỹ văn bản và trích xuất thông tin thành duy nhất một chuỗi JSON hợp lệ với đầy đủ 7 trường: loai_van_ban, so_ky_hieu, co_quan_ban_hanh, ngay_ban_hanh, trich_yeu, nguoi_ky, cong_bao."
            },
            {
                "role": "user",
                "content": info.get("full_prompt", "")
            },
            {
                "role": "assistant",
                "content": json.dumps(assistant_output, ensure_ascii=False, indent=2)
            }
        ]
    }

    # Lưu vào train.jsonl
    jsonl_path = out_dir / "train.jsonl"
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(chat_sample, ensure_ascii=False) + "\n")

    # Lưu thêm 1 file .txt thuần túy chứa đúng prompt để inference trực tiếp
    stem_name = Path(file_name).stem
    txt_path = out_dir / f"{stem_name}_context.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(info.get("full_prompt", ""))

    return str(jsonl_path), str(txt_path)


if __name__ == "__main__":
    import argparse

    project_root = Path(__file__).resolve().parents[1]

    # Tự động tìm metadata.csv (Ưu tiên HF Cache -> Kaggle Input -> local)
    default_csv = r"E:\SGOD\21_9_2026\sgod_model_ocr\metadata\metadata.csv"
    # 1. Kiểm tra Hugging Face Cache
    hf_bases = [
        Path("/root/.cache/huggingface/hub"),
        Path(os.path.expanduser("~/.cache/huggingface/hub"))
    ]
    found_hf = False
    for hf_base in hf_bases:
        if hf_base.exists():
            found = list(hf_base.rglob("metadata.csv"))
            if found:
                default_csv = found[0]
                found_hf = True
                break

    # 2. Kiểm tra Kaggle Input nếu chưa thấy ở HF
    if not found_hf:
        kaggle_csv = Path("/kaggle/input")
        if kaggle_csv.exists():
            found = list(kaggle_csv.rglob("metadata.csv"))
            if found:
                default_csv = found[0]

    parser = argparse.ArgumentParser(description="Trích xuất Prompt và chuẩn bị Dataset Qwen từ file OCR JSON")
    parser.add_argument("--json_path", type=str, default=r"E:\SGOD\results\ocr_output", help="Đường dẫn file .json hoặc thư mục chứa các file OCR JSON")
    parser.add_argument("--metadata_csv", type=str, default=r"E:\SGOD\dataset_congbao\metadata.csv", help="Đường dẫn file metadata.csv")
    parser.add_argument("--dataset_dir", type=str, default=r"E:\SGOD\results\data_qwen", help="Thư mục xuất dataset")  
    parser.add_argument("--reset", action="store_true", help="Xóa train.jsonl cũ trước khi sinh mới (tránh trùng lặp khi chạy lại)")

    args = parser.parse_args()

    input_p = Path(args.json_path)
    csv_path = args.metadata_csv if os.path.exists(args.metadata_csv) else None
    dataset_p = Path(args.dataset_dir)
    dataset_p.mkdir(parents=True, exist_ok=True)

    # 1. Tìm danh sách file JSON cần xử lý
    json_files: List[Path] = []
    if input_p.is_file() and input_p.suffix.lower() == ".json":
        json_files = [input_p]
    elif input_p.is_dir():
        json_files = sorted(list(input_p.glob("*.json")))

    if not json_files:
        print(f"[!] Không tìm thấy file JSON nào tại: {input_p}")
        sys.exit(0)

    # Nếu có cờ --reset hoặc chạy batch thư mục, xóa file train.jsonl cũ để tránh duplicate
    train_jsonl = dataset_p / "train.jsonl"
    if args.reset and train_jsonl.exists():
        train_jsonl.unlink()
        print(f"[*] Đã reset file dataset cũ: {train_jsonl}")

    print("=" * 60)
    print(f"[*] BẮT ĐẦU XỬ LÝ {len(json_files)} FILE OCR JSON SANG DATASET QWEN")
    print(f"[*] Metadata CSV: {csv_path if csv_path else 'Không tìm thấy (nhãn sẽ để null)'}")
    print(f"[*] Thư mục đích: {dataset_p.resolve()}")
    print("=" * 60)

    success_cnt = 0
    for idx, jf in enumerate(json_files, 1):
        try:
            info = build_qwen_input_context(str(jf))
            if "error" in info:
                print(f"[{idx}/{len(json_files)}] [!] Bỏ qua {jf.name}: {info['error']}")
                continue

            jsonl_path, txt_path = save_dataset_for_qwen(
                info,
                dataset_dir=str(dataset_p),
                metadata_csv_path=csv_path
            )
            success_cnt += 1
            if len(json_files) <= 10 or idx % 10 == 0 or idx == len(json_files):
                print(f"[{idx}/{len(json_files)}] [+] Đã xử lý: {jf.name}")
        except Exception as e:
            print(f"[{idx}/{len(json_files)}] [!] Lỗi khi xử lý {jf.name}: {e}")

    print("\n" + "=" * 60)
    print(f"[+] HOÀN THÀNH: {success_cnt}/{len(json_files)} file đã được lưu vào:")
    print(f"    - File JSONL huấn luyện: {dataset_p / 'train.jsonl'}")
    print(f"    - File TXT dễ đọc:      {dataset_p}")
    print("=" * 60)
