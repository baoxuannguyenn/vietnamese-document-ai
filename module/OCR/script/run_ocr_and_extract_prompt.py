#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipeline tổng hợp: OCR hàng loạt tài liệu (PDF / Ảnh) và Trích xuất Context Prompt cho Qwen.
Quy trình: File PDF -> OCR (YOLO + VietOCR) -> JSON OCR -> Làm sạch ngữ cảnh (Context) -> Prompt Qwen / Dataset ChatML.
"""

import os
import sys
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

# Thêm đường dẫn root module
CURRENT_DIR = Path(__file__).resolve().parent
MODULE_ROOT = CURRENT_DIR.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from pipeline.ocr_pipeline import OCRPipeline

# ==============================================================================
# CẤU HÌNH NHẬN DIỆN KHỐI CHỮ KÝ VĂN BẢN HÀNH CHÍNH
# ==============================================================================
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
    if is_first_page and line_idx < 15:
        return False
    if re.search(r"\b(của|theo|về|gửi|kính gửi|đề nghị|quy định|hướng dẫn|căn cứ)\b", line, re.IGNORECASE):
        return False
    for pat in SIG_PATTERNS:
        if re.search(pat, line, re.IGNORECASE):
            return True
    return False


def get_raw_signature_block(page_lines: List[str], is_first_page: bool = False) -> Optional[str]:
    """Trích xuất khối chữ ký/xác thực văn bản."""
    if not page_lines:
        return None

    # 1. Văn bản hợp nhất (VBHN)
    for i, line in enumerate(page_lines):
        if re.search(r"XÁC THỰC VĂN BẢN HỢP NHẤT|/VBHN", line, re.IGNORECASE):
            start_i = max(0, i - 2)
            while start_i < i and (re.search(r"^công báo", page_lines[start_i], re.IGNORECASE) or len(page_lines[start_i].split()) > 15):
                start_i += 1
            return "\n".join([l.strip() for l in page_lines[start_i:] if l.strip()])

    # 2. Văn bản thông thường
    for i, line in enumerate(page_lines):
        if is_signature_line(line, is_first_page=is_first_page, line_idx=i):
            return "\n".join([l.strip() for l in page_lines[max(0, i - 1):] if l.strip()])

    return None


def clean_signature_block(sig_text: str) -> str:
    """Lọc bỏ chữ ký số điện tử và header Công báo trong khối chữ ký."""
    if not sig_text:
        return ""
    filtered = []
    for line in sig_text.split("\n"):
        line = line.strip()
        lower = line.lower()
        if any(lower.startswith(p) for p in ["ký bởi:", "cơ quan:", "ngày ký:"]):
            continue
        if re.search(r"^công báo[\s\/]", lower):
            continue
        if line:
            filtered.append(line)
    return "\n".join(filtered)


def clean_page_1_header(raw_content: str, max_lines: int = 30) -> str:
    """Chỉ giữ lại phần Header + Tiêu đề / Trích yếu trang 1 (dừng trước phần Căn cứ...)."""
    raw_lines = [l.strip() for l in raw_content.split("\n") if l.strip()]
    filtered_lines = []

    for line in raw_lines:
        lower = line.lower()
        if any(lower.startswith(prefix) for prefix in ["ký bởi:", "cơ quan:", "ngày ký:"]):
            cb_match = re.search(r'(CÔNG BÁO\s*[\/\:]?\s*Số\s*\d+.*)', line, re.IGNORECASE)
            if cb_match:
                filtered_lines.append(cb_match.group(1).strip())
            continue
        filtered_lines.append(line)

    result_lines = []
    for i, line in enumerate(filtered_lines):
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


def build_qwen_context_from_ocr(ocr_data: Dict[str, Any]) -> Dict[str, Any]:
    """Tạo chuỗi ngữ cảnh rút gọn từ dữ liệu OCR."""
    file_name = ocr_data.get("file_name", "")
    content_list = ocr_data.get("content", [])
    total_pages = ocr_data.get("total_pages", len(content_list))

    if total_pages == 0:
        return {"error": "Tài liệu rỗng"}

    content_map = {item.get("page"): item for item in content_list}

    # 1. Trang 1
    raw_p1 = content_map.get(1, {}).get("content", "").strip()
    page_1_clean = clean_page_1_header(raw_p1, max_lines=30)

    # 2. Định vị trang chữ ký
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
        found_page = matched_pages[0][0]
        for p, l in matched_pages:
            if "XÁC THỰC" in l.upper() or "VBHN" in l.upper():
                found_page = p
                break

    p_content = content_map.get(found_page, {})
    lines = p_content.get("lines", [])
    raw_sig = get_raw_signature_block(lines, is_first_page=(found_page == 1))
    if not raw_sig and lines:
        raw_sig = "\n".join(lines[-15:])
    clean_sig = clean_signature_block(raw_sig)

    # 3. Ghép Context
    context_text = f"""[PHẦN ĐẦU VĂN BẢN (TRANG 1)]:
{page_1_clean}

[KHỐI CHỮ KÝ / XÁC THỰC VĂN BẢN (TRANG {found_page})]:
{clean_sig if clean_sig else "Không có khối chữ ký"}"""

    full_prompt = f"""Trích xuất thông tin hành chính từ văn bản dưới đây thành duy nhất 1 JSON gồm 7 trường: loai_van_ban, so_ky_hieu, co_quan_ban_hanh, ngay_ban_hanh, trich_yeu, nguoi_ky, cong_bao.

Quy tắc:
- loai_van_ban: viết hoa chữ cái đầu (Thông tư, Nghị định, Quyết định...). Nếu có "/VBHN-" thì là "Văn bản hợp nhất".
- so_ky_hieu: BẮT BUỘC là mã có dấu gạch chéo "/" sau chữ "Số:" (VD: 47/2026/TT-BCA, 50/2026/TT-BTC, 840/QĐ-TTg). TUYỆT ĐỐI KHÔNG lấy số đơn lẻ từ "CÔNG BÁO/Số...".
- co_quan_ban_hanh: tên cơ quan in hoa, bỏ Quốc hiệu và Tiêu ngữ; VBHN-VPQH là "VĂN PHÒNG QUỐC HỘI".
- ngay_ban_hanh: BẮT BUỘC lấy từ dòng địa danh (VD: "Hà Nội, ngày 12 tháng 5 năm 2026" -> "12/05/2026"). TUYỆT ĐỐI KHÔNG lấy ngày của Công Báo.
- trich_yeu: tóm tắt nội dung chính ngay dưới loại văn bản.
- nguoi_ky: chỉ họ tên người ký ở khối chữ ký, bỏ chức danh/quân hàm.
- cong_bao: từ dòng "CÔNG BÁO/Số [X]/Ngày [DD-MM-YYYY]" trên đầu trang, chuyển thành định dạng chuẩn "Công báo số [X] ngày YYYY-MM-DD". Nếu không có thì để null.

NGỮ CẢNH VĂN BẢN:
\"\"\"
{context_text}
\"\"\""""

    return {
        "file_name": file_name,
        "total_pages": total_pages,
        "signature_page": found_page,
        "context_text": context_text,
        "full_prompt": full_prompt
    }


def save_qwen_dataset(info: Dict[str, Any], dataset_dir: Path, metadata_df: Optional[Any] = None):
    """Lưu prompt ngữ cảnh vào file .txt và file train.jsonl (nếu có nhãn metadata)."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    file_name = info.get("file_name", "")
    stem_name = Path(file_name).stem

    # Lưu context text cho Qwen inference
    txt_path = dataset_dir / f"{stem_name}_context.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(info.get("full_prompt", ""))

    # Nếu có metadata để đối chiếu ground truth
    if metadata_df is not None:
        assistant_output = {
            "loai_van_ban": None, "so_ky_hieu": None, "co_quan_ban_hanh": None,
            "ngay_ban_hanh": None, "trich_yeu": None, "nguoi_ky": None, "cong_bao": None
        }
        try:
            import urllib.parse
            norm_name = urllib.parse.unquote(file_name).lower()
            matched = metadata_df[metadata_df["norm_fname"] == norm_name]
            if matched.empty:
                matched = metadata_df[metadata_df["norm_fname"].str.contains(re.escape(stem_name.lower()), na=False)]

            if not matched.empty:
                row = matched.iloc[0]
                assistant_output = {
                    "loai_van_ban": str(row.get("Loại văn bản", "")).strip() or None,
                    "so_ky_hieu": str(row.get("Số, ký hiệu", "")).strip() or None,
                    "co_quan_ban_hanh": str(row.get("Cơ quan ban hành", "")).strip() or None,
                    "ngay_ban_hanh": str(row.get("Ngày ban hành", "")).strip() or None,
                    "trich_yeu": str(row.get("Trích yếu", "")).strip() or None,
                    "nguoi_ky": str(row.get("Người ký", "")).strip() or None,
                    "cong_bao": str(row.get("Công báo", "")).strip() or None,
                }
        except Exception:
            pass

        sample = {
            "messages": [
                {"role": "system", "content": "Bạn là chuyên gia trích xuất thông tin hành chính từ văn bản quy phạm pháp luật Việt Nam."},
                {"role": "user", "content": info.get("full_prompt", "")},
                {"role": "assistant", "content": json.dumps(assistant_output, ensure_ascii=False, indent=2)}
            ]
        }
        with open(dataset_dir / "train.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Pipeline OCR và Tạo Context Qwen Trực Tiếp")
    parser.add_argument("--config", type=str, default=str(MODULE_ROOT / "config" / "ocr.yaml"), help="File config ocr.yaml")
    parser.add_argument("--input_dir", type=str, default=None, help="Thư mục chứa tài liệu PDF")
    parser.add_argument("--ocr_output", type=str, default=None, help="Thư mục lưu JSON kết quả OCR")
    parser.add_argument("--qwen_output", type=str, default=r"E:\SGOD\results\data_qwen", help="Thư mục lưu Prompt/Context Qwen")
    parser.add_argument("--metadata_csv", type=str, default=r"E:\SGOD\dataset_congbao\metadata.csv", help="File metadata.csv đối chiếu nhãn (nếu có)")
    parser.add_argument("--limit", type=int, default=0, help="Giới hạn số file xử lý (0 = tất cả)")
    parser.add_argument("--force", action="store_true", help="Ghi đè file đã xử lý")

    args = parser.parse_args()

    # 1. Khởi tạo OCR Pipeline
    pipeline = OCRPipeline(config_path=args.config)
    data_cfg = pipeline.config.get("dataset", {})

    input_dir = Path(pipeline._resolve_path(args.input_dir or data_cfg.get("raw_dir", "data/raw")))
    ocr_out_dir = Path(pipeline._resolve_path(args.ocr_output or data_cfg.get("output_dir", "results/ocr_output")))
    qwen_out_dir = Path(args.qwen_output)

    ocr_out_dir.mkdir(parents=True, exist_ok=True)
    qwen_out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("      PIPELINE TỔNG HỢP: OCR & TẠO CONTEXT PROMPT CHO QWEN")
    print("=" * 65)
    print(f"[*] Thư mục PDF đầu vào: {input_dir}")
    print(f"[*] Thư mục xuất OCR JSON : {ocr_out_dir}")
    print(f"[*] Thư mục Context Qwen : {qwen_out_dir}")
    print("-" * 65)

    if not input_dir.exists():
        print(f"[!] Thư mục đầu vào không tồn tại: {input_dir}")
        sys.exit(1)

    # Đọc metadata nếu có
    metadata_df = None
    if args.metadata_csv and os.path.exists(args.metadata_csv):
        try:
            import urllib.parse
            import pandas as pd
            metadata_df = pd.read_csv(args.metadata_csv)
            metadata_df["norm_fname"] = metadata_df["file_name"].apply(
                lambda x: urllib.parse.unquote(str(x)).lower().strip()
            )
            print(f"[+] Đã tải metadata ({len(metadata_df)} dòng) để tạo nhãn ground truth.")
        except Exception as e:
            print(f"[!] Không đọc được metadata.csv: {e}")

    # Lấy danh sách file PDF
    pdf_files = sorted([f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() == ".pdf"])
    if not pdf_files:
        print(f"[!] Không tìm thấy file PDF nào tại: {input_dir}")
        sys.exit(0)

    if args.limit > 0:
        pdf_files = pdf_files[:args.limit]

    total = len(pdf_files)
    print(f"[+] Tìm thấy: {total} tài liệu cần xử lý.")

    for idx, pdf_file in enumerate(pdf_files, 1):
        target_ocr_json = ocr_out_dir / f"{pdf_file.stem}.json"
        target_qwen_txt = qwen_out_dir / f"{pdf_file.stem}_context.txt"

        print(f"\n[{idx}/{total}] Đang xử lý: {pdf_file.name} ...")

        # Bước 1: OCR (nếu chưa có hoặc bật --force)
        ocr_data = None
        if target_ocr_json.exists() and not args.force:
            print("  - Đã có kết quả OCR, đang nạp từ file JSON...")
            with open(target_ocr_json, "r", encoding="utf-8") as f:
                ocr_data = json.load(f)
        else:
            print("  - Đang chạy OCR (YOLO + VietOCR)...")
            ocr_data = pipeline.predict(str(pdf_file), save_json=str(target_ocr_json))

        # Bước 2: Tạo Prompt và Context rút gọn cho Qwen
        info = build_qwen_context_from_ocr(ocr_data)
        if "error" in info:
            print(f"  [!] Lỗi trích xuất ngữ cảnh: {info['error']}")
            continue

        save_qwen_dataset(info, qwen_out_dir, metadata_df=metadata_df)
        print(f"  -> Đã tạo Prompt Context: {target_qwen_txt.name}")

    print("\n" + "=" * 65)
    print("HOÀN THÀNH TOÀN BỘ QUY TRÌNH!")
    print(f"  - Kết quả OCR JSON: {ocr_out_dir}")
    print(f"  - Context cho Qwen: {qwen_out_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()

