#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script chạy OCR hàng loạt (Batch OCR) cho tập tài liệu (PDF và ảnh):
- Đồng bộ hoàn toàn với file cấu hình config/ocr.yaml
- Hỗ trợ resume (tự động bỏ qua file đã có kết quả để tiết kiệm thời gian)
- Cho phép ghi đè các tham số linh hoạt từ dòng lệnh (CLI)
"""

import sys
import argparse
from pathlib import Path
from typing import List

# Đảm bảo đường dẫn import hoạt động độc lập
CURRENT_DIR = Path(__file__).resolve().parent
MODULE_ROOT = CURRENT_DIR.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from pipeline.ocr_pipeline import OCRPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Chạy OCR hàng loạt đồng bộ theo config/ocr.yaml")
    parser.add_argument(
        "--config", type=str, default=str(MODULE_ROOT / "config" / "ocr.yaml"),
        help="Đường dẫn file cấu hình YAML (mặc định: module/OCR/config/ocr.yaml)"
    )
    parser.add_argument(
        "--input_dir", type=str, default=None,
        help="Thư mục chứa tài liệu PDF/ảnh (mặc định lấy từ dataset.raw_dir trong config)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Thư mục lưu JSON kết quả (mặc định lấy từ dataset.output_dir trong config)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Thiết bị chạy ('cuda' hoặc 'cpu', mặc định lấy theo config)"
    )
    parser.add_argument(
        "--dpi", type=int, default=None,
        help="DPI khi render PDF ra ảnh (mặc định lấy theo config)"
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Giới hạn số file cần xử lý (0 = tất cả)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Chạy lại và ghi đè những file đã có kết quả JSON"
    )
    return parser.parse_args()


def collect_document_files(input_dir: Path) -> List[Path]:
    """Tìm tất cả các file PDF và hình ảnh trong thư mục đầu vào (không trùng lặp)."""
    valid_exts = {".pdf", ".png", ".jpg", ".jpeg"}
    files = [
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in valid_exts
    ]
    return sorted(files)


def main():
    args = parse_args()

    # 1. Thu thập các tham số override từ CLI (nếu người dùng chỉ định)
    overrides = {}
    if args.device:
        overrides["device"] = args.device

    # 2. Khởi tạo Pipeline đồng bộ từ config YAML
    pipeline = OCRPipeline(config_path=args.config, **overrides)

    # 3. Phân giải thư mục đầu vào và đầu ra (ưu tiên cờ CLI -> fallback sang config ocr.yaml)
    data_cfg = pipeline.config.get("dataset", {})
    raw_dir = args.input_dir or data_cfg.get("raw_dir", "data/raw")
    out_dir = args.output_dir or data_cfg.get("output_dir", "results/ocr_output")

    input_path = Path(pipeline._resolve_path(raw_dir))
    output_path = Path(pipeline._resolve_path(out_dir))
    dpi = args.dpi or pipeline.default_dpi

    print("=" * 65)
    print("        CHẠY OCR HÀNG LOẠT (VIETNAMESE DOCUMENT AI)")
    print("=" * 65)
    print(f"[*] Config file  : {args.config}")
    print(f"[*] Thiết bị     : {pipeline.device.upper()}")
    print(f"[*] Thư mục input: {input_path}")
    print(f"[*] Thư mục json : {output_path}")
    print(f"[*] DPI PDF      : {dpi}")
    print("-" * 65)

    if not input_path.exists():
        print(f"[!] Lỗi: Thư mục đầu vào không tồn tại: {input_path}")
        print(f"[!] Vui lòng đặt tài liệu vào '{input_path}' hoặc dùng cờ '--input_dir'.")
        sys.exit(1)

    # 4. Quét danh sách file cần OCR
    doc_files = collect_document_files(input_path)
    if not doc_files:
        print(f"[!] Không tìm thấy file PDF hay ảnh nào trong: {input_path}")
        sys.exit(0)

    if args.limit > 0:
        doc_files = doc_files[:args.limit]

    total_files = len(doc_files)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"[+] Tìm thấy tổng cộng: {total_files} tài liệu cần xử lý.")

    # 5. Xử lý từng tài liệu
    success_count, skipped_count, failed_count = 0, 0, 0

    for idx, doc_file in enumerate(doc_files, 1):
        target_json = output_path / f"{doc_file.stem}.json"

        # Resume: Bỏ qua nếu file kết quả đã tồn tại và không bật --force
        if target_json.exists() and not args.force:
            print(f"[{idx}/{total_files}] Bỏ qua (Đã xử lý): {doc_file.name}")
            skipped_count += 1
            continue

        print(f"\n[{idx}/{total_files}] Đang xử lý: {doc_file.name} ...")
        try:
            pipeline.predict(str(doc_file), dpi=dpi, save_json=str(target_json))
            print(f"    -> Đã xuất: {target_json.name}")
            success_count += 1
        except Exception as e:
            print(f"[!] Thất bại với file {doc_file.name}: {e}")
            failed_count += 1

    # 6. Tổng kết
    print("\n" + "=" * 65)
    print("HOÀN THÀNH BATCH OCR:")
    print(f"  - Tổng số file     : {total_files}")
    print(f"  - Xử lý thành công : {success_count}")
    print(f"  - Đã có từ trước   : {skipped_count}")
    print(f"  - Gặp lỗi          : {failed_count}")
    print(f"  - Thư mục kết quả  : {output_path.resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    main()
