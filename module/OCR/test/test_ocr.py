#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script kiểm thử nhanh cho OCR Pipeline:
- Kiểm tra nạp cấu hình config/ocr.yaml
- Kiểm tra nạp weights YOLO và VietOCR
- Chạy thử nghiệm nhận diện trên file chỉ định hoặc ảnh mẫu tự sinh
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import cv2

# Đảm bảo đường dẫn import hoạt động
CURRENT_DIR = Path(__file__).resolve().parent
MODULE_ROOT = CURRENT_DIR.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from pipeline.ocr_pipeline import OCRPipeline


def create_dummy_sample_image() -> np.ndarray:
    """Tạo một ảnh mẫu nền trắng có chữ để kiểm thử pipeline."""
    img = np.full((300, 800, 3), 255, dtype=np.uint8)
    cv2.putText(
        img, "CONG HOA XA HOI CHU NGHIA VIET NAM", (50, 80),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA
    )
    cv2.putText(
        img, "Doc lap - Tu do - Hanh phuc", (180, 140),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA
    )
    cv2.putText(
        img, "SO: 123/2026/QD-TTg", (50, 210),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA
    )
    return img


def main():
    parser = argparse.ArgumentParser(description="Kiểm thử OCR Pipeline")
    parser.add_argument("--config", type=str, default=str(MODULE_ROOT / "config" / "ocr.yaml"),
                        help="Đường dẫn file cấu hình ocr.yaml")
    parser.add_argument("--input", type=str, default=None,
                        help="Đường dẫn file ảnh hoặc PDF cần test")
    parser.add_argument("--device", type=str, default=None,
                        help="Thiết bị ('cuda' hoặc 'cpu')")
    args = parser.parse_args()

    print("[1] Đang khởi tạo OCRPipeline từ:", args.config)
    overrides = {}
    if args.device:
        overrides["device"] = args.device

    pipeline = OCRPipeline(config_path=args.config, **overrides)
    print(f"    - Thiết bị: {pipeline.device}")
    print(f"    - YOLO weights: {pipeline.detector.model_path}")
    print(f"    - VietOCR weights: {pipeline.recognizer.weight_path}")
    print(f"    - Sorter Y-ratio: {pipeline.y_threshold_ratio}")

    if args.input:
        target = Path(args.input)
        if not target.exists():
            print(f"[!] File không tồn tại: {target}")
            sys.exit(1)
        print(f"\n[2] Đang chạy test trên file: {target}")
        result = pipeline.predict(str(target))
    else:
        print("\n[2] Không truyền --input, đang chạy thử trên ảnh mẫu giả lập...")
        sample_img = create_dummy_sample_image()
        result = pipeline.predict(sample_img)

    print("\n[3] Kết quả OCR:")
    print("=" * 60)
    full_text = result.get("full_text") or (result.get("content", [{}])[0].get("content", ""))
    print(full_text if full_text else "[Không phát hiện được chữ]")
    print("=" * 60)
    print(f"[+] Kiểm thử thành công! Tổng số từ phát hiện: {len(result.get('results', []))}")


if __name__ == "__main__":
    main()

