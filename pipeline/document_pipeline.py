#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Document AI Pipeline Hoàn Chỉnh (End-to-End):
Tài liệu PDF/Ảnh -> Module OCR (YOLO + VietOCR) -> Module IE (Qwen3-VL 4-bit) -> 7 trường Metadata JSON
"""

import os
import sys
import re
import json
from pathlib import Path
from typing import Dict, Any, Union, Optional

# Thêm project root vào sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import Module OCR
from module.OCR.pipeline.ocr_pipeline import OCRPipeline
from module.OCR.script.run_ocr_and_extract_prompt import build_qwen_context_from_ocr

# Import Module IE
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel

DEFAULT_LORA_PATH = r"E:\SGOD\vietnamese-document-ai\module\IE\checkpoint\checkpoint-45"
DEFAULT_BASE_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_OCR_CONFIG = r"E:\SGOD\vietnamese-document-ai\module\OCR\config\ocr.yaml"


class DocumentPipeline:
    def __init__(
        self,
        ocr_config_path: str = DEFAULT_OCR_CONFIG,
        lora_path: str = DEFAULT_LORA_PATH,
        base_model_id: str = DEFAULT_BASE_MODEL,
        load_in_4bit: bool = True,
    ):
        """Khởi tạo toàn bộ hệ thống gồm cả OCR Engine và Information Extraction (LLM)."""
        print("=" * 65)
        print("      KHỞI TẠO HỆ THỐNG VIETNAMESE DOCUMENT AI PIPELINE")
        print("=" * 65)

        # 1. Khởi tạo OCR Pipeline
        print(f"[*] 1/2 Khởi tạo OCR Pipeline (Config: {ocr_config_path})...")
        self.ocr_pipeline = OCRPipeline(config_path=ocr_config_path)

        # 2. Khởi tạo IE Engine (Qwen3-VL 4-bit)
        print(f"[*] 2/2 Khởi tạo IE Engine (Base: {base_model_id}, 4-bit: {load_in_4bit})...")
        self.processor = AutoProcessor.from_pretrained(base_model_id)
        self.llm = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model_id,
            load_in_4bit=load_in_4bit,
            device_map="auto",
        )

        adapter_path = Path(lora_path)
        if adapter_path.exists() and (adapter_path / "adapter_config.json").exists():
            print(f"[*] Đang nạp LoRA weights: {adapter_path.name}")
            self.llm = PeftModel.from_pretrained(self.llm, str(adapter_path))
        else:
            print("[!] Cảnh báo: Không tìm thấy LoRA checkpoint, sử dụng Base Model.")

        self.llm.eval()
        print(f"[✓] Document Pipeline đã sẵn sàng hoạt động!")
        print("-" * 65)

    def extract_metadata_from_prompt(self, full_prompt: str, max_new_tokens: int = 512) -> Dict[str, Any]:
        """Đưa prompt ngữ cảnh vào mô hình LLM để trích xuất JSON."""
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Bạn là chuyên gia trích xuất thông tin hành chính từ văn bản quy phạm pháp luật Việt Nam. "
                            "Hãy đọc kỹ văn bản và trích xuất thông tin thành duy nhất một chuỗi JSON hợp lệ với đầy đủ 7 trường: "
                            "loai_van_ban, so_ky_hieu, co_quan_ban_hanh, ngay_ban_hanh, trich_yeu, nguoi_ky, cong_bao."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": full_prompt}],
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.llm.device)

        with torch.inference_mode():
            generated = self.llm.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )

        trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated)]
        raw_text = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        # Parse chuỗi JSON
        cleaned = re.sub(r"^```json\s*|```$", "", raw_text, flags=re.MULTILINE | re.IGNORECASE).strip()
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return {"raw_response": raw_text, "error": "JSON parse error"}

    def process(
        self,
        input_source: Union[str, Path],
        dpi: Optional[int] = None,
        save_result: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Xử lý tài liệu trọn gói: PDF/Ảnh -> Text OCR -> 7 trường Metadata JSON chuẩn.
        """
        source_path = Path(input_source)
        print(f"\n[>] Đang xử lý tài liệu: {source_path.name} ...")

        # Giai đoạn 1: OCR
        print("  [1/3] Đang chạy OCR nhận diện văn bản...")
        ocr_result = self.ocr_pipeline.predict(str(source_path), dpi=dpi)

        # Giai đoạn 2: Lọc ngữ cảnh thông minh (Header + Khối chữ ký)
        print("  [2/3] Đang lọc ngữ cảnh đầu văn bản & khối chữ ký...")
        context_info = build_qwen_context_from_ocr(ocr_result)
        if "error" in context_info:
            return {"error": context_info["error"]}

        # Giai đoạn 3: Trích xuất thông tin qua Qwen3-VL
        print("  [3/3] Qwen3-VL đang trích xuất 7 trường Metadata...")
        metadata = self.extract_metadata_from_prompt(context_info["full_prompt"])

        if save_result:
            target = Path(save_result)
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            print(f"  [✓] Đã lưu kết quả tại: {target}")

        return metadata


if __name__ == "__main__":
    test_dir = Path(r"E:\SGOD\test")
    output_dir = Path(r"E:\SGOD\results\full_pipeline_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Quét toàn bộ file .pdf trong thư mục test
    pdf_files = sorted([f for f in test_dir.iterdir() if f.is_file() and f.suffix.lower() == ".pdf"])

    if not pdf_files:
        print(f"[!] Không tìm thấy file PDF nào trong: {test_dir}")
        sys.exit(0)

    print(f"[*] Tìm thấy {len(pdf_files)} file PDF cần xử lý trong: {test_dir}")

    # Khởi tạo Pipeline một lần duy nhất
    pipeline = DocumentPipeline()

    # Chạy lặp qua từng file PDF
    for idx, pdf_file in enumerate(pdf_files, 1):
        target_json = output_dir / f"{pdf_file.stem}_result.json"
        print(f"\n[{idx}/{len(pdf_files)}] Đang xử lý: {pdf_file.name} ...")
        
        metadata = pipeline.process(pdf_file, save_result=str(target_json))
        
        print(f"[✓] Kết quả 7 trường JSON ({pdf_file.name}):")
        print(json.dumps(metadata, ensure_ascii=False, indent=2))

    print("\n" + "=" * 65)
    print(f"[✓] ĐÃ HOÀN THÀNH TẤT CẢ {len(pdf_files)} FILE PDF!")
    print(f"[✓] Kết quả lưu tại: {output_dir}")
    print("=" * 65)
