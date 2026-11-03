#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OCR Pipeline siêu tinh gọn - Chỉ 1 hàm predict duy nhất
Đồng bộ hoàn toàn với file cấu hình ocr.yaml
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Union, Optional, List, Tuple
import cv2
import yaml
import numpy as np

# Hỗ trợ import nội bộ linh hoạt
try:
    from ..model.detect_word import WordDetector
    from ..model.recognizer import TextRecognizer
    from ..utils.box_utils import sort_boxes_reading_order
except (ImportError, ValueError):
    from model.detect_word import WordDetector
    from model.recognizer import TextRecognizer
    from utils.box_utils import sort_boxes_reading_order


class OCRPipeline:
    def __init__(self, config_path: str = "module/OCR/config/ocr.yaml", **overrides):
        """Khởi tạo OCR Pipeline đọc đường dẫn và cấu hình trực tiếp từ file .yaml."""
        self.config_path = self._resolve_path(config_path)
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.config.update(overrides)

        # 1. Cấu hình phần cứng & tham số từ YAML
        self.device = self.config.get("device", "cuda")
        det_cfg = self.config.get("detector", {})
        rec_cfg = self.config.get("recognizer", {})
        sort_cfg = self.config.get("sorting", {})
        data_cfg = self.config.get("dataset", {})

        self.y_threshold_ratio = sort_cfg.get("y_threshold_ratio", 0.5)
        self.line_break = sort_cfg.get("line_break", True)
        self.default_dpi = data_cfg.get("dpi", 200)

        # 2. Khởi tạo Models từ các paths lấy trực tiếp trong file .yaml
        self.detector = WordDetector(
            model_path=self._resolve_path(det_cfg.get("model_path")),
            device=self.device,
            conf_thres=det_cfg.get("conf_thres", 0.25),
            iou_thres=det_cfg.get("iou_thres", 0.45),
            imgsz=det_cfg.get("imgsz", 1024)
        )
        self.recognizer = TextRecognizer(
            config_base=rec_cfg.get("config_base", "vgg_transformer"),
            weight_path=self._resolve_path(rec_cfg.get("weight_path")),
            device=self.device,
            beamsearch=rec_cfg.get("beamsearch", False)
        )

    def _resolve_path(self, path: Union[str, Path]) -> str:
        """Lấy đường dẫn từ file .yaml (hỗ trợ cả đường dẫn tương đối và tuyệt đối)."""
        if not path:
            return ""
        p = Path(path)
        if p.exists():
            return str(p)
        curr = Path(__file__).resolve()
        for parent in curr.parents:
            if (parent / path).exists():
                return str(parent / path)
        return str(path)

    def _load_pages(self, input_source: Union[str, np.ndarray], dpi: int) -> Tuple[List[np.ndarray], str]:
        """Tự động nạp danh sách ảnh trang từ PDF, file ảnh hoặc numpy array."""
        if isinstance(input_source, np.ndarray):
            return [input_source], "numpy_image"

        path_str = str(input_source)
        if path_str.lower().endswith(".pdf"):
            import fitz
            doc = fitz.open(path_str)
            pages = []
            for page in doc:
                pix = page.get_pixmap(dpi=dpi)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                pages.append(cv2.cvtColor(img, cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR))
            return pages, Path(path_str).name

        img = cv2.imread(path_str)
        if img is None:
            raise ValueError(f"Không thể đọc ảnh: {input_source}")
        return [img], Path(path_str).name

    def predict(
        self, 
        input_source: Union[str, np.ndarray], 
        dpi: Optional[int] = None, 
        save_json: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Hàm OCR duy nhất: Nhận diện cả PDF, file ảnh hoặc numpy array.
        """
        pages_imgs, file_name = self._load_pages(input_source, dpi=dpi or self.default_dpi)
        content_list, pages_list = [], []

        for page_idx, img in enumerate(pages_imgs, 1):
            h, w = img.shape[:2]
            boxes, crops = self.detector.detect(img)

            if not boxes:
                content_list.append({"page": page_idx, "content": "", "lines": []})
                pages_list.append({"page": page_idx, "image_size": {"width": w, "height": h}, "results": []})
                continue

            # Sắp xếp thứ tự đọc & gom dòng
            flat_idx, line_idx = sort_boxes_reading_order(boxes, self.y_threshold_ratio)
            texts = self.recognizer.predict_batch([crops[i] for i in flat_idx])
            pred_map = {idx: text.strip() for idx, text in zip(flat_idx, texts)}

            lines = [" ".join(pred_map[i] for i in line if pred_map[i]) for line in line_idx]
            lines = [l for l in lines if l]
            full_text = ("\n" if self.line_break else " ").join(lines)

            word_results = [
                {"box": [int(b) for b in boxes[i][:4]], "score": round(float(boxes[i][4]), 4), "text": pred_map[i]}
                for i in flat_idx if pred_map[i]
            ]
            content_list.append({"page": page_idx, "content": full_text, "lines": lines})
            pages_list.append({"page": page_idx, "image_size": {"width": w, "height": h}, "results": word_results})

        output = {
            "file_name": file_name,
            "total_pages": len(pages_imgs),
            "content": content_list,
            "pages": pages_list
        }

        if save_json:
            os.makedirs(os.path.dirname(os.path.abspath(save_json)), exist_ok=True)
            with open(save_json, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)

        return output
