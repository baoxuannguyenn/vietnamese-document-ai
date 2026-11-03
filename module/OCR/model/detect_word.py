#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module phát hiện từ/chữ sử dụng Ultralytics YOLO (v8 / 11).
Đồng bộ tham số trực tiếp với config/ocr.yaml.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import torch
from ultralytics import YOLO


class WordDetector:
    def __init__(
        self,
        model_path: str = "module/OCR/weights/best_yolo_ep100.pt",
        device: str = "cuda",
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        imgsz: int = 1024,
    ):
        """
        Word Detector sử dụng thư viện chính thức Ultralytics YOLO.

        Args:
            model_path: Đường dẫn tới file weights (.pt, .onnx, .engine).
            device: 'cuda' hoặc 'cpu' (tự fallback cpu nếu không có cuda).
            conf_thres: Ngưỡng độ tin cậy (Confidence threshold).
            iou_thres: Ngưỡng NMS IoU.
            imgsz: Kích thước ảnh đầu vào mặc định khi infer.
        """
        # Kiểm tra CUDA hợp lệ
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        self.model_path = model_path
        self.device = device
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.imgsz = imgsz

        # Khởi tạo YOLO model
        self.model = YOLO(model_path)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], device: Optional[str] = None) -> "WordDetector":
        """Khởi tạo WordDetector trực tiếp từ dictionary config detector trong ocr.yaml."""
        return cls(
            model_path=cfg.get("model_path", "module/OCR/weights/best_yolo_ep100.pt"),
            device=device or cfg.get("device", "cuda"),
            conf_thres=float(cfg.get("conf_thres", 0.25)),
            iou_thres=float(cfg.get("iou_thres", 0.45)),
            imgsz=int(cfg.get("imgsz", 1024)),
        )

    def detect(
        self, image: np.ndarray, imgsz: Optional[int] = None
    ) -> Tuple[List[List[float]], List[np.ndarray]]:
        """
        Dự đoán vị trí các từ trong ảnh và crop từng vùng từ.

        Args:
            image: Ảnh đầu vào dạng numpy array (BGR OpenCV).
            imgsz: Kích thước resize khi infer (mặc định lấy theo config self.imgsz).

        Returns:
            boxes: Danh sách bounding boxes dạng [x1, y1, x2, y2, score, class_id]
            cropped_words: Danh sách các patch ảnh đã crop sẵn
        """
        target_imgsz = imgsz if imgsz is not None else self.imgsz

        results = self.model.predict(
            source=image,
            imgsz=target_imgsz,
            conf=self.conf_thres,
            iou=self.iou_thres,
            device=self.device,
            verbose=False,
        )

        result = results[0]
        boxes_data = result.boxes.data.cpu().numpy()  # Tensor Nx6: [x1, y1, x2, y2, conf, cls]

        h_orig, w_orig = image.shape[:2]
        boxes: List[List[float]] = []
        cropped_words: List[np.ndarray] = []

        for box in boxes_data:
            x1, y1, x2, y2, conf, cls_id = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Giới hạn tọa độ trong kích thước ảnh gốc
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_orig, x2), min(h_orig, y2)

            # Crop vùng từ/chữ
            word_crop = image[y1:y2, x1:x2]

            if word_crop.size > 0:
                boxes.append([x1, y1, x2, y2, float(conf), int(cls_id)])
                cropped_words.append(word_crop)

        return boxes, cropped_words

    @staticmethod
    def train_model(
        data_yaml: str,
        model_name: str = "yolo11m.pt",
        epochs: int = 100,
        imgsz: int = 1024,
        batch_size: int = 16,
        device: str = "0",
        project: str = "runs/detect",
        name: str = "word_detection",
        **kwargs
    ):
        """Huấn luyện model YOLO Word Detection bằng Ultralytics."""
        model = YOLO(model_name)
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=device,
            project=project,
            name=name,
            max_det=1000,
            mosaic=0.5,
            degrees=5.0,
            scale=0.5,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            **kwargs
        )
        return results
