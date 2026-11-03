#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module nhận diện chữ (Text Recognition) sử dụng VietOCR.
Đồng bộ tham số trực tiếp với config/ocr.yaml.
"""

import PIL.Image
if not hasattr(PIL.Image, "ANTIALIAS"):
    PIL.Image.ANTIALIAS = PIL.Image.Resampling.LANCZOS

from typing import List, Union, Optional
import numpy as np
import torch
import cv2
from PIL import Image
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

# Cấu hình chuẩn mặc định cho vgg_transformer (hoạt động offline độc lập, không cần tải từ vocr.vn)
VGG_TRANSFORMER_CFG = {
    'vocab': 'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ ',
    'backbone': 'vgg19_bn',
    'cnn': {'pretrained': False, 'ss': [[2, 2], [2, 2], [2, 1], [2, 1], [1, 1]], 'ks': [[2, 2], [2, 2], [2, 1], [2, 1], [1, 1]], 'hidden': 256},
    'seq_modeling': 'transformer',
    'transformer': {'d_model': 256, 'nhead': 8, 'num_encoder_layers': 6, 'num_decoder_layers': 6, 'dim_feedforward': 2048, 'max_seq_length': 1024, 'pos_dropout': 0.1, 'trans_dropout': 0.1},
    'dataset': {'image_height': 32, 'image_min_width': 32, 'image_max_width': 512},
    'predictor': {'beamsearch': False}
}

class TextRecognizer:
    def __init__(
        self,
        config_base: str = "vgg_transformer",
        weight_path: Optional[str] = "module/OCR/weights/vgg_transformer_best.pth",
        device: str = "cuda",
        beamsearch: bool = False,
    ):
        dev = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        try:
            cfg = Cfg.load_config_from_name(config_base)
        except Exception:
            cfg = Cfg(VGG_TRANSFORMER_CFG.copy())

        if weight_path:
            cfg["weights"] = str(weight_path)
        cfg["device"] = dev
        cfg["predictor"]["beamsearch"] = beamsearch

        self.config_base = config_base
        self.weight_path = str(weight_path) if weight_path else ""
        self.config = cfg
        self.device = dev
        self.detector = Predictor(cfg)

    def predict(self, image: Union[np.ndarray, Image.Image]) -> str:
        """Nhận diện 1 ảnh patch chứa từ."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return self.detector.predict(image)

    def predict_batch(self, images: List[Union[np.ndarray, Image.Image]]) -> List[str]:
        """Nhận diện batch danh sách các ảnh patch từ."""
        if not images:
            return []
        pil_images = [
            Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) if isinstance(img, np.ndarray) else img
            for img in images
        ]
        return self.detector.predict_batch(pil_images)
