#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Vietnamese Document AI - OCR Module
Đồng bộ hoàn toàn với file cấu hình config/ocr.yaml
"""

from .pipeline.ocr_pipeline import OCRPipeline
from .model.detect_word import WordDetector
from .model.recognizer import TextRecognizer
from .utils.box_utils import sort_boxes_reading_order, draw_ocr_boxes

__all__ = [
    "OCRPipeline",
    "WordDetector",
    "TextRecognizer",
    "sort_boxes_reading_order",
    "draw_ocr_boxes",
]

