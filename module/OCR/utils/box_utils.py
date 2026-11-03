#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Các hàm tiện ích xử lý Bounding Box và sắp xếp thứ tự đọc (Reading Order).
"""

from typing import List, Tuple, Optional
import cv2
import numpy as np


def sort_boxes_reading_order(
    boxes: List[List[float]], 
    y_threshold_ratio: float = 0.5
) -> Tuple[List[int], List[List[int]]]:
    """
    Sắp xếp các bounding boxes [x1, y1, x2, y2, ...] theo thứ tự đọc tự nhiên:
    - Dòng trên trước, dòng dưới sau.
    - Trong cùng một dòng thì từ Trái sang Phải.
    
    Args:
        boxes: Danh sách bounding boxes [[x1, y1, x2, y2, conf, cls], ...]
        y_threshold_ratio: Tỷ lệ chênh lệch trục Y (so với chiều cao TB dòng) để xác định cùng dòng.
        
    Returns:
        flat_indices: Danh sách chỉ số (index) theo thứ tự đọc phẳng hoàn chỉnh.
        line_indices: Danh sách chỉ số gom theo từng dòng: [[id_w1, id_w2], [id_w3, id_w4], ...]
    """
    if not boxes:
        return [], []

    # Lưu chỉ số ban đầu để không bị lệch thứ tự khi map lại với crops ảnh
    indexed_boxes = list(enumerate(boxes))

    # Sắp xếp sơ bộ theo tọa độ y1 (từ trên xuống dưới)
    indexed_boxes.sort(key=lambda item: item[1][1])

    lines = []
    current_line = [indexed_boxes[0]]

    for item in indexed_boxes[1:]:
        box = item[1]
        # Chiều cao trung bình của dòng hiện tại
        avg_h = sum(b[1][3] - b[1][1] for b in current_line) / len(current_line)
        line_y_center = sum((b[1][1] + b[1][3]) / 2 for b in current_line) / len(current_line)
        box_y_center = (box[1] + box[3]) / 2

        # Nếu chênh lệch tâm Y nhỏ hơn ngưỡng cho phép -> cùng dòng
        if abs(box_y_center - line_y_center) < (avg_h * y_threshold_ratio):
            current_line.append(item)
        else:
            # Sắp xếp các từ trong dòng hiện tại từ trái sang phải (theo x1)
            current_line.sort(key=lambda x: x[1][0])
            lines.append(current_line)
            current_line = [item]

    if current_line:
        current_line.sort(key=lambda x: x[1][0])
        lines.append(current_line)

    # Tách ra danh sách chỉ số theo dòng và chỉ số phẳng
    line_indices = [[item[0] for item in line] for line in lines]
    flat_indices = [item[0] for line in lines for item in line]

    return flat_indices, line_indices


def draw_ocr_boxes(
    image: np.ndarray,
    boxes: List[List[float]],
    texts: Optional[List[str]] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Vẽ bounding boxes và text lên ảnh phục vụ debug/visualize.
    """
    annotated = image.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box[:4]]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        if texts and i < len(texts):
            text = texts[i]
            cv2.putText(
                annotated, text, (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA
            )
    return annotated

