#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script chuyển đổi file metadata từ định dạng Parquet sang CSV.
"""

from pathlib import Path
import pandas as pd

# ==============================================================================
# ĐIỀU CHỈNH ĐƯỜNG DẪN TRỰC TIẾP TẠI ĐÂY
# ==============================================================================
PARQUET_PATH = Path(r"E:\SGOD\dataset_congbao\metadata.parquet")
CSV_PATH = Path(r"E:\SGOD\dataset_congbao\metadata.csv")
# ==============================================================================


def convert_parquet_to_csv(parquet_file: Path, csv_file: Path):
    if not parquet_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file nguồn: {parquet_file}")

    print(f"[*] Đang đọc file Parquet: {parquet_file} ...")
    df = pd.read_parquet(parquet_file)

    # Đảm bảo thư mục lưu tồn tại
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    # Xuất file CSV (dùng utf-8-sig để hiển thị chuẩn tiếng Việt trong Excel)
    df.to_csv(csv_file, index=False, encoding="utf-8-sig")

    print(f"[+] Chuyển đổi thành công!")
    print(f"  - Số dòng : {len(df):,}")
    print(f"  - Các cột : {df.columns.tolist()}")
    print(f"  - File CSV: {csv_file.resolve()}")


if __name__ == "__main__":
    convert_parquet_to_csv(PARQUET_PATH, CSV_PATH)
