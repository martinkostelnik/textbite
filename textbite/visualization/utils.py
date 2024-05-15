"""Visualization utilities

Date -- 15.05.2024
Author -- Martin Kostelnik
"""


import cv2
import numpy as np


COLORS = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (34, 139, 34),    # Forest Green
    (70, 130, 180),   # Steel Blue
    (255, 20, 147),   # Deep Pink
    (218, 112, 214),  # Orchid
    (255, 165, 0),    # Orange
    (173, 216, 230),  # Light Blue
    (255, 69, 0),     # Red-Orange
    (0, 191, 255),    # Deep Sky Blue
    (128, 0, 128),    # Purple
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Cyan
    (255, 99, 71),    # Tomato
    (255, 192, 203),  # Pink
    (32, 178, 170),   # Light Sea Green
    (250, 128, 114),  # Salmon
    (0, 128, 128),    # Teal
    (240, 230, 140)   # Khaki
]


def overlay_line(img, line, color, alpha):
    mask = np.zeros_like(img)
    pts = line.polygon
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], color)
    return cv2.addWeighted(img, 1, mask, 1-alpha, 0)
