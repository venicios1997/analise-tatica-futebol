"""Máscara do gramado por visão clássica: HSV + limiar + morfologia.

Serve para descartar detecções fora do campo (torcida, banco, gandulas),
que um YOLO genérico (COCO) também acha como "pessoa".
"""
from __future__ import annotations

import cv2
import numpy as np

# faixa de verde em HSV do OpenCV (H vai de 0 a 179)
VERDE_MIN = np.array([30, 40, 40])
VERDE_MAX = np.array([90, 255, 255])


def mascara_gramado(frame_bgr: np.ndarray) -> np.ndarray:
    """Devolve máscara binária (uint8 0/255) da região do campo."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    verde = cv2.inRange(hsv, VERDE_MIN, VERDE_MAX)

    # abertura tira pontinhos verdes soltos (placas, torcida);
    # fechamento tapa os "buracos" que os jogadores e as linhas deixam no gramado
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    verde = cv2.morphologyEx(verde, cv2.MORPH_OPEN, kernel)
    verde = cv2.morphologyEx(verde, cv2.MORPH_CLOSE, kernel, iterations=3)

    # o campo é a maior ilha verde; o casco convexo engloba os jogadores na borda
    contornos, _ = cv2.findContours(verde, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mascara = np.zeros_like(verde)
    if contornos:
        maior = max(contornos, key=cv2.contourArea)
        cv2.fillPoly(mascara, [cv2.convexHull(maior)], 255)
    return mascara


def dentro_do_campo(mascara: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    """Para cada caixa, verifica se o ponto dos pés (centro da base) está no campo."""
    if len(xyxy) == 0:
        return np.zeros(0, dtype=bool)
    h, w = mascara.shape
    x = ((xyxy[:, 0] + xyxy[:, 2]) / 2).astype(int).clip(0, w - 1)
    y = xyxy[:, 3].astype(int).clip(0, h - 1)
    return mascara[y, x] > 0
