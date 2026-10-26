"""Leitura/gravação de imagens que funciona com acentos e "°" no caminho.

No Windows, cv2.imread/cv2.imwrite não abrem caminhos fora do ASCII (a pasta
da disciplina se chama "15° disciplina") e falham em silêncio: imread devolve
None e imwrite devolve False. imdecode/imencode trabalham em memória e o
NumPy cuida do arquivo. (Vídeo via cv2.VideoCapture/VideoWriter não tem o
problema.)
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def ler_imagem(caminho: str | Path) -> np.ndarray:
    imagem = cv2.imdecode(np.fromfile(str(caminho), dtype=np.uint8), cv2.IMREAD_COLOR)
    if imagem is None:
        raise ValueError(f"não consegui ler a imagem {caminho}")
    return imagem


def salvar_imagem(caminho: str | Path, imagem: np.ndarray) -> None:
    ok, dados = cv2.imencode(Path(caminho).suffix, imagem)
    if not ok:
        raise ValueError(f"não consegui codificar a imagem {caminho}")
    dados.tofile(str(caminho))
