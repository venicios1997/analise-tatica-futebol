"""Geometria do campo oficial: os 32 pontos-chave e suas posições reais.

A ordem dos pontos é a do dataset de pontos do campo (Roboflow
football-field-detection), definida em roboflow/sports,
sports/configs/soccer.py. Unidade: metros. Origem no canto superior
esquerdo; x ao longo do comprimento (0-120), y ao longo da largura (0-70).
"""
from __future__ import annotations

import numpy as np

COMPRIMENTO, LARGURA = 120.0, 70.0
AREA_GRANDE_LARG, AREA_GRANDE_COMP = 41.0, 20.15
AREA_PEQ_LARG, AREA_PEQ_COMP = 18.32, 5.5
RAIO_CIRCULO, MARCA_PENALTI = 9.15, 11.0

_L, _W = COMPRIMENTO, LARGURA
_yg1, _yg2 = (_W - AREA_GRANDE_LARG) / 2, (_W + AREA_GRANDE_LARG) / 2
_yp1, _yp2 = (_W - AREA_PEQ_LARG) / 2, (_W + AREA_PEQ_LARG) / 2

PONTOS = np.array([
    (0, 0), (0, _yg1), (0, _yp1), (0, _yp2), (0, _yg2), (0, _W),                     # 1-6   linha de fundo esq.
    (AREA_PEQ_COMP, _yp1), (AREA_PEQ_COMP, _yp2),                                    # 7-8   pequena área esq.
    (MARCA_PENALTI, _W / 2),                                                         # 9     pênalti esq.
    (AREA_GRANDE_COMP, _yg1), (AREA_GRANDE_COMP, _yp1),
    (AREA_GRANDE_COMP, _yp2), (AREA_GRANDE_COMP, _yg2),                              # 10-13 grande área esq.
    (_L / 2, 0), (_L / 2, _W / 2 - RAIO_CIRCULO),
    (_L / 2, _W / 2 + RAIO_CIRCULO), (_L / 2, _W),                                   # 14-17 linha do meio
    (_L - AREA_GRANDE_COMP, _yg1), (_L - AREA_GRANDE_COMP, _yp1),
    (_L - AREA_GRANDE_COMP, _yp2), (_L - AREA_GRANDE_COMP, _yg2),                    # 18-21 grande área dir.
    (_L - MARCA_PENALTI, _W / 2),                                                    # 22    pênalti dir.
    (_L - AREA_PEQ_COMP, _yp1), (_L - AREA_PEQ_COMP, _yp2),                          # 23-24 pequena área dir.
    (_L, 0), (_L, _yg1), (_L, _yp1), (_L, _yp2), (_L, _yg2), (_L, _W),               # 25-30 linha de fundo dir.
    (_L / 2 - RAIO_CIRCULO, _W / 2), (_L / 2 + RAIO_CIRCULO, _W / 2),                # 31-32 círculo central
], dtype=np.float32)

# linhas do campo como pares de pontos (índices base 1, como no repositório)
LINHAS = [
    (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (7, 8),
    (10, 11), (11, 12), (12, 13), (14, 15), (15, 16),
    (16, 17), (18, 19), (19, 20), (20, 21), (23, 24),
    (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
    (1, 14), (2, 10), (3, 7), (4, 8), (5, 13), (6, 17),
    (14, 25), (18, 26), (23, 27), (24, 28), (21, 29), (17, 30),
]
