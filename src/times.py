"""Separação dos times pela cor da camisa (abordagem clássica).

Para cada jogador recorta o tronco, joga fora os pixels de grama e resume a
cor da camisa num vetor Lab. Um K-means separa as cores; os dois grupos
grandes viram os times e o resto (árbitro, banco, goleiros) vira "outro".
"""
from __future__ import annotations

from collections import Counter, defaultdict

import cv2
import numpy as np
from sklearn.cluster import KMeans

from campo import VERDE_MAX, VERDE_MIN

TIME_A, TIME_B, OUTRO = 0, 1, 2


def cor_camisa(frame_bgr: np.ndarray, xyxy: np.ndarray) -> np.ndarray | None:
    """Cor média (Lab) da região do tronco, ignorando a grama."""
    x1, y1, x2, y2 = xyxy.astype(int)
    w, h = x2 - x1, y2 - y1
    if w < 4 or h < 8:
        return None
    # tronco: faixa central, abaixo da cabeça e acima do calção
    tronco = frame_bgr[y1 + int(0.15 * h): y1 + int(0.50 * h),
                       x1 + int(0.25 * w): x1 + int(0.75 * w)]
    if tronco.size == 0:
        return None

    hsv = cv2.cvtColor(tronco, cv2.COLOR_BGR2HSV)
    nao_grama = cv2.inRange(hsv, VERDE_MIN, VERDE_MAX) == 0
    if nao_grama.sum() < 5:
        return None
    # Lab em vez de HSV: distância euclidiana faz sentido e não tem o
    # problema do matiz circular (vermelho em H=0 e H=179)
    lab = cv2.cvtColor(tronco, cv2.COLOR_BGR2LAB)
    return lab[nao_grama].mean(axis=0)


class ClassificadorTimes:
    """K-means com mais grupos que times; cada grupo é mapeado para A, B ou outro.

    Grupos extras absorvem árbitros, banco e variações de luz de uma mesma
    camisa; as variações são reagrupadas no time correspondente.
    """
    N_GRUPOS = 5
    PESO_BRILHO = 0.5              # L varia com sol/sombra; pesa menos que a cor (a, b)
    DIST_TIMES_DIFERENTES = 25.0   # distância mínima entre os times (espaço ponderado)
    CROMA_MESMA_CAMISA = 10.0      # diferença máxima de cor para juntar grupos
    BRILHO_MESMA_CAMISA = 30.0     # diferença máxima de brilho para juntar grupos

    def __init__(self, permitir_outro: bool = True):
        """permitir_outro=False quando o detector já separa árbitro e goleiro:
        aí todo "jogador" é de algum time, e grupos extras vão para o time mais próximo."""
        self.permitir_outro = permitir_outro
        self.kmeans = KMeans(n_clusters=self.N_GRUPOS, n_init=10, random_state=0)
        self.mapa = np.full(self.N_GRUPOS, OUTRO)
        self.votos: dict[int, Counter] = defaultdict(Counter)

    def _pesar(self, cores: np.ndarray) -> np.ndarray:
        return cores * np.array([self.PESO_BRILHO, 1.0, 1.0])

    def ajustar(self, cores: np.ndarray) -> None:
        grupos = self.kmeans.fit_predict(self._pesar(cores))
        centros = self.kmeans.cluster_centers_
        ordem = np.argsort(-np.bincount(grupos, minlength=self.N_GRUPOS))

        # A = grupo mais numeroso; B = maior grupo distante de A (ou o mais distante)
        a = ordem[0]
        dist = {g: float(np.linalg.norm(centros[g] - centros[a])) for g in ordem[1:]}
        candidatos = [g for g in ordem[1:] if dist[g] > self.DIST_TIMES_DIFERENTES]
        if candidatos:
            b = candidatos[0]
        else:
            b = max(dist, key=dist.get)
            print(f"AVISO: uniformes muito parecidos (distância {dist[b]:.0f} < "
                  f"{self.DIST_TIMES_DIFERENTES:.0f}); a separação dos times pode falhar.")
        self.mapa[a], self.mapa[b] = TIME_A, TIME_B

        for g in ordem:
            if g in (a, b):
                continue
            for centro_time, time_ in ((centros[a], TIME_A), (centros[b], TIME_B)):
                croma = np.linalg.norm(centros[g, 1:] - centro_time[1:])
                brilho = abs(centros[g, 0] - centro_time[0])
                if croma < self.CROMA_MESMA_CAMISA and brilho < self.BRILHO_MESMA_CAMISA:
                    self.mapa[g] = time_
            if self.mapa[g] == OUTRO and not self.permitir_outro:
                mais_perto_de_a = np.linalg.norm(centros[g] - centros[a]) <= np.linalg.norm(centros[g] - centros[b])
                self.mapa[g] = TIME_A if mais_perto_de_a else TIME_B

    def prever(self, cores: np.ndarray) -> np.ndarray:
        return self.mapa[self.kmeans.predict(self._pesar(cores))]

    def time_do_track(self, track_id: int, rotulo: int | None) -> int:
        """Voto da maioria ao longo do vídeo: um frame com sombra não troca o time.

        rotulo=None (cor não medida neste frame) não vota, só consulta o histórico.
        """
        if rotulo is not None:
            self.votos[track_id][rotulo] += 1
        if not self.votos[track_id]:
            return OUTRO
        return self.votos[track_id].most_common(1)[0][0]
