"""Os treinos do trabalho, com os parâmetros exatos usados.

    python src/treinar.py por_jogo   # detector de futebol (nível 2)      ~45 min
    python src/treinar.py por_foto   # mesmo detector, split com vazamento ~30 min
    python src/treinar.py campo      # 32 pontos do campo (homografia)     ~15 min
    python src/treinar.py campo_1280 # pontos do campo em 1280 px      ~55 min

Tempos medidos numa RTX 4060 (8 GB); os lotes são o máximo que cabe nessa placa.
Saída: modelos/treino/<nome>/weights/best.pt
"""
from __future__ import annotations

import sys
from pathlib import Path

from ultralytics import YOLO

RAIZ = Path(__file__).resolve().parent.parent
DADOS = RAIZ / "dados"

TREINOS = {
    # a bola tem ~11 px de largura em 1920 px; em 640 viraria ~4 px -> imgsz 1280
    "por_jogo": dict(model="yolo11m.pt", data=DADOS / "futebol_por_jogo/data.yaml",
                     imgsz=1280, batch=3, epochs=80, patience=20),
    "por_foto": dict(model="yolo11m.pt", data=DADOS / "futebol_por_foto/data.yaml",
                     imgsz=1280, batch=3, epochs=80, patience=20),
    # mosaic=0: o mosaico recorta e cola 4 imagens e destrói a geometria do campo inteiro
    "campo": dict(model="yolo11m-pose.pt", data=DADOS / "campo_sem_0bfacc/data.yaml",
                  imgsz=640, batch=8, epochs=100, patience=25, mosaic=0),
    # teste em 1280 px (ficou pior que o de 640 px, ver README). Sem parada antecipada
    # (patience=0): a caixa converge cedo e a pose só melhora depois, então o
    # critério de parada interrompia o treino antes de os pontos serem aprendidos
    "campo_1280": dict(model="yolo11m-pose.pt", data=DADOS / "campo_sem_0bfacc/data.yaml",
                       imgsz=1280, batch=3, epochs=120, patience=0, mosaic=0),
}


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in TREINOS:
        sys.exit(f"uso: python src/treinar.py [{'|'.join(TREINOS)}]")
    nome = sys.argv[1]
    cfg = dict(TREINOS[nome])
    modelo = YOLO(str(RAIZ / "modelos" / cfg.pop("model")))  # baixa os pesos COCO se faltarem
    modelo.train(data=str(cfg.pop("data")), project=str(RAIZ / "modelos" / "treino"), name=nome,
                 exist_ok=True, seed=0, workers=2, plots=True, **cfg)


if __name__ == "__main__":
    main()
