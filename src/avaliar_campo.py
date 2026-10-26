"""Erro da homografia em METROS no teste do dataset de pontos do campo.

Para cada imagem de teste: o modelo prevê os pontos do campo, calcula-se a
homografia imagem -> campo (como no tatico.py) e projetam-se os pontos
ROTULADOS à mão; o erro é a distância, em metros, até a posição oficial.
Isso responde: "quando o sistema põe um jogador no mapa, erra por quanto?"

Referência: a mesma conta com a homografia ajustada nos próprios rótulos
mostra o piso do erro (imprecisão de anotação, distorção da lente, campo fora
do tamanho padrão de 120 x 70 m).

Uso:
    python src/avaliar_campo.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from campo_modelo import PONTOS
from imagens import ler_imagem
from tatico import LIMIAR_RANSAC_M, homografia, projetar, resolucao_de_treino

RAIZ = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pesos", default=str(RAIZ / "modelos/treino/campo/weights/best.pt"))
    ap.add_argument("--ransac", type=float, default=LIMIAR_RANSAC_M, help="limiar do RANSAC em metros")
    a = ap.parse_args()

    base = RAIZ / "dados" / "campo_sem_0bfacc" / "test"
    modelo = YOLO(a.pesos)
    print(f"Modelo: {a.pesos} (imgsz {resolucao_de_treino(modelo)})")
    erro_modelo, erro_piso, sem_h = [], [], 0
    for img_path in sorted((base / "images").glob("*.jpg")):
        imagem = ler_imagem(img_path)
        h, w = imagem.shape[:2]
        v = np.loadtxt(base / "labels" / (img_path.stem + ".txt"))[5:].reshape(32, 3)
        vis = v[:, 2] > 0
        pts_img = v[vis, :2] * [w, h]

        H, _ = homografia(modelo, imagem, a.ransac)
        if H is None:
            sem_h += 1
            continue
        erro_modelo.append(np.linalg.norm(projetar(H, pts_img) - PONTOS[vis], axis=1).mean())

        H_rot, _ = cv2.findHomography(pts_img, PONTOS[vis], 0)
        erro_piso.append(np.linalg.norm(projetar(H_rot, pts_img) - PONTOS[vis], axis=1).mean())

    em, ep = np.array(erro_modelo), np.array(erro_piso)
    resumo = {
        "modelo": Path(a.pesos).parent.parent.name, "imgsz": resolucao_de_treino(modelo),
        "limiar_ransac_m": a.ransac, "imagens": len(em) + sem_h, "sem_homografia": sem_h,
        "erro_mediana_m": round(float(np.median(em)), 2), "erro_p90_m": round(float(np.percentile(em, 90)), 2),
        "erro_max_m": round(float(em.max()), 2),
        "piso_mediana_m": round(float(np.median(ep)), 2), "piso_p90_m": round(float(np.percentile(ep, 90)), 2),
    }
    destino = RAIZ / "saidas" / f"erro_homografia_{resumo['modelo']}_ransac{a.ransac:g}m.json"
    destino.write_text(json.dumps(resumo, indent=2), encoding="utf-8")
    print(f"Teste: {len(em) + sem_h} imagens; sem homografia: {sem_h}")
    print(f"Erro com pontos PREVISTOS (sistema): mediana {np.median(em):.2f} m | p90 {np.percentile(em, 90):.2f} m | máx {em.max():.2f} m")
    print(f"Piso (homografia nos rótulos):        mediana {np.median(ep):.2f} m | p90 {np.percentile(ep, 90):.2f} m")


if __name__ == "__main__":
    main()
