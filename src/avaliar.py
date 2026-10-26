"""Compara detectores no teste por jogo (jogos que nenhum modelo viu no treino).

O YOLO do COCO não sabe o que é goleiro ou árbitro, então a comparação é
feita num espaço comum de 2 classes: "pessoa em campo" e "bola".

Uso:
    python src/avaliar.py --treinado modelos/treino/por_jogo/weights/best.pt
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import supervision as sv
from supervision.metrics import MeanAveragePrecision
from ultralytics import YOLO

from campo import dentro_do_campo, mascara_gramado
from imagens import ler_imagem

RAIZ = Path(__file__).resolve().parent.parent
PESSOA, BOLA = 0, 1
# classes do dataset: 0 ball, 1 goalkeeper, 2 player, 3 referee
MAPA_DATASET = {0: BOLA, 1: PESSOA, 2: PESSOA, 3: PESSOA}
MAPA_COCO = {0: PESSOA, 32: BOLA}


def remapear(det: sv.Detections, mapa: dict[int, int]) -> sv.Detections:
    det = det[np.isin(det.class_id, list(mapa))]
    det.class_id = np.array([mapa[c] for c in det.class_id], dtype=int)
    return det


def avaliar(nome: str, modelo: YOLO, mapa: dict[int, int], dataset: sv.DetectionDataset,
            filtrar_campo: bool = False) -> dict:
    preds, alvos, tempos = [], [], []
    for caminho in dataset.image_paths:
        alvo = dataset.annotations[caminho]
        imagem = ler_imagem(caminho)
        t0 = time.perf_counter()
        r = modelo(imagem, imgsz=1280, conf=0.05, verbose=False)[0]
        tempos.append(time.perf_counter() - t0)
        det = remapear(sv.Detections.from_ultralytics(r), mapa)
        if filtrar_campo:
            det = det[dentro_do_campo(mascara_gramado(imagem), det.xyxy)]
        preds.append(det)
        alvos.append(remapear(alvo, MAPA_DATASET))

    res = MeanAveragePrecision().update(preds, alvos).compute()
    ap50 = dict(zip(res.matched_classes, res.ap_per_class[:, 0]))
    ap = dict(zip(res.matched_classes, res.ap_per_class.mean(axis=1)))

    # recall a um limiar de uso real (0.3): quantas bolas/pessoas o modelo acha de fato
    def recall(classe, conf=0.3, iou=0.5):
        achados = total = 0
        for p, a in zip(preds, alvos):
            p = p[(p.class_id == classe) & (p.confidence >= conf)]
            a = a[a.class_id == classe]
            total += len(a)
            if len(a) and len(p):
                achados += int((sv.box_iou_batch(a.xyxy, p.xyxy).max(axis=1) >= iou).sum())
        return achados / total if total else float("nan")

    return {
        "modelo": nome,
        "mAP50 pessoa": ap50.get(PESSOA, 0.0), "mAP50 bola": ap50.get(BOLA, 0.0),
        "mAP50-95 pessoa": ap.get(PESSOA, 0.0), "mAP50-95 bola": ap.get(BOLA, 0.0),
        "recall pessoa@0.3": recall(PESSOA), "recall bola@0.3": recall(BOLA),
        "ms/imagem": 1000 * np.median(tempos),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--treinado", nargs="+", default=[str(RAIZ / "modelos/treino/por_jogo/weights/best.pt")])
    ap.add_argument("--coco", default=str(RAIZ / "modelos" / "yolo11m.pt"))
    ap.add_argument("--sem-coco", action="store_true")
    ap.add_argument("--teste", default="futebol_por_jogo",
                    help="pasta em dados/ cujo test/ será usado (futebol_por_jogo = jogos inéditos)")
    a = ap.parse_args()

    base = RAIZ / "dados" / a.teste
    dataset = sv.DetectionDataset.from_yolo(str(base / "test/images"), str(base / "test/labels"),
                                            str(base / "data.yaml"))
    print(f"Teste {a.teste}: {len(dataset)} imagens\n")

    linhas = []
    if not a.sem_coco:
        coco = YOLO(a.coco)
        linhas += [avaliar("COCO genérico", coco, MAPA_COCO, dataset),
                   avaliar("COCO + máscara do gramado (nível 1)", coco, MAPA_COCO, dataset,
                           filtrar_campo=True)]
    for pesos in a.treinado:
        nome = f"fine-tuned ({Path(pesos).parent.parent.name})"
        linhas.append(avaliar(nome, YOLO(pesos), MAPA_DATASET, dataset))

    df = pd.DataFrame(linhas).set_index("modelo").round(3)
    print(df.to_string())
    df.to_csv(RAIZ / "saidas" / f"comparacao_detectores_{a.teste}.csv")


if __name__ == "__main__":
    main()
