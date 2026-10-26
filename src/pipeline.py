"""Detecção + tracking + separação de times, com dois detectores possíveis:

- nível 1 (padrão): YOLO do COCO ("pessoa", "bola") + máscara do gramado,
  porque o modelo genérico também vê torcida e banco;
- nível 2 (--pesos best.pt do fine-tuning): o próprio detector já distingue
  jogador, goleiro, árbitro e bola.

Uso:
    python src/pipeline.py dados/videos/0bfacc_0.mp4 --max-frames 300
    python src/pipeline.py dados/videos/0bfacc_0.mp4 --pesos modelos/treino/por_jogo/weights/best.pt

Gera em saidas/ (<tag> = coco ou finetuned):
    <nome>_<tag>_anotado.mp4   vídeo com jogadores coloridos por time, ID e bola
    <nome>_<tag>_tracks.csv    uma linha por pessoa por frame (base para o mapa tático)
    <nome>_mascara.jpg         exemplo da máscara do gramado (para o relatório)
"""
from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from trackers import ByteTrackTracker
from ultralytics import YOLO

from campo import dentro_do_campo, mascara_gramado
from imagens import salvar_imagem
from times import OUTRO, TIME_A, TIME_B, ClassificadorTimes, cor_camisa

CONF_PESSOA, CONF_BOLA = 0.30, 0.10
GOLEIRO = 3
NOMES_TIME = {TIME_A: "A", TIME_B: "B", OUTRO: "outro", GOLEIRO: "goleiro"}
CORES_TIME = sv.ColorPalette.from_hex(["#00BFFF", "#FF1493", "#FFD700", "#7CFC00"])

# classe do detector -> papel da pessoa em campo
PAPEIS_COCO = {0: "jogador", 32: "bola"}
PAPEIS_FUTEBOL = {"player": "jogador", "goalkeeper": "goleiro", "referee": "arbitro", "ball": "bola"}

RAIZ = Path(__file__).resolve().parent.parent
COLUNAS = ["frame", "tempo_s", "track_id", "time", "conf", "x1", "y1", "x2", "y2", "pe_x", "pe_y"]


def eh_modelo_futebol(modelo: YOLO) -> bool:
    return "player" in modelo.names.values()


def detectar(modelo: YOLO, frame: np.ndarray) -> tuple[sv.Detections, sv.Detections]:
    """Devolve (pessoas, bola). Cada pessoa leva o papel em det.data['papel']."""
    if eh_modelo_futebol(modelo):
        papeis = {i: PAPEIS_FUTEBOL[n] for i, n in modelo.names.items()}
        det = sv.Detections.from_ultralytics(
            modelo(frame, conf=CONF_BOLA, imgsz=1280, verbose=False)[0])
    else:
        papeis = PAPEIS_COCO
        det = sv.Detections.from_ultralytics(
            modelo(frame, conf=CONF_BOLA, classes=list(papeis), imgsz=1280, verbose=False)[0])
        # o COCO vê "pessoa", inclusive torcida e banco: a máscara do gramado filtra
        det = det[dentro_do_campo(mascara_gramado(frame), det.xyxy)]

    det.data["papel"] = np.array([papeis[c] for c in det.class_id], dtype=object)
    pessoas = det[(det.data["papel"] != "bola") & (det.confidence >= CONF_PESSOA)]
    bolas = det[det.data["papel"] == "bola"]
    if len(bolas) > 1:  # só existe uma bola: fica a mais confiante
        bolas = bolas[[int(np.argmax(bolas.confidence))]]
    return pessoas, bolas


def cores_validas(frame: np.ndarray, det: sv.Detections) -> tuple[np.ndarray, np.ndarray]:
    cores, idx = [], []
    for i, caixa in enumerate(det.xyxy):
        cor = cor_camisa(frame, caixa)
        if cor is not None:
            cores.append(cor)
            idx.append(i)
    return np.array(cores), np.array(idx, dtype=int)


def ajustar_times(modelo: YOLO, video: str, total_frames: int, n_amostras: int = 30) -> ClassificadorTimes:
    """Primeira passada: junta cores de camisa de frames espalhados pelo vídeo e treina o K-means.

    Espalhar as amostras pelo vídeo inteiro (e não só o começo) evita que um
    início com close-up ou replay deixe o K-means sem jogadores de um dos times.
    """
    todas = []
    frames = sv.get_video_frames_generator(video, stride=max(1, total_frames // n_amostras))
    for frame in islice(frames, n_amostras):
        pessoas, _ = detectar(modelo, frame)
        pessoas = pessoas[pessoas.data["papel"] == "jogador"]
        cores, _ = cores_validas(frame, pessoas)
        if len(cores):
            todas.append(cores)
    if not todas:
        raise SystemExit("Nenhum jogador encontrado nas amostras do vídeo: é uma transmissão de futebol?")
    # com o detector de futebol, árbitro e goleiro vêm da própria rede: todo "jogador" tem time
    classificador = ClassificadorTimes(permitir_outro=not eh_modelo_futebol(modelo))
    classificador.ajustar(np.vstack(todas))
    return classificador


def processar(video: str, pesos: str, max_frames: int | None = None, saidas: Path | None = None) -> Path:
    """Roda detecção + tracking + times e devolve o caminho do CSV de tracks."""
    saidas = saidas or RAIZ / "saidas"
    saidas.mkdir(parents=True, exist_ok=True)

    modelo = YOLO(pesos)
    nome = Path(video).stem
    tag = "finetuned" if eh_modelo_futebol(modelo) else "coco"
    info = sv.VideoInfo.from_video_path(video)
    if max_frames:
        info.total_frames = min(info.total_frames, max_frames)
    print("Ajustando a separação de times...")
    times = ajustar_times(modelo, video, info.total_frames)

    # limiares mais baixos que o padrão: o YOLO genérico dá confiança modesta
    # para jogadores pequenos no fundo do campo
    rastreador = ByteTrackTracker(frame_rate=info.fps, track_activation_threshold=0.4,
                                  high_conf_det_threshold=0.5)
    # caixa + rótulo "#ID confiança" em cada pessoa; caixa branca + "bola confiança" na bola
    anot_caixa = sv.BoxAnnotator(color=CORES_TIME, color_lookup=sv.ColorLookup.CLASS, thickness=2)
    anot_rotulo = sv.LabelAnnotator(color=CORES_TIME, color_lookup=sv.ColorLookup.CLASS,
                               text_position=sv.Position.TOP_LEFT, text_scale=0.4, text_padding=3)
    branco = sv.Color.from_hex("#FFFFFF")
    anot_caixa_bola = sv.BoxAnnotator(color=branco, thickness=2)
    anot_rotulo_bola = sv.LabelAnnotator(color=branco, text_color=sv.Color.BLACK,
                                    text_position=sv.Position.TOP_CENTER, text_scale=0.4, text_padding=3)

    linhas = []
    frames = sv.get_video_frames_generator(video, end=info.total_frames)
    with sv.VideoSink(str(saidas / f"{nome}_{tag}_anotado.mp4"), info) as gravador:
        for n, frame in enumerate(frames):
            if n == 0:
                salvar_imagem(saidas / f"{nome}_mascara.jpg",
                              np.hstack([frame, cv2.cvtColor(mascara_gramado(frame), cv2.COLOR_GRAY2BGR)]))

            pessoas, bolas = detectar(modelo, frame)
            pessoas = rastreador.update(pessoas)
            pessoas = pessoas[pessoas.tracker_id != -1]  # -1 = track ainda não confirmado

            # time de cada track: papel do detector (árbitro/goleiro) ou K-means na
            # cor da camisa (jogador), sempre com voto da maioria no histórico do ID
            papel = pessoas.data["papel"]
            rotulos: list[int | None] = [None] * len(pessoas)
            for i, p in enumerate(papel):
                if p == "arbitro":
                    rotulos[i] = OUTRO
                elif p == "goleiro":
                    rotulos[i] = GOLEIRO
            jogadores = np.where(papel == "jogador")[0]
            cores, idx = cores_validas(frame, pessoas[jogadores])
            if len(idx):
                for i, r in zip(jogadores[idx], times.prever(cores)):
                    rotulos[i] = int(r)
            pessoas.class_id = np.array(
                [times.time_do_track(t, r) for t, r in zip(pessoas.tracker_id, rotulos)], dtype=int
            )

            for caixa, tid, time_, conf in zip(pessoas.xyxy, pessoas.tracker_id,
                                               pessoas.class_id, pessoas.confidence):
                linhas.append({
                    "frame": n, "tempo_s": round(n / info.fps, 3), "track_id": int(tid),
                    "time": NOMES_TIME[time_], "conf": round(float(conf), 3),
                    "x1": caixa[0], "y1": caixa[1], "x2": caixa[2], "y2": caixa[3],
                    "pe_x": (caixa[0] + caixa[2]) / 2, "pe_y": caixa[3],
                })
            for caixa, conf in zip(bolas.xyxy, bolas.confidence):
                linhas.append({
                    "frame": n, "tempo_s": round(n / info.fps, 3), "track_id": -1,
                    "time": "bola", "conf": round(float(conf), 3),
                    "x1": caixa[0], "y1": caixa[1], "x2": caixa[2], "y2": caixa[3],
                    "pe_x": (caixa[0] + caixa[2]) / 2, "pe_y": (caixa[1] + caixa[3]) / 2,
                })

            anotado = anot_caixa.annotate(frame.copy(), pessoas)
            anotado = anot_rotulo.annotate(anotado, pessoas, labels=[
                f"#{t} {c:.0%}" for t, c in zip(pessoas.tracker_id, pessoas.confidence)])
            anotado = anot_caixa_bola.annotate(anotado, bolas)
            anotado = anot_rotulo_bola.annotate(anotado, bolas, labels=[f"bola {c:.0%}" for c in bolas.confidence])
            gravador.write_frame(anotado)

            if n % 50 == 0:
                print(f"frame {n}/{info.total_frames}")

    csv = saidas / f"{nome}_{tag}_tracks.csv"
    df = pd.DataFrame(linhas, columns=COLUNAS).round(1)
    df.to_csv(csv, index=False)
    resumir(df, info)
    return csv


def indicadores(df: pd.DataFrame, fps: float, total_frames: int) -> dict:
    """Indicadores de tracking para comparar detectores no mesmo vídeo."""
    pessoas = df[df.track_id >= 0]
    por_id = pessoas.groupby("track_id").agg(frames=("frame", "nunique"), time=("time", "last"))
    return {
        "frames": total_frames,
        "ids_pessoa": len(por_id),
        "duracao_mediana_id_s": round(por_id.frames.median() / fps, 1),
        "pessoas_por_frame": int(pessoas.groupby("frame").size().median()),
        "bola_frames_pct": round(100 * df[df.time == "bola"].frame.nunique() / total_frames, 1),
        "ids_por_categoria": {k: int(v) for k, v in por_id.time.value_counts().items()},
    }


def resumir(df: pd.DataFrame, info: sv.VideoInfo) -> None:
    ind = indicadores(df, info.fps, info.total_frames)
    print(f"\nPronto: {ind['frames']} frames")
    print(f"  IDs de pessoa:            {ind['ids_pessoa']}")
    print(f"  duração mediana de um ID: {ind['duracao_mediana_id_s']} s")
    print(f"  pessoas por frame (med.): {ind['pessoas_por_frame']}")
    print(f"  bola vista em:            {ind['bola_frames_pct']}% dos frames")
    print(f"  IDs por categoria:        {ind['ids_por_categoria']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--pesos", default=str(RAIZ / "modelos" / "yolo11m.pt"))
    ap.add_argument("--max-frames", type=int, default=None)
    a = ap.parse_args()
    processar(a.video, a.pesos, a.max_frames)
