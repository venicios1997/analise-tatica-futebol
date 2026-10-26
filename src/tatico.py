"""Etapa tática: homografia frame a frame -> posições em metros -> radar, calor e métricas.

Roda depois do pipeline.py com o detector de futebol (nível 2):
    python src/tatico.py dados/videos/0bfacc_0.mp4

Lê  saidas/<nome>_finetuned_tracks.csv e saidas/<nome>_finetuned_anotado.mp4
Gera em saidas/:
    <nome>_tatico.mp4     vídeo anotado + linhas do campo reprojetadas + radar 2D
    <nome>_posicoes.csv   tracks com posição no campo (metros), bruta e suavizada
    <nome>_jogadores.csv  distância e velocidade por jogador
    <nome>_calor.png      mapa de calor de cada time
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from ultralytics import YOLO

from campo_modelo import COMPRIMENTO, LARGURA, LINHAS, MARCA_PENALTI, PONTOS, RAIO_CIRCULO
from imagens import salvar_imagem

RAIZ = Path(__file__).resolve().parent.parent
CONF_PONTO = 0.5      # confiança mínima de um ponto do campo para entrar na homografia
MIN_PONTOS = 4        # homografia precisa de pelo menos 4 correspondências
LIMIAR_RANSAC_M = 3.0  # tolerância do RANSAC em metros
JANELA_SUAVIZACAO = 9  # frames (~0,36 s a 25 fps): mediana móvel tira o tremor da homografia
VEL_MAX_MS = 12.0     # ~43 km/h: acima disso é salto de ID/homografia, não corrida
RAIO_POSSE_M = 2.0    # jogador a até 2 m da bola é quem está com ela
SEGUNDOS_REAPROVEITO = 0.5  # por quanto tempo uma homografia antiga cobre frames sem campo visível

CORES_BGR = {"A": (255, 191, 0), "B": (147, 20, 255), "outro": (0, 215, 255),
             "goleiro": (0, 252, 124), "bola": (255, 255, 255)}


# ---------------------------------------------------------------- homografia

def resolucao_de_treino(modelo: YOLO) -> int:
    return int(modelo.ckpt.get("train_args", {}).get("imgsz", 640))


def homografia(modelo_campo: YOLO, frame: np.ndarray,
               limiar_ransac_m: float = LIMIAR_RANSAC_M) -> tuple[np.ndarray | None, int]:
    """Matriz imagem -> campo (metros) e quantos pontos sobraram após o RANSAC."""
    r = modelo_campo(frame, imgsz=resolucao_de_treino(modelo_campo), verbose=False)[0]
    if r.keypoints is None or len(r.keypoints) == 0 or r.keypoints.conf is None:
        return None, 0
    xy = r.keypoints.xy[0].cpu().numpy()
    conf = r.keypoints.conf[0].cpu().numpy()
    ok = conf > CONF_PONTO
    if ok.sum() < MIN_PONTOS:
        return None, int(ok.sum())
    H, inliers = cv2.findHomography(xy[ok], PONTOS[ok], cv2.RANSAC, limiar_ransac_m)
    if H is None or inliers.sum() < MIN_PONTOS:
        return None, 0
    return H, int(inliers.sum())


def projetar(H: np.ndarray, pontos: np.ndarray) -> np.ndarray:
    if len(pontos) == 0:
        return np.zeros((0, 2))
    return cv2.perspectiveTransform(pontos.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)


def segmentos_do_campo(passos_circulo: int = 48) -> list[np.ndarray]:
    """Linhas do campo em metros (retas + círculo central), para desenhar."""
    segs = [PONTOS[[a - 1, b - 1]] for a, b in LINHAS]
    ang = np.linspace(0, 2 * np.pi, passos_circulo)
    centro = np.array([COMPRIMENTO / 2, LARGURA / 2])
    segs.append((centro + RAIO_CIRCULO * np.c_[np.cos(ang), np.sin(ang)]).astype(np.float32))
    return segs


SEGMENTOS = segmentos_do_campo()


def desenhar_linhas_projetadas(frame: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Desenha o campo oficial de volta na imagem: se casar com as linhas reais, H está certa."""
    H_inv = np.linalg.inv(H)
    camada = frame.copy()
    for seg in SEGMENTOS:
        pts = projetar(H_inv, seg).astype(np.int32)
        cv2.polylines(camada, [pts], isClosed=False, color=(255, 0, 255), thickness=2, lineType=cv2.LINE_AA)
    return cv2.addWeighted(camada, 0.6, frame, 0.4, 0)


# ---------------------------------------------------------------- radar 2D

class Radar:
    def __init__(self, escala: float = 8.0, margem: float = 5.0):
        self.escala, self.margem = escala, margem
        largura = int((COMPRIMENTO + 2 * margem) * escala)
        altura = int((LARGURA + 2 * margem) * escala)
        self.base = np.full((altura, largura, 3), (60, 140, 60), dtype=np.uint8)
        for seg in SEGMENTOS:
            cv2.polylines(self.base, [self.px(seg)], False, (255, 255, 255), 2, cv2.LINE_AA)
        for x in (MARCA_PENALTI, COMPRIMENTO - MARCA_PENALTI, COMPRIMENTO / 2):
            cv2.circle(self.base, tuple(self.px(np.array([[x, LARGURA / 2]]))[0]), 3, (255, 255, 255), -1)

    def px(self, metros: np.ndarray) -> np.ndarray:
        return ((np.asarray(metros) + self.margem) * self.escala).astype(np.int32)

    def desenhar(self, linhas: pd.DataFrame) -> np.ndarray:
        img = self.base.copy()
        for _, r in linhas.iterrows():
            if np.isnan(r.cx):
                continue
            p = tuple(self.px(np.array([[r.cx, r.cy]]))[0])
            if r.time == "bola":
                cv2.circle(img, p, 7, (0, 0, 0), -1)
                cv2.circle(img, p, 5, CORES_BGR["bola"], -1)
            else:
                cv2.circle(img, p, 11, CORES_BGR.get(r.time, (128, 128, 128)), -1)
                cv2.circle(img, p, 11, (0, 0, 0), 2)
        return img

    def calor(self, x: np.ndarray, y: np.ndarray, titulo: str) -> np.ndarray:
        h, _, _ = np.histogram2d(y, x, bins=[int(LARGURA), int(COMPRIMENTO)],
                                 range=[[0, LARGURA], [0, COMPRIMENTO]])
        h = cv2.GaussianBlur(h, (0, 0), 2.0)
        h = (255 * h / h.max()).astype(np.uint8) if h.max() > 0 else h.astype(np.uint8)
        alto, largo = int(LARGURA * self.escala), int(COMPRIMENTO * self.escala)
        cor = cv2.applyColorMap(cv2.resize(h, (largo, alto)), cv2.COLORMAP_INFERNO)
        img = self.base.copy()
        m = int(self.margem * self.escala)
        regiao = img[m:m + alto, m:m + largo]
        img[m:m + alto, m:m + largo] = cv2.addWeighted(regiao, 0.35, cor, 0.65, 0)
        for seg in SEGMENTOS:  # linhas por cima do calor
            cv2.polylines(img, [self.px(seg)], False, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, titulo, (m, m - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        return img


# ---------------------------------------------------------------- métricas

def suavizar(df: pd.DataFrame) -> pd.DataFrame:
    """Mediana móvel por ID (e para a bola) sobre a posição em metros."""
    df = df.sort_values(["track_id", "frame"]).copy()
    grupos = df.groupby("track_id")
    for c in ("x", "y"):
        df["c" + c] = grupos["campo_" + c].transform(
            lambda s: s.rolling(JANELA_SUAVIZACAO, center=True, min_periods=1).median())
    return df


def metricas_jogadores(df: pd.DataFrame, fps: float) -> pd.DataFrame:
    jog = df[(df.track_id >= 0) & df.time.isin(["A", "B", "goleiro"])].sort_values(["track_id", "frame"])
    linhas = []
    for tid, g in jog.groupby("track_id"):
        g = g.dropna(subset=["cx"])
        if len(g) < fps * 5:  # IDs com menos de 5 s não dão métrica confiável
            continue
        consecutivo = g.frame.diff() == 1
        passo = np.hypot(g.cx.diff(), g.cy.diff())
        vel = passo * fps
        valido = consecutivo & (vel < VEL_MAX_MS)
        # velocidade máxima = maior média de 1 s (picos de 1 frame são ruído)
        vel_1s = vel.where(valido).rolling(int(fps), min_periods=int(fps * 0.8)).mean()
        duracao = len(g) / fps
        distancia = passo[valido].sum()
        linhas.append({
            "track_id": tid, "time": g.time.mode().iat[0], "duracao_s": round(duracao, 1),
            "distancia_m": round(distancia, 1),
            "vel_media_kmh": round(3.6 * distancia / duracao, 1),
            "vel_max_kmh": round(3.6 * vel_1s.max(), 1),
        })
    colunas = ["track_id", "time", "duracao_s", "distancia_m", "vel_media_kmh", "vel_max_kmh"]
    return pd.DataFrame(linhas, columns=colunas).sort_values("distancia_m", ascending=False)


def posse(df: pd.DataFrame) -> dict[str, float]:
    """% dos frames com a bola a até RAIO_POSSE_M de um jogador de cada time."""
    bola = df[df.time == "bola"].set_index("frame")[["cx", "cy"]].dropna()
    jog = df[df.time.isin(["A", "B"])].dropna(subset=["cx"])
    donos = []
    for frame, g in jog.groupby("frame"):
        if frame not in bola.index:
            continue
        b = bola.loc[frame]
        d = np.hypot(g.cx - b.cx, g.cy - b.cy)
        if d.min() <= RAIO_POSSE_M:
            donos.append(g.time.iat[int(np.argmin(d.values))])
    contagem = pd.Series(donos).value_counts(normalize=True)
    return {t: round(100 * float(contagem.get(t, 0.0)), 1) for t in ("A", "B")} | {"frames_com_dono": len(donos)}


# ---------------------------------------------------------------- principal

def processar(video: str, pesos_campo: str, saidas: Path | None = None) -> dict:
    """Homografia + radar + métricas; devolve o resumo (também salvo em JSON)."""
    nome = Path(video).stem
    saidas = saidas or RAIZ / "saidas"
    tracks = pd.read_csv(saidas / f"{nome}_finetuned_tracks.csv")
    anotado = str(saidas / f"{nome}_finetuned_anotado.mp4")
    info = sv.VideoInfo.from_video_path(anotado)
    modelo_campo = YOLO(pesos_campo)

    # 1ª passada: homografia de cada frame. Em falhas curtas reaproveita a
    # anterior; falhas longas (cortes de cena) ficam sem homografia.
    print("Calculando homografias...")
    max_reuso = int(info.fps * SEGUNDOS_REAPROVEITO)
    Hs, n_pontos = [], []
    H_anterior, falhas_seguidas = None, 0
    for frame in sv.get_video_frames_generator(video, end=info.total_frames):
        H, n = homografia(modelo_campo, frame)
        if H is None:
            falhas_seguidas += 1
            H = H_anterior if falhas_seguidas <= max_reuso else None
        else:
            falhas_seguidas = 0
            H_anterior = H
        Hs.append(H)
        n_pontos.append(n)

    # posições no campo
    tracks["campo_x"] = np.nan
    tracks["campo_y"] = np.nan
    for frame, idx in tracks.groupby("frame").groups.items():
        H = Hs[frame] if frame < len(Hs) else None
        if H is None:
            continue
        xy = projetar(H, tracks.loc[idx, ["pe_x", "pe_y"]].to_numpy())
        tracks.loc[idx, "campo_x"], tracks.loc[idx, "campo_y"] = xy[:, 0], xy[:, 1]
    tracks = suavizar(tracks)
    tracks.round(2).to_csv(saidas / f"{nome}_posicoes.csv", index=False)

    # 2ª passada: vídeo anotado + linhas reprojetadas, com o radar numa faixa abaixo
    print("Gerando vídeo tático...")
    radar = Radar()
    por_frame = dict(tuple(tracks.groupby("frame")))
    vazio = tracks.iloc[0:0]
    larg_radar = info.width // 2
    alt_radar = int(larg_radar * radar.base.shape[0] / radar.base.shape[1]) // 2 * 2
    saida = sv.VideoInfo(width=info.width, height=info.height + alt_radar, fps=info.fps)
    with sv.VideoSink(str(saidas / f"{nome}_tatico.mp4"), saida) as gravador:
        for n, frame in enumerate(sv.get_video_frames_generator(anotado)):
            mini = radar.desenhar(por_frame.get(n, vazio))
            if Hs[n] is not None:
                frame = desenhar_linhas_projetadas(frame, Hs[n])
            else:
                cv2.putText(mini, "sem visao do campo neste trecho", (40, mini.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)
            faixa = np.full((alt_radar, info.width, 3), 30, dtype=np.uint8)
            x0 = (info.width - larg_radar) // 2
            faixa[:, x0:x0 + larg_radar] = cv2.resize(mini, (larg_radar, alt_radar))
            gravador.write_frame(np.vstack([frame, faixa]))

    # mapas de calor e métricas
    calor = [radar.calor(g.cx.dropna().to_numpy(), g.cy.dropna().to_numpy(), f"Time {t}")
             for t, g in ((t, tracks[tracks.time == t]) for t in ("A", "B"))]
    salvar_imagem(saidas / f"{nome}_calor.png", np.vstack(calor))

    jogadores = metricas_jogadores(tracks, info.fps)
    jogadores.to_csv(saidas / f"{nome}_jogadores.csv", index=False)

    pessoas = tracks[tracks.track_id >= 0].dropna(subset=["campo_x"])
    dentro = ((pessoas.campo_x.between(-3, COMPRIMENTO + 3)) & (pessoas.campo_y.between(-3, LARGURA + 3))).mean() \
        if len(pessoas) else 0.0
    resumo = {
        "frames": len(Hs),
        "frames_homografia_propria": sum(n >= MIN_PONTOS for n in n_pontos),
        "frames_com_homografia": sum(H is not None for H in Hs),
        "pontos_apos_ransac_mediana": float(np.median(n_pontos)) if n_pontos else 0.0,
        "fps": info.fps,
        "posicoes_dentro_do_campo_pct": round(100 * float(dentro), 1),
        "posse": posse(tracks),
        "limiar_ransac_m": LIMIAR_RANSAC_M,
        "pesos_campo": Path(pesos_campo).parent.parent.name,
    }
    (saidas / f"{nome}_resumo_tatico.json").write_text(json.dumps(resumo, indent=2), encoding="utf-8")
    print(f"\nHomografia própria em {resumo['frames_homografia_propria']}/{len(Hs)} frames "
          f"(mediana de {resumo['pontos_apos_ransac_mediana']:.0f} pontos após RANSAC); "
          f"com reaproveitamento: {resumo['frames_com_homografia']}/{len(Hs)}")
    print(f"Posições de pessoas dentro do campo (±3 m): {dentro:.1%}")
    print(f"Posse de bola: {resumo['posse']}")
    print("\nJogadores (IDs com 5 s ou mais):")
    print(jogadores.to_string(index=False))
    return resumo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--pesos-campo", default=str(RAIZ / "modelos/treino/campo/weights/best.pt"))
    a = ap.parse_args()
    processar(a.video, a.pesos_campo)
