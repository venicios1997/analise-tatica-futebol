"""Analisa um vídeo de transmissão de futebol, de ponta a ponta.

    python src/analisar.py meu_jogo.mp4
    python src/analisar.py                 # abre uma janela para escolher o vídeo
    (ou dois cliques em analisar.bat / arrastar o vídeo para cima dele)

Etapas: detecção + tracking + times (modelo treinado) -> pontos do campo +
homografia -> radar, mapa de calor, distância, velocidade, posse -> relatório.
Tudo vai para resultados/<nome do vídeo>/ e o relatório abre no navegador.

O sistema foi feito para a câmera principal de TV (aberta, lateral). Close-ups,
replays e câmeras atrás do gol não têm campo suficiente para a homografia;
esses trechos aparecem como "sem visão do campo" e o relatório avisa.
"""
from __future__ import annotations

import argparse
import json
import time
import webbrowser
from datetime import datetime
from html import escape
from pathlib import Path

import pandas as pd
import supervision as sv

import pipeline
import tatico
from relatorio import CSS, ffmpeg, figura, tabela

RAIZ = Path(__file__).resolve().parent.parent
PESOS_DETECTOR = RAIZ / "modelos/treino/por_jogo/weights/best.pt"
PESOS_CAMPO = RAIZ / "modelos/treino/campo/weights/best.pt"
H264 = ["-c:v", "libx264", "-preset", "fast", "-crf", "26", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]


def escolher_video() -> str:
    from tkinter import Tk, filedialog
    raiz = Tk()
    raiz.withdraw()
    raiz.attributes("-topmost", True)
    caminho = filedialog.askopenfilename(
        title="Escolha um vídeo de futebol",
        filetypes=[("Vídeos", "*.mp4 *.avi *.mov *.mkv *.m4v"), ("Todos os arquivos", "*.*")])
    raiz.destroy()
    if not caminho:
        raise SystemExit("Nenhum vídeo escolhido.")
    return caminho


def alertas(ind: dict, resumo: dict) -> list[str]:
    """Sinais de que o vídeo foge do que o modelo conhece."""
    avisos = []
    cobertura = resumo["frames_com_homografia"] / max(resumo["frames"], 1)
    if cobertura < 0.8:
        avisos.append(f"Só {cobertura:.0%} dos frames mostram campo suficiente para o mapa tático. "
                      "Close-ups, replays ou câmeras atrás do gol ficam fora do radar e das métricas.")
    if ind["pessoas_por_frame"] < 10:
        avisos.append(f"Mediana de {ind['pessoas_por_frame']} pessoas por frame: pouco para uma câmera "
                      "aberta de transmissão. O vídeo pode ser de outro ângulo, e as métricas ficam incompletas.")
    if ind["bola_frames_pct"] < 30:
        avisos.append(f"Bola encontrada em só {ind['bola_frames_pct']:.0f}% dos frames; a posse de bola é pouco confiável.")
    if resumo["posicoes_dentro_do_campo_pct"] < 95:
        avisos.append(f"{100 - resumo['posicoes_dentro_do_campo_pct']:.0f}% das posições caíram fora do campo: "
                      "a homografia pode estar errando neste vídeo (ângulo ou campo incomum).")
    return avisos


def relatorio_video(pasta: Path, nome: str, info: sv.VideoInfo, ind: dict, resumo: dict,
                    segundos: float) -> Path:
    jog = pd.read_csv(pasta / f"{nome}_jogadores.csv").rename(columns={
        "track_id": "ID", "time": "time", "duracao_s": "tempo em cena (s)", "distancia_m": "distância (m)",
        "vel_media_kmh": "vel. média (km/h)", "vel_max_kmh": "vel. máx. 1 s (km/h)"})
    avisos = alertas(ind, resumo)
    bloco_avisos = "".join(f'<div class="aviso">{escape(a)}</div>' for a in avisos) or \
        '<p class="nota">Nenhum alerta: o vídeo se parece com uma transmissão de câmera principal.</p>'
    cobertura = resumo["frames_com_homografia"] / max(resumo["frames"], 1)
    posse = resumo["posse"]
    numeros = [
        (f"{info.total_frames / info.fps:.0f} s", f"{info.width}×{info.height}, {info.fps:.0f} fps"),
        (str(ind["pessoas_por_frame"]), "pessoas por frame (mediana)"),
        (f"{ind['bola_frames_pct']:.0f}%", "frames com a bola"),
        (f"{cobertura:.0%}", "frames com mapa tático"),
        (f"{posse['A']:.0f}% × {posse['B']:.0f}%", f"posse A × B ({posse['frames_com_dono']} frames com dono)"),
        (str(ind["ids_pessoa"]), "IDs de pessoa no vídeo"),
    ]
    cartoes = "".join(f"<div><span>{escape(v)}</span>{escape(r)}</div>" for v, r in numeros)
    tabela_jog = tabela(jog) if len(jog) else '<p class="nota">Nenhum jogador ficou 5 s ou mais em cena com mapa tático.</p>'

    html = f"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Análise: {escape(nome)}</title><style>{CSS}</style></head>
<body><header>
  <p class="sobre">Análise tática de futebol · resultado por vídeo</p>
  <h1>{escape(nome)}</h1>
  <p class="lead">Analisado em {datetime.now():%d/%m/%Y %H:%M}, em {segundos / 60:.1f} min.
  Caixas com a confiança de cada detecção; cores de time aprendidas deste vídeo
  (<strong>A</strong> em azul, <strong>B</strong> em rosa, arbitragem em amarelo, goleiros em verde).</p>
</header><main>
<section><h2>Resumo</h2><div class="numeros">{cartoes}</div>{bloco_avisos}</section>
<section><h2>Vídeo com detecções</h2>
  <video src="web/anotado.mp4" controls muted loop playsinline preload="metadata"></video>
  <p class="nota">Rótulo <code>#ID confiança</code>: o ID segue a pessoa ao longo do vídeo; a confiança é o quanto o modelo
  está seguro daquela detecção naquele frame.</p></section>
<section><h2>Mapa tático</h2>
  <video src="web/tatico.mp4" controls muted loop playsinline preload="metadata"></video>
  <p class="nota">Linhas rosa: o campo oficial reprojetado pela homografia. Quando elas casam com as linhas reais,
  as posições em metros são confiáveis (erro típico de ~3 m).</p>
  {figura(f"{nome}_calor.png", "Mapa de calor de cada time.", "estreita")}</section>
<section><h2>Jogadores (IDs com 5 s ou mais em cena)</h2>{tabela_jog}
  <p class="nota">Distância e velocidade herdam o erro da homografia; use para comparar jogadores do mesmo vídeo,
  não como medida exata.</p></section>
<section><h2>Arquivos</h2><ul>
  <li><code>{escape(nome)}_finetuned_tracks.csv</code>: cada detecção por frame (caixa em pixels, confiança, time).</li>
  <li><code>{escape(nome)}_posicoes.csv</code>: as mesmas detecções com posição no campo em metros.</li>
  <li><code>{escape(nome)}_jogadores.csv</code>: distância e velocidade por jogador.</li>
  <li><code>{escape(nome)}_resumo_tatico.json</code>: números do resumo.</li>
</ul></section>
</main><footer>Gerado por <code>src/analisar.py</code>. Modelos: detector YOLO11m treinado para futebol e modelo de 32 pontos do campo.</footer>
</body></html>"""
    destino = pasta / "relatorio.html"
    destino.write_text(html, encoding="utf-8")
    return destino


def analisar(video: str, max_frames: int | None = None, abrir: bool = True) -> Path:
    for pesos in (PESOS_DETECTOR, PESOS_CAMPO):
        if not pesos.exists():
            raise SystemExit(f"Modelo não encontrado: {pesos}\nTreine com: python src/treinar.py "
                             f"{'por_jogo' if pesos == PESOS_DETECTOR else 'campo'}")
    nome = Path(video).stem
    pasta = RAIZ / "resultados" / nome
    info = sv.VideoInfo.from_video_path(video)
    if max_frames:
        info.total_frames = min(info.total_frames, max_frames)
    print(f"Vídeo: {video}\n{info.total_frames} frames ({info.total_frames / info.fps:.0f} s), "
          f"{info.width}x{info.height}. Estimativa numa RTX 4060: ~{max(1, round(info.total_frames / 500))} min.\n")
    inicio = time.time()

    print("[1/3] Detecção, tracking e times")
    csv = pipeline.processar(video, str(PESOS_DETECTOR), max_frames, pasta)
    ind = pipeline.indicadores(pd.read_csv(csv), info.fps, info.total_frames)

    print("\n[2/3] Pontos do campo, homografia e métricas")
    resumo = tatico.processar(video, str(PESOS_CAMPO), pasta)

    print("\n[3/3] Relatório")
    (pasta / "web").mkdir(exist_ok=True)
    ffmpeg("-i", str(pasta / f"{nome}_finetuned_anotado.mp4"), "-vf", "scale='min(1280,iw)':-2,setsar=1",
           *H264, str(pasta / "web" / "anotado.mp4"))
    ffmpeg("-i", str(pasta / f"{nome}_tatico.mp4"), "-vf", "scale='min(1280,iw)':-2,setsar=1",
           *H264, str(pasta / "web" / "tatico.mp4"))
    (pasta / "indicadores.json").write_text(json.dumps(ind, indent=2), encoding="utf-8")
    relatorio = relatorio_video(pasta, nome, info, ind, resumo, time.time() - inicio)

    print(f"\nPronto em {(time.time() - inicio) / 60:.1f} min. Resultados em: {pasta}")
    for aviso in alertas(ind, resumo):
        print(f"  ALERTA: {aviso}")
    if abrir:
        webbrowser.open(relatorio.as_uri())
    return relatorio


def main() -> None:
    ap = argparse.ArgumentParser(description="Análise tática de um vídeo de futebol")
    ap.add_argument("video", nargs="?", help="caminho do vídeo; sem ele, abre uma janela para escolher")
    ap.add_argument("--max-frames", type=int, default=None, help="analisa só o começo (teste rápido)")
    ap.add_argument("--nao-abrir", action="store_true", help="não abre o relatório no navegador")
    a = ap.parse_args()
    analisar(a.video or escolher_video(), a.max_frames, abrir=not a.nao_abrir)


if __name__ == "__main__":
    main()
