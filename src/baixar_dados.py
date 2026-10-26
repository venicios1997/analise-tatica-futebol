"""Baixa o vídeo de demonstração e os dois datasets usados no trabalho.

- vídeo 0bfacc_0.mp4 (DFL Bundesliga), distribuído pelo repositório roboflow/sports
- martinjolif/football-player-detection (Hugging Face; espelho do Roboflow
  football-players-detection, CC BY 4.0): jogador, goleiro, árbitro, bola
- martinjolif/football-pitch-detection (Hugging Face; espelho do Roboflow
  football-field-detection, CC BY 4.0): 32 pontos-chave do campo

Uso:
    python src/baixar_dados.py
Depois: python src/preparar_dataset.py e python src/preparar_campo.py
"""
from __future__ import annotations

from pathlib import Path

import gdown
from huggingface_hub import snapshot_download

RAIZ = Path(__file__).resolve().parent.parent
DADOS = RAIZ / "dados"
VIDEO_ID = "12TqauVZ9tLAv8kWxTTBFWtgt2hNQ4_ZF"  # roboflow/sports, examples/soccer/setup.sh
DATASETS = {
    "hf_football_players": "martinjolif/football-player-detection",
    "hf_football_pitch": "martinjolif/football-pitch-detection",
}


def main() -> None:
    video = DADOS / "videos" / "0bfacc_0.mp4"
    if not video.exists():
        video.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(id=VIDEO_ID, output=str(video), quiet=False)
    print(f"vídeo: {video}")

    for pasta, repo in DATASETS.items():
        destino = DADOS / pasta
        snapshot_download(repo, repo_type="dataset", local_dir=str(destino), allow_patterns=["data/*"])
        n = len(list((destino / "data").glob("*/images/*.jpg")))
        print(f"{repo}: {n} imagens em {destino}")


if __name__ == "__main__":
    main()
