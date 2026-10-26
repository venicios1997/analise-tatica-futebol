"""Prepara o dataset de pontos do campo (pose YOLO, 32 pontos) sem o jogo do vídeo.

O dataset (martinjolif/football-pitch-detection, espelho do Roboflow
football-field-detection, CC BY 4.0) contém frames do jogo 0bfacc, o mesmo do
vídeo de demonstração. Esses frames saem de todos os splits para que o vídeo
continue sendo um jogo inédito também para o modelo de campo.

Uso:
    python src/preparar_campo.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
ORIGEM = RAIZ / "dados" / "hf_football_pitch" / "data"
DESTINO = RAIZ / "dados" / "campo_sem_0bfacc"
JOGOS_EXCLUIDOS = {"0bfacc"}

# do data.yaml original: espelhamento horizontal troca os lados do campo
FLIP_IDX = [24, 25, 26, 27, 28, 29, 22, 23, 21, 17, 18, 19, 20, 13, 14, 15, 16,
            9, 10, 11, 12, 8, 6, 7, 0, 1, 2, 3, 4, 5, 31, 30]


def main() -> None:
    if DESTINO.exists():
        shutil.rmtree(DESTINO)
    for split in ("train", "valid", "test"):
        (DESTINO / split / "images").mkdir(parents=True)
        (DESTINO / split / "labels").mkdir(parents=True)
        mantidas = excluidas = 0
        for img in sorted((ORIGEM / split / "images").glob("*.jpg")):
            if img.name[:6] in JOGOS_EXCLUIDOS:
                excluidas += 1
                continue
            rotulo = ORIGEM / split / "labels" / (img.stem + ".txt")
            shutil.copy2(img, DESTINO / split / "images" / img.name)
            shutil.copy2(rotulo, DESTINO / split / "labels" / rotulo.name)
            mantidas += 1
        print(f"{split:5s}: {mantidas} imagens ({excluidas} do jogo do vídeo removidas)")

    (DESTINO / "data.yaml").write_text(
        f"path: {DESTINO.as_posix()}\ntrain: train/images\nval: valid/images\ntest: test/images\n"
        f"kpt_shape: [32, 3]\nflip_idx: {FLIP_IDX}\nnames:\n  0: football_pitch\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
