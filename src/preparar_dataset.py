"""Refaz o split do dataset de jogadores POR JOGO, não por foto.

O dataset original (Roboflow, espelhado em martinjolif/football-player-detection
no Hugging Face) separou treino/validação/teste por frame: frames do mesmo jogo,
às vezes a poucos segundos de distância, caem em lados diferentes. O teste
fica "fácil" (vazamento) e o mAP sai otimista.

O nome de cada imagem começa com o código do jogo (ex.: 4b770a_1_6_png...),
então agrupamos por esse código e usamos GroupShuffleSplit.

Para medir o tamanho do vazamento, também gera um split POR FOTO (aleatório)
usando só os jogos de treino/validação: um modelo treinado nele pode ser
avaliado no próprio teste aleatório (jogos já vistos) e no teste por jogo
(jogos inéditos); a diferença é o otimismo causado pelo vazamento.

Uso:
    python src/preparar_dataset.py
Gera dados/futebol_por_jogo/ e dados/futebol_por_foto/ (formato YOLO + data.yaml).
"""
from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

RAIZ = Path(__file__).resolve().parent.parent
ORIGEM = RAIZ / "dados" / "hf_football_players" / "data"
DESTINO = RAIZ / "dados" / "futebol_por_jogo"
DESTINO_POR_FOTO = RAIZ / "dados" / "futebol_por_foto"
CLASSES = ["ball", "goalkeeper", "player", "referee"]
SEMENTE = 42


def escrever_yaml(caminho: Path, base: Path) -> None:
    linhas = [f"path: {base.as_posix()}", "train: train/images",
              "val: valid/images", "test: test/images", "names:"]
    linhas += [f"  {i}: {nome}" for i, nome in enumerate(CLASSES)]
    caminho.write_text("\n".join(linhas) + "\n", encoding="utf-8")


def main() -> None:
    imagens = sorted(ORIGEM.glob("*/images/*.jpg"))
    jogos = np.array([img.name[:6] for img in imagens], dtype=object)
    idx = np.arange(len(imagens))

    # 1º corte: ~20% dos jogos para teste; 2º corte: ~15% do resto para validação
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEMENTE)
    resto, teste = next(gss.split(idx, groups=jogos))
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEMENTE)
    treino, valid = next(gss.split(resto, groups=jogos[resto]))
    treino, valid = resto[treino], resto[valid]

    print("== split por jogo")
    gravar_split(DESTINO, imagens, jogos, {"train": treino, "valid": valid, "test": teste})
    assert not set(jogos[treino]) & set(jogos[teste]), "jogo vazou para o teste"
    assert not set(jogos[valid]) & set(jogos[teste]), "jogo vazou para o teste"

    # split por foto, só com os jogos de treino+validação (o teste por jogo fica intocado)
    rng = np.random.default_rng(SEMENTE)
    embaralhado = rng.permutation(resto)
    n_teste, n_valid = int(0.2 * len(resto)), int(0.1 * len(resto))
    print("\n== split por foto (com vazamento)")
    gravar_split(DESTINO_POR_FOTO, imagens, jogos, {
        "test": embaralhado[:n_teste],
        "valid": embaralhado[n_teste:n_teste + n_valid],
        "train": embaralhado[n_teste + n_valid:],
    })


def gravar_split(destino: Path, imagens: list[Path], jogos: np.ndarray,
                 partes: dict[str, np.ndarray]) -> None:
    if destino.exists():
        shutil.rmtree(destino)
    for nome, ids in partes.items():
        (destino / nome / "images").mkdir(parents=True)
        (destino / nome / "labels").mkdir(parents=True)
        for i in ids:
            img = imagens[i]
            rotulo = img.parent.parent / "labels" / (img.stem + ".txt")
            shutil.copy2(img, destino / nome / "images" / img.name)
            shutil.copy2(rotulo, destino / nome / "labels" / rotulo.name)
        print(f"{nome:5s}: {len(ids):3d} imagens, {len(set(jogos[ids]))} jogos {dict(Counter(jogos[ids]))}")
    escrever_yaml(destino / "data.yaml", destino)


if __name__ == "__main__":
    main()
