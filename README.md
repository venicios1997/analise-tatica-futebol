# Do pixel ao mapa tático: análise tática de futebol com visão computacional

Trabalho final da disciplina de Visão Computacional, MBA em Ciência de Dados.

A partir de um vídeo de transmissão de futebol, o sistema detecta jogadores,
goleiros, árbitros e a bola, segue cada pessoa ao longo do vídeo, separa os
times pela cor da camisa e projeta tudo numa vista de cima do campo, em
metros. Daí saem radar tático, mapa de calor, distância percorrida,
velocidade e posse de bola.

O projeto foi construído em camadas, para comparar com números a abordagem
**clássica/genérica** com a **aprendida**:

- **Nível 1**: YOLO genérico (COCO) + visão clássica (máscara HSV do gramado,
  morfologia, K-means de cores).
- **Nível 2**: YOLO treinado para futebol (fine-tuning), com divisão dos
  dados **por jogo**.
- **Mapa tático**: modelo de 32 pontos do campo + homografia com RANSAC.

Todas as métricas foram medidas em **jogos que os modelos nunca viram**.

## Por onde começar

| Arquivo | O que é |
|---|---|
| `saidas/relatorio.html` | **Relatório com todos os resultados**: tabelas, vídeos, gráficos de treino, mapa de calor, métricas por jogador. Abrir no navegador. |
| `README.md` | Este arquivo: resumo, estrutura e como reproduzir. |

## Analisar outro vídeo

```
analisar.bat                         (dois cliques: abre uma janela para escolher o vídeo)
analisar.bat C:\videos\meu_jogo.mp4  (ou arraste o vídeo para cima do analisar.bat)
.venv\Scripts\python src\analisar.py meu_jogo.mp4 [--max-frames 300]
```

Roda tudo (detecção com caixa e confiança, tracking, times, mapa tático,
métricas) e abre `resultados/<nome do vídeo>/relatorio.html`. Numa RTX 4060,
30 s de vídeo levam cerca de 1,5 minuto.

**Para que vídeos serve.** O sistema é genérico para **transmissões de
futebol com a câmera principal** (aberta, lateral, mostrando boa parte do
campo): as cores dos times são aprendidas de cada vídeo e os modelos não
dependem de time ou estádio. Close-ups, replays e câmeras atrás do gol não têm
campo suficiente para a homografia: esses trechos aparecem como "sem visão do
campo" e ficam fora das métricas. O relatório gera **alertas automáticos**
quando o vídeo foge do esperado (pouca cobertura do campo, poucas pessoas por
frame, bola rara, posições fora do campo). Outras ligas, jogos noturnos e
filmagens amadoras não foram testados.

**Teste de generalização** (`resultados/573e61_0/`): um jogo que nenhum dos
dois modelos viu no treino, com uniformes branco × grená. O teste revelou uma
falha na separação de times: a regra exigia que os times diferissem na cor
(a,b), e aqui eles diferem quase só no brilho. A regra foi corrigida para usar a distância
completa no espaço do K-means. Resultado: times separados, 96% das posições
dentro do campo, bola em 70% dos frames.

## Resultados principais

**Detecção nos 3 jogos de teste** (90 imagens, 64 bolas; classes unificadas em
pessoa/bola, porque o COCO não conhece goleiro nem árbitro):

| Detector | mAP50 pessoa | mAP50 bola | mAP50-95 pessoa | bolas achadas (conf ≥ 0,3) |
|---|---|---|---|---|
| COCO genérico | 0,873 | 0,432 | 0,716 | 22% |
| COCO + máscara do gramado (nível 1) | 0,883 | 0,458 | 0,723 | 22% |
| **Treinado para futebol (nível 2)** | **0,989** | **0,711** | **0,841** | **72%** |

**Tracking no vídeo inédito** (jogo `0bfacc`, 30 s, 750 frames):

| Indicador | Nível 1 | Nível 2 |
|---|---|---|
| IDs de pessoa | 46 | 36 |
| Tempo mediano de um ID | 12,8 s | 29,8 s |
| Frames com a bola | 50,5% | 76,5% |
| Banco e treinadores rastreados | sim (como "outro") | não |

**Homografia** (28 imagens de teste): erro mediano de **3,38 m** na posição
projetada; o piso (homografia ajustada nos próprios rótulos) é 0,91 m. No
vídeo, 99,8% das posições caem dentro do campo, velocidade média dos
jogadores de linha de 6,6 a 10,2 km/h e máxima (média de 1 s) de 15,7 a
22,8 km/h. Posse: 98,6% para o time de camisa branca, o que confere com o vídeo.

**Experimento de vazamento**: um modelo treinado com divisão por foto *não*
teve resultado inflado no teste com vazamento (bola: 0,60 contra 0,70 no teste
honesto). A hipótese não se confirmou neste dataset; a explicação está no relatório.

**Experimento de resolução**: o modelo de pontos do campo treinado em 1280 px
(em vez de 640) ficou pior: 4,61 m de erro mediano contra 3,38 m, e sem
homografia em 2 de 28 imagens de teste. O modelo de 640 px foi mantido.

## Estrutura

```
analise_tatica_futebol/
├── README.md                 este arquivo
├── analisar.bat              analisa qualquer vídeo (dois cliques ou arrastar o vídeo)
├── requirements.txt          versões usadas
├── src/
│   ├── baixar_dados.py       vídeo + 2 datasets (Hugging Face / Google Drive)
│   ├── preparar_dataset.py   split por jogo (GroupShuffleSplit) e split por foto
│   ├── preparar_campo.py     dataset de pontos do campo sem o jogo do vídeo
│   ├── treinar.py            os treinos, com os parâmetros usados
│   ├── campo.py              máscara do gramado (HSV + morfologia + contornos)
│   ├── times.py              cor da camisa em Lab + K-means (5 grupos) + voto por ID
│   ├── pipeline.py           detecção + tracking + times (níveis 1 e 2)
│   ├── avaliar.py            comparação dos detectores no teste por jogo
│   ├── campo_modelo.py       os 32 pontos do campo oficial em metros
│   ├── tatico.py             homografia, radar, calor, distância, velocidade, posse
│   ├── avaliar_campo.py      erro da homografia em metros
│   ├── analisar.py           análise completa de um vídeo qualquer + relatório próprio
│   ├── relatorio.py          gera saidas/relatorio.html
│   └── imagens.py            leitura/gravação de imagens com acentos no caminho
├── saidas/                   resultados do trabalho (CSV, JSON, imagens, relatório, vídeos web)
├── resultados/<vídeo>/       análises feitas com analisar.bat (ex.: 573e61_0)
└── modelos/treino/           gráficos dos treinos (curvas, matriz de confusão)
```

Ficam fora do pacote de entrega, por tamanho, e são recriados pelos scripts:
`dados/` (`baixar_dados.py`), os pesos `modelos/treino/*/weights/best.pt`
(`treinar.py`) e os vídeos anotados em resolução cheia (`pipeline.py`, `tatico.py`).

## Como reproduzir

Testado em Windows 11, Python 3.13, NVIDIA RTX 4060 (8 GB).

```
python -m venv .venv
.venv\Scripts\pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\pip install -r requirements.txt

.venv\Scripts\python src\baixar_dados.py
.venv\Scripts\python src\preparar_dataset.py
.venv\Scripts\python src\preparar_campo.py

.venv\Scripts\python src\treinar.py por_jogo      # ~45 min
.venv\Scripts\python src\treinar.py por_foto      # ~30 min (só para o experimento de vazamento)
.venv\Scripts\python src\treinar.py campo         # ~15 min
.venv\Scripts\python src\treinar.py campo_1280    # ~55 min (só para o experimento de resolução)

.venv\Scripts\python src\pipeline.py dados\videos\0bfacc_0.mp4                                               # nível 1
.venv\Scripts\python src\pipeline.py dados\videos\0bfacc_0.mp4 --pesos modelos\treino\por_jogo\weights\best.pt  # nível 2
.venv\Scripts\python src\avaliar.py --treinado modelos\treino\por_jogo\weights\best.pt modelos\treino\por_foto\weights\best.pt
.venv\Scripts\python src\avaliar.py --sem-coco --teste futebol_por_foto --treinado modelos\treino\por_jogo\weights\best.pt modelos\treino\por_foto\weights\best.pt
.venv\Scripts\python src\avaliar_campo.py --ransac 1
.venv\Scripts\python src\avaliar_campo.py --ransac 3
.venv\Scripts\python src\avaliar_campo.py --ransac 3 --pesos modelos\treino\campo_1280\weights\best.pt
.venv\Scripts\python src\tatico.py dados\videos\0bfacc_0.mp4
.venv\Scripts\python src\relatorio.py
```

Sem GPU os treinos são inviáveis; a inferência roda em CPU, só que bem mais devagar.

## Decisões técnicas e problemas resolvidos

- **Máscara do gramado**: o YOLO genérico detecta "pessoa", inclusive torcida e
  banco. Verde em HSV → abertura → fechamento → maior contorno → casco convexo;
  fica quem tem os pés dentro.
- **Separação de times**: K-means com 2 grupos absorvia árbitro e banco num dos
  times (~35% de contaminação). Solução: 5 grupos, peso 0,5 no brilho (L do Lab,
  que muda com sol e sombra), times decididos pela distância completa entre
  grupos (um segundo jogo, branco × grená, mostrou que só a cor não basta) e
  regras para juntar grupos da mesma camisa. Com o detector treinado, árbitro e
  goleiro vêm da rede, então todo "jogador" é atribuído a um dos dois times.
- **Caixa e confiança no vídeo**: cada pessoa aparece com `#ID confiança` na cor
  do time; a bola com `bola confiança`.
- **Divisão por jogo**: o dataset original misturava frames do mesmo jogo em
  treino e teste; refeito com `GroupShuffleSplit`.
- **Resolução 1280 px** no detector: a bola tem ~11 px de largura em 1920 px.
- **Memória da GPU**: com lote 4, o treino excedia 8 GB e o Windows passava a
  usar RAM compartilhada sem avisar (180 s por época contra 33 s com lote 3).
- **Ordem dos 32 pontos do campo** conferida antes de usar: erro de reprojeção
  de 11,6 px com a ordem certa contra 368 px com a ordem embaralhada.
- **Jogo do vídeo removido** do dataset de pontos do campo (13 imagens).
- **RANSAC com tolerância de 3 m**: com 1 m, metade dos pontos bons era
  descartada e uma imagem teve homografia degenerada (erro de 4,2 km).
- **Caminhos com "°" no Windows**: `cv2.imread`/`imwrite` falham em silêncio;
  `src/imagens.py` lê e grava pela memória.
- **Vídeo no navegador**: o OpenCV grava em mp4v; o relatório converte para H.264.

## Limitações

- Erro de posição de ~3,4 m (piso 0,9 m): suficiente para mapa de calor e
  formação, mas soma ruído à distância percorrida. O mesmo modelo treinado em
  1280 px ficou pior (4,61 m; sem homografia em 2 de 28 imagens), então o de
  640 px foi mantido. Próximos passos possíveis: mais dados de pontos do campo
  ou segmentação das linhas.
- Bola no ar é projetada no chão (a homografia vale só para o plano do gramado).
- Uma liga e um estilo de transmissão; outras câmeras e jogos noturnos não foram testados.
- Times pela cor dependem de uniformes contrastantes; goleiros ficam sem time.

## Créditos e licenças dos dados

- Vídeo: DFL Bundesliga Data Shootout (Kaggle), distribuído pelo repositório
  [roboflow/sports](https://github.com/roboflow/sports).
- Datasets: Roboflow *football-players-detection* e *football-field-detection*
  (CC BY 4.0), espelhados no Hugging Face por `martinjolif`.
- Coordenadas dos 32 pontos do campo: `roboflow/sports`, `sports/configs/soccer.py`.
- Modelos: Ultralytics YOLO11 (pesos iniciais COCO). Tracking: ByteTrack (`trackers`).
