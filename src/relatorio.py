"""Gera saidas/relatorio.html reunindo todos os resultados do projeto.

Lê os arquivos produzidos pelos outros scripts (CSVs, JSONs, imagens, vídeos);
não roda modelo nenhum. Os vídeos do OpenCV (codec mp4v) não tocam no
navegador, então são convertidos para H.264 em saidas/web/ com o ffmpeg do
pacote imageio-ffmpeg.

Uso (depois de pipeline.py, avaliar.py, avaliar_campo.py e tatico.py):
    python src/relatorio.py
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime
from html import escape
from pathlib import Path

import imageio_ffmpeg
import pandas as pd

from pipeline import indicadores

RAIZ = Path(__file__).resolve().parent.parent
SAIDAS = RAIZ / "saidas"
WEB = SAIDAS / "web"
NOME = "0bfacc_0"
FPS = 25.0


# ---------------------------------------------------------------- vídeos

def ffmpeg(*args: str) -> None:
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", *args], check=True)


def desatualizado(destino: Path, *fontes: Path) -> bool:
    return not destino.exists() or any(f.stat().st_mtime > destino.stat().st_mtime for f in fontes)


def videos_web() -> dict[str, str]:
    WEB.mkdir(exist_ok=True)
    h264 = ["-c:v", "libx264", "-preset", "fast", "-crf", "26", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    coco, fino, tatico = (SAIDAS / f"{NOME}_{s}.mp4" for s in ("coco_anotado", "finetuned_anotado", "tatico"))
    lado = WEB / "nivel1_vs_nivel2.mp4"
    if desatualizado(lado, coco, fino):
        print("convertendo vídeo lado a lado...")
        ffmpeg("-i", str(coco), "-i", str(fino), "-filter_complex",
               "[0:v]scale=960:-2[a];[1:v]scale=960:-2[b];[a][b]hstack", *h264, str(lado))
    tat = WEB / "tatico.mp4"
    if desatualizado(tat, tatico):
        print("convertendo vídeo tático...")
        ffmpeg("-i", str(tatico), "-vf", "scale=1280:-2,setsar=1", *h264, str(tat))
    return {"lado": "web/nivel1_vs_nivel2.mp4", "tatico": "web/tatico.mp4"}


# ---------------------------------------------------------------- dados

def ler_json(nome: str) -> dict:
    return json.loads((SAIDAS / nome).read_text(encoding="utf-8"))


def splits(pasta: str) -> pd.DataFrame:
    linhas = []
    for parte in ("train", "valid", "test"):
        imgs = list((RAIZ / "dados" / pasta / parte / "images").glob("*.jpg"))
        linhas.append({"parte": {"train": "treino", "valid": "validação", "test": "teste"}[parte],
                       "imagens": len(imgs), "jogos": len({i.name[:6] for i in imgs}),
                       "códigos dos jogos": ", ".join(sorted({i.name[:6] for i in imgs}))})
    return pd.DataFrame(linhas)


def tabela(df: pd.DataFrame, destaque: int | None = None, pct: tuple[str, ...] = ()) -> str:
    """DataFrame -> <table>. `destaque` = índice da linha a realçar."""
    cab = "".join(f"<th>{escape(str(c))}</th>" for c in df.columns)
    corpo = []
    for i, (_, linha) in enumerate(df.iterrows()):
        celulas = []
        for c, v in linha.items():
            if isinstance(v, float):
                if c in pct:
                    txt = f"{100 * v:.0f}%"
                elif v.is_integer():  # o pandas vira int em float ao misturar colunas
                    txt = f"{v:.0f}"
                else:
                    txt = f"{v:.3f}".rstrip("0").rstrip(".") if abs(v) < 10 else f"{v:.1f}"
                celulas.append(f'<td class="num">{txt}</td>')
            elif isinstance(v, int):
                celulas.append(f'<td class="num">{v}</td>')
            else:
                celulas.append(f"<td>{escape(str(v))}</td>")
        classe = ' class="destaque"' if i == destaque else ""
        corpo.append(f"<tr{classe}>{''.join(celulas)}</tr>")
    return f'<div class="tabela"><table><thead><tr>{cab}</tr></thead><tbody>{"".join(corpo)}</tbody></table></div>'


def figura(src: str, legenda: str, classe: str = "") -> str:
    cls = f' class="{classe}"' if classe else ""
    return (f'<figure{cls}><a href="{src}" target="_blank"><img src="{src}" alt="{escape(legenda)}" '
            f'loading="lazy"></a><figcaption>{legenda}</figcaption></figure>')


def kpi(rotulo: str, antes: str, depois: str, nota: str) -> str:
    return (f'<div class="kpi"><div class="kpi-rotulo">{rotulo}</div>'
            f'<div class="kpi-valor"><span class="antes">{antes}</span> → <strong>{depois}</strong></div>'
            f'<div class="kpi-nota">{nota}</div></div>')


# ---------------------------------------------------------------- página

def montar() -> str:
    v = videos_web()
    det = pd.read_csv(SAIDAS / "comparacao_detectores_futebol_por_jogo.csv")
    vaz = pd.read_csv(SAIDAS / "comparacao_detectores_futebol_por_foto.csv")
    jog = pd.read_csv(SAIDAS / f"{NOME}_jogadores.csv")
    tat = ler_json(f"{NOME}_resumo_tatico.json")
    # os nomes A/B saem da ordem dos grupos do K-means (arbitrária); no vídeo
    # 0bfacc o time com a bola é o de camisa branca, seja qual for a letra
    dono_posse = "A" if tat["posse"]["A"] >= tat["posse"]["B"] else "B"
    h1, h3 = ler_json("erro_homografia_campo_ransac1m.json"), ler_json("erro_homografia_campo_ransac3m.json")
    h1280 = ler_json("erro_homografia_campo_1280_ransac3m.json")
    trk = {t: indicadores(pd.read_csv(SAIDAS / f"{NOME}_{t}_tracks.csv"), FPS, 750) for t in ("coco", "finetuned")}

    linha = lambda nome: det[det.modelo.str.startswith(nome)].iloc[0]
    n1, n2 = linha("COCO + máscara"), linha("fine-tuned (por_jogo)")

    # tabela de detectores
    det_t = det[det.modelo != "fine-tuned (por_foto)"].rename(columns={"modelo": "detector"})[
        ["detector", "mAP50 pessoa", "mAP50 bola", "mAP50-95 pessoa", "mAP50-95 bola",
         "recall pessoa@0.3", "recall bola@0.3"]]
    det_t["detector"] = det_t["detector"].replace({"fine-tuned (por_jogo)": "fine-tuned (nível 2)"})

    trk_t = pd.DataFrame([
        {"indicador": "IDs de pessoa", "nível 1": trk["coco"]["ids_pessoa"], "nível 2": trk["finetuned"]["ids_pessoa"]},
        {"indicador": "duração mediana de um ID (s)", "nível 1": trk["coco"]["duracao_mediana_id_s"],
         "nível 2": trk["finetuned"]["duracao_mediana_id_s"]},
        {"indicador": "bola vista (% dos frames)", "nível 1": trk["coco"]["bola_frames_pct"],
         "nível 2": trk["finetuned"]["bola_frames_pct"]},
        {"indicador": "pessoas por frame (mediana)", "nível 1": trk["coco"]["pessoas_por_frame"],
         "nível 2": trk["finetuned"]["pessoas_por_frame"]},
    ])

    vaz_t = pd.DataFrame([
        {"modelo treinado no split por foto, avaliado em": "teste com vazamento (frames de jogos já vistos)",
         "mAP50 bola": float(vaz[vaz.modelo == "fine-tuned (por_foto)"]["mAP50 bola"].iat[0]),
         "recall bola@0.3": float(vaz[vaz.modelo == "fine-tuned (por_foto)"]["recall bola@0.3"].iat[0]),
         "mAP50 pessoa": float(vaz[vaz.modelo == "fine-tuned (por_foto)"]["mAP50 pessoa"].iat[0])},
        {"modelo treinado no split por foto, avaliado em": "teste honesto (3 jogos inéditos)",
         "mAP50 bola": float(det[det.modelo == "fine-tuned (por_foto)"]["mAP50 bola"].iat[0]),
         "recall bola@0.3": float(det[det.modelo == "fine-tuned (por_foto)"]["recall bola@0.3"].iat[0]),
         "mAP50 pessoa": float(det[det.modelo == "fine-tuned (por_foto)"]["mAP50 pessoa"].iat[0])},
    ])

    hom_t = pd.DataFrame([
        {"homografia": "piso: ajustada nos próprios rótulos", "erro mediano (m)": h3["piso_mediana_m"],
         "p90 (m)": h3["piso_p90_m"], "máximo (m)": "—"},
        {"homografia": f"sistema, RANSAC {h1['limiar_ransac_m']:g} m", "erro mediano (m)": h1["erro_mediana_m"],
         "p90 (m)": h1["erro_p90_m"], "máximo (m)": f"{h1['erro_max_m']:.0f}"},
        {"homografia": f"sistema, RANSAC {h3['limiar_ransac_m']:g} m (adotado)", "erro mediano (m)": h3["erro_mediana_m"],
         "p90 (m)": h3["erro_p90_m"], "máximo (m)": f"{h3['erro_max_m']:.1f}"},
        {"homografia": f"modelo em 1280 px, RANSAC {h1280['limiar_ransac_m']:g} m "
                       f"(sem homografia em {h1280['sem_homografia']} de {h1280['imagens']} imagens)",
         "erro mediano (m)": h1280["erro_mediana_m"], "p90 (m)": h1280["erro_p90_m"],
         "máximo (m)": f"{h1280['erro_max_m']:.1f}"},
    ])

    jog_t = jog.rename(columns={"track_id": "ID", "time": "time", "duracao_s": "tempo em cena (s)",
                                "distancia_m": "distância (m)", "vel_media_kmh": "vel. média (km/h)",
                                "vel_max_kmh": "vel. máx. 1 s (km/h)"})

    pct_det = ("recall pessoa@0.3", "recall bola@0.3")
    agora = datetime.now().strftime("%d/%m/%Y %H:%M")

    return f"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Análise Tática de Futebol</title>
<style>{CSS}</style>
</head>
<body>
<header>
  <p class="sobre">MBA Ciência de Dados · Visão Computacional</p>
  <h1>Análise tática de futebol: do pixel ao mapa tático</h1>
  <p class="autor">Marcos Venicios de Andrade</p>
  <p class="lead">A partir de um vídeo de transmissão, o sistema detecta jogadores, árbitros e bola, segue cada
  pessoa ao longo do tempo, separa os times pela cor da camisa e projeta tudo numa vista de cima do campo, em metros.
  Todas as métricas abaixo foram medidas em <strong>jogos que os modelos nunca viram</strong>.</p>
  <nav>
    <a href="#pipeline">Pipeline</a><a href="#nivel1">Nível 1</a><a href="#nivel2">Nível 2</a>
    <a href="#vazamento">Vazamento</a><a href="#tatico">Mapa tático</a><a href="#limites">Limitações</a>
  </nav>
</header>

<main>
<section class="kpis">
  {kpi("mAP50 da bola", f"{n1['mAP50 bola']:.2f}", f"{n2['mAP50 bola']:.2f}", "teste em 3 jogos inéditos")}
  {kpi("Bolas encontradas", f"{100 * n1['recall bola@0.3']:.0f}%", f"{100 * n2['recall bola@0.3']:.0f}%", "recall com confiança ≥ 0,3")}
  {kpi("Duração de um ID", f"{trk['coco']['duracao_mediana_id_s']:.0f} s", f"{trk['finetuned']['duracao_mediana_id_s']:.0f} s", "mediana, num vídeo de 30 s")}
  {kpi("Erro da homografia", f"{h1['erro_mediana_m']:.1f} m", f"{h3['erro_mediana_m']:.1f} m", f"mediana; piso de {h3['piso_mediana_m']:.1f} m")}
</section>

<section id="pipeline">
  <h2>Pipeline</h2>
  <div class="tabela"><table>
    <thead><tr><th>etapa</th><th>técnica</th><th>conteúdo da disciplina</th></tr></thead>
    <tbody>
      <tr><td>Máscara do gramado</td><td>HSV, limiar, abertura e fechamento, maior contorno, casco convexo</td><td>canais de cor, limiar, morfologia</td></tr>
      <tr><td>Detecção</td><td>YOLO11m: genérico do COCO (nível 1) ou treinado para futebol (nível 2)</td><td>CNN, detecção, IoU, NMS, mAP</td></tr>
      <tr><td>Tracking</td><td>ByteTrack: associação de caixas entre frames</td><td>tracking por detecção</td></tr>
      <tr><td>Times</td><td>cor do tronco em Lab sem a grama, K-means com 5 grupos, voto da maioria por ID</td><td>espaços de cor, clustering</td></tr>
      <tr><td>Vista de cima</td><td>32 pontos do campo (YOLO-pose) e homografia com RANSAC</td><td>geometria, homografia</td></tr>
      <tr><td>Métricas</td><td>trajetória suavizada (mediana móvel): distância, velocidade, calor, posse</td><td>—</td></tr>
    </tbody></table></div>
</section>

<section id="nivel1">
  <h2>Nível 1: modelo genérico e visão clássica</h2>
  <p>O YOLO pré-treinado no COCO sabe o que é <em>pessoa</em>, não <em>jogador</em>: detecta torcida, banco e gandulas.
  Uma máscara clássica do gramado (verde em HSV, limpa com morfologia) descarta quem está fora do campo.</p>
  {figura(f"{NOME}_mascara.jpg", "Frame original e máscara do gramado (HSV + abertura/fechamento + maior contorno).")}
  <h3>O que deu errado e virou aprendizado</h3>
  <ol>
    <li><strong>K-means com 2 grupos falhou com o árbitro.</strong> O grupo "vermelho" absorvia árbitro, bandeirinhas e o pessoal de roupa escura do banco (cerca de 35% de contaminação).
    Com 5 grupos, árbitros e banco ganham grupos próprios; os dois maiores grupos claramente diferentes viram os times e os demais só entram num time se tiverem a mesma cor (a mesma camisa na sombra).</li>
    <li><strong>Brilho pesa, mas menos.</strong> No espaço Lab, o canal L (sol e sombra) dominava a distância: um árbitro amarelo com o mesmo brilho de um jogador vermelho ficava "perto" dele.
    Com peso 0,5 em L, o K-means passou a separar vermelho, branco, roupa escura e amarelo.</li>
    <li><strong>Um segundo jogo mostrou o limite da regra.</strong> No jogo <code>573e61</code> (branco × grená) os times diferem quase só no brilho, e a regra antiga, que exigia diferença de cor, não separava os times.
    A decisão passou a usar a distância completa no espaço do K-means, e o número de grupos foi de 4 para 5.</li>
    <li><strong>Voto da maioria por ID:</strong> um frame com sombra não troca o time de um jogador.</li>
  </ol>
</section>

<section id="nivel2">
  <h2>Nível 2: detector treinado para futebol</h2>
  <p>YOLO11m ajustado (fine-tuning) em 372 frames da Bundesliga com as classes bola, goleiro, jogador e árbitro
  (dataset Roboflow <em>football-players-detection</em>, CC BY 4.0). Treino em 1280 px, porque a bola tem cerca de 11 px de largura
  em 1920 px. O split original era por foto; aqui foi refeito <strong>por jogo</strong>, com <code>GroupShuffleSplit</code>:</p>
  {tabela(splits("futebol_por_jogo"))}

  <h3>Detecção no teste (3 jogos inéditos)</h3>
  <p>Classes unificadas em <em>pessoa</em> e <em>bola</em>, porque o COCO não conhece goleiro nem árbitro.</p>
  {tabela(det_t.reset_index(drop=True), destaque=len(det_t) - 1, pct=pct_det)}
  <p>A máscara clássica ajuda um pouco o modelo genérico, mas não substitui o aprendizado. O maior ganho está na bola, o objeto mais difícil.</p>

  <h3>No vídeo inédito: nível 1 (esquerda) × nível 2 (direita)</h3>
  <video src="{v['lado']}" controls muted loop playsinline preload="metadata"></video>
  {tabela(trk_t)}
  <p>Com o modelo treinado, a maioria dos jogadores mantém o mesmo ID durante os 30 s. Banco e treinadores deixam de ser rastreados
  sem máscara nenhuma, e árbitro e goleiro passam a vir do próprio detector, não de uma regra de cor.</p>

  <div class="grade">
    {figura("../modelos/treino/por_jogo/results.png", "Curvas de treino: perdas e mAP por época (melhor época: 60).")}
    {figura("../modelos/treino/por_jogo/confusion_matrix_normalized.png", "Matriz de confusão normalizada (validação).")}
  </div>
</section>

<section id="vazamento">
  <h2>Experimento: o split por foto infla o resultado?</h2>
  <p>Hipótese: com frames do mesmo jogo no treino e no teste, a métrica sai otimista. Para testar, um segundo modelo foi treinado num split
  aleatório por foto, usando só os jogos de treino e validação, e avaliado em dois testes:</p>
  {tabela(vaz_t, pct=("recall bola@0.3",))}
  <div class="aviso"><strong>Hipótese não confirmada.</strong> Se o vazamento inflasse a métrica, o primeiro resultado seria maior, e foi o contrário.
  Com cerca de 50 a 60 bolas por teste, o recall varia uns 7 pontos percentuais para cima ou para baixo, então a diferença está perto do ruído.
  Leitura provável: todos os jogos são da mesma liga, com a mesma câmera e o mesmo estilo de transmissão. Um "jogo novo" não é uma grande mudança de domínio aqui.
  O split por jogo continua sendo o protocolo correto, porque responde à pergunta "funciona num jogo que nunca vi?".</div>
</section>

<section id="tatico">
  <h2>Mapa tático: homografia, radar e métricas</h2>
  <p>Um modelo YOLO-pose detecta 32 pontos do campo (cantos, áreas, círculo central). Com as posições oficiais desses pontos em metros,
  o RANSAC estima a homografia imagem → campo a cada frame, e os pés de cada jogador são projetados no gramado.
  O jogo do vídeo foi <strong>removido</strong> do dataset de pontos do campo (13 imagens) para continuar inédito.</p>
  <video src="{v['tatico']}" controls muted loop playsinline preload="metadata"></video>
  <p class="nota">Linhas rosa: o campo oficial reprojetado no vídeo. Quando elas casam com as linhas reais, a homografia está certa.</p>

  <h3>Qualidade da homografia (28 imagens de teste)</h3>
  {tabela(hom_t, destaque=2)}
  <p>Com limiar de 1 m, o RANSAC descartava metade dos pontos bons. Num caso, sobraram pontos quase alinhados e a homografia degenerou, com erro de quilômetros.
  Com 3 m, não há mais catástrofes. O gargalo restante é a precisão dos pontos detectados, cerca de 25 px por ponto com o modelo em 640 px.</p>
  <div class="aviso"><strong>Mais resolução não ajudou.</strong> O mesmo modelo de pontos treinado em 1280 px, para localizar os pontos com mais precisão,
  ficou pior: {h1280['erro_mediana_m']:.1f} m de erro mediano contra {h3['erro_mediana_m']:.1f} m, e sem homografia em {h1280['sem_homografia']} imagens.
  Na validação, mAP50 de pose de 0,69 contra 0,83. Duas explicações prováveis, não testadas: os pontos do campo dependem de contexto global, e em 1280 px
  cada região que a rede enxerga cobre uma fração menor do campo; e o treino foi mais ruidoso (lote de 3 imagens contra 8). O modelo de 640 px foi mantido.</div>

  <div class="numeros">
    <div><span>{tat['frames_com_homografia']}/{tat['frames']}</span>frames com homografia</div>
    <div><span>{tat['posicoes_dentro_do_campo_pct']:.1f}%</span>das posições dentro do campo (±3 m)</div>
    <div><span>{max(tat['posse']['A'], tat['posse']['B']):.0f}%</span>posse do time {dono_posse} (camisa branca)</div>
  </div>

  {figura(f"{NOME}_calor.png", "Mapa de calor de cada time ao longo dos 30 s.", "estreita")}

  <h3>Jogadores (IDs com 5 s ou mais em cena)</h3>
  {tabela(jog_t)}
  <p>Velocidade média de 7 a 10 km/h e máxima (média de 1 s) de 15 a 23 km/h: faixa realista para futebol profissional.
  A posse quase total do time branco confere com o vídeo, em que ele troca passes o trecho inteiro.</p>
</section>

<section id="limites">
  <h2>Limitações e próximos passos</h2>
  <ul>
    <li>A homografia erra cerca de {h3['erro_mediana_m']:.1f} m (o piso é {h3['piso_mediana_m']:.1f} m). Isso basta para o mapa tático, mas soma ruído à distância percorrida.
    Treinar em 1280 px piorou o resultado; próximos passos possíveis são mais dados de pontos do campo ou um modelo de segmentação das linhas.</li>
    <li>A bola no ar é projetada no chão, porque a homografia só vale para o plano do gramado.</li>
    <li>Todos os dados vêm de uma liga e de um estilo de transmissão. Um teste mais forte seria outra liga, outra câmera ou um jogo noturno.</li>
    <li>A separação de times por cor depende de uniformes contrastantes. Uma alternativa a comparar são embeddings visuais (CLIP/SigLIP) com clustering.</li>
  </ul>
</section>
</main>

<footer>Gerado por <code>src/relatorio.py</code> em {agora}. Vídeo de demonstração: jogo <code>0bfacc</code> (DFL Bundesliga, via roboflow/sports).</footer>
</body>
</html>"""


CSS = """
:root{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#5f5e58;--line:#e2dfd6;--card:#ffffff;--accent:#1f7a4d;--accent-soft:#e3f1e9;--warn:#fff4dc;--warn-line:#e9c46a}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#161714;--fg:#ecebe6;--muted:#a3a29b;--line:#34352f;--card:#1f201c;--accent:#5cc28d;--accent-soft:#1d3327;--warn:#2e2816;--warn-line:#8a7430}}
:root[data-theme="dark"]{--bg:#161714;--fg:#ecebe6;--muted:#a3a29b;--line:#34352f;--card:#1f201c;--accent:#5cc28d;--accent-soft:#1d3327;--warn:#2e2816;--warn-line:#8a7430}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.6 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header,main,footer{max-width:1080px;margin:0 auto;padding:0 16px}
header{padding-top:48px;padding-bottom:8px}
.autor{font-weight:600;font-size:16px;margin:4px 0 12px}
.sobre{color:var(--accent);font-weight:600;letter-spacing:.04em;text-transform:uppercase;font-size:13px;margin:0}
h1{font-size:clamp(28px,4vw,40px);line-height:1.15;margin:8px 0 12px}
h2{font-size:26px;margin:0 0 12px;padding-top:8px}
h3{font-size:18px;margin:28px 0 8px}
.lead{font-size:18px;color:var(--muted);max-width:760px}
nav{display:flex;flex-wrap:wrap;gap:6px 18px;margin:20px 0 0;padding:12px 0;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}
nav a{color:var(--fg);text-decoration:none;font-weight:500}
nav a:hover{color:var(--accent)}
section{padding:36px 0;border-bottom:1px solid var(--line)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;border:0;padding-top:28px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px}
.kpi-rotulo{font-size:13px;color:var(--muted);font-weight:600}
.kpi-valor{font-size:26px;margin:4px 0;font-variant-numeric:tabular-nums}
.kpi-valor .antes{color:var(--muted);font-size:20px}
.kpi-valor strong{color:var(--accent)}
.kpi-nota{font-size:13px;color:var(--muted)}
.tabela{overflow-x:auto;margin:12px 0}
table{border-collapse:collapse;width:100%;font-size:14px;background:var(--card)}
th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{font-size:12px;text-transform:uppercase;letter-spacing:.03em;color:var(--muted);font-weight:600}
td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
tr.destaque td{background:var(--accent-soft);font-weight:600}
figure{margin:16px 0}
figure img{width:100%;height:auto;border:1px solid var(--line);border-radius:8px;display:block}
figcaption{font-size:13px;color:var(--muted);margin-top:6px}
figure.estreita{max-width:560px}
.grade{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}
video{width:100%;border-radius:8px;border:1px solid var(--line);background:#000;margin:8px 0}
.aviso{background:var(--warn);border-left:4px solid var(--warn-line);padding:14px 16px;border-radius:6px;margin:16px 0}
.numeros{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:20px 0}
.numeros div{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;color:var(--muted);font-size:14px}
.numeros span{display:block;font-size:26px;color:var(--fg);font-weight:600;font-variant-numeric:tabular-nums}
.nota{font-size:14px;color:var(--muted)}
code{font-size:.9em;background:var(--accent-soft);padding:1px 5px;border-radius:4px}
ol li,ul li{margin-bottom:8px}
footer{padding:28px 16px 48px;color:var(--muted);font-size:13px}
"""


def main() -> None:
    destino = SAIDAS / "relatorio.html"
    destino.write_text(montar(), encoding="utf-8")
    print(f"Relatório: {destino}")


if __name__ == "__main__":
    main()
