"""
============================================================
    SISTEMA INTELIGENTE DE ANÁLISE DE IMAGENS METÁLICAS
    Atividade Integradora – Visão Computacional + Análise de Defeitos
============================================================

Etapas implementadas:
    1. Aquisição de imagem, vídeo ou webcam
    2. Processamento (escala de cinza, blur, bordas)
    3. Análise de cor (HSV, canais)
    4. Histograma (análise de iluminação)
    5. Binarização (threshold + segmentação)
    6. Heurística para oxidação e dano superficial
    7. Resultado final com informações consolidadas

Requisitos:
    pip install opencv-python numpy matplotlib
    opcional: pip install ultralytics Pillow
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import tkinter as tk
from tkinter import filedialog, messagebox

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


MODELO_DEFEITOS_PATH = "modelo_metal.pt"


# ============================================================
# UTILITÁRIOS
# ============================================================

def interpretar_histograma(gray: np.ndarray) -> str:
    """
    Analisa o histograma de uma imagem em escala de cinza e
    retorna uma descrição da distribuição de iluminação.
    """
    media = gray.mean()
    desvio = gray.std()

    # Limiares empiricos para classificar exposicao e contraste.
    if media < 80:
        qualidade = "Imagem SUBEXPOSTA (escura)"
    elif media > 180:
        qualidade = "Imagem SUPEREXPOSTA (clara)"
    else:
        qualidade = "Imagem com exposição NORMAL"

    if desvio < 30:
        contraste = "baixo contraste"
    elif desvio > 70:
        contraste = "alto contraste"
    else:
        contraste = "contraste moderado"

    return f"{qualidade} — {contraste} (média={media:.1f}, σ={desvio:.1f})"


def interpretar_hsv(hsv: np.ndarray) -> str:
    """
    Extrai o matiz dominante e a saturação média da imagem HSV.
    """
    hue_medio = hsv[:, :, 0].mean()          # 0-179 no OpenCV
    saturacao_media = hsv[:, :, 1].mean()    # 0-255
    brilho_medio = hsv[:, :, 2].mean()       # 0-255

    # Faixas aproximadas de matiz no espaco HSV do OpenCV (0-179), circular.
    faixas_hue = [
        (0,   10,  "Vermelho"),
        (11,  25,  "Laranja"),
        (26,  35,  "Amarelo"),
        (36,  85,  "Verde"),
        (86, 130,  "Ciano/Azul"),
        (131, 155, "Azul"),
        (156, 170, "Roxo/Magenta"),
        (171, 179, "Vermelho"),
    ]
    cor_dominante = "Indefinida"
    for lo, hi, nome in faixas_hue:
        if lo <= hue_medio <= hi:
            cor_dominante = nome
            break

    return (
        f"Matiz dominante: {cor_dominante} (H={hue_medio:.1f})  |  "
        f"Saturação média: {saturacao_media:.1f}  |  Brilho médio: {brilho_medio:.1f}"
    )


def eh_video(caminho: str) -> bool:
    extensoes_video = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v")
    return caminho.lower().endswith(extensoes_video)


def tem_modelo_customizado() -> bool:
    return YOLO is not None and os.path.exists(MODELO_DEFEITOS_PATH)


def detectar_defeitos_modelo(img: np.ndarray):
    """
    Executa um modelo customizado, caso o arquivo de pesos exista.
    Espera um YOLO treinado para classes de defeitos metálicos, como oxidação,
    fissura, risco ou amassado.
    """
    if not tem_modelo_customizado():
        return None

    model = YOLO(MODELO_DEFEITOS_PATH)
    # A inferencia retorna caixas, classes e confiancas; plot() gera anotacao visual.
    results = model(img, conf=0.25, verbose=False)
    annotated_bgr = results[0].plot()
    annotated = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    deteccoes = []
    nomes = model.names
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        classe = nomes[cls_id]
        confianca = float(box.conf[0])
        bbox = box.xyxy[0].cpu().numpy().astype(int).tolist()
        deteccoes.append({"classe": classe, "confiança": round(confianca, 3), "bbox": bbox})

    contagem = {}
    for d in deteccoes:
        contagem[d["classe"]] = contagem.get(d["classe"], 0) + 1

    if contagem:
        resumo = "Modelo customizado detectou: " + ", ".join(
            f"{v}x {k}" for k, v in sorted(contagem.items(), key=lambda x: -x[1])
        )
    else:
        resumo = "Modelo customizado não encontrou defeitos com confiança suficiente"

    return {
        "annotated": annotated,
        "deteccoes": deteccoes,
        "resumo": resumo,
        "tem_modelo": True,
    }


def criar_overlay_defeitos(img_bgr: np.ndarray, rust_mask: np.ndarray, damage_mask: np.ndarray):
    overlay = img_bgr.copy()

    if rust_mask is not None and rust_mask.any():
        overlay[rust_mask > 0] = (0, 0, 255)

    if damage_mask is not None and damage_mask.any():
        overlay[damage_mask > 0] = (0, 255, 255)

    blended = cv2.addWeighted(img_bgr, 0.70, overlay, 0.30, 0)
    return blended


def analisar_defeitos_metalicos(img: np.ndarray):
    """
    Detecta indícios simples de oxidação e dano superficial por heurística.

    A abordagem não é um classificador treinado; ela procura padrões comuns:
    tons de ferrugem, manchas escuras e regiões com contraste/texture anormais.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Categorizacao de ferrugem: pixels HSV com matiz laranja-avermelhado
    # e saturacao/valor moderados entram na mascara de oxidacao.
    # Duas faixas cobrem variacoes de iluminacao e tons de ferrugem.
    rust_mask_1 = cv2.inRange(hsv, (5, 45, 35), (25, 255, 220))
    rust_mask_2 = cv2.inRange(hsv, (0, 30, 35), (18, 255, 180))
    rust_mask = cv2.bitwise_or(rust_mask_1, rust_mask_2)

    # Reduz ruido e pequenos pontos isolados.
    rust_mask = cv2.medianBlur(rust_mask, 5)
    rust_mask = cv2.morphologyEx(
        rust_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )

    # Black-hat realca pontos escuros; top-hat realca riscos claros.
    blackhat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, blackhat_kernel)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, blackhat_kernel)

    _, dark_spots = cv2.threshold(blackhat, 18, 255, cv2.THRESH_BINARY)
    _, bright_scratch = cv2.threshold(tophat, 18, 255, cv2.THRESH_BINARY)

    damage_mask = cv2.bitwise_or(dark_spots, bright_scratch)
    # Fecha pequenos buracos e conecta fragmentos.
    damage_mask = cv2.morphologyEx(
        damage_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )

    rust_pct = float(np.count_nonzero(rust_mask)) / rust_mask.size * 100.0
    damage_pct = float(np.count_nonzero(damage_mask)) / damage_mask.size * 100.0

    contraste_local = cv2.Laplacian(gray, cv2.CV_64F).var()
    brilho_medio = float(gray.mean())
    saturacao_media = float(hsv[:, :, 1].mean())

    # Scores combinam area detectada e indicadores globais.
    oxidation_score = min(100.0, rust_pct * 6.0 + max(0.0, (35.0 - saturacao_media) * 0.5))
    damage_score = min(100.0, damage_pct * 8.0 + max(0.0, contraste_local - 120.0) * 0.03)

    if oxidation_score >= 35:
        oxidacao = "Oxidação provável"
    elif oxidation_score >= 15:
        oxidacao = "Possível oxidação"
    else:
        oxidacao = "Sem indícios fortes de oxidação"

    if damage_score >= 35:
        dano = "Dano superficial provável"
    elif damage_score >= 15:
        dano = "Possível dano superficial"
    else:
        dano = "Sem indícios fortes de dano"

    if oxidation_score < 15 and damage_score < 15:
        conclusao = "Nenhuma anomalia relevante detectada"
    else:
        conclusao = f"{oxidacao} | {dano}"

    resumo = (
        f"Oxidação: {oxidacao} (score={oxidation_score:.1f}, área={rust_pct:.2f}%) | "
        f"Dano: {dano} (score={damage_score:.1f}, área={damage_pct:.2f}%) | "
        f"Brilho médio={brilho_medio:.1f} | Contraste local={contraste_local:.1f}"
    )

    overlay = criar_overlay_defeitos(img, rust_mask, damage_mask)

    modelo_custom = detectar_defeitos_modelo(img)
    if modelo_custom is not None:
        overlay = cv2.cvtColor(modelo_custom["annotated"], cv2.COLOR_RGB2BGR)
        resumo = resumo + f" | {modelo_custom['resumo']}"
        if modelo_custom["deteccoes"]:
            conclusao = resumo

    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    return {
        "rust_mask": rust_mask,
        "damage_mask": damage_mask,
        "overlay": overlay_rgb,
        "rust_pct": rust_pct,
        "damage_pct": damage_pct,
        "oxidation_score": oxidation_score,
        "damage_score": damage_score,
        "resumo": resumo,
        "conclusao": conclusao,
        "tem_modelo_customizado": modelo_custom is not None,
    }


# ============================================================
# ETAPA 1 – AQUISIÇÃO DE IMAGEM
# ============================================================

def adquirir_imagem(fonte: str = "arquivo", caminho: str = "imagem.jpg"):
    """
    Carrega uma imagem de arquivo ou captura um frame da webcam.

    Parâmetros
    ----------
    fonte   : "arquivo" ou "webcam"
    caminho : caminho do arquivo quando fonte="arquivo"

    Retorna
    -------
    img     : imagem BGR (numpy array)
    img_rgb : imagem RGB para exibição com matplotlib
    """
    if fonte == "webcam":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Não foi possível abrir a webcam.")
        ret, img = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError("Falha ao capturar frame da webcam.")
        print("[INFO] Frame capturado da webcam com sucesso.")
    else:
        if not os.path.exists(caminho):
            raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
        img = cv2.imread(caminho)
        if img is None:
            raise ValueError(f"OpenCV não conseguiu ler o arquivo: {caminho}")
        print(f"[INFO] Imagem carregada: {caminho}  |  tamanho: {img.shape[1]}x{img.shape[0]} px")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, img_rgb


def adquirir_video(caminho: str = "video.mp4"):
    if caminho == "webcam":
        cap = cv2.VideoCapture(0)
    else:
        if not os.path.exists(caminho):
            raise FileNotFoundError(f"Arquivo de vídeo não encontrado: {caminho}")
        cap = cv2.VideoCapture(caminho)

    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir a fonte de vídeo: {caminho}")

    return cap


# ============================================================
# ETAPA 2 – PROCESSAMENTO DE IMAGEM
# ============================================================

def processar_imagem(img: np.ndarray):
    """
    Aplica escala de cinza, blur gaussiano e detecção de bordas (Canny).

    Retorna
    -------
    gray  : imagem em escala de cinza
    blur  : imagem com suavização gaussiana
    edges : mapa de bordas detectadas pelo Canny
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # GaussianBlur: kernel 5×5, sigma automático (0)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny: limiares 100 (baixo) e 200 (alto)
    edges = cv2.Canny(blur, 100, 200)

    print("[INFO] Processamento concluído: escala de cinza, blur, bordas Canny.")
    return gray, blur, edges


# ============================================================
# ETAPA 3 – ANÁLISE DE COR (HSV)
# ============================================================

def analisar_cor(img: np.ndarray):
    """
    Converte para HSV, altera saturação (+50) e separa os canais.

    Retorna
    -------
    hsv          : imagem HSV original
    img_sat_alta : imagem RGB com saturação aumentada
    h, s, v      : canais individuais
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Aumentar saturação em 50 unidades (clip em 255)
    s_alta = np.clip(s.astype(np.int32) + 50, 0, 255).astype(np.uint8)
    hsv_modificado = cv2.merge([h, s_alta, v])

    img_sat_alta = cv2.cvtColor(hsv_modificado, cv2.COLOR_HSV2RGB)

    descricao = interpretar_hsv(hsv)
    print(f"[INFO] Análise HSV → {descricao}")
    return hsv, img_sat_alta, h, s, v


# ============================================================
# ETAPA 4 – HISTOGRAMA
# ============================================================

def gerar_histograma(gray: np.ndarray) -> plt.Figure:
    """
    Gera histograma de intensidade e retorna a figura matplotlib.
    """
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(gray.ravel(), bins=256, range=(0, 256), color="steelblue", alpha=0.85)
    ax.axvline(gray.mean(), color="red", linestyle="--", linewidth=1.5,
               label=f"Média = {gray.mean():.1f}")
    ax.set_title("Histograma de Intensidade (Canal Cinza)", fontsize=11)
    ax.set_xlabel("Intensidade (0 = preto, 255 = branco)")
    ax.set_ylabel("Número de pixels")
    ax.legend()
    fig.tight_layout()

    descricao = interpretar_histograma(gray)
    print(f"[INFO] Histograma gerado → {descricao}")
    return fig


# ============================================================
# ETAPA 5 – BINARIZAÇÃO
# ============================================================

def binarizar(gray: np.ndarray, limiar: int = 127):
    """
    Aplica threshold simples e segmentação por contornos.

    Retorna
    -------
    thresh          : imagem binarizada
    img_contornos   : imagem BGR com contornos desenhados
    n_contornos     : número de objetos segmentados
    """
    _, thresh = cv2.threshold(gray, limiar, 255, cv2.THRESH_BINARY)

    # Segmentação: encontrar contornos externos
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Criar imagem colorida para visualizar contornos
    img_contornos = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_contornos, contornos, -1, (0, 255, 0), 2)

    n_contornos = len(contornos)
    print(f"[INFO] Binarização concluída — limiar={limiar} — {n_contornos} contornos encontrados.")
    return thresh, img_contornos, n_contornos


# ============================================================
# ETAPA 6 – IA COM YOLO
# ============================================================

def detectar_objetos_yolo(img: np.ndarray, conf: float = 0.25):
    """
    Executa YOLOv8n para detecção de objetos.

    Parâmetros
    ----------
    img  : imagem BGR
    conf : confiança mínima para aceitar detecções

    Retorna
    -------
    annotated   : imagem RGB com anotações do YOLO
    deteccoes   : lista de dicionários {classe, confiança, bbox}
    resumo_ia   : string com resumo das detecções
    """
    if YOLO is None:
        raise RuntimeError("Ultralytics/YOLO não está disponível neste ambiente.")

    print("[INFO] Carregando modelo YOLOv8n...")
    model = YOLO("yolov8n.pt")   # download automático na 1ª execução

    results = model(img, conf=conf, verbose=False)
    annotated_bgr = results[0].plot()
    annotated = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    deteccoes = []
    nomes = model.names
    boxes = results[0].boxes

    for box in boxes:
        cls_id = int(box.cls[0])
        classe = nomes[cls_id]
        confianca = float(box.conf[0])
        bbox = box.xyxy[0].cpu().numpy().astype(int).tolist()
        deteccoes.append({"classe": classe, "confiança": round(confianca, 3), "bbox": bbox})

    # Contagem por classe
    contagem = {}
    for d in deteccoes:
        contagem[d["classe"]] = contagem.get(d["classe"], 0) + 1

    if contagem:
        resumo_ia = "Objetos detectados: " + ", ".join(
            f'{v}x {k}' for k, v in sorted(contagem.items(), key=lambda x: -x[1])
        )
    else:
        resumo_ia = "Nenhum objeto detectado com confiança >= {:.0%}".format(conf)

    print(f"[INFO] YOLO → {resumo_ia}")
    return annotated, deteccoes, resumo_ia


def processar_frame_metalico(img: np.ndarray):
    # Agrega todas as etapas do pipeline em um unico dicionario.
    gray, blur, edges = processar_imagem(img)
    hsv, img_sat_alta, h, s, v = analisar_cor(img)
    descricao_hist = interpretar_histograma(gray)

    thresh, img_contornos, n_contornos = binarizar(gray)
    img_contornos_rgb = cv2.cvtColor(img_contornos, cv2.COLOR_BGR2RGB)

    defeitos = analisar_defeitos_metalicos(img)

    return {
        "gray": gray,
        "blur": blur,
        "edges": edges,
        "hsv": hsv,
        "img_sat_alta": img_sat_alta,
        "descricao_hist": descricao_hist,
        "thresh": thresh,
        "img_contornos_rgb": img_contornos_rgb,
        "n_contornos": n_contornos,
        "defeitos": defeitos,
    }


# ============================================================
# ETAPA 7 – RESULTADO FINAL (PAINEL CONSOLIDADO)
# ============================================================

def exibir_resultado_final(
    img_rgb, gray, edges, img_sat_alta,
    thresh, img_contornos_rgb,
    overlay_defeitos, resumo_defeitos, descricao_hist, descricao_hsv,
    n_contornos, conclusao_defeitos
):
    """
    Monta um painel com todas as etapas do pipeline e
    exibe um resumo textual no terminal.
    """
    # ------ Painel visual ------
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle("Sistema Inteligente de Análise de Imagens com IA", fontsize=14, fontweight="bold")

    imagens = [
        (img_rgb,           "1 – Original (RGB)"),
        (gray,              "2 – Escala de Cinza"),
        (edges,             "3 – Bordas (Canny)"),
        (img_sat_alta,      "4 – HSV: Saturação +50"),
        (thresh,            "5 – Binarização (Threshold)"),
        (img_contornos_rgb, "6 – Segmentação (Contornos)"),
        (overlay_defeitos,  "7 – Destaque de Oxidação/Dano"),
    ]

    for idx, (ax, (imagem, titulo)) in enumerate(zip(axes.ravel(), imagens)):
        cmap = "gray" if imagem.ndim == 2 else None
        ax.imshow(imagem, cmap=cmap)
        ax.set_title(titulo, fontsize=9)
        ax.axis("off")

    # Ultima celula: texto resumo
    ax_info = axes.ravel()[-1]
    ax_info.axis("off")
    info_texto = (
        f"RESUMO DO PIPELINE\n"
        f"{'─'*28}\n"
        f"{descricao_hist}\n\n"
        f"{descricao_hsv}\n\n"
        f"Contornos segmentados: {n_contornos}\n\n"
        f"{resumo_defeitos}\n\n"
        f"Conclusão: {conclusao_defeitos}"
    )
    ax_info.text(0.05, 0.95, info_texto, transform=ax_info.transAxes,
                 fontsize=7.5, verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    # Salva o painel consolidado para referencia.
    plt.savefig("resultado_final.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[INFO] Resultado salvo em: resultado_final.png")

    # ------ Resumo no terminal ------
    print("\n" + "=" * 60)
    print("  RESULTADO FINAL DO SISTEMA")
    print("=" * 60)
    print(f"  Iluminação  : {descricao_hist}")
    print(f"  Cores       : {descricao_hsv}")
    print(f"  Segmentação : {n_contornos} regiões detectadas via threshold")
    print(f"  Defeitos    : {resumo_defeitos}")
    print(f"  Conclusão   : {conclusao_defeitos}")
    print("=" * 60 + "\n")


def processar_video_metalico(caminho_video: str):
    cap = adquirir_video(caminho_video)
    print("[INFO] Processando vídeo. Pressione 'q' para sair.")

    total_frames = 0
    frames_com_oxidacao = 0
    frames_com_dano = 0
    ultimo_resumo = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1
        # Processa 1 a cada 5 frames para acelerar.
        if total_frames % 5 != 0:
            continue

        resultados = processar_frame_metalico(frame)
        defeitos = resultados["defeitos"]

        if defeitos["oxidation_score"] >= 15:
            frames_com_oxidacao += 1
        if defeitos["damage_score"] >= 15:
            frames_com_dano += 1

        ultimo_resumo = defeitos["conclusao"]

        overlay_bgr = cv2.cvtColor(defeitos["overlay"], cv2.COLOR_RGB2BGR)
        indicio_oxidacao = "SIM" if defeitos["oxidation_score"] >= 15 else "NAO"
        texto = f"Oxidacao: {indicio_oxidacao} (score={defeitos['oxidation_score']:.1f})"
        cv2.putText(
            overlay_bgr,
            texto,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255) if indicio_oxidacao == "SIM" else (0, 200, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Analise de Metal - Video", overlay_bgr)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("  RESUMO DO VÍDEO")
    print("=" * 60)
    print(f"  Frames lidos       : {total_frames}")
    print(f"  Frames com oxidação: {frames_com_oxidacao}")
    print(f"  Frames com dano     : {frames_com_dano}")
    print(f"  Última conclusão    : {ultimo_resumo}")
    print("=" * 60 + "\n")


def selecionar_arquivo_imagem():
    caminho = filedialog.askopenfilename(
        title="Selecionar imagem metálica",
        filetypes=[
            ("Imagens", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
            ("Todos os arquivos", "*.*"),
        ],
    )
    if caminho:
        executar_pipeline(fonte="arquivo", caminho=caminho)


def selecionar_arquivo_video():
    caminho = filedialog.askopenfilename(
        title="Selecionar vídeo metálico",
        filetypes=[
            ("Vídeos", "*.mp4 *.avi *.mov *.mkv *.wmv *.m4v"),
            ("Todos os arquivos", "*.*"),
        ],
    )
    if caminho:
        executar_pipeline(fonte="video", caminho=caminho)


def abrir_interface_grafica():
    janela = tk.Tk()
    janela.title("Análise de Partes Metálicas")
    janela.geometry("460x300")
    janela.resizable(False, False)

    titulo = tk.Label(
        janela,
        text="Sistema de Inspeção de Metal",
        font=("Segoe UI", 18, "bold"),
        pady=18,
    )
    titulo.pack()

    subtitulo = tk.Label(
        janela,
        text="Escolha uma imagem, um vídeo ou a webcam para inspeção de oxidação e dano.",
        font=("Segoe UI", 10),
        wraplength=380,
        justify="center",
    )
    subtitulo.pack(pady=6)

    painel = tk.Frame(janela)
    painel.pack(pady=18)

    botoes = [
        ("Selecionar imagem", selecionar_arquivo_imagem),
        ("Selecionar vídeo", selecionar_arquivo_video),
        ("Usar webcam", lambda: executar_pipeline(fonte="webcam", caminho="webcam")),
        ("Sair", janela.destroy),
    ]

    for texto, acao in botoes:
        tk.Button(
            painel,
            text=texto,
            command=acao,
            width=24,
            pady=6,
            bg="#1f4e79" if texto != "Sair" else "#6b2f2f",
            fg="white",
            relief="flat",
            cursor="hand2",
        ).pack(pady=5)

    rodape = tk.Label(
        janela,
        text=(
            f"Modelo customizado: {'ativo' if tem_modelo_customizado() else 'não encontrado'}\n"
            f"Arquivo esperado: {MODELO_DEFEITOS_PATH}"
        ),
        font=("Segoe UI", 9),
        justify="center",
    )
    rodape.pack(side="bottom", pady=10)

    janela.mainloop()


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def executar_pipeline(fonte: str = "arquivo", caminho: str = "imagem.jpg"):
    """
    Executa todas as etapas do sistema integrado.
    """
    print("\n🔷 SISTEMA INTELIGENTE DE ANÁLISE DE IMAGENS COM IA 🔷\n")

    if fonte in ("video", "webcam"):
        processar_video_metalico(caminho)
        return

    # Etapa 1
    img, img_rgb = adquirir_imagem(fonte, caminho)

    resultados = processar_frame_metalico(img)
    descricao_hsv = interpretar_hsv(resultados["hsv"])
    descricao_hist = resultados["descricao_hist"]

    fig_hist = gerar_histograma(resultados["gray"])
    fig_hist.savefig("histograma.png", dpi=120)
    plt.close(fig_hist)

    defeitos = resultados["defeitos"]

    exibir_resultado_final(
        img_rgb, resultados["gray"], resultados["edges"], resultados["img_sat_alta"],
        resultados["thresh"], resultados["img_contornos_rgb"],
        defeitos["overlay"], defeitos["resumo"], descricao_hist, descricao_hsv,
        resultados["n_contornos"], defeitos["conclusao"]
    )


# ============================================================
# PONTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    # Uso:
    #   python sistema_visao_ia.py imagem.jpg        → analisa arquivo
    #   python sistema_visao_ia.py webcam            → captura da câmera
    #   python sistema_visao_ia.py video.mp4         → analisa vídeo
    #   python sistema_visao_ia.py gui               → abre a interface

    if len(sys.argv) >= 2:
        arg = sys.argv[1]
        if arg.lower() == "gui":
            abrir_interface_grafica()
            sys.exit(0)
        if arg.lower() == "webcam":
            executar_pipeline(fonte="webcam", caminho="webcam")
        elif eh_video(arg):
            executar_pipeline(fonte="video", caminho=arg)
        else:
            executar_pipeline(fonte="arquivo", caminho=arg)
    else:
        abrir_interface_grafica()
