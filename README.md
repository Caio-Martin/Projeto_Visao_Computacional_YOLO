# Sistema Inteligente de Analise de Imagens Metalicas

Desenvolvido para a disciplina de Computação Gráfica & Processamento de Imagens no [Centro Universitário Padre Anchieta](https://www.anchieta.br/) sob a supervisão do docente [Daniel Feitoza Ruis da Silva](https://www.linkedin.com/in/daniel-feitoza-a5216490/), no curso de Ciência da Computação (5º Semestre - 2026).

---
## Participantes 

- [Aline da Silva de Azevedo](https://github.com/asazeved)
- [Ana Júlia Lima Formiga](https://github.com/AnaJuliaFormiga)
- [Caio Martin do Nascimento](https://github.com/Caio-Martin)


## Visao geral
Este trabalho implementa um pipeline completo de visao computacional para inspecao de pecas metalicas, combinando tecnicas classicas (pre-processamento, histograma, binarizacao e contornos) com heuristicas para oxidação/dano e suporte opcional a um modelo customizado YOLO.

## Objetivo
- Detectar indícios de oxidacao e danos superficiais em imagens ou videos.
- Fornecer um painel visual consolidado com todas as etapas do pipeline.
- Disponibilizar interface grafica simples para uso sem linha de comando.

## Requisitos
- Python 3.9
- Dependencias principais:
  - opencv-python
  - numpy
  - matplotlib

Dependencias opcionais:
- ultralytics (para YOLOv8)
- Pillow (caso o ambiente exija para imagens)

Instalacao rapida:
```bash
pip install -r requirements.txt
```
Ou, manualmente:
```bash
pip install opencv-python numpy matplotlib
```
Opcional:
```bash
pip install ultralytics Pillow
```

## Como executar
### 1) Interface grafica
```bash
python sistema_visao_ia.py gui
```
A interface permite selecionar imagem, video ou webcam.

### 2) CLI
- Imagem:
```bash
python sistema_visao_ia.py caminho/para/imagem.jpg
```
- Webcam:
```bash
python sistema_visao_ia.py webcam
```
- Video:
```bash
python sistema_visao_ia.py caminho/para/video.mp4
```

## Saidas geradas
- `resultado_final.png`: painel com as etapas e o resumo final.
- `histograma.png`: histograma da imagem em tons de cinza.

## Estrutura do projeto
- [sistema_visao_ia.py](sistema_visao_ia.py): pipeline principal e interface grafica.
- [requirements.txt](requirements.txt): dependencias.
- [README.md](README.md): documentacao.

## Pipeline implementado (resumo tecnico)
1. **Aquisicao**: arquivo, video ou webcam.
2. **Pre-processamento**: escala de cinza, blur gaussiano e bordas (Canny).
3. **Analise de cor**: HSV, saturacao media e matiz dominante.
4. **Histograma**: avaliacao de iluminacao e contraste.
5. **Binarizacao**: threshold simples e segmentacao por contornos.
6. **Heuristicas**: oxidacao e dano superficial por mascaras de cor e morfologia.
7. **Painel final**: visualizacao consolidada e conclusao textual.

## Heuristica de oxidacao e dano
A deteccao de defeitos metalicos nao e um classificador treinado. Ela utiliza:
- Mascara de “tons de ferrugem” no HSV (intervalos de matiz/valor/saturacao).
- Morfologia (opening/closing) para reduzir ruido.
- Operadores top-hat e black-hat para realcar riscos e manchas.
- Medidas de area relativa das mascaras e contraste local.

Pontuacoes (score) calculadas:
- **Oxidacao**: combina area de ferrugem e baixa saturacao.
- **Dano**: combina area de manchas/riscos e contraste local.

As pontuacoes sao normalizadas e limitadas em 100.

Equacoes (conceito):

$$
score_{ox} = \min(100, 6 \cdot area_{ferrugem} + 0.5 \cdot \max(0, 35 - saturacao))
$$
$$
score_{dano} = \min(100, 8 \cdot area_{dano} + 0.03 \cdot \max(0, contraste - 120))
$$

## Modelo customizado (YOLO)
Se existir o arquivo `modelo_metal.pt` no mesmo diretorio, o sistema ativa um detector YOLO customizado para defeitos. O resultado anotado substitui o overlay das heuristicas e entra no resumo final.

Para usar:
- Coloque o arquivo `modelo_metal.pt` ao lado de [sistema_visao_ia.py](sistema_visao_ia.py).
- Instale `ultralytics`.

## Parametros importantes
- **Canny**: limiares 100 e 200.
- **Threshold**: limiar 127 para binarizacao.
- **HSV**: saturacao aumentada em +50 para visualizacao.
- **YOLO**: confianca padrao 0.25.

## Observacoes
- Para video/webcam, pressione `q` para encerrar.
- O modelo YOLOv8n e baixado automaticamente na primeira execucao (quando `ultralytics` esta instalado).
