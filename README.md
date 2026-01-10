# 🎬 VIDEO-ANALYSIS — Análise de Ações, Rosto e Sentimentos

Uma ferramenta para analisar vídeos com foco em:
- detecção e qualificação de ações (movimentos) 🕺
- detecção de rostos e análise de emoções (DeepFace) 🙂😢😠
- geração de arquivos `.jsonl` com os eventos detectados e posterior anotação dos frames 🎯

Este README foi adaptado à estrutura atual do projeto (conforme imagem fornecida).

---

## 🗂 Estrutura do projeto (visão geral)

- app/
  - faces/
    - main_faces.py        — script principal para detecção/análise de faces
    - overlays/            — utilitários / assets para desenhar no frame
    - requirements_b.txt   — dependências específicas (faces)
  - movements/
    - main_movements.py    — script principal para detecção/qualificação de ações
    - actions/             — lógica/definições de ações
    - core/                — núcleo do processamento de movimentos
    - pipeline/            — scripts auxiliares / etapas do pipeline
    - excluir_esse_arquivo*.py — arquivos temporários / de limpeza
    - requirements_a.txt   — dependências específicas (movements)
  - utils/                 — utilitários compartilhados
    - report.py — script responsável por criar o relatório das ações emoções.
    - video_activity_classifier.py — script responsável por cortar trechos do vídeo.
- data/                    — vídeos de entrada, conjuntos de teste
- yolov8n.pt, yolo11n.pt, yolo11n-pose.pt — pesos de modelos presentes no repositório
- README.en.md
- README.pt.md

> Observação: Há ambientes virtuais locais dentro de alguns subdirs (ex: `.venv_faces`, `.venv_move`). Recomenda-se usar um único ambiente no nível do projeto ou gerenciar ambientes por componente conforme preferir.

---

## 🚀 Como funciona (fluxo resumido)

1. Detectar e qualificar ações (movements)
   - Processa o vídeo frame a frame.
   - Gera um arquivo `.jsonl` com—por frame—labels de ações, bounding boxes e metadados.

2. Detectar rostos e analisar emoções (faces)
   - Pode reutilizar bounding boxes do `.jsonl` ou detectar faces novamente por frame.
   - Usa DeepFace para inferir emoções por rosto.

3. Anotar frames / gerar vídeo final
   - Um passo final consome os `.jsonl` e desenha labels/caixas nos frames; produz vídeo anotado e/ou relatórios agregados.

---

## 🧰 Principais bibliotecas usadas

- YOLO (detecção de objetos/ações) 🧭
- DeepFace (análise facial / emoções) 😊😢😡
- OpenCV (cv2) — leitura/escrita de vídeo e manipulação de frames 🎞️
- MediaPipe — landmarks, tracking ✨
- Outras dependências estão listadas em:
  - `app/movements/requirements_a.txt`
  - `app/faces/requirements_b.txt`

Recomenda-se consolidar as dependências num `requirements.txt` na raiz para facilitar a instalação (exemplo abaixo).

---

## ⚙️ Instalação (sugestão)

1. Clone o repositório:
```bash
git clone https://github.com/guillherms/video-analysis.git
cd video-analysis
```

2. Crie e ative um ambiente virtual (recomendado):
```bash
python -m venv .venv
source .venv/bin/activate    # macOS / Linux
# ou
.venv\Scripts\activate       # Windows
```

3. Instale dependências (exemplo consolidado):
Crie um `requirements.txt` ou instale manualmente as libs essenciais:
```bash
pip install -r app/movements/requirements_a.txt
pip install -r app/faces/requirements_b.txt
```
ou (instalação direta de pacotes principais):
```bash
pip install opencv-python deepface mediapipe ultralytics
```

Nota: Para melhor performance em inferência (YOLO / DeepFace), use GPU com CUDA quando disponível.

---

## ▶️ Exemplos de uso (com base na estrutura atual)

Detectar ações (movements) e gerar `.jsonl`:
```bash
python app/movements/main_movements.py --input data/input_video.mp4 --output outputs/actions.jsonl --model yolov8n.pt
```

Detectar faces e analisar emoções (faces):
```bash
python app/faces/main_faces.py --input data/input_video.mp4 --output outputs/faces.jsonl
```

Pipeline de anotação (exemplo sugerido — se ainda não existir, crie em `app/pipeline/`):
```bash
python app/pipeline/annotate.py --video data/input_video.mp4 --actions outputs/actions.jsonl --faces outputs/faces.jsonl --out outputs/annotated.mp4
```

A sintaxe exata das flags depende dos argumentos implementados nos scripts `main_*`. Ajuste conforme o código.

---

## 📝 Formato de saída sugerido (.jsonl)

Exemplo de linha em `.jsonl`:
```json
{"frame": 123, "timestamp": 4.1, "actions":[{"label":"walking","bbox":[x,y,w,h],"confidence":0.92}], "faces":[{"bbox":[x,y,w,h],"emotion":"happy","confidence":0.88}]}
```

Um arquivo `.jsonl` facilita streaming e consumo por etapas posteriores do pipeline.

---