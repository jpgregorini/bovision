# BoviSight MVP — Design Spec

> Sistema de pesagem automatizada de bovinos por visao computacional.
> MVP end-to-end com dados do Kaggle e mock de hardware.

---

## 1. Decisoes de Escopo

| Decisao | Escolha |
|---|---|
| Escopo | Pipeline end-to-end completo, simplificado |
| Deteccao | YOLOv8n-seg prova de conceito com dataset Kaggle |
| Regressao peso | XGBoost com dataset real do Kaggle |
| Hardware | Mock local (imagens de diretorio, sem TensorRT) |
| Dashboard | Funcional minimo, tela unica, sem auth |
| Comunicacao | HTTP direto (REST), sem MQTT |
| Banco | SQLite via SQLAlchemy (migravel para PostgreSQL) |

---

## 2. Dados e Modelos de IA

### 2.1 Dataset de Deteccao (YOLOv8)

**Fonte:** Kaggle — datasets de bovinos com imagens anotadas (bounding boxes ou segmentacao). Prioridade para datasets com anotacoes no formato YOLO ou COCO (convertivel).

**Modelo:** YOLOv8n-seg (nano + segmentacao). Nano por ser o menor e mais rapido — ideal para prova de conceito e para o Jetson futuro. Segmentacao porque o sistema precisa da mascara do animal para extrair medidas.

**Treino:** Script `scripts/train_detection.py`:
- Baixa/organiza o dataset na estrutura `data/annotated/`
- Gera o `data.yaml`
- Treina com a biblioteca `ultralytics`
- Exporta `best.pt` e `best.onnx` para `models/detection/`

**Avaliacao:** Script `scripts/evaluate.py` roda no split de test e reporta mAP, precisao e recall.

### 2.2 Dataset de Regressao de Peso (XGBoost)

**Fonte:** Kaggle — datasets com medidas corporais de bovinos + peso real. Features buscadas: comprimento, altura, perimetro toracico, largura. As features disponiveis serao mapeadas para as features que o sistema RealSense extrairia.

**Modelo:** XGBoost regressor com StandardScaler nas features.

**Treino:** Script `scripts/train_regression.py`:
- Le `data/weighings/dataset.csv`
- Split train/test 80/20
- Cross-validation 5-fold
- Salva `weight_model.pkl` + `scaler.pkl` em `models/regression/`
- Reporta MAE, RMSE, R-quadrado, erro percentual medio

### 2.3 Fora do MVP
- Fine-tune por fazenda (`finetune_farm.py`)
- TensorRT / `.engine`
- Data augmentation avancado via Roboflow

---

## 3. Pipeline de Campo (Mock Local)

### 3.1 Visao Geral

Cinco modulos em `src/`, cada um com uma responsabilidade. Modulos que dependem de hardware possuem modo mock ativado por `BOVISION_ENV=dev`.

```
capture.py  ->  detect.py  ->  measure.py  ->  predict.py
         \____________________________________________/
                        pipeline.py (orquestra tudo)
```

### 3.2 capture.py — Captura de Frames

**Modo producao (futuro):** Conecta nas cameras RealSense via `pyrealsense2`, retorna frame RGB + depth map.

**Modo mock (MVP):** Le imagens `.jpg` de `data/sample_images/`. Para o depth map, gera um mapa sintetico uniforme simulando distancia fixa de 3m.

```python
class FrameCapture:
    def get_frame(self) -> tuple[np.ndarray, np.ndarray]:
        """Retorna (rgb, depth)"""
```

### 3.3 detect.py — Deteccao + Segmentacao

Carrega o modelo YOLOv8 (`best.pt` no MVP). Recebe frame RGB, retorna mascara binaria + bounding box + confianca. Retorna `None` se confianca < threshold (padrao 0.5).

```python
class BovineDetector:
    def detect(self, rgb_frame: np.ndarray) -> Detection | None

@dataclass
class Detection:
    mask: np.ndarray             # binaria, mesmo tamanho do frame
    box: tuple[int,int,int,int]  # x1, y1, x2, y2
    confidence: float
```

### 3.4 measure.py — Extracao de Medidas

**Modo producao (futuro):** Combina mascara + depth map + calibracao da camera para calcular medidas reais em metros.

**Modo mock (MVP):** Usa a mascara para extrair medidas em pixels (comprimento, altura, largura do bounding box da mascara, area, perimetro). Aplica fator de conversao fixo pixels-para-metros baseado na distancia simulada.

```python
class MorphologyMeasurer:
    def measure(self, mask: np.ndarray, depth: np.ndarray) -> Measurements

@dataclass
class Measurements:
    comprimento_m: float
    altura_m: float
    largura_m: float
    area_m2: float
    perimetro_m: float
```

### 3.5 predict.py — Estimativa de Peso

Carrega `weight_model.pkl` + `scaler.pkl`. Recebe `Measurements`, aplica o scaler, prediz com XGBoost. Retorna peso em kg + confianca (derivada do desvio entre as arvores do ensemble).

```python
class WeightPredictor:
    def predict(self, measurements: Measurements) -> WeightEstimate

@dataclass
class WeightEstimate:
    weight_kg: float
    confidence: float
```

### 3.6 pipeline.py — Orquestrador

Loop continuo:
1. Captura frame via `FrameCapture`
2. Detecta bovino via `BovineDetector`
3. Se detectado com confianca suficiente, acumula N frames de confirmacao (padrao 3)
4. Mede o animal via `MorphologyMeasurer`
5. Estima peso via `WeightPredictor`
6. Salva foto de evidencia em `data/evidence/` (nomeada pelo timestamp Unix)
7. Envia resultado para API via HTTP POST em `/api/weighings`
8. Aguarda cooldown entre pesagens (padrao 30s)

---

## 4. API e Banco de Dados

### 4.1 Stack

- FastAPI com SQLAlchemy ORM + SQLite
- Alembic para migracoes de schema
- Pydantic para validacao de entrada/saida

### 4.2 Modelos do Banco

**Tabela `animals`:**

| Coluna | Tipo | Notas |
|---|---|---|
| id | INTEGER PK | autoincrement |
| rfid | TEXT UNIQUE | identificador do animal |
| raca | TEXT | |
| sexo | TEXT | 'M' ou 'F' |
| created_at | DATETIME | |

**Tabela `weighings`:**

| Coluna | Tipo | Notas |
|---|---|---|
| id | INTEGER PK | autoincrement |
| animal_id | INTEGER FK nullable | -> animals.id |
| weight_kg | FLOAT | |
| confidence | FLOAT | |
| comprimento_m | FLOAT | |
| altura_m | FLOAT | |
| largura_m | FLOAT | |
| area_m2 | FLOAT | |
| perimetro_m | FLOAT | |
| evidence_path | TEXT | caminho local da foto |
| created_at | DATETIME | |

**Tabela `farms`:**

| Coluna | Tipo | Notas |
|---|---|---|
| id | INTEGER PK | autoincrement |
| name | TEXT | |
| location | TEXT | |
| created_at | DATETIME | |

`animal_id` em weighings e nullable porque o pipeline nao identifica qual animal especifico e — apenas detecta um bovino. A associacao animal-pesagem sera manual ou via RFID no futuro. Nao ha FK de animal para farm no MVP. A tabela farms existe para manter a estrutura, mas o sistema funciona como single-farm.

### 4.3 Estrutura de Arquivos

```
api/
├── main.py          # FastAPI app, CORS, startup/shutdown
├── database.py      # Engine SQLAlchemy, SessionLocal, Base
├── models.py        # Modelos SQLAlchemy (tabelas)
├── schemas.py       # Schemas Pydantic (request/response)
└── routers/
    ├── weighings.py # CRUD pesagens
    ├── animals.py   # CRUD animais
    └── reports.py   # Alertas e estatisticas
```

Removidos vs README: `auth.py` (sem autenticacao) e `farms.py` (rota minima em main.py se necessario). Adicionados vs README: `models.py` e `schemas.py` (boa pratica FastAPI).

### 4.4 Endpoints

**Pesagens:**
- `POST /api/weighings` — pipeline envia pesagem nova
- `GET /api/weighings` — lista pesagens (paginado, filtro por data)
- `GET /api/weighings/{id}` — uma pesagem especifica

**Animais:**
- `POST /api/animals` — cadastrar animal
- `GET /api/animals` — listar animais
- `GET /api/animals/{id}` — detalhe + historico de pesagens

**Relatorios:**
- `GET /api/reports/summary` — total de pesagens, peso medio, ultima pesagem
- `GET /api/reports/alerts` — animais com variacao de peso > 5% negativa
- `GET /api/reports/weight-history` — series temporais para graficos

### 4.5 Fora do MVP
- Autenticacao JWT
- Multi-fazenda (filtro farm_id)
- Upload de fotos para MinIO
- MQTT broker
- Exportacao PDF/Excel

---

## 5. Dashboard

### 5.1 Stack

- React com Vite
- Tailwind CSS
- Recharts para graficos
- Fetch API nativa (sem axios)

### 5.2 Tela Unica — Visao Geral

Quatro blocos verticais, sem navegacao:

1. **Stats Cards (topo):** 4 cards — Total de Pesagens, Peso Medio, Ultima Pesagem, Alertas Ativos
2. **Grafico de Historico:** Linha Recharts mostrando peso x tempo nos ultimos 30 dias
3. **Tabela de Pesagens Recentes:** ID, peso, confianca, data, link para evidencia
4. **Lista de Alertas:** Animais com perda de peso significativa

### 5.3 Estrutura de Arquivos

```
dashboard/
├── index.html
├── package.json
├── vite.config.js
├── tailwind.config.js
├── postcss.config.js
└── src/
    ├── main.jsx
    ├── App.jsx
    ├── api.js
    └── components/
        ├── StatsCards.jsx
        ├── WeightChart.jsx
        ├── WeighingsTable.jsx
        └── AlertsList.jsx
```

### 5.4 Fluxo de Dados

`App.jsx` monta a pagina e faz 4 chamadas via `useEffect`:
- `api.getSummary()` -> `StatsCards`
- `api.getWeightHistory()` -> `WeightChart`
- `api.getWeighings()` -> `WeighingsTable`
- `api.getAlerts()` -> `AlertsList`

Cada componente recebe dados via props. Sem estado global (Redux/Zustand). Auto-refresh via `setInterval` a cada 30 segundos.

### 5.5 Fora do MVP
- Roteamento (React Router)
- Autenticacao / login
- Pagina de detalhe por animal
- Modo escuro
- Responsividade mobile
- Internacionalizacao

---

## 6. Infraestrutura

### 6.1 Fluxo de Conexao

```
data/sample_images/*.jpg
        |
        | le imagens
        v
pipeline.py (loop continuo)
capture -> detect -> measure -> predict -> salvar evidence
        |
        | POST /api/weighings
        v
API FastAPI (:8000) + SQLite bovision.db
        |
        | GET /api/*
        v
Dashboard React (:5173) + auto-refresh 30s
```

### 6.2 Como Rodar

Tres terminais:

```
Terminal 1 — API:
  cd api && uvicorn main:app --reload --port 8000

Terminal 2 — Dashboard:
  cd dashboard && npm run dev

Terminal 3 — Pipeline:
  cd src && python pipeline.py
```

### 6.3 Configuracao (.env)

```env
# Ambiente
BOVISION_ENV=dev

# Caminhos
SAMPLE_IMAGES_DIR=data/sample_images
MODEL_DETECTION_PATH=models/detection/best.pt
MODEL_REGRESSION_PATH=models/regression/weight_model.pkl
SCALER_PATH=models/regression/scaler.pkl
CALIBRATION_PATH=calibration/camera_params.npy

# API
API_URL=http://localhost:8000
API_HOST=0.0.0.0
API_PORT=8000
DATABASE_URL=sqlite:///./bovision.db

# Pipeline
CONFIDENCE_THRESHOLD=0.5
CONFIRMATION_FRAMES=3
COOLDOWN_SECONDS=30

# Dashboard
VITE_API_URL=http://localhost:8000/api
```

### 6.4 Dependencias Python (requirements.txt)

```
# Deteccao
ultralytics>=8.2.0
torch>=2.1.0
opencv-python>=4.8.0

# Regressao
xgboost>=2.0.0
scikit-learn>=1.3.0
pandas>=2.1.0

# API
fastapi>=0.111.0
uvicorn>=0.30.0
sqlalchemy>=2.0.0
alembic>=1.13.0
pydantic>=2.0.0

# Utilitarios
python-dotenv>=1.0.0
numpy>=1.24.0
requests>=2.31.0
```

---

## 7. Estrutura Final de Pastas

```
bovision/
├── .env.example
├── .env                         (gitignored)
├── .gitignore
├── requirements.txt
├── data/
│   ├── raw/                     fotos originais Kaggle
│   ├── annotated/               dataset formatado YOLO
│   │   ├── images/{train,val,test}/
│   │   ├── labels/{train,val,test}/
│   │   └── data.yaml
│   ├── weighings/
│   │   └── dataset.csv          medidas + peso para XGBoost
│   ├── sample_images/           imagens para mock do pipeline
│   └── evidence/                fotos de evidencia do pipeline
├── models/
│   ├── detection/
│   │   ├── best.pt
│   │   └── best.onnx
│   └── regression/
│       ├── weight_model.pkl
│       └── scaler.pkl
├── src/
│   ├── capture.py
│   ├── detect.py
│   ├── measure.py
│   ├── predict.py
│   └── pipeline.py
├── api/
│   ├── main.py
│   ├── database.py
│   ├── models.py
│   ├── schemas.py
│   └── routers/
│       ├── weighings.py
│       ├── animals.py
│       └── reports.py
├── dashboard/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       ├── api.js
│       └── components/
│           ├── StatsCards.jsx
│           ├── WeightChart.jsx
│           ├── WeighingsTable.jsx
│           └── AlertsList.jsx
├── scripts/
│   ├── collect_data.py
│   ├── train_detection.py
│   ├── train_regression.py
│   └── evaluate.py
├── docker/                      vazio no MVP
│   ├── Dockerfile.api
│   └── docker-compose.yml
└── calibration/
    └── camera_params.npy        vazio no MVP
```

---

## 8. Diferencas MVP vs README

| Item | README | MVP |
|---|---|---|
| api/auth.py | JWT multi-fazenda | Removido |
| api/routers/farms.py | CRUD fazendas | Removido |
| api/models.py + schemas.py | Nao existiam | Adicionados |
| data/sample_images/ | Nao existia | Adicionado para mock |
| Dockerfile.edge | Container Jetson | Removido |
| MinIO | Armazenamento de fotos | Fotos em disco local |
| MQTT | Broker Mosquitto | HTTP direto |
| finetune_farm.py | Fine-tune por fazenda | Removido |
| Dashboard | 5+ telas com auth | Tela unica sem auth |

---

## 9. Fases de Implementacao

```
Fase 1 — Dados
  Baixar datasets Kaggle, organizar em data/

Fase 2 — Modelo de Deteccao
  Treinar YOLOv8n-seg, avaliar, exportar

Fase 3 — Modelo de Regressao
  Treinar XGBoost, avaliar, salvar

Fase 4 — Modulos do Pipeline
  Implementar capture, detect, measure, predict (com mocks)

Fase 5 — API + Banco
  FastAPI + SQLite + endpoints + migracoes Alembic

Fase 6 — Pipeline Orquestrador
  pipeline.py conectando tudo + envio HTTP para API

Fase 7 — Dashboard
  React + Vite + Tailwind + componentes + conexao com API

Fase 8 — Integracao e Teste
  Rodar tudo junto, validar o fluxo end-to-end
```
