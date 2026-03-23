# BoviSight — Arquitetura do Sistema: Cada Pasta e Arquivo

> O que é, para que serve e onde roda cada parte do sistema.

---

## Visão Geral

O sistema é dividido em 3 lugares físicos diferentes. Antes de entrar nas pastas, é importante saber onde cada coisa vive:

| Onde | O que roda |
|---|---|
| **Seu computador** | Treino dos modelos, scripts de preparação |
| **Jetson (campo)** | Câmeras, detecção, medição, estimativa de peso |
| **Nuvem (servidor)** | API, banco de dados, dashboard, armazenamento de fotos |

---

## Estrutura Completa

```
bovision/
├── data/
│   ├── raw/
│   ├── annotated/
│   │   ├── images/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   ├── labels/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   └── data.yaml
│   ├── weighings/
│   │   └── dataset.csv
│   └── evidence/
├── models/
│   ├── detection/
│   │   ├── best.pt
│   │   ├── best.onnx
│   │   └── best.engine
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
│   ├── auth.py
│   └── routers/
│       ├── weighings.py
│       ├── animals.py
│       ├── farms.py
│       └── reports.py
├── dashboard/
│   ├── src/
│   │   ├── components/
│   │   └── pages/
│   └── public/
├── scripts/
│   ├── collect_data.py
│   ├── train_detection.py
│   ├── train_regression.py
│   ├── finetune_farm.py
│   └── evaluate.py
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.edge
│   └── docker-compose.yml
├── calibration/
│   └── camera_params.npy
├── requirements.txt
└── .env.example
```

---

## `data/` — Todos os dados do sistema

Esta pasta existe **apenas no seu computador de desenvolvimento**. Nunca vai para a nuvem nem para o Jetson — os arquivos são grandes demais (gigabytes de fotos) e o Jetson não precisa deles para funcionar.

---

### `data/raw/`

**O que é:** Depósito das fotos originais tiradas em campo, exatamente como saíram da câmera. Nada é alterado aqui.

**O que contém:** Milhares de arquivos `.jpg` nomeados pelo ID do animal e ângulo.

**Exemplo de conteúdo:**
```
BOI_001_lateral_esq.jpg
BOI_001_lateral_dir.jpg
BOI_001_frontal.jpg
BOI_001_dorsal.jpg
BOI_002_lateral_esq.jpg
...
```

**Por que existe separado:** Você nunca quer perder a foto original. Todo processamento posterior usa cópias, não os originais.

---

### `data/annotated/`

**O que é:** O dataset já processado e organizado para treinar o YOLOv8. Esta pasta é gerada automaticamente quando você exporta o projeto do Roboflow.

**Por que existe separado do `raw/`:** As fotos aqui já passaram por augmentation (variações artificiais de luz, rotação, etc.) e estão no formato exato que o YOLOv8 espera. São arquivos diferentes das fotos originais.

---

### `data/annotated/images/train/`

**O que é:** As fotos usadas para o YOLOv8 aprender. Corresponde a 80% do total de imagens anotadas.

**Exemplo de conteúdo:**
```
BOI_001_lateral_esq.jpg
BOI_001_lateral_esq_flip.jpg        ← versão espelhada (augmentation)
BOI_001_lateral_esq_bright.jpg      ← versão mais clara (augmentation)
BOI_003_lateral_dir.jpg
...
```

---

### `data/annotated/images/val/`

**O que é:** Fotos usadas para avaliar o modelo **durante** o treino, a cada epoch. O modelo nunca aprende com essas fotos — elas existem para o sistema perceber se está melhorando ou piorando. Corresponde a 10% do total.

**Por que é necessário:** Sem essa separação, o modelo poderia "decorar" as fotos de treino e ter um desempenho falso de 100% — mas falharia com fotos novas. Isso se chama overfitting.

---

### `data/annotated/images/test/`

**O que é:** Fotos guardadas a sete chaves e usadas **uma única vez** — só no final, para a avaliação definitiva do modelo. Nem o treino nem a validação tocam nessas fotos. Corresponde a 10% do total.

**Por que existe:** É o "exame final" do modelo. Simula como ele vai se comportar com fotos que nunca viu na vida — que é exatamente o que acontece em campo.

---

### `data/annotated/labels/train/`, `labels/val/`, `labels/test/`

**O que é:** Os arquivos de anotação — um arquivo `.txt` para cada foto correspondente. Contém as coordenadas do contorno do boi que você desenhou no Roboflow.

**Exemplo de conteúdo de `BOI_001_lateral_esq.txt`:**
```
0 0.312 0.445 0.387 0.421 0.501 0.398 0.634 0.412 0.701 0.489
```
O primeiro número (`0`) é a classe — bovino. Os pares de números seguintes são as coordenadas X,Y do contorno do animal, normalizadas entre 0 e 1 em relação ao tamanho da imagem.

---

### `data/annotated/data.yaml`

**O que é:** O arquivo de configuração do dataset. Diz ao YOLOv8 onde ficam as pastas de imagens, quantas classes existem e o nome de cada classe.

**Exemplo de conteúdo:**
```yaml
path: ./data/annotated
train: images/train
val:   images/val
test:  images/test

nc: 1
names: ['bovino']
```

`nc: 1` significa que o modelo só precisa reconhecer uma coisa: bovino. Quanto menos classes, mais fácil e preciso o treino.

---

### `data/weighings/dataset.csv`

**O que é:** A planilha mais importante do projeto. Cada linha é um animal fotografado com as medidas extraídas pelo sistema **e** o peso real medido na balança convencional logo em seguida. É com esses dados que o XGBoost aprende a estimar peso.

**Exemplo de conteúdo:**
```
id,      comprimento_m, altura_m, largura_m, area_m2, perimetro_m, peso_kg, raca,   sexo
BOI_001, 1.43,          1.31,     0.49,      1.92,    1.84,        487,     nelore, M
BOI_002, 1.28,          1.19,     0.41,      1.61,    1.71,        391,     nelore, F
BOI_003, 1.39,          1.35,     0.52,      1.97,    1.89,        524,     angus,  M
```

**Por que é o arquivo mais valioso:** Sem ele não existe modelo de peso. É o único lugar onde o sistema aprende a relação entre o que vê (medidas) e o que precisa saber (quilogramas). Quanto mais linhas, mais preciso o modelo.

---

### `data/evidence/`

**O que é:** Pasta onde o Jetson salva uma foto de cada pesagem realizada, com o contorno do boi destacado em verde. Serve como prova de que a pesagem aconteceu e foi processada corretamente.

**Exemplo de conteúdo:**
```
1745162127.jpg    ← nome é o timestamp Unix da pesagem
1745162891.jpg
1745163445.jpg
```

**Onde fica:** Tanto no Jetson (temporariamente) quanto na nuvem após sincronização.

---

## `models/` — Os cérebros do sistema

Esta pasta contém os arquivos resultantes do treino. É o que você copia para o Jetson após treinar.

---

### `models/detection/best.pt`

**O que é:** O modelo YOLOv8 treinado, salvo no formato nativo do PyTorch. É o arquivo gerado ao final do treino no Google Colab.

**Onde fica:** Seu computador e Google Colab. **Não vai para o Jetson** — o Jetson usa as versões otimizadas abaixo.

**Tamanho típico:** 50–200 MB dependendo do tamanho do modelo escolhido.

---

### `models/detection/best.onnx`

**O que é:** O mesmo modelo YOLOv8, convertido para o formato universal ONNX. Roda em qualquer hardware sem depender do PyTorch.

**Onde fica:** Seu computador e opcionalmente no Jetson como fallback.

**Quando usar:** Em hardwares que não são Jetson (como Raspberry Pi ou servidores de nuvem).

---

### `models/detection/best.engine`

**O que é:** O modelo YOLOv8 convertido e otimizado especificamente para a GPU do Jetson usando TensorRT. É 4–5x mais rápido que o ONNX no Jetson.

**Onde fica:** Exclusivamente no Jetson. Não funciona em nenhum outro hardware — é gerado pelo próprio Jetson a partir do arquivo ONNX.

**Tamanho típico:** 30–120 MB.

---

### `models/regression/weight_model.pkl`

**O que é:** O modelo XGBoost treinado e serializado — "fotografado" em disco para poder ser carregado mais tarde. Contém tudo que o modelo aprendeu sobre a relação entre medidas morfológicas e peso.

**Onde fica:** Seu computador e **no Jetson** (é pequeno, menos de 5 MB).

**Analogia:** É como salvar um jogo — você treinou o modelo, agora salva o estado para não precisar treinar de novo toda vez.

---

### `models/regression/scaler.pkl`

**O que é:** O normalizador das medidas. Durante o treino, as medidas são ajustadas para uma escala padrão (média 0, desvio padrão 1) para o XGBoost funcionar melhor. O scaler guarda os parâmetros desse ajuste para aplicar a mesma transformação nas medidas novas em campo.

**Por que é necessário:** Se você treinou com medidas normalizadas e usa medidas brutas na hora de prever, o resultado fica errado. O scaler garante que a transformação seja sempre idêntica.

**Onde fica:** Seu computador e **no Jetson**, sempre junto com o `weight_model.pkl`.

---

## `src/` — O código que roda no Jetson

Esta é a pasta que vai literalmente dentro do Jetson. Todo arquivo aqui roda em campo, em tempo real, 24 horas por dia.

---

### `src/capture.py`

**O que é:** O código que conversa com as câmeras RealSense. É responsável por abrir a conexão com as 3 câmeras, sincronizar os frames RGB e de profundidade, aplicar os filtros de suavização do depth e entregar os frames prontos para o restante do sistema.

**Input:** Câmeras físicas conectadas via USB.

**Output:** Par de imagens a cada chamada — foto colorida (RGB) + mapa de profundidade em metros.

---

### `src/detect.py`

**O que é:** O código que carrega o modelo YOLOv8 (`best.engine`) e o usa para analisar cada frame da câmera. Identifica se tem um boi na imagem, onde ele está e desenha a máscara de contorno.

**Input:** Frame RGB colorido da câmera.

**Output:**
```
{
  mask:       imagem binária (branco onde é boi, preto onde é fundo)
  box:        [187, 134, 1043, 598]   ← coordenadas do retângulo em volta do boi
  confidence: 0.94                    ← certeza da detecção (0 a 1)
}
```
Retorna `None` se nenhum boi for detectado no frame.

---

### `src/measure.py`

**O que é:** O código que combina a máscara do boi (onde ele está na imagem) com o mapa de profundidade (a que distância está cada ponto) para calcular as medidas reais em metros. Usa os parâmetros de calibração da câmera para converter pixels em metros com precisão.

**Input:**
- Máscara binária do boi (saída do `detect.py`)
- Frame de profundidade em metros (saída do `capture.py`)

**Output:**
```
{
  comprimento_m:        1.43
  altura_m:             1.31
  largura_m:            0.49
  area_m2:              1.92
  perimetro_toracico_m: 1.84
  dist_camera_m:        3.21
  volume_proxy_m3:      0.916
}
```

---

### `src/predict.py`

**O que é:** O código que carrega o XGBoost (`weight_model.pkl`) e o scaler, recebe as medidas morfológicas e retorna o peso estimado em kg com um índice de confiança.

**Input:** Dicionário de medidas (saída do `measure.py`).

**Output:**
```
{
  weight_kg:  487.4
  confidence: 0.91
}
```

---

### `src/pipeline.py`

**O que é:** O maestro. Chama todos os outros módulos na ordem certa, gerencia a lógica de "quando uma pesagem é válida" (quantos frames confirmar, tempo mínimo entre pesagens, confiança mínima), salva a foto de evidência e envia o resultado para a API na nuvem via MQTT. É o único arquivo que precisa estar rodando para o sistema funcionar — ele chama todo o resto.

**Fluxo interno:**
```
1. capture.py  → pegar frame das câmeras
2. detect.py   → tem boi? onde?
3. (acumular 5 frames confirmando presença)
4. measure.py  → medir o animal
5. predict.py  → estimar peso
6. salvar foto de evidência
7. enviar para nuvem via MQTT
8. aguardar próximo animal
```

---

## `api/` — O servidor na nuvem

Esta pasta roda em um servidor VPS na internet. O Jetson envia os dados para cá. O dashboard consome daqui.

---

### `api/main.py`

**O que é:** O ponto de entrada da API. Configura o servidor FastAPI, registra todas as rotas e define middlewares como CORS (que permite o dashboard acessar a API de outro domínio).

---

### `api/database.py`

**O que é:** A configuração da conexão com o banco de dados PostgreSQL. Define o pool de conexões — quantas conexões simultâneas a API pode abrir com o banco.

---

### `api/auth.py`

**O que é:** O sistema de autenticação. Gera e valida tokens JWT — cada fazenda tem suas credenciais e só consegue ver seus próprios dados. Sem isso, qualquer pessoa com o link da API conseguiria ver os dados de todas as fazendas.

---

### `api/routers/weighings.py`

**O que é:** Todas as rotas relacionadas a pesagens.

**Endpoints que expõe:**
```
POST /weighings                    → receber nova pesagem do Jetson
GET  /weighings/animal/{rfid}      → histórico de um animal
GET  /weighings/farm/{id}/today    → pesagens do dia de uma fazenda
```

---

### `api/routers/animals.py`

**O que é:** Rotas para gerenciar o cadastro de animais.

**Endpoints que expõe:**
```
POST /animals                      → cadastrar novo animal
GET  /animals/{rfid}               → dados de um animal
GET  /animals/farm/{id}            → todos os animais de uma fazenda
```

---

### `api/routers/farms.py`

**O que é:** Rotas para gerenciar fazendas e usuários.

**Endpoints que expõe:**
```
POST /farms                        → cadastrar nova fazenda
GET  /farms/{id}/summary           → resumo geral da fazenda
```

---

### `api/routers/reports.py`

**O que é:** Rotas para relatórios e alertas — as mais importantes para o fazendeiro.

**Endpoints que expõe:**
```
GET /reports/alerts?farm_id=X      → animais com perda de peso > 5%
GET /reports/growth?farm_id=X      → curva de crescimento do lote
GET /reports/export?farm_id=X      → gerar PDF/Excel para frigorífico
```

---

## `dashboard/` — A interface do fazendeiro

Código React que roda no navegador do fazendeiro. Hospedado na nuvem junto com a API.

---

### `dashboard/src/components/`

**O que é:** Componentes reutilizáveis da interface — gráficos, tabelas, cards de alerta, botões. Cada arquivo é um pedaço visual da tela.

**Exemplos de arquivos:**
```
WeightCurve.jsx      → gráfico de evolução de peso de um animal
AlertCard.jsx        → card vermelho mostrando animal com perda de peso
AnimalTable.jsx      → tabela com todos os animais da fazenda
WeighingBadge.jsx    → indicador de última pesagem (há quanto tempo)
```

---

### `dashboard/src/pages/`

**O que é:** As telas completas da aplicação — cada arquivo é uma página inteira.

**Exemplos de arquivos:**
```
Overview.jsx         → tela inicial com resumo da fazenda
AnimalDetail.jsx     → tela de detalhe de um animal com curva de peso
Alerts.jsx           → tela de alertas ativos
Reports.jsx          → tela de relatórios e exportação
Settings.jsx         → configurações da fazenda e câmeras
```

---

## `scripts/` — Ferramentas de preparação

Estes arquivos rodam **apenas no seu computador**, uma vez cada, durante o desenvolvimento. Nunca vão para o Jetson nem para a nuvem.

---

### `scripts/collect_data.py`

**O que é:** Script auxiliar para registrar os dados durante a coleta em campo. Você usa no celular ou laptop enquanto está na fazenda fotografando e pesando os animais. Garante que cada foto seja associada ao peso correto no CSV.

---

### `scripts/train_detection.py`

**O que é:** Script que executa o treino do YOLOv8 no Google Colab. Você sobe este arquivo no Colab, aponta para o dataset anotado e roda. Ao final, baixa o `best.pt` e `best.onnx`.

---

### `scripts/train_regression.py`

**O que é:** Script que treina o XGBoost usando o `dataset.csv`. Roda no seu computador em minutos (não precisa de GPU). Gera o `weight_model.pkl` e o `scaler.pkl`.

---

### `scripts/finetune_farm.py`

**O que é:** Versão especial do treino de regressão para fazer o ajuste fino por fazenda. Recebe o CSV de calibração de uma fazenda específica (30–50 pesagens paralelas) e gera um modelo personalizado para aquele rebanho.

---

### `scripts/evaluate.py`

**O que é:** Script de avaliação que compara as estimativas do sistema com os pesos reais da balança. Gera o relatório de validação com MAE, RMSE, erro percentual e gráficos de dispersão. Você usa isso para aprovar o sistema antes de entregar ao cliente.

**Exemplo de output:**
```
Animais avaliados: 100
MAE:              7.8 kg
Erro percentual:  1.9%
% dentro de ±10kg: 91%
% dentro de ±20kg: 99%
RESULTADO: ✅ APROVADO
```

---

## `docker/` — Empacotamento do sistema

---

### `docker/Dockerfile.api`

**O que é:** Receita de como montar o container da API na nuvem. Lista o sistema operacional base, instala as dependências Python e define como iniciar o servidor. Com este arquivo, qualquer servidor de nuvem consegue rodar a API com um único comando.

---

### `docker/Dockerfile.edge`

**O que é:** Receita de como montar o container que roda no Jetson. Parte de uma imagem base da NVIDIA (que já tem CUDA e TensorRT instalados) e adiciona o código do `src/`.

---

### `docker/docker-compose.yml`

**O que é:** O arquivo que sobe todos os serviços da nuvem de uma vez — API, banco de dados PostgreSQL, MinIO para fotos e broker MQTT. Em vez de configurar cada um separadamente, um único comando sobe tudo junto já conectado.

---

## `calibration/` — Parâmetros das câmeras

---

### `calibration/camera_params.npy`

**O que é:** Arquivo gerado uma única vez por kit de câmeras instalado, durante o processo de calibração com o tabuleiro de xadrez. Contém os parâmetros ópticos da câmera — distância focal, centro da imagem e coeficientes de distorção da lente.

**Onde fica:** No Jetson de cada fazenda. Cada unidade tem o seu próprio arquivo, porque cada câmera é ligeiramente diferente das outras.

**Por que é crítico:** Sem este arquivo, o sistema não consegue converter pixels em metros corretamente. Um erro de calibração de 5% aqui se traduz em erro de 5% no peso final.

---

## `requirements.txt`

**O que é:** Lista de todas as bibliotecas Python que o sistema precisa, com as versões exatas. Quando você instala o projeto em uma máquina nova — seja seu computador, o Jetson ou o servidor — este arquivo garante que todos usem exatamente as mesmas versões, evitando incompatibilidades.

**Exemplo de conteúdo:**
```
ultralytics==8.2.0
torch==2.3.0
opencv-python==4.10.0
pyrealsense2==2.55.1
xgboost==2.0.3
fastapi==0.111.0
```

---

## `.env.example`

**O que é:** Modelo do arquivo de variáveis de ambiente — senhas, chaves de API, endereços de servidor. O arquivo real (`.env`) nunca é enviado para o GitHub porque contém informações sensíveis. Este arquivo de exemplo mostra quais variáveis existem sem revelar os valores reais.

**Exemplo de conteúdo:**
```
DB_PASSWORD=coloque_sua_senha_aqui
JWT_SECRET=chave_de_256_bits_aqui
MQTT_BROKER=mqtt.bovision.com.br
API_URL=https://api.bovision.com.br
MINIO_PASSWORD=coloque_sua_senha_aqui
```

---

## Resumo — O que vai para onde

| Pasta / Arquivo | Seu computador | Jetson (campo) | Nuvem (servidor) |
|---|:---:|:---:|:---:|
| `data/` | ✅ | ❌ | ❌ |
| `models/detection/best.pt` | ✅ | ❌ | ❌ |
| `models/detection/best.onnx` | ✅ | ⚠️ fallback | ❌ |
| `models/detection/best.engine` | ❌ | ✅ | ❌ |
| `models/regression/*.pkl` | ✅ | ✅ | ❌ |
| `src/` | ✅ | ✅ | ❌ |
| `api/` | ✅ | ❌ | ✅ |
| `dashboard/` | ✅ | ❌ | ✅ |
| `scripts/` | ✅ | ❌ | ❌ |
| `docker/` | ✅ | ✅ | ✅ |
| `calibration/` | ✅ | ✅ | ❌ |

---

*BoviSight — Cada arquivo no lugar certo, cada lugar com um propósito.*