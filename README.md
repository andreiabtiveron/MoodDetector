# MoodDetector 🎭

CNN para detecção de emoções faciais usando o dataset FER2013.

> **Nota:** Rode o `melhortreino.py` primeiro, depois o `predicao.py`

## 📋 Estrutura do Projeto

```text
MoodDetector/
│
├── fer2013/               ← dataset precisa ser baixado (veja instruções abaixo)
│   ├── train/
│   │   ├── angry/
│   │   ├── disgust/
│   │   ├── fear/
│   │   ├── happy/
│   │   ├── neutral/
│   │   ├── sad/
│   │   └── surprise/
│   └── test/
│       ├── angry/
│       ├── disgust/
│       ├── fear/
│       ├── happy/
│       ├── neutral/
│       ├── sad/
│       └── surprise/
│
├── outputs/              ← gerado automaticamente (modelos e gráficos)
│
├── preprocesso.py        ← carrega e processa datasets
├── melhortreino.py       ← treina a CNN
├── predicao.py           ← prediz emoção de uma imagem
├── utils.py              ← funções auxiliares
│
├── Dockerfile            ← imagem Docker otimizada
├── docker-compose.yml    ← orquestração
├── requirements.txt      ← dependências Python
├── .dockerignore
│
├── foto_teste.png        ← imagem exemplo para predição
│
└── README.md
```

---

## 📦 Como Baixar o Dataset FER2013

O dataset pode ser baixado diretamente no Kaggle:

🔗 **https://www.kaggle.com/datasets/msambare/fer2013**

### Passo a passo:

1. Acesse o link acima
2. Clique em **Download**
3. Você receberá um arquivo chamado `fer2013.zip`
4. Extraia o `.zip`

Após extrair, você terá:

```text
train/
test/
```

5. **Mova essas duas pastas para dentro do diretório `fer2013/` no seu projeto**

### Estrutura final esperada:

```text
fer2013/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
│
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```


```text
fer2013/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
│
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

---

## 🐳 Execução com Docker (Recomendado)

### Pré-requisitos
- Docker e Docker Compose instalados
- Dataset FER2013 baixado e colocado em `./fer2013/`

### Build da imagem (apenas uma vez)

```bash
docker compose build
```

> **Otimizações do build:**
> - Multi-stage build reduz tamanho final da imagem
> - Cache de layers do pip para rebuilds rápidos
> - Base TensorFlow oficial otimizada

### Treinar o modelo

```bash
docker compose up train
```

**Saídas geradas em `./outputs/`:**
- `emotion_cnn_melhor.h5` - Modelo treinado final
- `best_model.h5` - Melhor checkpoint durante treino
- `training_curves.png` - Gráficos de acurácia e loss
- `confusion_matrix.png` - Matriz de confusão

### Fazer predição em uma imagem

Certifique-se de que:
- O modelo já foi treinado (`outputs/emotion_cnn_melhor.h5` existe)
- Existe uma imagem `foto_teste.png` no diretório raiz

```bash
docker compose run --rm predict
```

Para prever outra imagem:

```bash
docker compose run --rm predict python -c "from predicao import predict_image; predict_image('caminho/imagem.png')"
```

### Recursos e Performance

O `docker-compose.yml` está configurado com:
- **Train**: até 4 CPUs, 8GB RAM (mínimo 2 CPUs, 4GB)
- **Predict**: até 2 CPUs, 2GB RAM

Ajuste conforme sua máquina editando `deploy.resources` no `docker-compose.yml`.

---

## 🐍 Execução Local (sem Docker)

### 1. Criar ambiente virtual

```bash
python -m venv .venv
source .venv/bin/activate.fish  # fish shell
# ou: source .venv/bin/activate  # bash/zsh
```

### 2. Instalar dependências

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Treinar o modelo

```bash
python melhortreino.py
```

**Arquivos gerados:**
- `emotion_cnn_melhor.h5`
- `best_model.h5`
- `training_curves.png`
- `confusion_matrix.png`

### 4. Fazer predição

```bash
python predicao.py
```

Ou para outra imagem:

```bash
python -c 'from predicao import predict_image; predict_image("sua_imagem.png")'
```

---

## 🏗️ Arquitetura da CNN

```text
Input (48x48x1 grayscale)
    ↓
Conv2D(64) → BatchNorm → Conv2D(64) → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Conv2D(128) → BatchNorm → Conv2D(128) → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Conv2D(256) → BatchNorm → Conv2D(256) → BatchNorm → MaxPool → Dropout(0.30)
    ↓
Flatten
    ↓
Dense(256) → BatchNorm → Dropout(0.5)
    ↓
Dense(7, softmax) → [angry, disgust, fear, happy, neutral, sad, surprise]
```

**Técnicas aplicadas:**
- Data augmentation (rotação, shift, zoom, flip)
- Class weights balanceados
- Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
- Batch Normalization
- He initialization

---

## 📊 Classes de Emoções

| ID | Emoção    |
|----|-----------|
| 0  | angry     |
| 1  | disgust   |
| 2  | fear      |
| 3  | happy     |
| 4  | sad       |
| 5  | surprise  |
| 6  | neutral   |

> ⚠️ **Importante:** Verifique o mapeamento impresso por `load_datasets()` durante o treino (`Classes detectadas: {...}`). Se a ordem diferir da lista `EMOTIONS` em `predicao.py`, ajuste-a.

---

## 🚀 Dicas de Performance

### Com GPU (NVIDIA)
Para usar GPU com Docker, instale o [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) e adicione no `docker-compose.yml`:

```yaml
services:
  train:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Build mais rápido
- Use `DOCKER_BUILDKIT=1 docker compose build` para builds paralelos
- O `.dockerignore` exclui arquivos desnecessários do contexto

---

## 📝 Dependências

- **TensorFlow** 2.15.0 (inclui Keras)
- **NumPy** 1.26.2
- **Pillow** 10.1.0 (manipulação de imagens)
- **scikit-learn** 1.3.2 (métricas e class weights)
- **matplotlib** 3.8.2 (visualizações)
- **seaborn** 0.13.0 (matriz de confusão)
- **h5py** 3.10.0 (salvar modelos)

Veja versões fixas em `requirements.txt`.

---

## 🤝 Contribuindo

1. Certifique-se de que o dataset está estruturado corretamente
2. Use Docker para ambiente reproduzível
3. Verifique os gráficos gerados em `outputs/` após o treino
4. Ajuste hiperparâmetros em `melhortreino.py` conforme necessário

---

## 📄 Licença

Este projeto usa o dataset [FER2013](https://www.kaggle.com/datasets/msambare/fer2013) disponível no Kaggle.
