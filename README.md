# MoodDetector
Roda o melhortreino.py, depois o predicao.py

## Estrutura do Projeto

```text
FER2013-Emotion-Recognition/
│
├── fer2013/              ← dataset precisa ser baixado (segue instruções abaixo)
│   ├── train/
│   └── test/
│
├── preprocesso.py
├── treino.py
├── predicao.py
├── utils.py
│
├── foto_teste.png
│
└── README.md
```

##  Como Baixar o Dataset FER2013

O dataset pode ser baixado diretamente no Kaggle:

🔗 https://www.kaggle.com/datasets/msambare/fer2013

### Passo a passo:

1. Acesse o link acima.
2. Clique em **Download**.
3. Você receberá um arquivo chamado `fer2013.zip`.
4. Extraia o `.zip`.

Após extrair, você terá:
```text 
train/
test/
```

5. Mova essas duas pastas para dentro do seu projeto, ficando assim:


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
