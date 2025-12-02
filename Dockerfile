FROM tensorflow/tensorflow:2.15.0 AS builder

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

FROM tensorflow/tensorflow:2.15.0

LABEL maintainer="MoodDetector Team"
LABEL description="CNN para detecção de emoções faciais - FER2013"

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

COPY preprocesso.py .
COPY melhortreino.py .
COPY predicao.py .
COPY utils.py .

RUN mkdir -p fer2013 outputs

ENV TF_CPP_MIN_LOG_LEVEL=2 \
    TF_ENABLE_ONEDNN_OPTS=1 \
    PYTHONUNBUFFERED=1

VOLUME ["/app/fer2013", "/app/outputs"]

CMD ["python", "melhortreino.py"]
