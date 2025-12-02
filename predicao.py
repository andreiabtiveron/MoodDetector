import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Diretório de saída (compatível com Docker e execução local)
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")


def predict_image(img_path):
    print(f"Lendo imagem: {img_path}")

    # carrega imagem
    img = image.load_img(img_path, target_size=(48, 48), color_mode="grayscale")

    # converte pra array
    img = np.array(img).reshape(1, 48, 48, 1) / 255.0

    # carrega modelo
    model_path = os.path.join(OUTPUT_DIR, "emotion_cnn_melhor.h5")
    print(f"Carregando modelo: {model_path}")
    model = load_model(model_path)

    #prediz  
    pred = model.predict(img)
    emotion_id = pred.argmax()

    print(f"Emoção prevista: {EMOTIONS[emotion_id]}")
    return EMOTIONS[emotion_id]


if __name__ == "__main__":
    predict_image("foto_teste.png")

