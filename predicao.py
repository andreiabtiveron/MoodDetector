import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def predict_image(img_path):
    print(f"Lendo imagem: {img_path}")

    # carrega imagem
    img = image.load_img(img_path, target_size=(48, 48), color_mode="grayscale")

    # converte pra array
    img = np.array(img).reshape(1, 48, 48, 1) / 255.0

    # carrega modelo
    model = load_model("emotion_cnn.h5")

    #prediz  
    pred = model.predict(img)
    emotion_id = pred.argmax()

    print(f"Emoção prevista: {EMOTIONS[emotion_id]}")
    return EMOTIONS[emotion_id]


if __name__ == "__main__":
    predict_image("foto_teste.png")

