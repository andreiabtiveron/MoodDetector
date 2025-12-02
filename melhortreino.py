# treino.py — versão otimizada com CNN melhor, class weights e callbacks

from preprocesso import load_datasets
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Diretório de saída (compatível com Docker e execução local)
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

#bloco convolucional para reduzir dimensionalidade e dropout
def conv_block(filters):
    return [
        layers.Conv2D(filters, (3,3), padding="same", activation="relu", kernel_initializer="he_normal"),
        layers.BatchNormalization(),
        layers.Conv2D(filters, (3,3), padding="same", activation="relu", kernel_initializer="he_normal"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25)
    ]
#construir a arquitetura da cnn
def build_model():

    model = models.Sequential([
        layers.Input(shape=(48, 48, 1)),

        *conv_block(64),
        *conv_block(128),

        layers.Conv2D(256, (3,3), padding="same", activation="relu", kernel_initializer="he_normal"),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3,3), padding="same", activation="relu", kernel_initializer="he_normal"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.30),

        layers.Flatten(),

        layers.Dense(256, activation="relu", kernel_initializer="he_normal"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(7, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()
    return model

# treinamentoooooooo
def train_model():

    # Carrega dataset já com augmentation (treino) e normalização (teste)
    train_gen, test_gen = load_datasets()

    # calcula class weights 
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weights = dict(enumerate(class_weights))
    print("Class weights:", class_weights)

    # callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(os.path.join(OUTPUT_DIR, 'best_model.h5'), monitor='val_loss', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-7)
    ]

    # Cria modelo
    model = build_model()

    # treino
    print("\nTreinando CNN\n")
    history = model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=60,#adicionei mais epocas 
        class_weight=class_weights,
        callbacks=callbacks
    )

    # avaliação
    loss, acc = model.evaluate(test_gen)
    print(f"Acurácia final: {acc:.4f}")

     

    # Curvas de acurácia e loss
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Treino")
    plt.plot(history.history["val_accuracy"], label="Validação")
    plt.title("Acurácia")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Treino")
    plt.plot(history.history["val_loss"], label="Validação")
    plt.title("Loss")
    plt.legend()

    plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"))
    plt.close()

    
    #matriz de confusão 
    
    y_pred = model.predict(test_gen).argmax(axis=1)
    y_true = test_gen.classes

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title("Matriz de Confusão")
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plt.close()

    print("\nRelatório de Classificação:\n")
    print(classification_report(y_true, y_pred))

    # alva modelo
    print(f"Salvando modelo final em '{OUTPUT_DIR}/emotion_cnn_melhor.h5'")
    model.save(os.path.join(OUTPUT_DIR, "emotion_cnn_melhor.h5"))

    print("\nTreinamento concluído\n")


if __name__ == "__main__":
    train_model()

