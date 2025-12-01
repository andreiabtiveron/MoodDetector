# treino.py — versão otimizada com CNN melhor, class weights e callbacks

from preprocesso import load_datasets
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def build_model():

    model = models.Sequential([

        layers.Input(shape=(48, 48, 1)),

        # bloco 1
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        # bloco 2
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        # blc 3
        layers.Conv2D(256, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Flatten(),

        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),

        layers.Dense(7, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print(model.summary())
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
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3)
    ]

    # Cria modelo
    model = build_model()

    # treino
    print("\nTreinando CNN\n")
    history = model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=30,
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

    plt.savefig("training_curves.png")
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
    plt.savefig("confusion_matrix.png")
    plt.close()

    print("\nRelatório de Classificação:\n")
    print(classification_report(y_true, y_pred))

    # alva modelo
    print("Salvando modelo final em 'emotion_cnn_melhor.h5'")
    model.save("emotion_cnn_melhor.h5")

    print("\nTreinamento concluído\n")


if __name__ == "__main__":
    train_model()

