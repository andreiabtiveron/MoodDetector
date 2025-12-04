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

# FORCE GPU USAGE
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"GPU FOUND: {physical_devices}")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # Use mixed precision for 2x speed boost
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
else:
    print("WARNING: NO GPU DETECTED!")

# Diretório de saída
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def conv_block(filters, dropout_rate=0.25):
    return [
        layers.Conv2D(filters, (3,3), padding="same", activation="relu", 
                     kernel_initializer="he_normal",
                     kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(dropout_rate)
    ]

def build_model():
    model = models.Sequential([
        layers.Input(shape=(48, 48, 1)),
        
        # Simplified architecture - fewer layers = faster training
        *conv_block(64, dropout_rate=0.20),
        *conv_block(128, dropout_rate=0.25),
        *conv_block(256, dropout_rate=0.30),
        
        layers.GlobalAveragePooling2D(),
        
        layers.Dense(256, activation="relu", 
                    kernel_initializer="he_normal",
                    kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        # Output layer must be float32 for mixed precision
        layers.Dense(7, activation="softmax", dtype='float32')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Higher LR for faster convergence
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"]
    )
    
    model.summary()
    return model

def train_model():
    # Load datasets
    train_gen, test_gen = load_datasets()
    
    # Calculate class weights
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weights = dict(enumerate(class_weights))
    print("Class weights:", class_weights)
    
    # Aggressive callbacks for fast training
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
        ModelCheckpoint(os.path.join(OUTPUT_DIR, 'best_model.h5'), 
                       monitor='val_accuracy', 
                       save_best_only=True, 
                       mode='max',
                       verbose=1),
        ReduceLROnPlateau(monitor='val_loss', 
                         factor=0.5,
                         patience=3,
                         min_lr=1e-7,
                         verbose=1)
    ]
    
    model = build_model()
    
    print("\nTreinando CNN Otimizada com GPU\n")
    history = model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=30,  # Reduced epochs - quality over quantity
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
        workers=4,  # Parallel data loading
        use_multiprocessing=True,
        max_queue_size=20  # Prefetch more batches
    )
    
    loss, acc = model.evaluate(test_gen)
    print(f"Acurácia final: {acc:.4f}")
    
    # Plot training curves
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
    
    # Confusion matrix
    print("\nGerando matriz de confusão...")
    y_pred = model.predict(test_gen, workers=4, use_multiprocessing=True).argmax(axis=1)
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
    
    print(f"Salvando modelo final em '{OUTPUT_DIR}/emotion_cnn_melhor.h5'")
    model.save(os.path.join(OUTPUT_DIR, "emotion_cnn_melhor.h5"))
    
    print("\nTreinamento concluído\n")


if __name__ == "__main__":
    train_model()
