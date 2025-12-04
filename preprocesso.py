# carrega as pastas fer2013/train e fer2013/test usando ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_datasets(
    base_path="fer2013",
    img_size=(48, 48),
    batch_size=64
):

    #carrega dataset dividido em /train e /test
    # aí normaliza as imagens(rescale=1/255)

    train_dir = f"{base_path}/train"
    test_dir = f"{base_path}/test"

    # data augmentation (mais agressivo para melhor generalização)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,          # aumentado de 10 para 20
        width_shift_range=0.25,     # aumentado de 0.2 para 0.25
        height_shift_range=0.25,    # aumentado de 0.2 para 0.25
        shear_range=0.2,
        zoom_range=0.25,            # aumentado de 0.2 para 0.25
        horizontal_flip=True,
        brightness_range=[0.8, 1.2], # novo: variação de brilho
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator(
            rescale=1./255
    )

    # carrega pastas
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=False
    )

    print("Classes detectadas:", train_generator.class_indices)
    return train_generator, test_generator

