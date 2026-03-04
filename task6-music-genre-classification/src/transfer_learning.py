"""
transfer_learning.py
Transfer learning with a frozen VGG16 base for music genre classification
on mel spectrogram images.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix

from config import VIS_DIR, MODELS_DIR, SPEC_DIR
from cnn_model import _plot_training_curves, _plot_confusion_matrix

IMG_SIZE   = (128, 128)
BATCH_SIZE = 32
EPOCHS     = 30


def run_transfer_learning(genres):
    """
    Fine-tune a VGG16-based model (frozen conv layers + custom head)
    on mel spectrogram images. Returns test accuracy or None.
    """
    print("\n" + "=" * 70)
    print("STEP 8: Transfer Learning with VGG16")
    print("=" * 70)

    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        from tensorflow.keras.applications import VGG16
        from tensorflow.keras.models import Model
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    except ImportError:
        print("  TensorFlow not available — skipping transfer learning.")
        return None

    train_dir = os.path.join(SPEC_DIR, 'train')
    test_dir  = os.path.join(SPEC_DIR, 'test')

    train_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
    test_datagen  = ImageDataGenerator(rescale=1. / 255)

    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', subset='training',
    )
    val_gen = train_datagen.flow_from_directory(
        train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', subset='validation',
    )
    test_gen = test_datagen.flow_from_directory(
        test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', shuffle=False,
    )

    num_classes = len(train_gen.class_indices)

    # Frozen VGG16 base with a custom classification head
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    preds = layers.Dense(num_classes, activation='softmax')(x)

    transfer_model = Model(inputs=base_model.input, outputs=preds)
    transfer_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6),
        ModelCheckpoint(os.path.join(MODELS_DIR, 'transfer_best.keras'),
                        monitor='val_accuracy', save_best_only=True, mode='max'),
    ]

    print("\n  Training VGG16 Transfer Learning model...")
    history = transfer_model.fit(
        train_gen, epochs=EPOCHS, validation_data=val_gen,
        callbacks=callbacks, verbose=1,
    )

    transfer_model.save(os.path.join(MODELS_DIR, 'transfer_final.keras'))

    _plot_training_curves(history, prefix='transfer')

    test_loss, test_acc = transfer_model.evaluate(test_gen)
    print(f"\n  VGG16 Transfer Test Accuracy: {test_acc:.4f}")

    test_gen.reset()
    y_pred = np.argmax(transfer_model.predict(test_gen), axis=1)
    y_true = test_gen.classes
    class_names = list(test_gen.class_indices.keys())

    print(classification_report(y_true, y_pred, target_names=class_names))
    _plot_confusion_matrix(y_true, y_pred, class_names, test_acc,
                           title='Confusion matrix — VGG16 Transfer',
                           cmap='Greens', filename='transfer_confusion_matrix.png')

    return test_acc
