"""
cnn_model.py
Custom CNN trained on mel spectrogram images for music genre classification.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix

from config import VIS_DIR, MODELS_DIR, SPEC_DIR

IMG_SIZE   = (128, 128)
BATCH_SIZE = 32
EPOCHS     = 40


def run_cnn(genres):
    """
    Build, train, and evaluate a custom CNN on spectrogram images.
    Returns (test_accuracy, model) or (None, None) if TensorFlow is unavailable.
    """
    print("\n" + "=" * 70)
    print("STEP 7: Custom CNN on spectrogram images")
    print("=" * 70)

    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    except ImportError:
        print("  TensorFlow not available — skipping CNN.")
        return None, None

    train_dir = os.path.join(SPEC_DIR, 'train')
    test_dir  = os.path.join(SPEC_DIR, 'test')

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=5,
        width_shift_range=0.1,
        height_shift_range=0.1,
        validation_split=0.2,
    )
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', subset='training', shuffle=True,
    )
    val_gen = train_datagen.flow_from_directory(
        train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', subset='validation', shuffle=True,
    )
    test_gen = test_datagen.flow_from_directory(
        test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', shuffle=False,
    )

    num_classes = len(train_gen.class_indices)
    print(f"  Classes : {train_gen.class_indices}")
    print(f"  Train   : {train_gen.samples}  |  Val: {val_gen.samples}  |  Test: {test_gen.samples}")

    model = Sequential([
        layers.Conv2D(32,  (3, 3), activation='relu', padding='same', input_shape=(*IMG_SIZE, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32,  (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64,  (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64,  (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax'),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6),
        ModelCheckpoint(os.path.join(MODELS_DIR, 'cnn_best.keras'),
                        monitor='val_accuracy', save_best_only=True, mode='max'),
    ]

    print("\n  Training CNN...")
    history = model.fit(
        train_gen, epochs=EPOCHS, validation_data=val_gen,
        callbacks=callbacks, verbose=1,
    )

    model.save(os.path.join(MODELS_DIR, 'cnn_final.keras'))

    _plot_training_curves(history, prefix='cnn')

    test_loss, test_acc = model.evaluate(test_gen)
    print(f"\n  CNN Test Accuracy: {test_acc:.4f}")

    test_gen.reset()
    y_pred = np.argmax(model.predict(test_gen), axis=1)
    y_true = test_gen.classes
    class_names = list(test_gen.class_indices.keys())

    print(classification_report(y_true, y_pred, target_names=class_names))
    _plot_confusion_matrix(y_true, y_pred, class_names, test_acc,
                           title='Confusion matrix — Custom CNN',
                           cmap='Purples', filename='cnn_confusion_matrix.png')

    return test_acc, model


# ── Shared plotting helpers ───────────────────────────────────────────────────

def _plot_training_curves(history, prefix):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history['accuracy'],     label='Train',      linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    axes[0].set_title(f'{prefix.upper()} — Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['loss'],     label='Train',      linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[1].set_title(f'{prefix.upper()} — Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, f'{prefix}_training_history.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  saved: {prefix}_training_history.png")


def _plot_confusion_matrix(y_true, y_pred, class_names, acc, title, cmap, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True label')
    plt.title(f'{title} (Acc={acc:.3f})')
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, filename), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  saved: {filename}")
