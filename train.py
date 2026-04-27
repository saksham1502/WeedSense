"""
Train CNN binary classifier on the local dataset.
Classes: soybean → 1 (Crop), broadleaf/grass/soil → 0 (Weed/Other)
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

DATASET_PATH = "dataset"
IMG_SIZE     = (128, 128)
BATCH_SIZE   = 32
EPOCHS       = 15

# ── Custom binary labels: soybean=1, everything else=0 ────────────────────────
# We use a custom flow so we can remap the 4 classes to binary.

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

# Keras flow_from_directory with class_mode='binary' needs exactly 2 folders.
# We'll create a simple custom generator using tf.data instead.

import pathlib

def build_dataset(root, subset_frac=0.8, seed=42):
    root = pathlib.Path(root)
    all_files, all_labels = [], []

    for folder in root.iterdir():
        if not folder.is_dir():
            continue
        label = 1 if folder.name == "soybean" else 0
        for f in folder.glob("*.tif"):
            all_files.append(str(f))
            all_labels.append(label)

    # shuffle
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(all_files))
    all_files  = [all_files[i]  for i in idx]
    all_labels = [all_labels[i] for i in idx]

    split = int(len(all_files) * subset_frac)
    return (all_files[:split],  all_labels[:split],
            all_files[split:],  all_labels[split:])


def load_image_py(path, label):
    """Load .tif via Pillow — works with any format TF can't decode."""
    from PIL import Image as PILImage
    img = PILImage.open(path.decode()).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr, np.int64(label)


def load_image(path, label):
    image, label = tf.numpy_function(
        func=load_image_py,
        inp=[path, label],
        Tout=[tf.float32, tf.int64]
    )
    image.set_shape((IMG_SIZE[0], IMG_SIZE[1], 3))
    label.set_shape(())
    return image, label


def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, 0.1)
    return image, label


def make_tf_dataset(files, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((files, labels))
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(1024)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


# ── Build datasets ─────────────────────────────────────────────────────────────
print("Loading file paths…")
tr_f, tr_l, va_f, va_l = build_dataset(DATASET_PATH)
print(f"  Train: {len(tr_f)}  |  Val: {len(va_f)}")
print(f"  Crop (1): {sum(tr_l)}  |  Other (0): {len(tr_l)-sum(tr_l)}")

train_ds = make_tf_dataset(tr_f, tr_l, training=True)
val_ds   = make_tf_dataset(va_f, va_l, training=False)

# ── Model ──────────────────────────────────────────────────────────────────────
model = Sequential([
    Conv2D(32,  (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64,  (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# ── Callbacks ──────────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(patience=4, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(patience=2, factor=0.5, verbose=1),
    ModelCheckpoint("weed_model.keras", save_best_only=True, verbose=1)
]

# ── Train ──────────────────────────────────────────────────────────────────────
print("\nStarting training…")
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=callbacks
)

print("\nDone! Model saved as weed_model.keras")

# ── Quick eval ─────────────────────────────────────────────────────────────────
loss, acc = model.evaluate(val_ds, verbose=0)
print(f"Val accuracy: {acc*100:.2f}%  |  Val loss: {loss:.4f}")
