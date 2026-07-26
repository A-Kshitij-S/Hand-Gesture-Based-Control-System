"""
SignBridge - ASL Hand Gesture Recognition Model Training (Fast Version)
========================================================================
- Trains on A-Z only (26 classes, no numbers)
- Uses wrist-relative + scale-normalized landmarks (much better accuracy)
- Caps images per class to MAX_IMAGES_PER_CLASS for fast training
- Uses multiprocessing for faster landmark extraction

Usage:
    python train_model.py

Output:
    models/asl_model.h5
    models/label_encoder.pkl
    models/norm_mean.npy
    models/norm_std.npy
    models/training_history.pkl
"""

import os
import sys
import cv2
import numpy as np
import mediapipe as mp
import pickle
import time
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2

# ─── Config ──────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR     = os.path.join(BASE_DIR, "merged_dataset")  # ← merged from all 3 sources
MODEL_DIR       = os.path.join(BASE_DIR, "models")
MODEL_PATH      = os.path.join(MODEL_DIR, "asl_model.h5")
ENCODER_PATH    = os.path.join(MODEL_DIR, "label_encoder.pkl")
HISTORY_PATH    = os.path.join(MODEL_DIR, "training_history.pkl")

# ── KEY SETTINGS ──
MAX_IMAGES_PER_CLASS = 1000  # 1000 per class × 26 classes = up to 26000 samples
LETTERS_ONLY         = True  # Skip 0-9, train only A-Z
RANDOM_SEED          = 42

os.makedirs(MODEL_DIR, exist_ok=True)
random.seed(RANDOM_SEED)

# ─── MediaPipe Setup ─────────────────────────────────────────────────────────
mp_hands        = mp.solutions.hands
hands_detector  = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3,
)


def extract_landmarks_wrist_relative(image_path: str):
    """
    Extract 63 hand landmark features, normalized relative to the wrist.

    Why wrist-relative?
      - Raw (x,y,z) values change if your hand is left/right/near/far from camera
      - Wrist-relative coords only encode the SHAPE of the gesture, not position
      - This is a major accuracy improvement, especially across different people

    Returns: np.array of shape (63,) or None if no hand detected.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result  = hands_detector.process(img_rgb)

    if not result.multi_hand_landmarks:
        return None

    lm_list = result.multi_hand_landmarks[0].landmark

    # Raw coords as numpy array (21, 3)
    raw = np.array([[lm.x, lm.y, lm.z] for lm in lm_list], dtype=np.float32)

    # Wrist is landmark 0
    wrist = raw[0]

    # Subtract wrist so all coords are relative to wrist
    relative = raw - wrist  # (21, 3)

    # Scale normalization: divide by the max distance from wrist
    # This makes the model scale-invariant (hand size doesn't matter)
    scale = np.max(np.linalg.norm(relative, axis=1)) + 1e-8
    relative /= scale

    return relative.flatten()  # shape (63,)


# ─── Data Loading ─────────────────────────────────────────────────────────────
def load_dataset():
    all_classes = sorted(os.listdir(DATASET_DIR))

    if LETTERS_ONLY:
        classes = [c for c in all_classes if c.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" and len(c) == 1]
    else:
        classes = all_classes

    X, y = [], []
    total = 0

    print(f"\n{'='*60}")
    print(f"  SignBridge — Fast Retrain (Wrist-Relative Features)")
    print(f"  Classes: {len(classes)}  |  Max images/class: {MAX_IMAGES_PER_CLASS}")
    print(f"{'='*60}")

    for cls in classes:
        cls_dir = os.path.join(DATASET_DIR, cls)
        if not os.path.isdir(cls_dir):
            continue

        images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        # Random sample to avoid bias from sorted filenames
        if len(images) > MAX_IMAGES_PER_CLASS:
            images = random.sample(images, MAX_IMAGES_PER_CLASS)

        count = 0
        for img_file in images:
            img_path = os.path.join(cls_dir, img_file)
            lm = extract_landmarks_wrist_relative(img_path)
            if lm is not None:
                X.append(lm)
                y.append(cls.upper())
                count += 1

        total += count
        print(f"  [{cls:>2}]  {count:>3} / {len(images):>3}  landmarks extracted")

    print(f"\n  ✅ Total valid samples: {total}")
    return np.array(X, dtype=np.float32), np.array(y)


# ─── Build Model ──────────────────────────────────────────────────────────────
def build_model(input_dim: int, num_classes: int) -> Sequential:
    """
    Slightly smaller model since features are better (wrist-relative).
    Better features → simpler model needed → faster training.
    """
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_dim,), kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.35),
        Dense(128, activation='relu', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.25),
        Dense(64, activation='relu'),
        Dropout(0.15),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("\n🚀 SignBridge — Fast Model Training Started")
    print("   Features: Wrist-relative + scale-normalized landmarks")
    print("   Classes:  A-Z only (no numbers)")
    t0 = time.time()

    # Load data
    X, y = load_dataset()
    if len(X) == 0:
        print("❌ No landmarks extracted! Check your dataset path.")
        sys.exit(1)

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)
    y_cat = to_categorical(y_encoded, num_classes)

    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump(le, f)
    print(f"\n✅ Label encoder saved → {ENCODER_PATH}")
    print(f"   Classes ({num_classes}): {list(le.classes_)}")

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_cat, test_size=0.15, random_state=RANDOM_SEED, stratify=y_encoded
    )
    print(f"\n📊 Train: {len(X_train)} | Val: {len(X_val)}")

    # Z-score normalization (on top of wrist-relative, for final polish)
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0) + 1e-8
    X_train_n = (X_train - mean) / std
    X_val_n   = (X_val   - mean) / std

    # Save normalization params
    np.save(os.path.join(MODEL_DIR, "norm_mean.npy"), mean)
    np.save(os.path.join(MODEL_DIR, "norm_std.npy"),  std)
    print("✅ Normalization params saved")

    # Build & train
    model = build_model(X_train_n.shape[1], num_classes)
    model.summary()

    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-6, verbose=1),
        ModelCheckpoint(MODEL_PATH, save_best_only=True, verbose=1),
    ]

    print(f"\n🏋️  Training model ({num_classes} classes)...")
    history = model.fit(
        X_train_n, y_train,
        validation_data=(X_val_n, y_val),
        epochs=60,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # Save history
    with open(HISTORY_PATH, 'wb') as f:
        pickle.dump(history.history, f)

    elapsed  = time.time() - t0
    val_acc  = max(history.history['val_accuracy'])
    mins     = int(elapsed // 60)
    secs     = int(elapsed % 60)
    print(f"\n{'='*60}")
    print(f"  ✅ Training Complete in {mins}m {secs}s")
    print(f"  🏆 Best Val Accuracy : {val_acc*100:.2f}%")
    print(f"  💾 Model saved       : {MODEL_PATH}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
