import os
import tensorflow as tf

from model import build_model
from data_pipeline import (
    get_train_dataset,
    get_validation_dataset
)


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 8

INITIAL_LEARNING_RATE = 0.001

EPOCHS = 1

CHECKPOINT_DIR = "outputs/checkpoints"
MODEL_PATH = "outputs/efficientnet_b0_best.keras"


# ============================================================
# GPU CHECK
# ============================================================

print("=" * 60)
print("EFFICIENTNET-B0 TRAINING")
print("=" * 60)

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print("\nGPU detected:")
    for gpu in gpus:
        print(" ", gpu)
else:
    print("\nNo GPU detected.")
    print("Training will use CPU.")


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)

os.makedirs(
    "outputs",
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading datasets...")

train_dataset = get_train_dataset(
    shuffle=True
)

validation_dataset = get_validation_dataset()

print("✓ Training dataset loaded.")
print("✓ Validation dataset loaded.")


# ============================================================
# BUILD MODEL
# ============================================================

print("\nBuilding EfficientNet-B0...")

model = build_model()

print("✓ Model created.")


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=INITIAL_LEARNING_RATE
    ),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[
        tf.keras.metrics.MeanAbsoluteError(
            name="mae"
        ),
        tf.keras.metrics.RootMeanSquaredError(
            name="rmse"
        )
    ]
)

print("✓ Model compiled.")


# ============================================================
# CALLBACKS
# ============================================================

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=MODEL_PATH,
    monitor="val_mae",
    mode="min",
    save_best_only=True,
    verbose=1
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_mae",
    mode="min",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_mae",
    mode="min",
    factor=0.5,
    patience=2,
    min_lr=1e-7,
    verbose=1
)


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_model_path = "outputs/efficientnet_b0_final.keras"

model.save(
    final_model_path
)

print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print("\nBest model:")
print(MODEL_PATH)

print("\nFinal model:")
print(final_model_path)