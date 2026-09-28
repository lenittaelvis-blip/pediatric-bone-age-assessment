import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, Model

from data_pipeline import (
    get_train_dataset,
    get_validation_dataset,
    get_test_dataset
)


INPUT_SHAPE = (512, 512, 3)
BATCH_SIZE = 8


def build_model():

    # EfficientNet-B0 backbone
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=INPUT_SHAPE
    )

    # Freeze backbone for initial transfer learning
    base_model.trainable = False

    # Model input
    inputs = layers.Input(
        shape=INPUT_SHAPE
    )

    # Feature extraction
    x = base_model(
        inputs,
        training=False
    )

    # Global feature vector
    x = layers.GlobalAveragePooling2D()(x)

    # Regularization
    x = layers.Dropout(0.3)(x)

    # Bone-age regression output
    outputs = layers.Dense(
        1,
        name="bone_age"
    )(x)

    model = Model(
        inputs,
        outputs
    )

    return model


if __name__ == "__main__":

    # ========================================================
    # BUILD MODEL
    # ========================================================

    model = build_model()

    # ========================================================
    # COMPILE MODEL
    # ========================================================

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="mse",
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(
                name="mae"
            ),
            tf.keras.metrics.RootMeanSquaredError(
                name="rmse"
            )
        ]
    )

    print("\nModel compiled successfully.")

    print("Loss   : MSE")
    print("Metric : MAE")
    print("Metric : RMSE")

    # ========================================================
    # LOAD VERIFIED DATA PIPELINE
    # ========================================================

    print("\nLoading training dataset...")

    train_dataset = get_train_dataset(
        shuffle=False
    )

    # ========================================================
    # TAKE ONE BATCH
    # ========================================================

    images, labels = next(
        iter(train_dataset)
    )

    print("\n--- ONE BATCH TEST ---")

    print(
        "Images shape :",
        images.shape
    )

    print(
        "Labels shape :",
        labels.shape
    )

    # ========================================================
    # FORWARD PASS
    # ========================================================

    predictions = model(
        images,
        training=False
    )

    print(
        "\nPredictions shape :",
        predictions.shape
    )

    print(
        "\nFirst 5 predictions:"
    )

    print(
        predictions[:5]
        .numpy()
        .flatten()
    )

    print(
        "\nFirst 5 actual bone ages:"
    )

    print(
        labels[:5]
        .numpy()
    )

    # ========================================================
    # CALCULATE LOSS / METRICS
    # ========================================================

        # ========================================================
    # CALCULATE LOSS / METRICS
    # ========================================================

    actual = labels
    predicted = tf.squeeze(
        predictions,
        axis=-1
    )

    mse_fn = tf.keras.losses.MeanSquaredError()

    mse = mse_fn(
        actual,
        predicted
    )

    mae = tf.reduce_mean(
        tf.abs(
            actual - predicted
        )
    )

    rmse = tf.sqrt(mse)

    print("\n--- INITIAL METRICS ---")

    print(
        "MSE  :",
        mse.numpy()
    )

    print(
        "MAE  :",
        mae.numpy()
    )

    print(
        "RMSE :",
        rmse.numpy()
    )

    print(
        "\n✓ Model successfully processed "
        "one real training batch."
    )