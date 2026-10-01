
from pathlib import Path

import pandas as pd
import tensorflow as tf


# ============================================================
# PATHS
# ============================================================

TRAIN_CSV = Path(
    "outputs/data_splits/train_split.csv"
)

VALIDATION_CSV = Path(
    "outputs/data_splits/validation_split.csv"
)

TEST_CSV = Path(
    "outputs/data_splits/test_split.csv"
)

# Google Drive dataset
IMAGE_DIR = Path("/content/bone_age_train_images")


# ============================================================
# SETTINGS
# ============================================================

IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
IMAGE_CHANNELS = 3

BATCH_SIZE = 8


# ============================================================
# LOAD SPLIT
# ============================================================

def load_split(csv_path):
    return pd.read_csv(csv_path)


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def load_and_preprocess(image_id):
    """
    Load one RSNA hand X-ray and preprocess it.

    Output:
        512 x 512 x 3 float32 image
        Pixel range remains 0-255 because
        EfficientNet-B0 performs its own rescaling.
    """

    image_path = tf.strings.join(
        [
            tf.constant(str(IMAGE_DIR) + "/"),
            tf.as_string(image_id),
            tf.constant(".png")
        ]
    )

    image = tf.io.read_file(image_path)

    image = tf.image.decode_png(
        image,
        channels=1
    )

    image = tf.image.resize_with_pad(
        image,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        method="lanczos3"
    )

    # Convert grayscale -> RGB
    image = tf.image.grayscale_to_rgb(image)
    image = tf.clip_by_value(image, 0.0, 255.0)

    image = tf.cast(
        image,
        tf.float32
    )

    return image


# ============================================================
# AUGMENTATION
# ============================================================

augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomTranslation(
            height_factor=0.0,
            width_factor=0.10,
            fill_mode="constant",
            fill_value=0
        )
    ],
    name="bone_age_augmentation"
)


# ============================================================
# CREATE DATASET
# ============================================================

def create_dataset(
    csv_path,
    training=False,
    shuffle=False
):

    df = load_split(csv_path)

    image_ids = df["id"].astype("int32").values
    bone_ages = df["boneage"].astype("float32").values

    image_ids = tf.constant(image_ids)
    bone_ages = tf.constant(bone_ages)

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_ids, bone_ages)
    )

    if training and shuffle:
        dataset = dataset.shuffle(
            buffer_size=256,
            reshuffle_each_iteration=True
        )

    def process(image_id, bone_age):

        image = load_and_preprocess(
            image_id
        )

        if training:
            image = augmentation(
                image,
                training=True
            )

        return image, bone_age

    dataset = dataset.map(
        process,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(
        BATCH_SIZE,
        drop_remainder=False
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# ============================================================
# PUBLIC DATASET FUNCTIONS
# ============================================================

def get_train_dataset(shuffle=True):

    return create_dataset(
        TRAIN_CSV,
        training=True,
        shuffle=shuffle
    )


def get_validation_dataset():

    return create_dataset(
        VALIDATION_CSV,
        training=False,
        shuffle=False
    )


def get_test_dataset():

    return create_dataset(
        TEST_CSV,
        training=False,
        shuffle=False
    )
