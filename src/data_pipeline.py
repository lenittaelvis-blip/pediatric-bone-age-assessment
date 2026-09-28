from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from preprocess import preprocess_image
from augmentation import augment_image


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

IMAGE_DIR = Path(
    "dataset/train_images"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
IMAGE_CHANNELS = 3

BATCH_SIZE = 8


# ============================================================
# LOAD SPLIT CSV
# ============================================================

def load_split(csv_path):
    """
    Load one of the previously created dataset splits.
    """

    df = pd.read_csv(csv_path)

    return df


# ============================================================
# LOAD AND PREPROCESS ONE IMAGE
# ============================================================

def load_and_preprocess(image_id):
    """
    Load one X-ray and apply the existing preprocessing pipeline.

    Output:
        512 × 512 × 3 image
    """

    image_path = IMAGE_DIR / f"{image_id}.png"

    image = preprocess_image(image_path)

    return image


# ============================================================
# CREATE DATASET
# ============================================================

def create_dataset(
    csv_path,
    training=False,
    shuffle=True
):
    """
    Create a TensorFlow dataset.

    Training:
        preprocessing + augmentation

    Validation/Test:
        preprocessing only
    """

    df = load_split(csv_path)

    image_ids = df["id"].astype(str).values

    bone_ages = (
        df["boneage"]
        .astype(np.float32)
        .values
    )

    def generator():

        for image_id, bone_age in zip(
            image_ids,
            bone_ages
        ):

            # --------------------------------------------
            # PREPROCESS
            # --------------------------------------------

            image = load_and_preprocess(
                image_id
            )

            # --------------------------------------------
            # AUGMENT TRAINING IMAGES ONLY
            # --------------------------------------------

            if training:

                image = augment_image(
                    image
                )

                image = image.numpy()

            # --------------------------------------------
            # CONVERT TO FLOAT32
            # --------------------------------------------

            image = image.astype(
                np.float32
            )

            bone_age = np.float32(
                bone_age
            )

            yield image, bone_age

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(
                shape=(
                    IMAGE_HEIGHT,
                    IMAGE_WIDTH,
                    IMAGE_CHANNELS
                ),
                dtype=tf.float32
            ),

            tf.TensorSpec(
                shape=(),
                dtype=tf.float32
            )
        )
    )

    if training and shuffle:

        dataset = dataset.shuffle(
        buffer_size=256,
        reshuffle_each_iteration=True
    )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# ============================================================
# CREATE TRAINING DATASET
# ============================================================

def get_train_dataset(shuffle=True):

    return create_dataset(
        TRAIN_CSV,
        training=True,
        shuffle=shuffle
    )


# ============================================================
# CREATE VALIDATION DATASET
# ============================================================

def get_validation_dataset():

    return create_dataset(
        VALIDATION_CSV,
        training=False
    )


# ============================================================
# CREATE TEST DATASET
# ============================================================

def get_test_dataset():

    return create_dataset(
        TEST_CSV,
        training=False
    )


# ============================================================
# TEST THE PIPELINE
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("DATA PIPELINE TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # TRAINING DATA
    # --------------------------------------------------------

    print("\nCreating training dataset...")

    train_dataset = get_train_dataset()

    train_images, train_labels = next(
        iter(train_dataset)
    )

    print("\nTRAINING BATCH")
    print("-" * 40)

    print(
        "Image batch shape :",
        train_images.shape
    )

    print(
        "Image data type   :",
        train_images.dtype
    )

    print(
        "Pixel minimum     :",
        tf.reduce_min(train_images).numpy()
    )

    print(
        "Pixel maximum     :",
        tf.reduce_max(train_images).numpy()
    )

    print(
        "Bone-age labels   :",
        train_labels.numpy()
    )

    # --------------------------------------------------------
    # VALIDATION DATA
    # --------------------------------------------------------

    print("\nCreating validation dataset...")

    validation_dataset = (
        get_validation_dataset()
    )

    validation_images, validation_labels = next(
        iter(validation_dataset)
    )

    print("\nVALIDATION BATCH")
    print("-" * 40)

    print(
        "Image batch shape :",
        validation_images.shape
    )

    print(
        "Image data type   :",
        validation_images.dtype
    )

    print(
        "Pixel minimum     :",
        tf.reduce_min(validation_images).numpy()
    )

    print(
        "Pixel maximum     :",
        tf.reduce_max(validation_images).numpy()
    )

    print(
        "Bone-age labels   :",
        validation_labels.numpy()
    )

    # --------------------------------------------------------
    # TEST DATA
    # --------------------------------------------------------

    print("\nCreating test dataset...")

    test_dataset = get_test_dataset()

    test_images, test_labels = next(
        iter(test_dataset)
    )

    print("\nTEST BATCH")
    print("-" * 40)

    print(
        "Image batch shape :",
        test_images.shape
    )

    print(
        "Image data type   :",
        test_images.dtype
    )

    print(
        "Pixel minimum     :",
        tf.reduce_min(test_images).numpy()
    )

    print(
        "Pixel maximum     :",
        tf.reduce_max(test_images).numpy()
    )

    print(
        "Bone-age labels   :",
        test_labels.numpy()
    )

    print("\n" + "=" * 60)
    print("DATA PIPELINE TEST COMPLETED")
    print("=" * 60)