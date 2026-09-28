from pathlib import Path

import numpy as np
import tensorflow as tf

from data_pipeline import (
    get_train_dataset,
    get_validation_dataset,
    get_test_dataset
)


# ============================================================
# SETTINGS
# ============================================================

EXPECTED_SHAPE = (8, 512, 512, 3)

MAX_PIXEL_VALUE = 255.0
MIN_PIXEL_VALUE = 0.0


# ============================================================
# VERIFY DATASET
# ============================================================

def verify_dataset(name, dataset):

    print("\n" + "=" * 60)
    print(f"VERIFYING {name.upper()} DATASET")
    print("=" * 60)

    total_batches = 0
    total_images = 0

    failed_batches = []

    minimum_pixel = float("inf")
    maximum_pixel = float("-inf")

    for batch_number, (images, labels) in enumerate(
        dataset,
        start=1
    ):

        total_batches += 1

        batch_size = images.shape[0]

        total_images += batch_size

        # ----------------------------------------------------
        # SHAPE CHECK
        # ----------------------------------------------------

        expected_batch_shape = (
            batch_size,
            512,
            512,
            3
        )

        shape_ok = (
            tuple(images.shape)
            == expected_batch_shape
        )

        # ----------------------------------------------------
        # DATA TYPE CHECK
        # ----------------------------------------------------

        dtype_ok = (
            images.dtype == tf.float32
        )

        # ----------------------------------------------------
        # NAN / INFINITY CHECK
        # ----------------------------------------------------

        finite_ok = bool(
            tf.reduce_all(
                tf.math.is_finite(images)
            ).numpy()
        )

        # ----------------------------------------------------
        # PIXEL RANGE CHECK
        # ----------------------------------------------------

        batch_min = float(
            tf.reduce_min(images).numpy()
        )

        batch_max = float(
            tf.reduce_max(images).numpy()
        )

        pixel_range_ok = (
            batch_min >= MIN_PIXEL_VALUE
            and
            batch_max <= MAX_PIXEL_VALUE
        )

        # ----------------------------------------------------
        # LABEL CHECK
        # ----------------------------------------------------

        label_shape_ok = (
            len(labels.shape) == 1
            and
            labels.shape[0] == batch_size
        )

        label_finite_ok = bool(
            tf.reduce_all(
                tf.math.is_finite(labels)
            ).numpy()
        )

        # ----------------------------------------------------
        # OVERALL RESULT
        # ----------------------------------------------------

        batch_ok = (
            shape_ok
            and dtype_ok
            and finite_ok
            and pixel_range_ok
            and label_shape_ok
            and label_finite_ok
        )

        if not batch_ok:

            failed_batches.append(
                batch_number
            )

        minimum_pixel = min(
            minimum_pixel,
            batch_min
        )

        maximum_pixel = max(
            maximum_pixel,
            batch_max
        )

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if batch_number % 100 == 0:

            print(
                f"Checked batch "
                f"{batch_number}"
            )

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n" + "-" * 60)
    print(f"{name.upper()} VERIFICATION RESULT")
    print("-" * 60)

    print(
        "Total batches checked :",
        total_batches
    )

    print(
        "Total images checked  :",
        total_images
    )

    print(
        "Failed batches        :",
        len(failed_batches)
    )

    print(
        "Minimum pixel value   :",
        minimum_pixel
    )

    print(
        "Maximum pixel value   :",
        maximum_pixel
    )

    if failed_batches:

        print(
            "\nFailed batch numbers:"
        )

        print(
            failed_batches
        )

    else:

        print(
            "\n✓ All batches passed verification."
        )

    return failed_batches


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("FULL DATA PIPELINE BATCH VERIFICATION")
    print("=" * 60)

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    train_dataset = get_train_dataset(
    shuffle=False
)


    train_failures = verify_dataset(
        "Training",
        train_dataset
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    validation_dataset = (
        get_validation_dataset()
    )

    validation_failures = verify_dataset(
        "Validation",
        validation_dataset
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    test_dataset = get_test_dataset()

    test_failures = verify_dataset(
        "Test",
        test_dataset
    )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print("\n" + "=" * 60)
    print("FINAL BATCH VERIFICATION")
    print("=" * 60)

    total_failures = (
        len(train_failures)
        +
        len(validation_failures)
        +
        len(test_failures)
    )

    if total_failures == 0:

        print(
            "\n✓ ALL BATCHES PASSED."
        )

        print(
            "✓ Training pipeline verified."
        )

        print(
            "✓ Validation pipeline verified."
        )

        print(
            "✓ Test pipeline verified."
        )

    else:

        print(
            "\n✗ SOME BATCHES FAILED."
        )

        print(
            "Do not start model training yet."
        )