from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

from preprocess import preprocess_image


# ============================================================
# SETTINGS
# ============================================================

TRAIN_IMAGE_DIR = Path("dataset/train_images")
OUTPUT_DIR = Path("outputs/augmentation_samples")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# AUGMENTATION LAYER
# ============================================================

augmentation = tf.keras.layers.RandomTranslation(
    height_factor=0.0,
    width_factor=0.10,
    fill_mode="constant",
    fill_value=0
)


# ============================================================
# AUGMENTATION FUNCTION
# ============================================================

def augment_image(image):
    """
    Applies random horizontal translation.

    Input:
        image -> Tensor of shape (512, 512, 3)

    Output:
        Tensor of shape (512, 512, 3)
    """

    image = tf.cast(image, tf.float32)

    # Add batch dimension
    image = tf.expand_dims(image, axis=0)

    # Apply augmentation
    image = augmentation(
        image,
        training=True
    )

    # Remove batch dimension
    image = image[0]

    return image


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_augmentation(image_id):

    sample_path = (
        TRAIN_IMAGE_DIR /
        f"{image_id}.png"
    )

    if not sample_path.exists():

        print("\nERROR: Image not found.")
        print("Please enter a valid training image ID.")

        return

    # --------------------------------------------------------
    # LOAD ORIGINAL
    # --------------------------------------------------------

    with Image.open(sample_path) as img:
        original_image = img.convert("L").copy()

    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    processed_image = preprocess_image(
        sample_path
    )

    # --------------------------------------------------------
    # CREATE THREE AUGMENTED EXAMPLES
    # --------------------------------------------------------

    augmented_images = []

    for _ in range(3):

        augmented = augment_image(
            processed_image
        )

        augmented_images.append(
            tf.cast(
                augmented,
                tf.uint8
            ).numpy()
        )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    plt.figure(figsize=(20, 5))

    # Original
    plt.subplot(1, 5, 1)

    plt.imshow(
        original_image,
        cmap="gray"
    )

    plt.title(
        f"Original X-ray\n"
        f"ID: {image_id}\n"
        f"{original_image.size[0]} × "
        f"{original_image.size[1]}"
    )

    plt.axis("off")

    # Preprocessed
    plt.subplot(1, 5, 2)

    plt.imshow(
        processed_image
    )

    plt.title(
        "Preprocessed\n"
        "512 × 512 × 3"
    )

    plt.axis("off")

    # Augmented
    for i, augmented_image in enumerate(
        augmented_images
    ):

        plt.subplot(
            1,
            5,
            i + 3
        )

        plt.imshow(
            augmented_image
        )

        plt.title(
            f"Augmented {i + 1}\n"
            "Horizontal Translation"
        )

        plt.axis("off")

    plt.suptitle(
        f"Preprocessing and Augmentation "
        f"Visualization — ID {image_id}"
    )

    plt.tight_layout()

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_path = (
        OUTPUT_DIR /
        f"{image_id}_preprocessing_augmentation_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    # --------------------------------------------------------
    # INFORMATION
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print(
        "PREPROCESSING + AUGMENTATION VISUALIZATION"
    )
    print("=" * 60)

    print("\nImage ID:", image_id)

    print(
        "Original Size :",
        original_image.size
    )

    print(
        "Processed Shape :",
        processed_image.shape
    )

    print(
        "Augmented Shape :",
        augmented_images[0].shape
    )

    print(
        "\nAugmentation:"
    )

    print(
        "Horizontal Translation = up to 10%"
    )

    print(
        "\nThree random augmented examples were generated."
    )

    print(
        "\nVisualization saved to:"
    )

    print(output_path)

    plt.show()


# ============================================================
# RUN VISUALIZATION ONLY WHEN FILE IS EXECUTED DIRECTLY
# ============================================================

if __name__ == "__main__":

    image_id = input(
        "\nEnter training image ID to visualize "
        "(example: 10000): "
    ).strip()

    visualize_augmentation(image_id)