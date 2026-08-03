from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf

from preprocess import preprocess_image


# ============================================================
# SETTINGS
# ============================================================

TRAIN_IMAGE_DIR = Path("dataset/train_images")
OUTPUT_DIR = Path("outputs/augmentation_samples")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# AUGMENTATION
# ============================================================

# Base paper: horizontal translation by 10%
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomTranslation(
        height_factor=0.0,
        width_factor=0.10,
        fill_mode="constant",
        fill_value=0
    )
])


# ============================================================
# LOAD AND PREPROCESS ONE SAMPLE
# ============================================================

sample_path = TRAIN_IMAGE_DIR / "10000.png"

processed_image = preprocess_image(sample_path)

# Add batch dimension:
# (512, 512, 3) -> (1, 512, 512, 3)
image_batch = tf.expand_dims(processed_image, axis=0)


# ============================================================
# CREATE AUGMENTED EXAMPLES
# ============================================================

augmented_images = []

for _ in range(3):

    augmented = augmentation(
        image_batch,
        training=True
    )

    augmented_images.append(
        augmented[0].numpy().astype("uint8")
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(processed_image)
plt.title("Preprocessed Original")
plt.axis("off")


for i, augmented_image in enumerate(augmented_images):

    plt.subplot(2, 2, i + 2)

    plt.imshow(augmented_image)

    plt.title(
        f"Augmented Example {i + 1}\n"
        "Horizontal Translation"
    )

    plt.axis("off")


plt.tight_layout()


# ============================================================
# SAVE DEMONSTRATION
# ============================================================

output_path = (
    OUTPUT_DIR /
    "10000_augmentation_comparison.png"
)

plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)

print("=" * 60)
print("AUGMENTATION TEST")
print("=" * 60)

print("\nOriginal processed shape :", processed_image.shape)

print(
    "Augmented image shape   :",
    augmented_images[0].shape
)

print("\nAugmentation:")
print("Horizontal Translation = up to 10%")

print("\nComparison saved to:")
print(output_path)

plt.show()