from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (512, 512)

TRAIN_IMAGE_DIR = Path("dataset/train_images")
OUTPUT_DIR = Path("outputs/preprocessing_samples")

# Create output folder if it does not exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# PREPROCESSING FUNCTION
# ============================================================

def preprocess_image(image_path):

    # Open image
    with Image.open(image_path) as img:

        # Ensure grayscale
        img = img.convert("L")

        # Resize while preserving aspect ratio
        # and pad remaining area with black
        img = ImageOps.pad(
            img,
            IMAGE_SIZE,
            method=Image.Resampling.LANCZOS,
            color=0
        )

        # Convert grayscale image to RGB
        img = img.convert("RGB")

        # Convert image to NumPy array
        image_array = np.array(img)

    return image_array


# ============================================================
# TEST THE FUNCTION
# ============================================================
if __name__ == "__main__":

    sample_path = TRAIN_IMAGE_DIR / "10000.png"

    # Load original image for comparison
    with Image.open(sample_path) as img:
        original_image = img.copy()

    # Preprocess image
    processed_image = preprocess_image(sample_path)


    print("=" * 60)
    print("PREPROCESSING TEST")
    print("=" * 60)

    print("Original Size :", original_image.size)
    print("Original Mode :", original_image.mode)

    print("\nProcessed Shape :", processed_image.shape)
    print("Processed Type  :", processed_image.dtype)
    print("Minimum Value   :", processed_image.min())
    print("Maximum Value   :", processed_image.max())


    # ============================================================
    # VISUALIZATION
    # ============================================================

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)

    plt.imshow(original_image, cmap="gray")

    plt.title(
        f"Original X-ray\n"
        f"{original_image.size[0]} × {original_image.size[1]}"
    )

    plt.axis("off")


    plt.subplot(1, 2, 2)

    plt.imshow(processed_image)

    plt.title(
        f"Preprocessed X-ray\n"
        f"{processed_image.shape[1]} × {processed_image.shape[0]}"
    )

    plt.axis("off")

    plt.tight_layout()


    # Save comparison for project demonstration
    comparison_path = OUTPUT_DIR / "10000_preprocessing_comparison.png"

    plt.savefig(
        comparison_path,
        dpi=200,
        bbox_inches="tight"
    )

    print("\nComparison saved to:")
    print(comparison_path)

    plt.show()
    # ============================================================
    # VALIDATE PREPROCESSING ON ALL TRAINING IMAGES
    # ============================================================

    print("\n" + "=" * 60)
    print("VALIDATING PREPROCESSING PIPELINE")
    print("=" * 60)

    image_files = list(TRAIN_IMAGE_DIR.glob("*.png"))

    successful = 0
    failed_images = []

    for i, image_path in enumerate(image_files, start=1):

        try:
            processed = preprocess_image(image_path)

            # Check expected output
            if processed.shape == (512, 512, 3) and processed.dtype == np.uint8:
                successful += 1
            else:
                failed_images.append(image_path.name)

        except Exception as error:
            failed_images.append(image_path.name)
            print(f"\nError in {image_path.name}: {error}")

        # Show progress
        if i % 1000 == 0:
            print(f"Validated {i} / {len(image_files)} images")


    print("\n" + "=" * 60)
    print("VALIDATION RESULT")
    print("=" * 60)

    print("Total Images       :", len(image_files))
    print("Successfully Passed:", successful)
    print("Failed Images      :", len(failed_images))

    if failed_images:
        print("Failed Image Names :", failed_images)
    else:
        print("All training images passed preprocessing validation.")