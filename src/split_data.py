from pathlib import Path
import hashlib
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

TRAIN_CSV = Path("dataset/boneage-training-dataset.csv")

OUTPUT_DIR = Path("outputs/data_splits")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_DIR = Path("dataset/train_images")
# ============================================================
# LOAD LABELED DATASET
# ============================================================

df = pd.read_csv(TRAIN_CSV)

print("=" * 60)
print("DATASET SPLITTING")
print("=" * 60)

print("\nTotal labeled images:", len(df))


# ============================================================
# FIRST SPLIT
# 80% Development + 20% Final Test
# ============================================================

development_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42
)


# ============================================================
# SECOND SPLIT
# Development data -> 80% Train + 20% Validation
# ============================================================

train_df, validation_df = train_test_split(
    development_df,
    test_size=0.20,
    random_state=42
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("SPLIT RESULT")
print("=" * 60)

print("Training Images   :", len(train_df))
print("Validation Images :", len(validation_df))
print("Test Images       :", len(test_df))

print("\nTotal:",
      len(train_df) +
      len(validation_df) +
      len(test_df))


# ============================================================
# SAVE SPLITS
# ============================================================

train_df.to_csv(
    OUTPUT_DIR / "train_split.csv",
    index=False
)

validation_df.to_csv(
    OUTPUT_DIR / "validation_split.csv",
    index=False
)

test_df.to_csv(
    OUTPUT_DIR / "test_split.csv",
    index=False
)


print("\nSplit files saved to:")
print(OUTPUT_DIR)
# ============================================================
# ANALYZE SPLIT DISTRIBUTIONS
# ============================================================

print("\n" + "=" * 60)
print("SPLIT DISTRIBUTION ANALYSIS")
print("=" * 60)


def analyze_split(name, data):

    total = len(data)

    male_count = data["male"].sum()
    female_count = total - male_count

    male_percentage = (male_count / total) * 100
    female_percentage = (female_count / total) * 100

    print(f"\n{name}")
    print("-" * 40)

    print("Total Images       :", total)

    print(
        f"Male               : {male_count} "
        f"({male_percentage:.2f}%)"
    )

    print(
        f"Female             : {female_count} "
        f"({female_percentage:.2f}%)"
    )

    print("Minimum Bone Age   :", data["boneage"].min(), "months")
    print("Maximum Bone Age   :", data["boneage"].max(), "months")
    print("Mean Bone Age      :", round(data["boneage"].mean(), 2), "months")
    print("Median Bone Age    :", round(data["boneage"].median(), 2), "months")


analyze_split("TRAINING SET", train_df)

analyze_split("VALIDATION SET", validation_df)

analyze_split("TEST SET", test_df)
# ============================================================
# SPLIT OVERLAP CHECK
# ============================================================

print("\n" + "=" * 60)
print("SPLIT OVERLAP CHECK")
print("=" * 60)

train_ids = set(train_df["id"].astype(str))
validation_ids = set(validation_df["id"].astype(str))
test_ids = set(test_df["id"].astype(str))

train_validation_overlap = train_ids & validation_ids
train_test_overlap = train_ids & test_ids
validation_test_overlap = validation_ids & test_ids

print("\nTraining ∩ Validation :", len(train_validation_overlap))
print("Training ∩ Test       :", len(train_test_overlap))
print("Validation ∩ Test     :", len(validation_test_overlap))

if (
    len(train_validation_overlap) == 0
    and len(train_test_overlap) == 0
    and len(validation_test_overlap) == 0
):
    print("\n✓ No image IDs are shared between the three splits.")
else:
    print("\n⚠ Split overlap detected.")

# ============================================================
# EXACT DUPLICATE IMAGE CHECK
# ============================================================

print("\n" + "=" * 60)
print("EXACT DUPLICATE IMAGE CHECK")
print("=" * 60)

def get_image_hash(image_path):
    """
    Creates a hash from the actual image pixels.
    Images with identical pixels will receive the same hash.
    """

    with Image.open(image_path) as image:
        image = image.convert("L")

        hash_object = hashlib.md5()

        hash_object.update(str(image.size).encode())
        hash_object.update(image.tobytes())

        return hash_object.hexdigest()


image_hashes = {}

image_files = list(IMAGE_DIR.glob("*.png"))

print("\nImages to check:", len(image_files))

for index, image_path in enumerate(image_files, start=1):

    try:
        image_hash = get_image_hash(image_path)

        if image_hash in image_hashes:
            image_hashes[image_hash].append(image_path.stem)
        else:
            image_hashes[image_hash] = [image_path.stem]

    except Exception as e:
        print("Could not process:", image_path.name)

    if index % 1000 == 0:
        print(f"Checked {index} / {len(image_files)} images")


duplicate_groups = [
    ids
    for ids in image_hashes.values()
    if len(ids) > 1
]

duplicate_image_count = sum(
    len(group)
    for group in duplicate_groups
)

print("\nUnique image contents :", len(image_hashes))
print("Duplicate groups      :", len(duplicate_groups))
print("Images involved       :", duplicate_image_count)

if len(duplicate_groups) == 0:

    print("\n✓ No exact duplicate images detected.")

else:

    print("\n⚠ Exact duplicate images detected.")

    print("\nDuplicate groups:")

    for group in duplicate_groups[:20]:
        print(group)

# ============================================================
# DUPLICATE CHECK ACROSS SPLITS
# ============================================================

print("\n" + "=" * 60)
print("DUPLICATE IMAGES ACROSS SPLITS")
print("=" * 60)

split_lookup = {}

for image_hash, ids in image_hashes.items():

    locations = []

    for image_id in ids:

        if image_id in train_ids:
            locations.append("TRAIN")

        if image_id in validation_ids:
            locations.append("VALIDATION")

        if image_id in test_ids:
            locations.append("TEST")

    if len(set(locations)) > 1:
        split_lookup[image_hash] = {
            "ids": ids,
            "splits": list(set(locations))
        }

print(
    "Duplicate groups crossing splits :",
    len(split_lookup)
)

if len(split_lookup) == 0:

    print(
        "\n✓ No exact duplicate images were found across "
        "training, validation and test sets."
    )

else:

    print(
        "\n⚠ Exact duplicate images cross split boundaries."
    )

    for item in list(split_lookup.values())[:20]:
        print(item)