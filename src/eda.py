import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# ----------------------------
# DATASET PATHS
# ----------------------------
TRAIN_CSV = r"D:\Projects\main project\dataset\boneage-training-dataset.csv"
TEST_CSV = r"D:\Projects\main project\dataset\boneage-test-dataset.csv"
TRAIN_FOLDER = r"D:\Projects\main project\dataset\train_images"
TEST_FOLDER = r"D:\Projects\main project\dataset\test_images"

# ----------------------------
# LOAD CSV FILES
# ----------------------------
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

# ----------------------------
# BASIC DATASET INFORMATION
# ----------------------------
print("=" * 50)
print("TRAIN DATASET")
print("=" * 50)
print(train_df.head())
print("\n")
print("=" * 50)
print("TEST DATASET")
print("=" * 50)
print(test_df.head())

# ----------------------------
# DATASET INFORMATION
# ----------------------------

print("\n")
print("=" * 60)
print("TRAIN DATASET INFORMATION")
print("=" * 60)

print("\nShape of Training Dataset:")
print(train_df.shape)

print("\nTraining Dataset Columns:")
print(train_df.columns)

print("\nTraining Dataset Data Types:")
print(train_df.dtypes)

print("\n")

print("=" * 60)
print("TEST DATASET INFORMATION")
print("=" * 60)

print("\nShape of Test Dataset:")
print(test_df.shape)

print("\nTest Dataset Columns:")
print(test_df.columns)

print("\nTest Dataset Data Types:")
print(test_df.dtypes)

# ----------------------------
# MISSING VALUES
# ----------------------------

print("\n")
print("=" * 60)
print("MISSING VALUES")
print("=" * 60)

print("\nTraining Dataset Missing Values:")
print(train_df.isnull().sum())

print("\n")

print("Test Dataset Missing Values:")
print(test_df.isnull().sum())

# ----------------------------
# DUPLICATE ROWS
# ----------------------------

print("\n")
print("=" * 60)
print("DUPLICATE ROWS")
print("=" * 60)

print("\nDuplicate Rows in Training Dataset:")
print(train_df.duplicated().sum())

print("\nDuplicate Rows in Test Dataset:")
print(test_df.duplicated().sum())

# ----------------------------
# IMAGE VERIFICATION
# ----------------------------

print("\n")
print("=" * 60)
print("IMAGE VERIFICATION")
print("=" * 60)

# Get image names from folders
train_images = {
    os.path.splitext(file)[0]
    for file in os.listdir(TRAIN_FOLDER)
    if file.endswith(".png")
}

test_images = {
    os.path.splitext(file)[0]
    for file in os.listdir(TEST_FOLDER)
    if file.endswith(".png")
}

# Get IDs from CSV
train_ids = set(train_df["id"].astype(str))
test_ids = set(test_df["Case ID"].astype(str))
# Check for duplicate IDs in CSV files
duplicate_train_ids = train_df["id"].duplicated().sum()
duplicate_test_ids = test_df["Case ID"].duplicated().sum()

# Missing images
missing_train = train_ids - train_images
missing_test = test_ids - test_images

# Extra images
extra_train = train_images - train_ids
extra_test = test_images - test_ids

print("\nTraining Images Expected :", len(train_ids))
print("Training Images Found    :", len(train_images))
print("Missing Training Images  :", len(missing_train))
print("Extra Training Images    :", len(extra_train))
print("Duplicate Training IDs   :", duplicate_train_ids)
print("\nTest Images Expected     :", len(test_ids))
print("Test Images Found        :", len(test_images))
print("Missing Test Images      :", len(missing_test))
print("Extra Test Images        :", len(extra_test))
print("Duplicate Test IDs       :", duplicate_test_ids)
# ----------------------------
# IMAGE ANALYSIS
# ----------------------------

print("\n")
print("=" * 60)
print("IMAGE ANALYSIS")
print("=" * 60)

image_sizes = []

for image_name in os.listdir(TRAIN_FOLDER):

    image_path = os.path.join(TRAIN_FOLDER, image_name)

    try:
        with Image.open(image_path) as img:
            image_sizes.append(img.size)

    except Exception:
        print("Corrupted Image:", image_name)

print("\nTotal Images Checked :", len(image_sizes))

print("Smallest Image :", min(image_sizes))

print("Largest Image  :", max(image_sizes))
# ----------------------------
# IMAGE COLOR MODE ANALYSIS
# ----------------------------

print("\n")
print("=" * 60)
print("IMAGE COLOR MODE ANALYSIS")
print("=" * 60)

image_modes = {}

for image_name in os.listdir(TRAIN_FOLDER):

    if not image_name.lower().endswith(".png"):
        continue

    image_path = os.path.join(TRAIN_FOLDER, image_name)

    try:
        with Image.open(image_path) as img:

            mode = img.mode

            if mode in image_modes:
                image_modes[mode] += 1
            else:
                image_modes[mode] = 1

    except Exception as e:
        print("Could not read:", image_name)

print("\nImage Color Modes:")

for mode, count in image_modes.items():
    print(mode, ":", count)

# ----------------------------
# BONE AGE ANALYSIS
# ----------------------------

print("\n")
print("=" * 60)
print("BONE AGE ANALYSIS")
print("=" * 60)

print("\nBone Age Statistics:")

print("Minimum Bone Age :", train_df["boneage"].min(), "months")
print("Maximum Bone Age :", train_df["boneage"].max(), "months")
print("Mean Bone Age    :", round(train_df["boneage"].mean(), 2), "months")
print("Median Bone Age  :", train_df["boneage"].median(), "months")
print("Standard Deviation:", round(train_df["boneage"].std(), 2), "months")

# ----------------------------
# BONE AGE DISTRIBUTION
# ----------------------------

plt.figure(figsize=(10, 6))

plt.hist(train_df["boneage"], bins=30, edgecolor="black")

plt.title("Distribution of Bone Age")
plt.xlabel("Bone Age (Months)")
plt.ylabel("Number of Images")

plt.tight_layout()
plt.show()
# ----------------------------
# GENDER ANALYSIS
# ----------------------------

print("\n")
print("=" * 60)
print("GENDER ANALYSIS")
print("=" * 60)

male_count = train_df["male"].sum()
female_count = len(train_df) - male_count

male_percentage = (male_count / len(train_df)) * 100
female_percentage = (female_count / len(train_df)) * 100

print("\nMale Images   :", male_count)
print("Female Images :", female_count)

print("\nMale Percentage   :", round(male_percentage, 2), "%")
print("Female Percentage :", round(female_percentage, 2), "%")
# ----------------------------
# GENDER DISTRIBUTION GRAPH
# ----------------------------

gender_labels = ["Male", "Female"]
gender_counts = [male_count, female_count]

plt.figure(figsize=(6, 5))

plt.bar(gender_labels, gender_counts)

plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Number of Images")

plt.tight_layout()
plt.show()
# ----------------------------
# SAMPLE IMAGE VISUALIZATION
# ----------------------------

print("\n")
print("=" * 60)
print("SAMPLE IMAGE VISUALIZATION")
print("=" * 60)

sample_rows = train_df.sample(n=4, random_state=42)

plt.figure(figsize=(12, 10))

for i, (_, row) in enumerate(sample_rows.iterrows()):

    image_id = str(row["id"])
    bone_age = row["boneage"]
    gender = "Male" if row["male"] else "Female"

    image_path = os.path.join(TRAIN_FOLDER, image_id + ".png")

    image = Image.open(image_path)

    plt.subplot(2, 2, i + 1)

    plt.imshow(image, cmap="gray")

    plt.title(
        f"ID: {image_id}\n"
        f"Bone Age: {bone_age} months | {gender}"
    )

    plt.axis("off")

plt.tight_layout()
plt.show()
# ----------------------------
# IMAGE INTEGRITY CHECK
# ----------------------------

print("\n")
print("=" * 60)
print("IMAGE INTEGRITY CHECK")
print("=" * 60)

corrupted_images = []

for image_name in os.listdir(TRAIN_FOLDER):

    if not image_name.lower().endswith(".png"):
        continue

    image_path = os.path.join(TRAIN_FOLDER, image_name)

    try:
        with Image.open(image_path) as img:
            img.verify()

    except Exception as e:
        corrupted_images.append(image_name)

print("\nTraining Images Checked :", len(train_images))
print("Corrupted Images Found :", len(corrupted_images))

if corrupted_images:
    print("Corrupted Image Names:")
    print(corrupted_images)
else:
    print("No corrupted training images detected.")
# ----------------------------
# TEST IMAGE INTEGRITY CHECK
# ----------------------------

corrupted_test_images = []

for image_name in os.listdir(TEST_FOLDER):

    if not image_name.lower().endswith(".png"):
        continue

    image_path = os.path.join(TEST_FOLDER, image_name)

    try:
        with Image.open(image_path) as img:
            img.verify()

    except Exception as e:
        corrupted_test_images.append(image_name)

print("\nTest Images Checked     :", len(test_images))
print("Corrupted Test Images   :", len(corrupted_test_images))

if corrupted_test_images:
    print("Corrupted Test Image Names:")
    print(corrupted_test_images)
else:
    print("No corrupted test images detected.")

# ============================================================
# METADATA VALIDATION
# ============================================================

print("\n")
print("=" * 60)
print("METADATA VALIDATION")
print("=" * 60)

# Check for missing bone-age values
missing_boneage = train_df["boneage"].isnull().sum()

# Check for missing sex values
missing_sex = train_df["male"].isnull().sum()

# Check unique sex values
unique_sex_values = train_df["male"].unique()

# Check invalid sex values
valid_sex_values = {True, False}
invalid_sex_values = set(unique_sex_values) - valid_sex_values

# Check invalid bone-age values
invalid_boneage = (train_df["boneage"] <= 0).sum()

print("\nMissing Bone Age :", missing_boneage)
print("Missing Sex      :", missing_sex)

print("\nSex Values Found :", unique_sex_values)

print("Invalid Sex Values :", len(invalid_sex_values))
print("Invalid Bone Age Values :", invalid_boneage)

if (
    missing_boneage == 0
    and missing_sex == 0
    and len(invalid_sex_values) == 0
    and invalid_boneage == 0
):
    print("\n✓ Training metadata passed validation.")
else:
    print("\n⚠ Metadata validation problems detected.")

# ============================================================
# SEX ENCODING CHECK
# ============================================================

print("\n")
print("=" * 60)
print("SEX ENCODING CHECK")
print("=" * 60)

# Convert Boolean sex values to numerical values
train_df["sex_encoded"] = train_df["male"].astype(int)

print("\nSex Encoding:")
print("Female (False) → 0")
print("Male   (True)  → 1")

print("\nEncoded Sex Value Counts:")
print(train_df["sex_encoded"].value_counts().sort_index())

# Verify that only 0 and 1 are present
invalid_encoded_values = set(train_df["sex_encoded"].unique()) - {0, 1}

print("\nInvalid Encoded Values :", len(invalid_encoded_values))

if len(invalid_encoded_values) == 0:
    print("✓ Sex encoding completed successfully.")
else:
    print("⚠ Invalid encoded values detected.")
# ============================================================
# BONE AGE OUTLIER ANALYSIS
# ============================================================

print("\n")
print("=" * 60)
print("BONE AGE OUTLIER ANALYSIS")
print("=" * 60)

# Calculate quartiles
Q1 = train_df["boneage"].quantile(0.25)
Q3 = train_df["boneage"].quantile(0.75)

# Calculate IQR
IQR = Q3 - Q1

# Calculate outlier boundaries
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Find potential outliers
outliers = train_df[
    (train_df["boneage"] < lower_bound) |
    (train_df["boneage"] > upper_bound)
]

print("\nQ1 (25th percentile) :", round(Q1, 2))
print("Q3 (75th percentile) :", round(Q3, 2))
print("IQR                  :", round(IQR, 2))

print("\nLower Outlier Boundary :", round(lower_bound, 2))
print("Upper Outlier Boundary :", round(upper_bound, 2))

print("\nPotential Outliers      :", len(outliers))

if len(outliers) > 0:
    print("\nPotential Outlier IDs and Bone Ages:")
    print(
        outliers[["id", "boneage", "male"]]
        .sort_values("boneage")
        .to_string(index=False)
    )
else:
    print("\nNo statistical bone-age outliers detected.")

print("\nIMPORTANT:")
print("Statistical outliers are NOT automatically removed.")
print("They must be visually inspected before any removal decision.")
# ============================================================
# OUTLIER IMAGE VISUAL INSPECTION
# ============================================================

print("\n")
print("=" * 60)
print("OUTLIER IMAGE VISUAL INSPECTION")
print("=" * 60)

outlier_ids = [3806, 1398]

plt.figure(figsize=(10, 5))

for i, image_id in enumerate(outlier_ids):

    image_id = str(image_id)

    image_path = os.path.join(
        TRAIN_FOLDER,
        image_id + ".png"
    )

    row = train_df[
        train_df["id"].astype(str) == image_id
    ].iloc[0]

    image = Image.open(image_path)

    plt.subplot(1, 2, i + 1)

    plt.imshow(image, cmap="gray")

    gender = "Male" if row["male"] else "Female"

    plt.title(
        f"ID: {image_id}\n"
        f"Bone Age: {row['boneage']} months | {gender}"
    )

    plt.axis("off")

plt.tight_layout()

plt.savefig(
    "outputs/preprocessing_samples/bone_age_outliers.png",
    dpi=150
)

plt.show()

print(
    "\nOutlier visualization saved to:"
)

print(
    "outputs/preprocessing_samples/bone_age_outliers.png"
)
