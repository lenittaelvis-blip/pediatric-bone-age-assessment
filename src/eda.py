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

print("\nTest Images Expected     :", len(test_ids))
print("Test Images Found        :", len(test_images))
print("Missing Test Images      :", len(missing_test))
print("Extra Test Images        :", len(extra_test))
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