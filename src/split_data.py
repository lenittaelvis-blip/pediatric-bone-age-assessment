from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

TRAIN_CSV = Path("dataset/boneage-training-dataset.csv")

OUTPUT_DIR = Path("outputs/data_splits")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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