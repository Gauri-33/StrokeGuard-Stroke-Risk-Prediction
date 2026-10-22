import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "stroke_risk_prediction_dataset.csv"
FIGURES_DIR = PROJECT_DIR / "reports" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
if not DATA_PATH.exists():
    # Fallback check if old filename or path is still present
    alt_path = PROJECT_DIR / "data" / "healthcare-dataset-stroke-data.csv"
    if alt_path.exists():
        DATA_PATH = alt_path

df = pd.read_csv(DATA_PATH)

# Create binary stroke target for classification analysis
if "stroke" not in df.columns:
    if "Stroke_Risk" in df.columns:
        df["stroke"] = (df["Stroke_Risk"] == "High").astype(int)
    elif "Stroke" in df.columns:
        df["stroke"] = df["Stroke"].astype(int)

# Identify column name mappings (supporting both old & new datasets)
age_col = "Age" if "Age" in df.columns else "age"
glucose_col = "Blood_Glucose" if "Blood_Glucose" in df.columns else ("avg_glucose_level" if "avg_glucose_level" in df.columns else None)
bmi_col = "BMI" if "BMI" in df.columns else "bmi"

print("\n========== STROKEGUARD: EXPLORATORY DATA ANALYSIS ==========\n")

# 1. Dataset overview
print("1. DATASET OVERVIEW")
print("-" * 50)
print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

# 2. Dataset information
print("\n2. DATASET INFORMATION")
print("-" * 50)
df.info()

# 3. Statistical summary
print("\n3. DESCRIPTIVE STATISTICS")
print("-" * 50)
print(df.describe(include="all").T)

# 4. Missing values
print("\n4. MISSING VALUES")
print("-" * 50)
missing = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100

missing_table = pd.DataFrame({
    "Missing Values": missing,
    "Missing Percentage": missing_percent.round(2)
})

print(missing_table[missing_table["Missing Values"] > 0])
missing_table.to_csv(PROJECT_DIR / "reports" / "missing_values.csv")

# 5. Duplicate records
print("\n5. DUPLICATE RECORDS")
print("-" * 50)
print("Number of duplicate rows:", df.duplicated().sum())

# 6. Stroke class distribution
print("\n6. STROKE CLASS DISTRIBUTION")
print("-" * 50)
print(df["stroke"].value_counts())
print("\nPercentage distribution:")
print((df["stroke"].value_counts(normalize=True) * 100).round(2))

plt.figure(figsize=(7, 5))
df["stroke"].value_counts().sort_index().plot(kind="bar", rot=0, color=["#087f8c", "#e0564c"])
plt.title("Stroke Class Distribution (High Risk vs Non-High)")
plt.xlabel("Stroke Risk Status (0 = Low/Moderate, 1 = High)")
plt.ylabel("Number of Individuals")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "eda_stroke_distribution.png", dpi=300)
plt.close()

# 7. Age distribution
if age_col in df.columns:
    plt.figure(figsize=(8, 5))
    df[age_col].dropna().hist(bins=30, edgecolor="black", color="#087f8c")
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eda_age_distribution.png", dpi=300)
    plt.close()

# 8. Glucose level distribution
if glucose_col and glucose_col in df.columns:
    plt.figure(figsize=(8, 5))
    df[glucose_col].dropna().hist(bins=30, edgecolor="black", color="#087f8c")
    plt.title(f"{glucose_col.replace('_', ' ')} Distribution")
    plt.xlabel("Blood Glucose Level")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eda_glucose_distribution.png", dpi=300)
    plt.close()

# 9. BMI distribution
if bmi_col in df.columns:
    plt.figure(figsize=(8, 5))
    df[bmi_col].dropna().hist(bins=30, edgecolor="black", color="#087f8c")
    plt.title("BMI Distribution")
    plt.xlabel("BMI")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eda_bmi_distribution.png", dpi=300)
    plt.close()

# 10. Stroke cases by age group
if age_col in df.columns:
    df["age_group"] = pd.cut(
        df[age_col],
        bins=[0, 18, 35, 50, 65, 120],
        labels=["0-18", "19-35", "36-50", "51-65", "66+"],
        include_lowest=True
    )

    age_stroke = df.groupby("age_group", observed=False)["stroke"].mean() * 100
    print("\n7. STROKE RATE BY AGE GROUP (%)")
    print("-" * 50)
    print(age_stroke.round(2))

    plt.figure(figsize=(8, 5))
    age_stroke.plot(kind="bar", rot=0, color="#087f8c")
    plt.title("Stroke Rate by Age Group")
    plt.xlabel("Age Group")
    plt.ylabel("Individuals with Stroke Risk (%)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eda_stroke_by_age_group.png", dpi=300)
    plt.close()

# 11. Stroke rate by selected categorical features
candidate_categoricals = [
    "Gender", "gender",
    "Hypertension", "hypertension",
    "Heart_Disease", "heart_disease",
    "Smoking_Status", "smoking_status",
    "Work_Type", "work_type",
    "Residence_Type", "Residence_type",
    "Diet_Quality",
    "Family_History_Stroke"
]

print("\n8. STROKE RATE BY CATEGORICAL FEATURES")
print("-" * 50)

for column in candidate_categoricals:
    if column in df.columns:
        rate = df.groupby(column, dropna=False, observed=False)["stroke"].mean() * 100
        print(f"\nStroke rate by {column} (%):")
        print(rate.round(2))

        plt.figure(figsize=(9, 5))
        rate.plot(kind="bar", rot=30, color="#087f8c")
        plt.title(f"Stroke Rate by {column}")
        plt.xlabel(column.replace("_", " ").title())
        plt.ylabel("High Risk Proportion (%)")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"eda_stroke_by_{column}.png", dpi=300)
        plt.close()

# 12. Numeric feature correlation heatmap
numeric_df = df.select_dtypes(include="number").drop(
    columns=["Patient_ID", "id", "stroke"], errors="ignore"
)

correlation = numeric_df.corr()

print("\n9. NUMERIC FEATURE CORRELATION")
print("-" * 50)
print(correlation.round(2))

plt.figure(figsize=(14, 11))
image = plt.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(image, label="Correlation")
plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=45, ha="right", fontsize=8)
plt.yticks(range(len(correlation.columns)), correlation.columns, fontsize=8)
plt.title("Correlation Heatmap of Numeric Features")

for i in range(len(correlation.columns)):
    for j in range(len(correlation.columns)):
        plt.text(j, i, f"{correlation.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "eda_correlation_heatmap.png", dpi=300)
plt.close()

# 13. Boxplots for key numerical features against stroke
box_cols = [c for c in [age_col, glucose_col, bmi_col, "Blood_Pressure_Systolic", "Total_Cholesterol"] if c and c in df.columns]
for column in box_cols:
    plot_data = df[[column, "stroke"]].dropna()
    plt.figure(figsize=(7, 5))
    plot_data.boxplot(column=column, by="stroke")
    plt.title(f"{column.replace('_', ' ').title()} by Stroke Status")
    plt.suptitle("")
    plt.xlabel("Stroke Risk (0 = Low/Mod, 1 = High)")
    plt.ylabel(column.replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"eda_{column}_by_stroke.png", dpi=300)
    plt.close()

# 14. Save summaries
df.describe(include="all").T.to_csv(PROJECT_DIR / "reports" / "eda_summary.csv")

print("\n========== EDA COMPLETED SUCCESSFULLY ==========")
print("Figures saved in:", FIGURES_DIR)
print("Summary saved in reports/eda_summary.csv")
print("Missing-value report saved in reports/missing_values.csv")