import os
import pandas as pd
from scipy.stats import ttest_ind, f_oneway

# === CONFIG ===
DATA_FOLDER = "/home/hemanth/Desktop/training/training/CSV/train"
GROUP_COLUMN = "Protocol Type"
VALUE_COLUMN = "Rate"
OUTPUT_FILE = "/home/hemanth/Desktop/training/pvalue_results.csv"

def clean_data(df):
    df = df.dropna(subset=[GROUP_COLUMN, VALUE_COLUMN])
    df = df.drop_duplicates()
    df = df[pd.to_numeric(df[VALUE_COLUMN], errors='coerce').notnull()]
    df[VALUE_COLUMN] = df[VALUE_COLUMN].astype(float)
    return df

def analyze_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        df = clean_data(df)

        groups = df[GROUP_COLUMN].unique()
        if len(groups) < 2:
            return os.path.basename(file_path), "Not enough groups", None

        data_groups = [df[df[GROUP_COLUMN] == g][VALUE_COLUMN] for g in groups]

        if len(groups) == 2:
            stat, pval = ttest_ind(*data_groups, equal_var=False)
            test = "T-test"
        else:
            stat, pval = f_oneway(*data_groups)
            test = "ANOVA"

        return os.path.basename(file_path), test, pval

    except Exception as e:
        return os.path.basename(file_path), "Error", str(e)

# === MAIN ===
results = []
for file in os.listdir(DATA_FOLDER):
    if file.endswith(".csv") and os.path.getsize(os.path.join(DATA_FOLDER, file)) > 0:
        path = os.path.join(DATA_FOLDER, file)
        result = analyze_dataset(path)
        results.append(result)

# === Export to CSV ===
df_results = pd.DataFrame(results, columns=["File", "Test", "P-Value"])
df_results.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Results saved to: {OUTPUT_FILE}")
