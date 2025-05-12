import os
import pandas as pd

# === CONFIGURATION ===
DATA_FOLDER = "/home/hemanth/Desktop/training/training/CSV/train"
OUTPUT_FILE = "/home/hemanth/Desktop/training/combined_labeled.csv"

combined_data = []

for file in os.listdir(DATA_FOLDER):
    path = os.path.join(DATA_FOLDER, file)

    if file.endswith(".csv") and os.path.getsize(path) > 10:  # Skip empty files
        label = "Normal" if "Benign" in file else "Suspicious"
        try:
            df = pd.read_csv(path)

            if df.empty or df.shape[1] == 0:
                print(f"⚠️ Skipping invalid/empty file: {file}")
                continue

            # Add the label
            df["Label"] = label

            # Select numeric columns
            numeric_cols = df.select_dtypes(include="number").columns.tolist()

            # Optionally keep 'Protocol Type' if it exists
            if "Protocol Type" in df.columns:
                numeric_cols.append("Protocol Type")

            # Add 'Label' column to keep
            numeric_cols.append("Label")

            # Subset, drop missing & duplicates
            df = df[numeric_cols].dropna().drop_duplicates()

            # Append to list
            combined_data.append(df)

            print(f"✅ Processed: {file} — Rows: {df.shape[0]}")

        except Exception as e:
            print(f"❌ Error reading {file}: {e}")

# === Combine all into one DataFrame ===
if combined_data:
    final_df = pd.concat(combined_data, ignore_index=True)
    print(f"📊 Final dataset shape: {final_df.shape}")

    # Export in chunks (efficient on large datasets)
    final_df.to_csv(OUTPUT_FILE, index=False, chunksize=100000)
    print(f"✅ Saved combined labeled data to: {OUTPUT_FILE}")
else:
    print("⚠️ No valid CSV files were processed.")
