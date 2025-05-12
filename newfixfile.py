import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import joblib
import os

# === CONFIG ===
DATA_PATH = "/home/hemanth/Desktop/training/combined_labeled.csv"
OUTPUT_PATH = "/home/hemanth/Desktop/training/suspicious_predictions.csv"
MODEL_PATH = "/home/hemanth/Desktop/training/rf_model.joblib"
CHUNK_SIZE = 250_000

def train_model():
    print("📥 Loading full dataset for training...")
    df = pd.read_csv(DATA_PATH, usecols=lambda c: c != "Protocol Type")

    label_encoder = LabelEncoder()
    df["Label"] = label_encoder.fit_transform(df["Label"])

    # Balance the sample
    normal = df[df["Label"] == 0].sample(n=100_000, random_state=42)
    suspicious = df[df["Label"] == 1].sample(n=100_000, random_state=42)
    sample_df = pd.concat([normal, suspicious])

    X = sample_df.drop(columns=["Label"])
    y = sample_df["Label"]

    print("🧠 Training model...")
    model = RandomForestClassifier(n_estimators=40, n_jobs=1, random_state=42)
    model.fit(X, y)

    joblib.dump((model, label_encoder), MODEL_PATH)
    print(f"✅ Model saved: {MODEL_PATH}")

def run_prediction():
    print("\n🔍 Running chunked predictions...")
    model, label_encoder = joblib.load(MODEL_PATH)

    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)

    chunk_iter = pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE)
    first_chunk = True
    suspicious_count = 0

    for i, chunk in enumerate(tqdm(chunk_iter, desc="📦 Processing chunks")):
        if "Protocol Type" in chunk.columns:
            chunk = chunk.drop(columns=["Protocol Type"])

        orig = chunk.copy()
        chunk["Label"] = label_encoder.transform(chunk["Label"])
        X = chunk.drop(columns=["Label"])

        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1]

        orig["Predicted_Label"] = label_encoder.inverse_transform(preds)
        orig["Suspicion_Score"] = probs

        suspicious_rows = orig[orig["Predicted_Label"] == "Suspicious"]
        suspicious_count += len(suspicious_rows)

        mode = "w" if first_chunk else "a"
        suspicious_rows.to_csv(OUTPUT_PATH, mode=mode, index=False, header=first_chunk)
        first_chunk = False

    print(f"\n✅ Total suspicious rows saved: {suspicious_count:,}")
    print(f"📁 Output file: {OUTPUT_PATH}")

if __name__ == "__main__":
    train_model()
    run_prediction()
