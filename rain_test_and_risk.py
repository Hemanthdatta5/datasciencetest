import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import matplotlib.pyplot as plt
import joblib
import os

# === CONFIG ===
DATA_PATH = "/home/hemanth/Desktop/training/combined_labeled.csv"
OUTPUT_PATH = "/home/hemanth/Desktop/training/suspicious_predictions.csv"
MODEL_PATH = "/home/hemanth/Desktop/training/rf_model.joblib"
TOP_OUTPUT = "/home/hemanth/Desktop/training/top_50_suspicious.csv"
CHUNK_SIZE = 250_000

# === TRAIN MODEL ===
def train_model():
    print("\n📥 Loading full dataset for training...")
    df = pd.read_csv(DATA_PATH, usecols=lambda c: c != "Protocol Type")

    label_encoder = LabelEncoder()
    df["Label"] = label_encoder.fit_transform(df["Label"])

    print("⚖️  Balancing data...")
    normal = df[df["Label"] == 0].sample(n=100_000, random_state=42)
    suspicious = df[df["Label"] == 1].sample(n=100_000, random_state=42)
    sample_df = pd.concat([normal, suspicious])

    X = sample_df.drop(columns=["Label"])
    y = sample_df["Label"]

    print("🧠 Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=40, random_state=42, n_jobs=1)
    model.fit(X, y)

    joblib.dump((model, label_encoder), MODEL_PATH)
    print(f"✅ Model saved: {MODEL_PATH}")

# === PREDICT IN CHUNKS ===
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

# === EXPORT TOP 50 ===
def export_top_50():
    print("\n📊 Exporting top 50 most suspicious records...")
    df = pd.read_csv(OUTPUT_PATH)
    top = df.sort_values("Suspicion_Score", ascending=False).head(50)
    top.to_csv(TOP_OUTPUT, index=False)
    print(f"✅ Top 50 saved to: {TOP_OUTPUT}")

# === PLOT SCORE DISTRIBUTION ===
def plot_scores():
    print("\n📈 Plotting suspicion score distribution...")
    df = pd.read_csv(OUTPUT_PATH)
    plt.hist(df["Suspicion_Score"], bins=50, color="red", edgecolor="black")
    plt.title("Distribution of Suspicion Scores")
    plt.xlabel("Suspicion Score")
    plt.ylabel("Number of Records")
    plt.tight_layout()
    plt.show()

# === MAIN ===
if __name__ == "__main__":
    train_model()
    run_prediction()
    export_top_50()
    plot_scores()