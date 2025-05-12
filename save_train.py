import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# === CONFIG ===
DATA_PATH = "/home/hemanth/Desktop/training/combined_labeled.csv"
OUTPUT_PATH = "/home/hemanth/Desktop/training/suspicious_predictions.csv"
MODEL_PATH = "/home/hemanth/Desktop/training/rf_model.joblib"
TOP_OUTPUT = "/home/hemanth/Desktop/training/top_50_suspicious.csv"
PLOT_PATH = "/home/hemanth/Desktop/training/suspicion_score_plot.png"
HISTOGRAM_CSV = "/home/hemanth/Desktop/training/histogram_summary.csv"
CONF_MATRIX_PATH = "/home/hemanth/Desktop/training/confusion_matrix_test.png"
TESTDATA_DIR = "/home/hemanth/Documents/testdata/testdata"
CHUNK_SIZE = 250_000

# === TRAIN MODEL ===
def train_model():
    print("\n📥 Loading dataset for training...")
    df = pd.read_csv(DATA_PATH, usecols=lambda c: c != "Protocol Type")

    label_encoder = LabelEncoder()
    df["Label"] = label_encoder.fit_transform(df["Label"])

    print("⚖️  Balancing samples...")
    normal = df[df["Label"] == 0].sample(n=100_000, random_state=42)
    suspicious = df[df["Label"] == 1].sample(n=100_000, random_state=42)
    sample_df = pd.concat([normal, suspicious])

    X = sample_df.drop(columns=["Label"])
    y = sample_df["Label"]

    print("🧠 Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=40, random_state=42, n_jobs=1)
    model.fit(X, y)

    joblib.dump((model, label_encoder), MODEL_PATH)
    print(f"✅ Model saved to: {MODEL_PATH}")

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

    print(f"\n✅ Suspicious rows saved: {suspicious_count:,}")
    print(f"📁 Output file: {OUTPUT_PATH}")

# === EXPORT TOP 50 ===
def export_top_50():
    print("\n📊 Exporting top 50 suspicious records...")
    df = pd.read_csv(OUTPUT_PATH)
    top = df.sort_values("Suspicion_Score", ascending=False).head(50)
    top.to_csv(TOP_OUTPUT, index=False)
    print(f"✅ Top 50 saved to: {TOP_OUTPUT}")

# === SAVE HISTOGRAM ===
def save_histogram():
    print("\n📈 Saving suspicion score histogram data and plot...")
    df = pd.read_csv(OUTPUT_PATH)
    hist_data = df["Suspicion_Score"].value_counts(bins=50).sort_index()
    hist_data.to_csv(HISTOGRAM_CSV)

    plt.figure(figsize=(8, 4))
    plt.hist(df["Suspicion_Score"], bins=50, color="red", edgecolor="black")
    plt.title("Distribution of Suspicion Scores")
    plt.xlabel("Suspicion Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    print(f"✅ Histogram saved to: {PLOT_PATH} and {HISTOGRAM_CSV}")

# === EVALUATE TEST DATA ===
def evaluate_model():
    print("\n🧪 Evaluating model on test data from", TESTDATA_DIR)

    test_files = [f for f in os.listdir(TESTDATA_DIR) if f.endswith(".csv")]
    test_df_list = []

    for file in test_files:
        path = os.path.join(TESTDATA_DIR, file)
        try:
            df = pd.read_csv(path, usecols=lambda c: c != "Protocol Type")
            # Infer label if missing
            if "Label" not in df.columns:
                if "Benign" in file or "Normal" in file:
                    df["Label"] = "Normal"
                else:
                    df["Label"] = "Suspicious"
            df["Source_File"] = file
            test_df_list.append(df)
        except Exception as e:
            print(f"❌ Failed to read {file}: {e}")

    if not test_df_list:
        print("⚠️ No valid CSV test files found.")
        return

    df = pd.concat(test_df_list, ignore_index=True)
    label_encoder = LabelEncoder()
    df["Label"] = label_encoder.fit_transform(df["Label"])

    X = df.drop(columns=["Label"])
    y = df["Label"]

    model, _ = joblib.load(MODEL_PATH)
    y_pred = model.predict(X)

    print("📋 Classification Report:\n")
    print(classification_report(y, y_pred, target_names=label_encoder.classes_))

    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title("Confusion Matrix (Test Data)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(CONF_MATRIX_PATH)
    print(f"✅ Test confusion matrix saved to: {CONF_MATRIX_PATH}")

# === MAIN ===
if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        train_model()
    run_prediction()
    export_top_50()
    save_histogram()
    evaluate_model()
