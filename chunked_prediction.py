import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib  # for saving/loading the model
from tqdm import tqdm

# === CONFIGURATION ===
DATA_PATH = "/home/hemanth/Desktop/training/combined_labeled.csv"
OUTPUT_PATH = "/home/hemanth/Desktop/training/suspicious_predictions.csv"
CHUNK_SIZE = 500_000

# === STEP 1: Load model (or train a small one if needed) ===
# Train on small subset for demo (or replace with joblib.load() if pre-trained model exists)
print("📥 Loading sample to train model for chunked prediction...")
sample_df = pd.read_csv(DATA_PATH, nrows=250_000)

if "Protocol Type" in sample_df.columns:
    sample_df = sample_df.drop(columns=["Protocol Type"])

label_encoder = LabelEncoder()
sample_df["Label"] = label_encoder.fit_transform(sample_df["Label"])

X_sample = sample_df.drop(columns=["Label"])
y_sample = sample_df["Label"]

print("🧠 Training model...")
model = RandomForestClassifier(n_estimators=50, n_jobs=1, random_state=42)
model.fit(X_sample, y_sample)

# Save model for reuse
joblib.dump(model, "rf_model.joblib")

# === STEP 2: Chunked Prediction ===
print("🔍 Starting chunked prediction...")
chunk_iter = pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE)
first_chunk = True
model = joblib.load("rf_model.joblib")

for i, chunk in enumerate(tqdm(chunk_iter, desc="📦 Processing chunks")):
    # Preprocess
    if "Protocol Type" in chunk.columns:
        chunk = chunk.drop(columns=["Protocol Type"])

    orig_chunk = chunk.copy()  # Save original rows for output

    # Encode label to drop it for prediction
    chunk["Label"] = label_encoder.transform(chunk["Label"])
    X = chunk.drop(columns=["Label"])

    # Predict label and probability
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]  # Probability of being Suspicious

    # Add predictions to output
    orig_chunk["Predicted_Label"] = label_encoder.inverse_transform(preds)
    orig_chunk["Suspicion_Score"] = probs

    # Append to CSV
    mode = "w" if first_chunk else "a"
    header = first_chunk
    orig_chunk.to_csv(OUTPUT_PATH, mode=mode, index=False, header=header)
    first_chunk = False

    print(f"✅ Saved chunk {i+1}")

print(f"\n🎉 All chunks processed and saved to {OUTPUT_PATH}")
