import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import numpy as np

# Enable progress bar for fitting
tqdm.pandas()

# === STEP 1: Load and Prepare Dataset ===
print("📥 Loading dataset...")
df = pd.read_csv("/home/hemanth/Desktop/training/combined_labeled.csv")

# Drop non-numeric or non-useful columns
if "Protocol Type" in df.columns:
    df = df.drop(columns=["Protocol Type"])

# Encode labels (Normal=0, Suspicious=1)
label_encoder = LabelEncoder()
df["Label"] = label_encoder.fit_transform(df["Label"])

X = df.drop(columns=["Label"])
y = df["Label"]

# === STEP 2: Split into Train/Test ===
print("🔀 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# === STEP 3: Train Model with Progress Bar ===
print("🧠 Training model...")
n_estimators = 100
# Create a forest manually with progress tracking
forest = []

for i in tqdm(range(n_estimators), desc="🌲 Training trees"):
    clf = RandomForestClassifier(n_estimators=1, warm_start=True, n_jobs=-1, random_state=42 + i)
    if i == 0:
        clf.fit(X_train, y_train)
    else:
        clf.set_params(n_estimators=i + 1)
        clf.fit(X_train, y_train)
    forest = clf  # Reuse the last model

# === STEP 4: Evaluate ===
print("\n📊 Model Evaluation:")
y_pred = forest.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["Normal", "Suspicious"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
