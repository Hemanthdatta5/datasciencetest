import os
import glob
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

warnings.filterwarnings("ignore")

def analyze_and_train(file_path):
    print(f"\n📄 File: {os.path.basename(file_path)}")
    try:
        df = pd.read_csv(file_path)

        # Basic cleanup
        df.dropna(thresh=len(df) * 0.5, axis=1, inplace=True)
        df.fillna(df.mean(numeric_only=True), inplace=True)

        if df.empty or df.shape[1] < 2:
            print("⚠️ Skipped: Not enough usable data.")
            return

        # Detect likely label column
        label_candidates = [col for col in df.columns if df[col].nunique() <= 20 and df[col].nunique() > 1]
        if not label_candidates:
            print("⚠️ Skipped: No clear label column found.")
            return

        label_col = label_candidates[0]
        X = df.drop(columns=[label_col])
        y = df[label_col]

        # Encode categorical features
        X = pd.get_dummies(X)

        # Task detection and model choice
        if y.dtype == object or y.nunique() <= 10:
            task = "classification"
            y_encoded = LabelEncoder().fit_transform(y)
            model = RandomForestClassifier()
        else:
            task = "regression"
            y_encoded = y
            model = RandomForestRegressor()

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42
        )

        # Train the model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Start writing the report
        report_lines = [
            f"📄 File: {os.path.basename(file_path)}",
            f"🔢 Shape: {df.shape[0]} rows × {df.shape[1]} columns",
            f"🎯 Target Column: {label_col}",
            f"🧪 Task Type: {task}",
            "",
            "📊 Columns Summary:"
        ]

        for col in df.columns:
            unique = df[col].nunique()
            missing = df[col].isnull().sum()
            dtype = df[col].dtype
            report_lines.append(f"- {col} | Type: {dtype} | Unique: {unique} | Missing: {missing}")

        report_lines.append("")

        if task == "classification":
            report_lines.append("✅ Classification Report:")
            report_lines.append(classification_report(y_test, y_pred))
        else:
            report_lines.append("✅ Regression Report:")
            report_lines.append(f"  - MSE : {mean_squared_error(y_test, y_pred):.4f}")
            report_lines.append(f"  - R²  : {r2_score(y_test, y_pred):.4f}")

        # Save report
        os.makedirs('reports', exist_ok=True)
        report_name = os.path.splitext(os.path.basename(file_path))[0] + '_report.txt'
        report_path = os.path.join('reports', report_name)
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))

        print(f"📝 Report saved to: {report_path}")

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")

# 📁 Directory of CSV files
data_dir = '/home/hemanth/Desktop/torchetest/test/training/CSV/train'
csv_files = glob.glob(os.path.join(data_dir, '*.csv'))

if not csv_files:
    print(f"❌ No CSV files found in {data_dir}")
else:
    print(f"✅ Found {len(csv_files)} datasets. Starting analysis and training...\n")
    for file in csv_files:
        analyze_and_train(file)
