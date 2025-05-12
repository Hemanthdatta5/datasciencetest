import os
import glob
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, mean_squared_error, r2_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

warnings.filterwarnings("ignore")

def assess_risk(task, y_true, y_pred, y_dist):
    """AI-like risk assessment based on results and label balance"""
    if task == "classification":
        imbalance_ratio = y_dist.max() / y_dist.sum()
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        if imbalance_ratio > 0.9 or f1 < 0.5 or precision < 0.5:
            return "High Risk"
        elif imbalance_ratio > 0.75 or f1 < 0.7:
            return "Moderate Risk"
        else:
            return "Low Risk"

    else:  # regression
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        if r2 < 0.3 or mse > np.var(y_true) * 0.7:
            return "High Risk"
        elif r2 < 0.6:
            return "Moderate Risk"
        else:
            return "Low Risk"

def analyze_and_train(file_path):
    print(f"\n📄 File: {os.path.basename(file_path)}")
    try:
        df = pd.read_csv(file_path)

        df.dropna(thresh=len(df) * 0.5, axis=1, inplace=True)
        df.fillna(df.mean(numeric_only=True), inplace=True)

        if df.empty or df.shape[1] < 2:
            print("⚠️ Skipped: Not enough usable data.")
            return

        label_candidates = [col for col in df.columns if df[col].nunique() <= 20 and df[col].nunique() > 1]
        if not label_candidates:
            print("⚠️ Skipped: No clear label column found.")
            return

        label_col = label_candidates[0]
        X = df.drop(columns=[label_col])
        y = df[label_col]

        X = pd.get_dummies(X)

        if y.dtype == object or y.nunique() <= 10:
            task = "classification"
            y_encoded = LabelEncoder().fit_transform(y)
            model = RandomForestClassifier()
        else:
            task = "regression"
            y_encoded = y
            model = RandomForestRegressor()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Risk evaluation
        if task == "classification":
            y_dist = pd.Series(y_encoded).value_counts()
            risk_level = assess_risk(task, y_test, y_pred, y_dist)
        else:
            risk_level = assess_risk(task, y_test, y_pred, None)

        # Build report
        report_lines = [
            f"📄 File: {os.path.basename(file_path)}",
            f"🔢 Shape: {df.shape[0]} rows × {df.shape[1]} columns",
            f"🎯 Target Column: {label_col}",
            f"🧪 Task Type: {task}",
            f"⚠️ Risk Level: {risk_level}",
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
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            report_lines.append("✅ Regression Report:")
            report_lines.append(f"  - MSE : {mse:.4f}")
            report_lines.append(f"  - R²  : {r2:.4f}")

        # Save report
        os.makedirs('reports', exist_ok=True)
        report_name = os.path.splitext(os.path.basename(file_path))[0] + '_report.txt'
        report_path = os.path.join('reports', report_name)
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))

        print(f"📝 Report saved to: {report_path}")

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")

# Run the analysis
data_dir = '/home/hemanth/Desktop/torchetest/test/training/CSV/train'
csv_files = glob.glob(os.path.join(data_dir, '*.csv'))

if not csv_files:
    print(f"❌ No CSV files found in {data_dir}")
else:
    print(f"✅ Found {len(csv_files)} datasets. Running risk analysis...\n")
    for file in csv_files:
        analyze_and_train(file)
