import pandas as pd

# Load prediction results
df = pd.read_csv("/home/hemanth/Desktop/training/suspicious_predictions.csv")

# Drop label and non-numeric columns
df_numeric = df.drop(columns=["Predicted_Label", "Protocol Type"], errors="ignore")

# Compute correlation with Suspicion_Score
correlations = df_numeric.corr(numeric_only=True)["Suspicion_Score"].sort_values(ascending=False)

# Save to file
correlation_file = "/home/hemanth/Desktop/training/suspicion_score_correlations.csv"
correlations.to_csv(correlation_file, header=True)

print(f"✅ Correlation scores saved to: {correlation_file}")
