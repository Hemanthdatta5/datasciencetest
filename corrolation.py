import pandas as pd

# Load the prediction data
df = pd.read_csv("/home/hemanth/Desktop/training/suspicious_predictions.csv")

# Drop non-numeric or irrelevant columns
df_numeric = df.drop(columns=["Predicted_Label", "Protocol Type"], errors="ignore")

# Compute correlation of all features with Suspicion_Score
correlations = df_numeric.corr(numeric_only=True)["Suspicion_Score"].sort_values(ascending=False)

# Save and display
correlations.to_csv("/home/hemanth/Desktop/training/suspicion_score_correlations.csv")
print("📈 Top correlated features with Suspicion Score:")
print(correlations.head(10))
