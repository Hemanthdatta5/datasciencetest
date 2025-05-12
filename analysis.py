import pandas as pd

# Load the predictions file
df = pd.read_csv("/home/hemanth/Desktop/training/suspicious_predictions.csv")

# Sort records by safest (lowest suspicion score)
safest = df.sort_values("Suspicion_Score", ascending=True).head(50)

# Sort records by riskiest (highest suspicion score)
riskiest = df.sort_values("Suspicion_Score", ascending=False).head(50)

# Save both results
safest.to_csv("/home/hemanth/Desktop/training/top_50_safest.csv", index=False)
riskiest.to_csv("/home/hemanth/Desktop/training/top_50_riskiest.csv", index=False)

print("✅ Saved top 50 safest and riskiest entries.")
