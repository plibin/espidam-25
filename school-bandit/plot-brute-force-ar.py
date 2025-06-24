import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

csv_path = "brute-force-ar.csv"
df = pd.read_csv(csv_path)

plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")

sns.boxplot(
    x=df["arm_index"].astype("category"),  # cast to categorical on the fly
    y="attack_rate",
    data=df,
    palette="vlag"
)

plt.xticks(rotation=90, fontsize=8)
plt.xlabel("Arm Index")
plt.ylabel("Attack Rate")
plt.title("Distribution of Attack Rates by Arm (0–71)")

plt.tight_layout()

plt.show()
