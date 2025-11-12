import pandas as pd
import os

data_dir = "C:/Users/HP/Desktop/project_ids_/data"

files = [
    "reduced_data_1.csv",
    "reduced_data_2.csv",
    "reduced_data_3.csv",
    "reduced_data_4.csv"
]

dfs = []
for f in files:
    path = os.path.join(data_dir, f)
    print(f"Reading {path} ...")
    df = pd.read_csv(path, low_memory=False)
    dfs.append(df)

# Combine all dataframes
combined_df = pd.concat(dfs, ignore_index=True)
print(f"Combined shape: {combined_df.shape}")

# Shuffle the data for randomness
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to new CSV
combined_df.to_csv(os.path.join(data_dir, "combined_data.csv"), index=False)
print("✅ Combined dataset saved to combined_data.csv")
