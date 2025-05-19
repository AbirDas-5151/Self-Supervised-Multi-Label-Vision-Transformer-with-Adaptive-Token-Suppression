import pandas as pd

csv_path = "/Users/user/Desktop/Transformer/CheXpert-v1.0/train.csv"
df = pd.read_csv(csv_path)

# Assuming the column is named 'Path' or 'path' - replace as needed
df['Path'] = df['Path'].str.replace('CheXpert-v1.0-small/', '', regex=False)

df.to_csv(csv_path, index=False)
print("Fixed paths in train.csv")
