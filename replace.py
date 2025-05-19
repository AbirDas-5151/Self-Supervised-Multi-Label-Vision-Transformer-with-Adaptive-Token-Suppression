import pandas as pd

csv_path = "/Users/user/Desktop/Transformer/CheXpert-v1.0/valid_fixed.csv"
df = pd.read_csv(csv_path)

# Assuming column name is 'Path' (check your CSV)
df['Path'] = df['Path'].str.replace('^valid/', '', regex=True)

df.to_csv("/Users/user/Desktop/Transformer/CheXpert-v1.0/valid_fixed_noprefix.csv", index=False)
