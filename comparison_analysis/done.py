import pandas as pd 

df = pd.read_csv(r"C:\Users\arisp\Downloads\codeforces_all_datasets_combined.csv")

df2 = df.sample(frac = 1).iloc[:10000]

df2.to_csv(r"C:\Users\arisp\Downloads\smaller_codeforces.csv")