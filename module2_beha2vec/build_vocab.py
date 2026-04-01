import pandas as pd
import json

df = pd.read_csv("input/data.csv")

urls = df["pageview_URL"].unique()

vocab = {url: idx for idx, url in enumerate(urls)}

with open("url_vocab.json", "w") as f:
    json.dump(vocab, f, indent=4)

print("✅ Vocab created:", len(vocab))