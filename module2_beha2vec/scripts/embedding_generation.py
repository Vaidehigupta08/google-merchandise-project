import torch
import pandas as pd
import json
from tqdm import tqdm
import logging

from scripts.model_training import TransformerModelTrainer

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    def __init__(self, model, url_vocab_size=1000, max_seq_len=512):
        self.model = model
        self.model.eval()
        self.url_vocab_size = url_vocab_size
        self.max_seq_len = max_seq_len

    def generate(self, input_csv, output_file="embeddings.json"):
        df = pd.read_csv(input_csv)

        embeddings = {}

        for user_id, group in tqdm(df.groupby("user_id"), desc="Generating embeddings"):

            with open("url_vocab.json", "r") as f:
                vocab = json.load(f)

            url_ids = [
                vocab.get(url, 0)
                for url in group["pageview_URL"]
            ][:self.max_seq_len]

            url_tensor   = torch.tensor(url_ids).unsqueeze(0)
            theme_tensor = torch.zeros_like(url_tensor)
            type_tensor  = torch.zeros((1, 3))

            # 🔥 FIX: pass user_id so model uses user_embedding
            uid_tensor = torch.tensor([int(user_id)], dtype=torch.long)

            with torch.no_grad():
                emb = self.model(url_tensor, theme_tensor, type_tensor, user_id=uid_tensor)

            embeddings[user_id] = emb.squeeze().numpy().tolist()

        with open(output_file, "w") as f:
            json.dump(embeddings, f, indent=4)

        logger.info(f"Embeddings saved to {output_file}")

        return embeddings


# -------------------------------
# MAIN EXECUTION
# -------------------------------
if __name__ == "__main__":
    print("🚀 Generating Embeddings...")

    import pandas as pd
    df = pd.read_csv("input/data.csv")
    num_users = df["user_id"].nunique()

    # 🔥 FIX: pass user_vocab_size when loading trainer
    trainer = TransformerModelTrainer(url_vocab_size=7, user_vocab_size=num_users)
    trainer.load_model("model.pth")
    model = trainer.model

    generator = EmbeddingGenerator(model)
    embeddings = generator.generate("input/data.csv")

    print("✅ Embeddings Generated!")
    print("Total Users:", len(embeddings))
    print("Vector Size:", len(list(embeddings.values())[0]))
    print("💾 Saved in embeddings.json")