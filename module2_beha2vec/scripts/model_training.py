import json
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import logging
import random
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from scripts.user_behavior_transformer import UserBehaviorTransformer

logger = logging.getLogger(__name__)

class TripletDataset(Dataset):
    """
    Builds (anchor, positive, negative) triplets from user sequences.
    Each user_id's data is split in half: anchor vs. positive.
    A negative comes from a different user.
    """

    def __init__(
        self,
        user_sequences_csv: str,
        url_vocab_size: int = 1000,
        theme_vocab_size: int = 50,
        use_theme: bool = False,
        use_type: bool = False,
        max_seq_len: int = 512
    ):
        self.df = pd.read_csv(user_sequences_csv)
        with open("url_vocab.json", "r") as f:
            self.vocab = json.load(f)
        self.url_vocab_size = url_vocab_size
        self.theme_vocab_size = theme_vocab_size
        self.use_theme = use_theme
        self.use_type = use_type
        self.max_seq_len = max_seq_len

        self.user_dict = {}
        for _, row in self.df.iterrows():
            uid = row['user_id']
            if uid not in self.user_dict:
                self.user_dict[uid] = []
            self.user_dict[uid].append(row)

        for uid in self.user_dict:
            self.user_dict[uid].sort(key=lambda x: x['timestamp'])

        self.all_users = list(self.user_dict.keys())

    def __len__(self):
        return len(self.all_users)

    def __getitem__(self, idx):
        uid = self.all_users[idx]
        data = self.user_dict[uid]

        if len(data) < 2:
            anchor_rows = data
            positive_rows = data
        else:
            mid = len(data) // 2
            anchor_rows = data[:mid]
            positive_rows = data[mid:]

        neg_uid = random.choice(self.all_users)
        while neg_uid == uid and len(self.all_users) > 1:
            neg_uid = random.choice(self.all_users)
        negative_rows = self.user_dict[neg_uid]

        anchor_input   = self._build_input(anchor_rows)
        positive_input = self._build_input(positive_rows)
        negative_input = self._build_input(negative_rows)

        # 🔥 FIX: pass user_id tensors so model can use user_embedding
        anchor_uid_tensor   = torch.tensor([int(uid)],     dtype=torch.long)
        positive_uid_tensor = torch.tensor([int(uid)],     dtype=torch.long)  # same user = positive
        negative_uid_tensor = torch.tensor([int(neg_uid)], dtype=torch.long)

        return (anchor_input, positive_input, negative_input,
                anchor_uid_tensor, positive_uid_tensor, negative_uid_tensor)

    def _build_input(self, rows):
        if len(rows) == 0:
            return (
                torch.tensor([], dtype=torch.long),
                torch.tensor([], dtype=torch.long),
                torch.zeros(3, dtype=torch.float)
            )

        url_ids = [
            self.vocab.get(r['pageview_URL'], 0)
            for r in rows
        ]
        url_ids = url_ids[: self.max_seq_len]
        url_seq = torch.tensor(url_ids, dtype=torch.long)

        theme_seq = torch.tensor([], dtype=torch.long)
        if self.use_theme and 'page_theme' in rows[0]:
            t = [hash(r['page_theme']) % self.theme_vocab_size for r in rows]
            t = t[: self.max_seq_len]
            theme_seq = torch.tensor(t, dtype=torch.long)

        type_vec = torch.zeros(3, dtype=torch.float)
        if self.use_type and 'page_type' in rows[0]:
            tv = [0, 0, 0]
            for rr in rows:
                ptype = rr['page_type'].lower()
                if "blog" in ptype:
                    tv[0] += 1
                elif "service" in ptype:
                    tv[1] += 1
                elif "home" in ptype:
                    tv[2] += 1
            type_vec = torch.tensor(tv, dtype=torch.float)

        return (url_seq, theme_seq, type_vec)


def triplet_collate_fn(batch):
    # 🔥 FIX: unpack 6 items now (3 inputs + 3 user_id tensors)
    anchors, positives, negatives, a_uids, p_uids, n_uids = zip(*batch)

    def pad_triplet(triplets):
        url_seqs   = [x[0] for x in triplets]
        theme_seqs = [x[1] for x in triplets]
        type_vecs  = [x[2] for x in triplets]

        padded_urls   = pad_sequence(url_seqs,   batch_first=True, padding_value=0)
        padded_themes = pad_sequence(theme_seqs, batch_first=True, padding_value=0)
        stacked_type  = torch.stack(type_vecs, dim=0)

        return (padded_urls, padded_themes, stacked_type)

    anchor_out   = pad_triplet(anchors)
    positive_out = pad_triplet(positives)
    negative_out = pad_triplet(negatives)

    # 🔥 FIX: stack user_id tensors
    a_uid_batch = torch.cat(a_uids, dim=0)
    p_uid_batch = torch.cat(p_uids, dim=0)
    n_uid_batch = torch.cat(n_uids, dim=0)

    return anchor_out, positive_out, negative_out, a_uid_batch, p_uid_batch, n_uid_batch


class TransformerModelTrainer:
    def __init__(
        self,
        url_vocab_size,
        theme_vocab_size=None,
        type_dim=None,
        embedding_dim=128,
        n_heads=4,
        n_layers=2,
        combined_dim=128,
        learning_rate=1e-4,
        max_seq_len=512,
        # 🔥 NEW
        user_vocab_size=None
    ):
        self.model = UserBehaviorTransformer(
            url_vocab_size, theme_vocab_size, type_dim,
            embedding_dim, n_heads, n_layers, combined_dim, max_seq_len,
            user_vocab_size=user_vocab_size   # 🔥 pass through
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.TripletMarginLoss(margin=1.0)

    def train(self, dataloader, epochs, output_file="model.pth"):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            # 🔥 FIX: unpack 6 items
            for anchor, positive, negative, a_uid, p_uid, n_uid in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                # 🔥 FIX: pass user_id to model
                a = self.model(*anchor, user_id=a_uid)
                p = self.model(*positive, user_id=p_uid)
                n = self.model(*negative, user_id=n_uid)

                loss = self.criterion(a, p, n)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        torch.save(self.model.state_dict(), output_file)
        logger.info(f"Model saved -> {output_file}")

    def load_model(self, file_path):
        self.model.load_state_dict(torch.load(file_path))
        self.model.eval()
        logger.info(f"Loaded model from {file_path}")


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    print("🚀 Starting Training...")

    with open("url_vocab.json", "r") as f:
        vocab = json.load(f)

    vocab_size = len(vocab)
    print("Vocab Size:", vocab_size)

    dataset = TripletDataset(
        user_sequences_csv="input/data.csv",
        url_vocab_size=vocab_size
    )

    # 🔥 FIX: user_vocab_size = total users
    num_users = len(dataset.all_users)
    print("Total Users:", num_users)

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=triplet_collate_fn
    )

    trainer = TransformerModelTrainer(
        url_vocab_size=vocab_size,
        user_vocab_size=num_users   # 🔥 pass here
    )

    trainer.train(dataloader, epochs=10)

    print("✅ Training Done!")