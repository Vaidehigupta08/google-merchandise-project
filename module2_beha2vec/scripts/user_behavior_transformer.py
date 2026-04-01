import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Basic positional encoding for sequences.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()*(-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos*div)
        pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer('pe', pe)

    def forward(self, x):
        slen = x.size(1)
        return x + self.pe[:slen,:].unsqueeze(0)

class UserBehaviorTransformer(nn.Module):
    """
    Transformer with:
      - URL embedding
      - 🔥 user_id embedding (FIX: ensures unique embedding per user even with same sequence)
      - optional theme embedding
      - optional type embedding
      - positional encoding
      - multi-head attention
    """
    def __init__(self, url_vocab_size=1000, theme_vocab_size=None, type_dim=None,
                 embedding_dim=128, n_heads=4, n_layers=2, combined_dim=128,
                 max_seq_len=512,
                 # 🔥 NEW: user vocab size
                 user_vocab_size=None):
        super().__init__()
        self.url_embedding = nn.Embedding(url_vocab_size, embedding_dim)
        self.theme_embedding = nn.Embedding(theme_vocab_size, embedding_dim) if theme_vocab_size else None
        self.type_fc = nn.Sequential(
            nn.Linear(type_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        ) if type_dim else None

        # 🔥 FIX: user_id embedding — each user gets unique learned bias
        self.user_embedding = nn.Embedding(user_vocab_size, embedding_dim) if user_vocab_size else None

        self.pos_enc = PositionalEncoding(embedding_dim, max_seq_len)
        enc_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.fc_out = nn.Linear(embedding_dim, combined_dim)

    def forward(self, url_seq, theme_seq=None, type_vec=None, user_id=None):
        ue = self.url_embedding(url_seq)
        ue = self.pos_enc(ue)
        ue = self.encoder(ue)
        rep = ue.mean(dim=1)

        if self.theme_embedding and theme_seq is not None:
            te = self.theme_embedding(theme_seq)
            te = self.pos_enc(te)
            te = self.encoder(te)
            rep += te.mean(dim=1)

        if self.type_fc and type_vec is not None:
            rep += self.type_fc(type_vec)

        # 🔥 FIX: add user embedding so same sequence → different output per user
        if self.user_embedding is not None and user_id is not None:
            user_emb = self.user_embedding(user_id)   # [B, embedding_dim]
            rep = rep + user_emb

        return self.fc_out(rep)