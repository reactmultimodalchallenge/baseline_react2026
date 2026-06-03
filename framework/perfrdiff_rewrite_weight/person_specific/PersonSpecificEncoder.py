import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super().__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False
        pos = torch.arange(0, max_len, device=device).float().unsqueeze(1)
        idx = torch.arange(0, d_model, step=2, device=device).float()
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (idx / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (idx / d_model)))

    def forward(self, x):
        return self.encoding[:x.shape[1], :].unsqueeze(0)


class Transformer(nn.Module):
    def __init__(
            self,
            device,
            in_features,
            embed_dim,
            num_heads,
            num_layers,
            mlp_dim,
            seq_len,
            proj_dim,
            proj_head="mlp",
            drop_prob=0.1,
            max_len=5000,
            pos_encoding="absolute",
            embed_layer="linear",
    ):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim if embed_layer == "linear" else in_features
        self.embed_layer = nn.Linear(in_features, embed_dim) if embed_layer == "linear" else nn.Identity()

        if pos_encoding == "learnable":
            self.pos_embed = nn.Parameter(torch.zeros(1, 1 + seq_len, self.embed_dim))
        elif pos_encoding == "absolute":
            self.pos_embed = PositionalEncoding(d_model=self.embed_dim, max_len=max_len, device=device)
        else:
            raise NotImplementedError(f"Unsupported positional encoding: {pos_encoding}")
        self.pos_encoding = pos_encoding

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.dropout = nn.Dropout(p=drop_prob)

        if proj_head == "linear":
            self.proj_head = nn.Linear(embed_dim, proj_dim)
        elif proj_head == "mlp":
            self.proj_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim, proj_dim),
            )
        else:
            self.proj_head = nn.Identity()

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.embed_layer(x)
        x = torch.cat([self.cls_token.expand(batch_size, -1, -1), x], dim=1)
        x = x + (self.pos_embed(x) if self.pos_encoding == "absolute" else self.pos_embed)
        x = self.dropout(x)
        x = self.transformer(x)
        feat = x[:, 0, :]
        proj = F.normalize(self.proj_head(feat), dim=-1)
        return feat, proj

    def get_model_name(self):
        return self.__class__.__name__
