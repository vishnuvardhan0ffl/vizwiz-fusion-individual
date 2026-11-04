"""All Project 2.2 models behind one interface.

Every model is built as Model(d_q, d_v, **hyperparameters), declares the inputs it `needs`, and maps a
batch dict to one logit per pair (positive class = unanswerable). Vector models take `features`:
"vit_bert" (ViT [CLS] + BERT mean, the default) or "clip" (CLIP image + text embeddings). Config entries
can reuse a class under a new name with `base`, e.g. clip_early = early fusion on CLIP features.
Late fusion has no trainable weights, so it lives in train.py and combines saved unimodal predictions.
Fusion levels follow the early / late / intermediate taxonomy of Baltrusaitis, Ahuja & Morency (2019).
"""
import torch
import torch.nn as nn

VECTOR_KEYS = {"vit_bert": ("q_vec", "v_vec"), "clip": ("q_clip", "v_clip")}


def vector_keys(features):
    if features not in VECTOR_KEYS:
        raise ValueError(f"features must be one of {sorted(VECTOR_KEYS)}, got '{features}'")
    return VECTOR_KEYS[features]


class MLPHead(nn.Module):
    """LayerNorm -> Linear -> GELU -> Dropout -> Linear, returning one logit."""

    def __init__(self, d_in, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class QuestionOnly(nn.Module):
    """Baseline: question embedding only (measures language priors)."""

    def __init__(self, d_q, d_v, hidden=256, dropout=0.3, features="vit_bert"):
        super().__init__()
        self.key = vector_keys(features)[0]
        self.needs = (self.key,)
        self.head = MLPHead(d_q, hidden, dropout)

    def forward(self, b):
        return self.head(b[self.key])


class ImageOnly(nn.Module):
    """Baseline: image embedding only (measures photo-quality cues)."""

    def __init__(self, d_q, d_v, hidden=256, dropout=0.3, features="vit_bert"):
        super().__init__()
        self.key = vector_keys(features)[1]
        self.needs = (self.key,)
        self.head = MLPHead(d_v, hidden, dropout)

    def forward(self, b):
        return self.head(b[self.key])


class EarlyFusion(nn.Module):
    """Concatenate question and image embeddings, then one MLP."""

    def __init__(self, d_q, d_v, hidden=256, dropout=0.3, features="vit_bert"):
        super().__init__()
        self.needs = vector_keys(features)
        self.head = MLPHead(d_q + d_v, hidden, dropout)

    def forward(self, b):
        q, v = (b[k] for k in self.needs)
        return self.head(torch.cat([q, v], -1))


class CrossAttentionBlock(nn.Module):
    """Question tokens attend to image regions, then image regions attend to the updated question tokens.
    Scaled dot-product attention (Vaswani et al. 2017) used across modalities as in Tsai et al. (2019)."""

    def __init__(self, d_model, heads, dropout):
        super().__init__()
        self.q_from_v = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.v_from_q = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.ln_q, self.ln_v = nn.LayerNorm(d_model), nn.LayerNorm(d_model)

    def forward(self, q, q_mask, v):
        q = self.ln_q(q + self.q_from_v(q, v, v)[0])
        v = self.ln_v(v + self.v_from_q(v, q, q, key_padding_mask=~q_mask)[0])
        return q, v


class CrossAttentionFusion(nn.Module):
    """Intermediate fusion: stacked bidirectional cross-attention over question tokens and image regions."""
    needs = ("q_tok", "q_mask", "v_tok")

    def __init__(self, d_q, d_v, d_model=256, heads=4, layers=1, dropout=0.1):
        super().__init__()
        self.q_proj, self.v_proj = nn.Linear(d_q, d_model), nn.Linear(d_v, d_model)
        self.blocks = nn.ModuleList(CrossAttentionBlock(d_model, heads, dropout) for _ in range(layers))
        self.head = MLPHead(2 * d_model, 128, 0.3)

    def forward(self, b):
        q, v, mask = self.q_proj(b["q_tok"]), self.v_proj(b["v_tok"]), b["q_mask"]
        for block in self.blocks:
            q, v = block(q, mask, v)
        m = mask.unsqueeze(-1).float()
        q_pool = (q * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(torch.cat([q_pool, v.mean(1)], -1))


class GatedFusion(nn.Module):
    """Gated Multimodal Unit (Arevalo et al. 2017) with modality dropout.

    The gate z (per pair, per hidden unit) weights the question representation by z and the photo
    representation by 1 - z. Modality dropout blanks the question or the photo for a share of training
    pairs so the gate learns to cope when one input is unreliable. The mean gate value per pair is kept
    in `last_gate_q` for analysis (higher = more weight on the question).
    """
    def __init__(self, d_q, d_v, hidden=256, dropout=0.3, modality_dropout=0.15, features="vit_bert"):
        super().__init__()
        self.needs = vector_keys(features)
        self.h_q, self.h_v = nn.Linear(d_q, hidden), nn.Linear(d_v, hidden)
        self.gate = nn.Linear(d_q + d_v, hidden)
        self.head = MLPHead(hidden, hidden, dropout)
        self.p = modality_dropout
        self.last_gate_q = None

    def forward(self, b):
        x_q, x_v = (b[k] for k in self.needs)
        if self.training and self.p > 0:
            u = torch.rand(len(x_q), 1, device=x_q.device)
            x_q = x_q * (u >= self.p / 2)                       # first half of the dropped share: no question
            x_v = x_v * ((u < self.p / 2) | (u >= self.p))      # second half: no photo
        z = torch.sigmoid(self.gate(torch.cat([x_q, x_v], -1)))
        h = z * torch.tanh(self.h_q(x_q)) + (1 - z) * torch.tanh(self.h_v(x_v))
        self.last_gate_q = z.mean(-1).detach()
        return self.head(h)


MODELS = {
    "question_only": QuestionOnly,
    "image_only": ImageOnly,
    "early": EarlyFusion,
    "cross_attention": CrossAttentionFusion,
    "gated": GatedFusion,
}


def build_model(name, dims, **hparams):
    """Build a model from its config entry. `base` picks the class (default: the name itself);
    `features` picks the input vectors, and with it the input sizes from `dims`."""
    hp = dict(hparams)
    base = hp.pop("base", name)
    if base not in MODELS:
        raise ValueError(f"unknown model '{base}'; choose from {sorted(MODELS)} (late fusion: see train.py)")
    features = hp.get("features", "vit_bert")
    if features == "clip":
        if "d_q_clip" not in dims:
            raise ValueError(f"'{name}' needs CLIP features, but the hand-off has no clip_txt / clip_img arrays")
        d_q, d_v = dims["d_q_clip"], dims["d_v_clip"]
    else:
        d_q, d_v = dims["d_q"], dims["d_v"]
    if base == "cross_attention" and features != "vit_bert":
        raise ValueError("cross_attention needs token features; only vit_bert has them")
    return MODELS[base](d_q, d_v, **hp)
