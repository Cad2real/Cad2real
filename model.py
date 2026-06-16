from typing import Dict, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- Small utilities ---------------------------------------------------------

class NumericProjector(nn.Module):
    """
    Projects a numeric feature vector R^{in_dim} -> R^{d_model}.
    Applies LayerNorm on input dimension for some scale invariance.
    """
    def __init__(self, in_dim: int, d_model: int):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_dim)
        return self.proj(self.ln(x))


class OptionalGlobalImageEncoder(nn.Module):
    """
    Placeholder image encoder that returns a single global token (B, d_model)
    if an image tensor is provided. If None, returns None.

    You can swap this for a ViT or a CNN->pool block later.
    """
    def __init__(self, d_model: int):
        super().__init__()
        # Minimal conv encoder for a global embedding; replace with a real backbone as needed.
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.proj = nn.Linear(128, d_model)

    def forward(self, img: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if img is None:
            return None
        # img: (B, 3, H, W), values assumed 0..1
        h = self.conv(img).flatten(1)         # (B, 128)
        g = self.proj(h)                      # (B, d_model)
        return g


# ---- Main model --------------------------------------------------------------

class MultiModalActTransformer(nn.Module):
    """
    Fuses object poses, pre-timestamp signals (hands + arms), optional image,
    and a task token. Predicts per-actor actions via actor-specific heads.

    Tokens in the sequence:
      [TASK] [D1] [D2] [PRE_NOVA2] [PRE_NOVA5] [PRE_LEFT] [PRE_RIGHT] [IMG?] [Q_NOVA2] [Q_NOVA5] [Q_LEFT] [Q_RIGHT] [Q_THREAD] [Q_STOP]
    """
    ACTOR_ORDER = ["nova2", "nova5", "LeftHand", "RightHand", "thread", "stop"]

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_tasks: int = 64,      # task vocab size (e.g., draw/place/...)
        use_images: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_images = use_images

        # Embeddings / projectors for numeric tokens
        self.task_embed = nn.Embedding(num_tasks, d_model)

        self.d1_proj = NumericProjector(7, d_model)
        self.d2_proj = NumericProjector(7, d_model)
        self.pre_nova2_proj = NumericProjector(6, d_model)
        self.pre_nova5_proj = NumericProjector(6, d_model)
        self.pre_left_proj  = NumericProjector(10, d_model)
        self.pre_right_proj = NumericProjector(10, d_model)

        # Optional image -> one global token
        self.img_enc = OptionalGlobalImageEncoder(d_model) if use_images else None

        # Learnable query tokens: one per actor
        self.num_queries = len(self.ACTOR_ORDER)
        self.query_tokens = nn.Parameter(torch.randn(self.num_queries, d_model) * 0.02)

        # Positional embeddings (max tokens ~ 1+2+4+1+6 = 14; allocate a bit more)
        self.max_tokens = 24
        self.pos_embed = nn.Parameter(torch.randn(self.max_tokens, d_model) * 0.02)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,  # Transformer expects (S, B, E)
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Output heads
        self.head_nova2    = nn.Linear(d_model, 6)
        self.head_nova5    = nn.Linear(d_model, 6)
        self.head_left     = nn.Linear(d_model, 10)
        self.head_right    = nn.Linear(d_model, 10)
        self.head_thread   = nn.Linear(d_model, 3)  # classes: 0=none, 1=left, 2=right
        self.head_stop     = nn.Linear(d_model, 1)  # sigmoid later

        # Final norm to stabilize
        self.final_norm = nn.LayerNorm(d_model)

    def _build_sequence(
        self,
        task_id: torch.Tensor,         # (B,)
        d1_pose: torch.Tensor,         # (B,7)
        d2_pose: torch.Tensor,         # (B,7)
        pre_nova2: torch.Tensor,       # (B,6)
        pre_nova5: torch.Tensor,       # (B,6)
        pre_left: torch.Tensor,        # (B,10)
        pre_right: torch.Tensor,       # (B,10)
        img: Optional[torch.Tensor],   # (B,3,H,W) or None
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Returns:
          seq: (S, B, d_model)  where S = token_len
          q_pos: list of int indices (length = num_queries) of query tokens in seq
        """
        B = d1_pose.shape[0]
        tokens = []

        # Core tokens
        t_task    = self.task_embed(task_id)                # (B,d)
        t_d1      = self.d1_proj(d1_pose)                  # (B,d)
        t_d2      = self.d2_proj(d2_pose)                  # (B,d)
        t_pre_n2  = self.pre_nova2_proj(pre_nova2)         # (B,d)
        t_pre_n5  = self.pre_nova5_proj(pre_nova5)         # (B,d)
        t_pre_l   = self.pre_left_proj(pre_left)           # (B,d)
        t_pre_r   = self.pre_right_proj(pre_right)         # (B,d)

        tokens.extend([t_task, t_d1, t_d2, t_pre_n2, t_pre_n5, t_pre_l, t_pre_r])

        # Optional image token
        if self.use_images:
            t_img = self.img_enc(img)                      # (B,d) or None
            if t_img is not None:
                tokens.append(t_img)

        # Add query tokens at the end; repeat across batch
        q_pos = []
        for _ in self.ACTOR_ORDER:
            q_pos.append(len(tokens))
            tokens.append(self.query_tokens.unsqueeze(0).expand(B, -1, -1)[:, q_pos[-1]-len(tokens), :])  # placeholder

        # The line above is a bit awkward; simpler: manually expand per query
        tokens = tokens[:-self.num_queries]  # remove the awkward placeholders
        q_pos = []
        for qi in range(self.num_queries):
            # (d,) -> (B,d)
            q_tok = self.query_tokens[qi].unsqueeze(0).expand(B, -1)
            tokens.append(q_tok)
            q_pos.append(len(tokens) - 1)

        # Stack -> (B, S, d)
        seq_bsd = torch.stack(tokens, dim=1)
        S = seq_bsd.shape[1]

        # Positional add
        pos = self.pos_embed[:S].unsqueeze(0)              # (1,S,d)
        seq_bsd = seq_bsd + pos

        # To (S,B,d) for Transformer (batch_first=False)
        seq_sbd = seq_bsd.transpose(0, 1)                  # (S,B,d)
        return seq_sbd, q_pos

    def forward(
        self,
        task_id: torch.Tensor,
        d1_pose: torch.Tensor,
        d2_pose: torch.Tensor,
        pre_nova2: torch.Tensor,
        pre_nova5: torch.Tensor,
        pre_left: torch.Tensor,
        pre_right: torch.Tensor,
        img: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns a dict of predictions:
          'nova2': (B,6), 'nova5': (B,6), 'LeftHand': (B,10), 'RightHand': (B,10),
          'thread_logits': (B,3), 'stop_logit': (B,1)
        """
        seq_sbd, q_pos = self._build_sequence(task_id, d1_pose, d2_pose, pre_nova2, pre_nova5, pre_left, pre_right, img)
        enc = self.encoder(seq_sbd)                        # (S,B,d)
        enc = self.final_norm(enc)

        # Gather query hidden states: (B,d) per query
        q_hiddens = [enc[pos_idx, :, :] for pos_idx in q_pos]   # each (B,d)
        # Order matches ACTOR_ORDER
        h_nova2, h_nova5, h_left, h_right, h_thread, h_stop = q_hiddens

        # Heads
        out = {
            "nova2":        self.head_nova2(h_nova2),
            "nova5":        self.head_nova5(h_nova5),
            "LeftHand":     self.head_left(h_left),
            "RightHand":    self.head_right(h_right),
            "thread_logits":self.head_thread(h_thread),
            "stop_logit":   self.head_stop(h_stop),
        }
        return out







