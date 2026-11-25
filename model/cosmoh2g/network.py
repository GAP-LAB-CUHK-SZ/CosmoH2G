import numpy as np
import torch
import torch.nn as nn

from cosmoh2g.pointtransformer import PointEncoder

LATENT_DIM = 512
TIME_EMB_DIM = 128
NUM_LAYERS = 5
NUM_HEADS = 8
DIM_FEEDFORWARD = 1024
DROPOUT = 0.1


def sinusoidal_table(max_len, dim):
    pe = torch.zeros(max_len, dim)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0).transpose(0, 1)


class PositionalEncoding(nn.Module):
    def __init__(self, dim, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.register_buffer("pe", sinusoidal_table(max_len, dim))

    def forward(self, x):
        return self.dropout(x + self.pe[: x.shape[0]])


class TimestepEmbedder(nn.Module):
    def __init__(self, d_model, time_embed_dim, max_len=1000):
        super().__init__()
        self.register_buffer("pe", sinusoidal_table(max_len, time_embed_dim))
        self.time_embed = nn.Sequential(
            nn.Linear(time_embed_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, timesteps):
        return self.time_embed(self.pe[timesteps])


class CosmoH2GDenoiser(nn.Module):
    """Transformer denoiser conditioned on the object, the contact map and the hand sequence.

    stage 1: x is the start keyframe, [B, 1, 9] = center/left/right keypoints.
    stage 2: x is the finger-direction sequence, [B, T, 6] = left/right unit directions,
             additionally conditioned on the start and end finger directions.
    """

    def __init__(self, stage):
        super().__init__()
        if stage not in (1, 2):
            raise ValueError(f"stage must be 1 or 2, got {stage}")
        self.stage = stage
        motion_dim = 9 if stage == 1 else 6

        self.timestep_embedder = TimestepEmbedder(LATENT_DIM, TIME_EMB_DIM)
        self.cond_token = nn.Parameter(torch.zeros(1, 1, LATENT_DIM))
        self.hand_encoder = PointEncoder(point_feat_dim=0)
        self.hand_adapter = nn.Linear(256, LATENT_DIM)
        self.obj_encoder = PointEncoder(point_feat_dim=1)
        self.obj_adapter = nn.Linear(256, LATENT_DIM)

        self.motion_adapter = nn.Linear(motion_dim, LATENT_DIM)
        if stage == 1:
            self.output_motion_adapter = self.motion_adapter
        else:
            self.output_motion_adapter = nn.Linear(motion_dim, LATENT_DIM)
        self.positional_encoder = PositionalEncoding(LATENT_DIM, dropout=DROPOUT)
        self.self_attn_layer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=LATENT_DIM,
                nhead=NUM_HEADS,
                dim_feedforward=DIM_FEEDFORWARD,
                dropout=DROPOUT,
                activation="gelu",
                batch_first=True,
            ),
            enable_nested_tensor=False,
            num_layers=NUM_LAYERS,
        )
        self.motion_layer = nn.Linear(LATENT_DIM, motion_dim)

    def forward(self, x, timesteps, c_obj_pts, c_hoi_cont, c_hand_seq, c_gripper_start=None, c_gripper_end=None):
        batch = x.shape[0]
        tokens = [
            self.timestep_embedder(timesteps),
            self.cond_token.expand(batch, -1, -1),
            self.obj_adapter(self.obj_encoder(c_obj_pts.float(), c_hoi_cont.float())),
        ]
        hand_feat = torch.zeros((batch, c_hand_seq.shape[2], 0), device=x.device)
        hand = [self.hand_encoder(c_hand_seq[:, t].float(), hand_feat) for t in range(c_hand_seq.shape[1])]
        tokens.append(self.hand_adapter(torch.cat(hand, dim=1)))
        if self.stage == 2:
            tokens.append(self.motion_adapter(c_gripper_start.float().reshape(batch, 1, -1)))
            tokens.append(self.motion_adapter(c_gripper_end.float().reshape(batch, 1, -1)))
        tokens.append(self.output_motion_adapter(x.float()))

        h = torch.cat(tokens, dim=1)
        h = self.positional_encoder(h.permute(1, 0, 2)).permute(1, 0, 2)
        h = self.self_attn_layer(h)
        return self.motion_layer(h[:, -x.shape[1]:])
