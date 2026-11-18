"""Cross-attention fusion with availability masks and modality dropout (02 section 4.3).

P0 version: complete data path and masking semantics, small sizes. P3-C1 extends and
tunes it; the masking contract below must not change:

* Masked steps of a stream are replaced by a learned missing token before attention,
  so values at masked steps can never influence the output.
* Key-padding masks hide unavailable steps from the other stream's queries.
* A gate per stream, conditioned on availability, weights the attended streams; a
  stream with no available step in the window gets weight 0.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from driverisk.data import schema as S


class _CrossBlock(nn.Module):
    """Queries from one stream attend to keys/values of the other, then a small FFN."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, q: Tensor, kv: Tensor, kv_available: Tensor) -> tuple[Tensor, Tensor]:
        """q, kv: [B, T, d]; kv_available: [B, T] bool. Returns ([B, T, d], weights [B, T, T])."""
        pad = ~kv_available
        # A fully missing stream would make every key padded (NaN softmax). Its steps are
        # missing tokens already, so let attention see them; the gate zeroes the stream.
        pad = pad & ~pad.all(dim=1, keepdim=True)
        out, weights = self.attn(q, kv, kv, key_padding_mask=pad, need_weights=True)
        x = self.norm1(q + out)
        return self.norm2(x + self.ffn(x)), weights


class CrossAttentionFusion(nn.Module):
    """Inputs follow contract section 7; output `logits [B, K]`, `z [B, d_model]`, `gates [B, 2]`.

    Args:
        d_model: width of both streams after projection.
        n_heads: attention heads per cross block.
        n_classes: K, the number of window classes.
        dropout: dropout inside attention and FFN.
        modality_dropout_p: training-time probability of hiding each stream per sample.
        max_len: longest window (steps) for the learned positional embedding.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_classes: int = S.K_CLASSES,
        dropout: float = 0.1,
        modality_dropout_p: float = 0.15,
        max_len: int = S.WINDOW_STEPS,
    ) -> None:
        super().__init__()
        self.modality_dropout_p = modality_dropout_p
        # Each stream sees its features and per-feature NaN indicators (contract section 6.2).
        self.proj_v = nn.Linear(2 * S.DV, d_model)
        self.proj_m = nn.Linear(2 * S.DM, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.missing_v = nn.Parameter(torch.zeros(d_model))
        self.missing_m = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.missing_v, std=0.02)
        nn.init.normal_(self.missing_m, std=0.02)
        self.motion_queries_vision = _CrossBlock(d_model, n_heads, dropout)
        self.vision_queries_motion = _CrossBlock(d_model, n_heads, dropout)
        self.gate = nn.Linear(2 * d_model + 2, 2)
        self.pool_score = nn.Linear(d_model, 1)
        self.head = nn.Linear(d_model, n_classes)

    def _modality_dropout(self, mask_v: Tensor, mask_m: Tensor) -> tuple[Tensor, Tensor]:
        """Hide whole streams per sample; never both, and never the only available one."""
        if not self.training or self.modality_dropout_p <= 0:
            return mask_v, mask_m
        b = mask_v.shape[0]
        has_v, has_m = mask_v.any(1), mask_m.any(1)
        drop_v = (torch.rand(b, device=mask_v.device) < self.modality_dropout_p) & has_m
        drop_m = (torch.rand(b, device=mask_v.device) < self.modality_dropout_p) & has_v
        both = drop_v & drop_m
        keep_v = torch.rand(b, device=mask_v.device) < 0.5
        drop_v = drop_v & ~(both & keep_v)
        drop_m = drop_m & ~(both & ~keep_v)
        return mask_v & ~drop_v[:, None], mask_m & ~drop_m[:, None]

    def _embed(
        self, x: Tensor, nan: Tensor, mask: Tensor, proj: nn.Linear, missing: Tensor
    ) -> Tensor:
        t = x.shape[1]
        h = proj(torch.cat([x, nan.to(x.dtype)], dim=-1))
        h = torch.where(mask[..., None], h, missing.expand_as(h))
        return h + self.pos[:, :t]

    def forward(
        self, x_v: Tensor, x_m: Tensor, nan_v: Tensor, nan_m: Tensor, mask_v: Tensor, mask_m: Tensor
    ) -> dict[str, Tensor]:
        """x_v [B,T,16], x_m [B,T,14] float32; nan_* same shapes bool; mask_* [B,T] bool."""
        mask_v, mask_m = self._modality_dropout(mask_v, mask_m)
        h_v = self._embed(x_v, nan_v, mask_v, self.proj_v, self.missing_v)
        h_m = self._embed(x_m, nan_m, mask_m, self.proj_m, self.missing_m)

        a_m, w_mv = self.motion_queries_vision(
            h_m, h_v, mask_v
        )  # what visual evidence explains motion
        a_v, w_vm = self.vision_queries_motion(
            h_v, h_m, mask_m
        )  # what motion evidence supports vision

        avail_v = mask_v.float().mean(1, keepdim=True)  # [B, 1]
        avail_m = mask_m.float().mean(1, keepdim=True)
        gate_in = torch.cat([a_v.mean(1), a_m.mean(1), avail_v, avail_m], dim=-1)
        gates = (
            torch.sigmoid(self.gate(gate_in)) * torch.cat([avail_v > 0, avail_m > 0], -1).float()
        )
        fused = gates[:, 0, None, None] * a_v + gates[:, 1, None, None] * a_m  # [B, T, d]

        any_avail = mask_v | mask_m
        scores = self.pool_score(fused).squeeze(-1)
        scores = scores.masked_fill(~any_avail & any_avail.any(1, keepdim=True), float("-inf"))
        z = (torch.softmax(scores, dim=1)[..., None] * fused).sum(1)
        return {"logits": self.head(z), "z": z, "gates": gates, "attn_mv": w_mv, "attn_vm": w_vm}


def forward_batch(
    model: CrossAttentionFusion, batch: dict, device: torch.device | str = "cpu"
) -> dict[str, Tensor]:
    """Run the model on a `WindowDataset` batch dict."""
    keys = ("x_v", "x_m", "nan_v", "nan_m", "mask_v", "mask_m")
    return model(*(batch[k].to(device) for k in keys))
