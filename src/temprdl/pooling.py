"""Extracted original Temp-RDL definitions; only imports/packaging are adapted."""
from __future__ import annotations

import torch


_BN_FREEZE_DIAGNOSTIC_PRINTED = False

class freeze_batchnorm_running_stats:
    """Freeze only BatchNorm running statistics.

    Gradients through interval forwards remain enabled. BatchNorm
    affine parameters and all GNN parameters can still receive
    gradients; running_mean/running_var are restored to their
    previous training/evaluation mode afterwards.
    """

    def __init__(self, module):
        self.module = module
        self.states = []

    def __enter__(self):
        global _BN_FREEZE_DIAGNOSTIC_PRINTED
        bn_base = torch.nn.modules.batchnorm._BatchNorm
        for submodule in self.module.modules():
            if isinstance(submodule, bn_base):
                self.states.append((submodule, submodule.training))
                submodule.eval()
        if not _BN_FREEZE_DIAGNOSTIC_PRINTED:
            print(f'[interval BN freeze] batchnorm_modules={len(self.states)} running_stats_updated_by_interval=false interval_gradients_enabled=true', flush=True)
            _BN_FREEZE_DIAGNOSTIC_PRINTED = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for (submodule, was_training) in self.states:
            submodule.train(was_training)
        return False

class IntervalPoolNoFull(torch.nn.Module):
    """Mean or positional-attention interval pooling without G_full."""

    def __init__(self, channels, boundaries, history_days, pooling, hidden=128):
        super().__init__()
        self.channels = int(channels)
        self.history_days = float(history_days)
        self.boundaries = tuple((float(v) for v in boundaries))
        self.pooling = str(pooling)
        if not self.boundaries or self.boundaries[-1] != self.history_days:
            raise RuntimeError(f'Interval boundaries must be nonempty and end at H: boundaries={self.boundaries} H={self.history_days}')
        if self.pooling not in {'mean', 'positional_attention'}:
            raise ValueError(f'Unsupported no-full pooling: {self.pooling}')
        if self.pooling == 'positional_attention':
            self.token_score = torch.nn.Sequential(torch.nn.Linear(self.channels + 5, max(32, hidden // 2)), torch.nn.Tanh(), torch.nn.Linear(max(32, hidden // 2), 1))
        else:
            self.token_score = None
        self._diagnostic_logged = False
        self._audit_enabled = False
        self._reset_audit_storage()

    def _reset_audit_storage(self):
        self._audit_base = []
        self._audit_final = []
        self._audit_output_delta = []
        self._audit_output_ratio = []
        self._audit_alpha = []

    def reset_audit(self):
        self._reset_audit_storage()
        self._audit_enabled = True

    def finish_audit(self):
        self._audit_enabled = False

        def cat(values):
            return None if not values else torch.cat(values, dim=0)

        def stats(value):
            if value is None:
                return None
            value = value.float()
            return {'mean': float(value.mean()), 'std': float(value.std(unbiased=False)), 'mean_abs': float(value.abs().mean()), 'max_abs': float(value.abs().max())}
        base = cat(self._audit_base)
        final = cat(self._audit_final)
        output_delta = cat(self._audit_output_delta)
        output_ratio = cat(self._audit_output_ratio)
        alpha = cat(self._audit_alpha)
        self._reset_audit_storage()
        return {'base_stats': stats(base), 'final_stats': stats(final), 'scaled_residual_stats': stats(output_delta), 'scaled_residual_ratio_mean': None if output_ratio is None else float(output_ratio.mean()), 'representation_delta_norm_stats': None, 'representation_delta_ratio_mean': None, 'gate': None, 'alpha_mean': None if alpha is None else alpha.float().mean(dim=0).tolist(), 'alpha_std': None if alpha is None else alpha.float().std(dim=0, unbiased=False).tolist()}

    def _position_features(self, batch_size, device, dtype):
        end = torch.tensor(self.boundaries, device=device, dtype=dtype)
        start = torch.cat([torch.zeros_like(end[:1]), end[:-1]], dim=0)
        width = end - start
        horizon = end[-1].clamp_min(torch.finfo(dtype).eps)
        center = 0.5 * (start + end)
        position = torch.stack([start / horizon, end / horizon, width / horizon, 1.0 - center / horizon], dim=-1)
        return position.unsqueeze(0).expand(batch_size, -1, -1)

    def forward(self, base_logit, h_full, h_seq, logit_seq=None, tpf_feat=None, task_head=None, valid_mask=None):
        if task_head is None:
            raise RuntimeError('No-full interval pooling requires the native task head')
        if h_seq.ndim != 3 or h_seq.size(2) != self.channels:
            raise RuntimeError(f'Expected h_seq [B,K,{self.channels}], got {tuple(h_seq.shape)}')
        if h_seq.size(1) != len(self.boundaries):
            raise RuntimeError(f'Interval token count {h_seq.size(1)} != boundaries {len(self.boundaries)}')
        batch_size = h_seq.size(0)
        num_tokens = h_seq.size(1)
        if self.pooling == 'mean':
            alpha = masked_alpha(h_seq, valid_mask)
            h_temporal = masked_mean(h_seq, alpha, valid_mask)
        else:
            position = self._position_features(batch_size, h_seq.device, h_seq.dtype)
            token_norm = h_seq.norm(dim=-1, keepdim=True)
            alpha = masked_alpha(h_seq, valid_mask, self.token_score(torch.cat([h_seq, position, token_norm], dim=-1)).squeeze(-1))
            h_temporal = (alpha.unsqueeze(-1) * h_seq).sum(dim=1)
        pred = task_head(h_temporal)
        representation_ratio = h_seq.new_zeros(batch_size)
        pred_flat = pred.reshape(pred.size(0), -1)
        if self._audit_enabled:
            if base_logit is None:
                raise RuntimeError('No-full audit requires base_logit during evaluation')
            base_flat = base_logit.reshape(base_logit.size(0), -1)
            output_delta = pred_flat - base_flat
            output_ratio = output_delta.abs().mean(dim=-1) / (1.0 + base_flat.detach().abs().mean(dim=-1))
            self._audit_base.append(base_flat.detach().float().cpu())
            self._audit_final.append(pred_flat.detach().float().cpu())
            self._audit_output_delta.append(output_delta.detach().float().cpu())
            self._audit_output_ratio.append(output_ratio.detach().float().cpu())
            self._audit_alpha.append(alpha.detach().float().cpu())
        if not self._diagnostic_logged:
            print(f'[interval pooling no full] pooling={self.pooling} h_seq={tuple(h_seq.shape)} h_temporal={tuple(h_temporal.shape)} boundaries={list(self.boundaries)} uses_g_full_anchor=false gfull_token=false interval_logits_used=false base_aux_loss=false representation_fusion=native_task_head_direct task_head_shared=true', flush=True)
            self._diagnostic_logged = True
        return (pred, representation_ratio, alpha)

def masked_alpha(seq, valid_mask, scores=None):
    if valid_mask is None:
        if scores is None:
            return seq.new_full(seq.shape[:2], 1.0 / seq.size(1))
        return torch.softmax(scores, dim=1)
    assert valid_mask.dtype == torch.bool and valid_mask.shape == seq.shape[:2]
    effective = valid_mask | ~valid_mask.any(dim=1, keepdim=True)
    if scores is None:
        weights = effective.to(seq.dtype)
        return weights / weights.sum(dim=1, keepdim=True)
    return torch.softmax(scores.masked_fill(~effective, float('-inf')), dim=1)

def masked_mean(seq, alpha, valid_mask):
    original = seq.mean(dim=1)
    if valid_mask is None:
        return original
    weighted = (alpha.unsqueeze(-1) * seq).sum(dim=1)
    unchanged = valid_mask.all(dim=1) | ~valid_mask.any(dim=1)
    return torch.where(unchanged.unsqueeze(-1), original, weighted)

class FullTokenPool(IntervalPoolNoFull):

    def forward(self, base_logit, h_full, h_seq, logit_seq=None, tpf_feat=None, task_head=None, valid_mask=None):
        if task_head is None or h_full is None:
            raise RuntimeError('Full-token pooling requires full representation and task head')
        assert h_seq.ndim == 3 and h_seq.size(2) == self.channels
        assert h_seq.size(1) == len(self.boundaries)
        assert h_full.shape == (h_seq.size(0), self.channels)
        if valid_mask is None:
            valid_mask = torch.ones(h_seq.shape[:2], dtype=torch.bool, device=h_seq.device)
        assert valid_mask.dtype == torch.bool
        assert valid_mask.shape == h_seq.shape[:2]
        tokens = torch.cat([h_seq, h_full.unsqueeze(1)], dim=1)
        mask = torch.cat([valid_mask, torch.ones_like(valid_mask[:, :1])], dim=1)
        if self.pooling == 'mean':
            alpha = masked_alpha(tokens, mask)
            pooled = masked_mean(tokens, alpha, mask)
        else:
            position = self._position_features(h_seq.size(0), h_seq.device, h_seq.dtype)
            full_position = h_seq.new_tensor([0.0, 1.0, 1.0, 0.5]).view(1, 1, 4)
            position = torch.cat([position, full_position.expand(h_seq.size(0), 1, 4)], dim=1)
            scores = self.token_score(torch.cat([tokens, position, tokens.norm(dim=-1, keepdim=True)], dim=-1)).squeeze(-1)
            alpha = masked_alpha(tokens, mask, scores)
            pooled = (alpha.unsqueeze(-1) * tokens).sum(dim=1)
        pred = task_head(pooled)
        if not self._diagnostic_logged:
            print('[FULL TOKEN POOL]', self.pooling, 'token_count=', tokens.size(1), 'full_token=last always_valid=true empty_fallback=full_only', flush=True)
            self._diagnostic_logged = True
        return (pred, h_seq.new_zeros(h_seq.size(0)), alpha)

class MeanPoolWithFullToken(FullTokenPool):

    def __init__(self, **kwargs):
        super().__init__(pooling='mean', **kwargs)

class AttentionPoolWithFullToken(FullTokenPool):

    def __init__(self, **kwargs):
        super().__init__(pooling='positional_attention', **kwargs)

