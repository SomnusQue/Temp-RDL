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

MAX_RESIDUAL_GATE = 0.3

INITIAL_GATE_FRACTION = 0.1

class AttentionPoolWithFullAnchor(torch.nn.Module):
    """Clean full-anchor positional representation residual Z1.

    G_full is used only as the stable representation anchor. Temporal tokens
    are the K mutually-exclusive intervals returned by the helper; no G_full
    token or interval logit is added to temporal attention.
    """

    def __init__(self, channels, boundaries, history_days, hidden=128, dropout=0.1, force_zero_temporal_gate=False):
        super().__init__()
        self.channels = int(channels)
        self.history_days = float(history_days)
        self.boundaries = tuple((float(v) for v in boundaries))
        self.force_zero_temporal_gate = bool(force_zero_temporal_gate)
        if not self.boundaries or self.boundaries[-1] != self.history_days:
            raise RuntimeError(f'Z1 boundaries must be nonempty and end at H: boundaries={self.boundaries} H={self.history_days}')
        self.token_score = torch.nn.Sequential(torch.nn.Linear(self.channels + 5, max(32, hidden // 2)), torch.nn.Tanh(), torch.nn.Linear(max(32, hidden // 2), 1))
        self.delta_head = torch.nn.Sequential(torch.nn.Linear(self.channels * 4, hidden), torch.nn.ReLU(), torch.nn.Dropout(dropout), torch.nn.Linear(hidden, self.channels))
        torch.nn.init.zeros_(self.delta_head[-1].weight)
        torch.nn.init.zeros_(self.delta_head[-1].bias)
        self.gate_logit = torch.nn.Parameter(torch.logit(torch.tensor(INITIAL_GATE_FRACTION, dtype=torch.float32)))
        self._diagnostic_logged = False
        self._audit_enabled = False
        self._reset_audit_storage()

    def _reset_audit_storage(self):
        self._audit_base = []
        self._audit_final = []
        self._audit_output_delta = []
        self._audit_output_ratio = []
        self._audit_representation_delta_norm = []
        self._audit_representation_ratio = []
        self._audit_alpha = []

    def reset_audit(self):
        self._reset_audit_storage()
        self._audit_enabled = True

    def finish_audit(self):
        self._audit_enabled = False

        def cat(xs):
            return None if not xs else torch.cat(xs, dim=0)

        def stats(x):
            if x is None:
                return None
            x = x.float()
            return {'mean': float(x.mean()), 'std': float(x.std(unbiased=False)), 'mean_abs': float(x.abs().mean()), 'max_abs': float(x.abs().max())}
        base = cat(self._audit_base)
        final = cat(self._audit_final)
        output_delta = cat(self._audit_output_delta)
        output_ratio = cat(self._audit_output_ratio)
        rep_norm = cat(self._audit_representation_delta_norm)
        rep_ratio = cat(self._audit_representation_ratio)
        alpha = cat(self._audit_alpha)
        self._reset_audit_storage()
        return {'base_stats': stats(base), 'final_stats': stats(final), 'scaled_residual_stats': stats(output_delta), 'scaled_residual_ratio_mean': None if output_ratio is None else float(output_ratio.mean()), 'representation_delta_norm_stats': stats(rep_norm), 'representation_delta_ratio_mean': None if rep_ratio is None else float(rep_ratio.mean()), 'gate': float((MAX_RESIDUAL_GATE * torch.sigmoid(self.gate_logit.detach())).cpu()), 'alpha_mean': None if alpha is None else alpha.float().mean(dim=0).tolist(), 'alpha_std': None if alpha is None else alpha.float().std(dim=0, unbiased=False).tolist()}

    def _position_features(self, batch_size, device, dtype):
        end = torch.tensor(self.boundaries, device=device, dtype=dtype)
        start = torch.cat([torch.zeros_like(end[:1]), end[:-1]], dim=0)
        width = end - start
        horizon = end[-1].clamp_min(torch.finfo(dtype).eps)
        center = 0.5 * (start + end)
        pos = torch.stack([start / horizon, end / horizon, width / horizon, 1.0 - center / horizon], dim=-1)
        return pos.unsqueeze(0).expand(batch_size, -1, -1)

    def forward(self, base_logit, h_full, h_seq, logit_seq=None, tpf_feat=None, task_head=None):
        if h_full is None or task_head is None:
            raise RuntimeError('Z1 requires h_full and the native task head')
        if h_seq.ndim != 3 or h_seq.size(2) != self.channels:
            raise RuntimeError(f'Expected h_seq [B,K,{self.channels}], got {tuple(h_seq.shape)}')
        if h_seq.size(1) != len(self.boundaries):
            raise RuntimeError(f'Interval token count {h_seq.size(1)} != boundaries {len(self.boundaries)}')
        batch_size = h_seq.size(0)
        position = self._position_features(batch_size, h_seq.device, h_seq.dtype)
        token_norm = h_seq.norm(dim=-1, keepdim=True)
        alpha = torch.softmax(self.token_score(torch.cat([h_seq, position, token_norm], dim=-1)).squeeze(-1), dim=1)
        h_temporal = (alpha.unsqueeze(-1) * h_seq).sum(dim=1)
        features = torch.cat([h_full, h_temporal, h_temporal - h_full, h_temporal * h_full], dim=-1)
        raw_delta_h = self.delta_head(features)
        anchor_rms = h_full.detach().pow(2).mean(dim=-1, keepdim=True).add(1e-06).sqrt()
        bounded_delta_h = anchor_rms * torch.tanh(raw_delta_h / anchor_rms)
        gate = MAX_RESIDUAL_GATE * torch.sigmoid(self.gate_logit)
        scaled_delta_h = gate * bounded_delta_h
        if self.force_zero_temporal_gate:
            scaled_delta_h = scaled_delta_h * 0.0
        h_final = h_full + scaled_delta_h
        pred = task_head(h_final)
        rep_norm = scaled_delta_h.norm(dim=-1)
        rep_ratio = rep_norm / h_full.detach().norm(dim=-1).add(1e-06)
        pred_flat = pred.reshape(pred.size(0), -1)
        base_flat = None
        output_delta = None
        output_ratio = None
        if base_logit is not None:
            base_flat = base_logit.reshape(base_logit.size(0), -1)
            output_delta = pred_flat - base_flat
            output_ratio = output_delta.abs().mean(dim=-1) / (1.0 + base_flat.detach().abs().mean(dim=-1))
        if self._audit_enabled:
            if base_flat is None:
                raise RuntimeError('Z1 audit requires base_logit during evaluation')
            self._audit_base.append(base_flat.detach().float().cpu())
            self._audit_final.append(pred_flat.detach().float().cpu())
            self._audit_output_delta.append(output_delta.detach().float().cpu())
            self._audit_output_ratio.append(output_ratio.detach().float().cpu())
            self._audit_representation_delta_norm.append(rep_norm.detach().float().cpu())
            self._audit_representation_ratio.append(rep_ratio.detach().float().cpu())
            self._audit_alpha.append(alpha.detach().float().cpu())
        if not self._diagnostic_logged:
            print(f'[z1 clean representation residual] h_full={tuple(h_full.shape)} h_seq={tuple(h_seq.shape)} h_final={tuple(h_final.shape)} boundaries={list(self.boundaries)} gate={float(gate.detach())} force_zero={self.force_zero_temporal_gate} uses_g_full_anchor=true gfull_token=false interval_logits_used=false base_aux_loss=false position_features=actual_boundaries correction_space=representation task_head_shared=true', flush=True)
            self._diagnostic_logged = True
        return (pred, rep_ratio, alpha)

class MeanPoolWithFullAnchor(AttentionPoolWithFullAnchor):
    """Mean interval pooling with the unchanged full-anchor Z1 residual."""

    def __init__(self, channels, boundaries, history_days, hidden=128, dropout=0.1, force_zero_temporal_gate=False):
        super().__init__(channels=channels, boundaries=boundaries, history_days=history_days, hidden=hidden, dropout=dropout, force_zero_temporal_gate=force_zero_temporal_gate)
        self.token_score = None

    def forward(self, base_logit, h_full, h_seq, logit_seq=None, tpf_feat=None, task_head=None):
        if h_full is None or task_head is None:
            raise RuntimeError('Mean-with-anchor Z1 requires h_full and the native task head')
        if h_seq.ndim != 3 or h_seq.size(2) != self.channels:
            raise RuntimeError(f'Expected h_seq [B,K,{self.channels}], got {tuple(h_seq.shape)}')
        if h_seq.size(1) != len(self.boundaries):
            raise RuntimeError(f'Interval token count {h_seq.size(1)} != boundaries {len(self.boundaries)}')
        batch_size = h_seq.size(0)
        alpha = h_seq.new_full((batch_size, h_seq.size(1)), 1.0 / float(h_seq.size(1)))
        h_temporal = h_seq.mean(dim=1)
        features = torch.cat([h_full, h_temporal, h_temporal - h_full, h_temporal * h_full], dim=-1)
        raw_delta_h = self.delta_head(features)
        anchor_rms = h_full.detach().pow(2).mean(dim=-1, keepdim=True).add(1e-06).sqrt()
        bounded_delta_h = anchor_rms * torch.tanh(raw_delta_h / anchor_rms)
        gate = MAX_RESIDUAL_GATE * torch.sigmoid(self.gate_logit)
        scaled_delta_h = gate * bounded_delta_h
        if self.force_zero_temporal_gate:
            scaled_delta_h = scaled_delta_h * 0.0
        h_final = h_full + scaled_delta_h
        pred = task_head(h_final)
        rep_norm = scaled_delta_h.norm(dim=-1)
        rep_ratio = rep_norm / h_full.detach().norm(dim=-1).add(1e-06)
        pred_flat = pred.reshape(pred.size(0), -1)
        base_flat = None
        output_delta = None
        output_ratio = None
        if base_logit is not None:
            base_flat = base_logit.reshape(base_logit.size(0), -1)
            output_delta = pred_flat - base_flat
            output_ratio = output_delta.abs().mean(dim=-1) / (1.0 + base_flat.detach().abs().mean(dim=-1))
        if self._audit_enabled:
            if base_flat is None:
                raise RuntimeError('Mean-with-anchor audit requires base_logit during evaluation')
            self._audit_base.append(base_flat.detach().float().cpu())
            self._audit_final.append(pred_flat.detach().float().cpu())
            self._audit_output_delta.append(output_delta.detach().float().cpu())
            self._audit_output_ratio.append(output_ratio.detach().float().cpu())
            self._audit_representation_delta_norm.append(rep_norm.detach().float().cpu())
            self._audit_representation_ratio.append(rep_ratio.detach().float().cpu())
            self._audit_alpha.append(alpha.detach().float().cpu())
        if not self._diagnostic_logged:
            print(f'[z1 mean representation residual] h_full={tuple(h_full.shape)} h_seq={tuple(h_seq.shape)} h_final={tuple(h_final.shape)} boundaries={list(self.boundaries)} gate={float(gate.detach())} force_zero={self.force_zero_temporal_gate} temporal_pooling=mean uses_g_full_anchor=true gfull_token=false interval_logits_used=false base_aux_loss=false correction_space=representation task_head_shared=true', flush=True)
            self._diagnostic_logged = True
        return (pred, rep_ratio, alpha)

