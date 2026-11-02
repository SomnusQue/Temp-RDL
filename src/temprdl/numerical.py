"""Extracted original Temp-RDL definitions; only imports/packaging are adapted."""
from __future__ import annotations

from types import MethodType

import torch

from torch_frame.nn import LinearEncoder

FIX = 'linear_missing_zero_safe_backward_v1'

def safe_encode(self, feat, col_names=None):
    if torch.isinf(feat).any():
        raise RuntimeError(f'输入含Inf，未自动修复：{self._nanfix_name}')

    missing = torch.isnan(feat)

    # 临时填均值，避免NaN参与可学习乘法。
    # 随后将缺失项的整个编码置零，保留原来的前向行为。
    safe = torch.where(missing, self.mean, feat)
    out = LinearEncoder.encode_forward(self, safe, col_names)
    out = out.masked_fill(missing.unsqueeze(-1), 0)

    if not self._nanfix_verified and feat.numel():
        with torch.no_grad():
            reference = LinearEncoder.encode_forward(self, feat, col_names)
            reference = torch.nan_to_num(reference, nan=0.0)
            if not torch.isfinite(out).all():
                raise RuntimeError(f'修复后编码仍非有限：{self._nanfix_name}')
            torch.testing.assert_close(out, reference, rtol=0, atol=0)

        self._nanfix_verified = True
        print(
            '[ENCODER FORWARD MATCH]', self._nanfix_name,
            'missing=', int(missing.sum()), flush=True,
        )

    return out

def install(model):
    count = 0
    for name, module in model.named_modules():
        if type(module) is not LinearEncoder:
            continue

        if module.na_strategy is not None:
            raise RuntimeError(f'已有缺失值策略，需要单独核对：{name}')

        if not (
            torch.isfinite(module.mean).all()
            and torch.isfinite(module.std).all()
            and (module.std > 0).all()
        ):
            raise RuntimeError(f'列统计量异常，未自动填补：{name}')

        module._nanfix_name = name
        module._nanfix_verified = False
        module.encode_forward = MethodType(safe_encode, module)
        count += 1

    if count == 0:
        raise RuntimeError('未找到预期的LinearEncoder')

    model.architecture_config['numerical_missing_fix'] = FIX
    print('[NUMERICAL FIX]', FIX, 'encoders=', count, flush=True)
    return model

