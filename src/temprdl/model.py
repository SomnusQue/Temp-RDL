"""Shared-backbone Temp-RDL, independent of any concrete prediction task."""
import inspect
import torch
from .fusion import freeze_batchnorm_running_stats
from .numerical import install
from .temporal import extract_rdl_hidden


class TempRDL(torch.nn.Module):
    def __init__(self,backbone,fusion):
        super().__init__()
        if not hasattr(backbone,"architecture_config"):backbone.architecture_config={}
        self.backbone=install(backbone)
        self.fusion=fusion
        self._accepts_valid_mask="valid_mask" in inspect.signature(fusion.forward).parameters
        self.representation_ratio=None
        self.temporal_weights=None

    def forward(self,full_batch,interval_batches,entity_table,valid_mask=None):
        if not interval_batches:raise ValueError("At least one temporal graph view is required")
        full=extract_rdl_hidden(self.backbone,full_batch,entity_table)
        with freeze_batchnorm_running_stats(self.backbone):
            sequence=torch.stack([extract_rdl_hidden(self.backbone,view,entity_table) for view in interval_batches],dim=1)
        options=dict(task_head=self.backbone.head)
        if self._accepts_valid_mask:options["valid_mask"]=valid_mask
        prediction,ratio,weights=self.fusion(None,full,sequence,**options)
        self.representation_ratio=ratio
        self.temporal_weights=weights
        return prediction
