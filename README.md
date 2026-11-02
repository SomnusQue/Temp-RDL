# Temp-RDL

Generic model components for **Temp-RDL (Mean pooling)** and **Temp-RDL (Attention)**.

**Temp-RDL (Mean pooling)** uses a shared GNN to encode the full relational graph and temporal subgraphs. Temporal representations are aggregated by mean pooling and combined with the full-graph representation before the original prediction head.

**Temp-RDL (Attention)** uses the same shared GNN and temporal decomposition, with attention-based aggregation of temporal representations.

The code retains the original full-anchor residual and full-token fusion components. Graph views, model dimensions and temporal boundaries are supplied by the caller.

## Source

- `src/temprdl/backbone.py`: native relational GNN and prediction head.
- `src/temprdl/rdl_nn.py`: relational feature encoders and message passing.
- `src/temprdl/temporal.py`: generic temporal graph-view operations.
- `src/temprdl/fusion.py`: Mean and Attention full-anchor fusion.
- `src/temprdl/pooling.py`: Mean and Attention full-token pooling.
- `src/temprdl/model.py`: shared-backbone model assembly.

This repository contains method source only. Dataset-specific code, task adapters, experimental configurations, tuning records, training scripts and checkpoints are not included.

## Attribution

Temp-RDL builds on [RelBench](https://github.com/stanford-star/relbench), [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) and [PyTorch Frame](https://github.com/pyg-team/pytorch-frame). See `THIRD_PARTY_NOTICES.md` and `licenses/RelBench-APACHE-2.0.txt` for upstream attribution.
