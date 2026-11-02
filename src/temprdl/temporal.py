"""Extracted original Temp-RDL definitions; only imports/packaging are adapted."""
from __future__ import annotations

import torch

_MIXED_EDGE_TIME_DIAGNOSTIC_PRINTED = False

def _edge_time_with_source(batch, edge_type):
    """Resolve an edge's real event time, including foreign-key relations."""
    if edge_type not in batch.edge_types:
        return (None, 'edge_type_missing')
    store = batch[edge_type]
    edge_index = store.edge_index
    num_edges = int(edge_index.size(1))
    for key in ('edge_time', 'time', 'timestamp', 'history_time'):
        value = getattr(store, key, None)
        if value is not None and hasattr(value, 'numel') and (value.numel() == num_edges):
            return (value, f'edge_store.{key}')
    (src_type, relation, dst_type) = edge_type
    relation = str(relation)
    if relation.startswith('rev_f2p_'):
        child_type = dst_type
        child_index = edge_index[1]
        side = 'dst'
    elif relation.startswith('f2p_'):
        child_type = src_type
        child_index = edge_index[0]
        side = 'src'
    else:
        return (None, 'static_non_fkey_relation')
    if child_type not in batch.node_types:
        return (None, f'child_node_missing[{child_type}]')
    if child_index.numel() == 0:
        return (child_index.new_empty((0,)), f'{side}_child[{child_type}].empty')
    node_store = batch[child_type]
    max_index = int(child_index.max().item())
    for key in ('time', 'timestamp', 'history_time'):
        value = getattr(node_store, key, None)
        if value is not None and hasattr(value, 'numel') and (value.numel() > max_index):
            return (value[child_index], f'{side}_child[{child_type}].{key}')
    return (None, f'static_child_no_event_time[{child_type}]')

def _edge_prediction_time(batch, edge_type, seed_time_flat):
    """Map every sampled edge to its root query's prediction time."""
    global _MIXED_EDGE_TIME_DIAGNOSTIC_PRINTED
    edge_index = batch[edge_type].edge_index
    num_edges = int(edge_index.size(1))
    if num_edges == 0:
        return seed_time_flat.new_empty((0,))
    (src_type, _, dst_type) = edge_type
    assignments = []
    for (node_type, node_index, side) in ((src_type, edge_index[0], 'src'), (dst_type, edge_index[1], 'dst')):
        node_batch = getattr(batch[node_type], 'batch', None)
        if node_batch is None:
            continue
        node_batch = node_batch.reshape(-1)
        if node_index.numel():
            max_index = int(node_index.max().item())
            if node_batch.numel() <= max_index:
                raise RuntimeError(f'Node-to-query assignment is shorter than sampled {side} indices for {edge_type}')
        assignments.append((side, node_batch[node_index].long()))
    if not assignments:
        if torch.unique(seed_time_flat).numel() == 1:
            return seed_time_flat[:1].expand(num_edges)
        raise RuntimeError(f'Mixed-timestamp exact intervals require disjoint NeighborLoader node-to-query assignments for {edge_type}')
    edge_query = assignments[0][1]
    for (side, candidate) in assignments[1:]:
        if not torch.equal(edge_query, candidate):
            raise RuntimeError(f'A sampled edge crosses disjoint query subgraphs for {edge_type} ({side})')
    if edge_query.numel() != num_edges:
        raise RuntimeError('Edge-to-query assignment length mismatch')
    if edge_query.numel():
        query_min = int(edge_query.min().item())
        query_max = int(edge_query.max().item())
        if query_min < 0 or query_max >= seed_time_flat.numel():
            raise RuntimeError(f'Edge-to-query assignment outside seed-time vector: min={query_min} max={query_max} seeds={seed_time_flat.numel()}')
    unique_count = int(torch.unique(seed_time_flat).numel())
    if unique_count > 1 and (not _MIXED_EDGE_TIME_DIAGNOSTIC_PRINTED):
        print(f'[temporal edge assignment] batch_size={seed_time_flat.numel()} unique_seed_times={unique_count} edge_to_query_assignment=disjoint per_edge_prediction_time=true', flush=True)
        _MIXED_EDGE_TIME_DIAGNOSTIC_PRINTED = True
    return seed_time_flat[edge_query]

def _mask_edge_store_attrs(store, keep):
    """Mask common edge-level tensor attributes using an edge keep mask."""
    num_edges = int(keep.numel())
    for (key, value) in list(store.items()):
        if key == 'edge_index' or str(key).startswith('edge_label'):
            continue
        if torch.is_tensor(value) and value.dim() > 0 and (value.size(0) == num_edges):
            store[key] = value[keep]

def _filter_num_sampled_edges(original_store, snapshot_store, keep):
    """Synchronize per-hop sampled-edge counts after temporal filtering."""
    counts = getattr(original_store, 'num_sampled_edges', None)
    if counts is None:
        return
    if torch.is_tensor(counts):
        values = [int(x) for x in counts.detach().reshape(-1).cpu().tolist()]
    else:
        values = [int(x) for x in counts]
    edge_count = int(keep.numel())
    if sum(values) != edge_count:
        raise RuntimeError(f'num_sampled_edges metadata does not match edge_index: sum_counts={sum(values)} edge_count={edge_count}')
    filtered = []
    offset = 0
    for count in values:
        segment = keep[offset:offset + count]
        filtered.append(int(segment.sum().item()))
        offset += count
    if offset != edge_count:
        raise RuntimeError(f'Per-hop edge segmentation incomplete: {offset} != {edge_count}')
    if torch.is_tensor(counts):
        snapshot_store.num_sampled_edges = counts.new_tensor(filtered)
    elif isinstance(counts, tuple):
        snapshot_store.num_sampled_edges = tuple(filtered)
    else:
        snapshot_store.num_sampled_edges = filtered

def _apply_edge_keep(batch, snapshot, edge_type, keep):
    original_store = batch[edge_type]
    snapshot_store = snapshot[edge_type]
    snapshot_store.edge_index = snapshot_store.edge_index[:, keep]
    _mask_edge_store_attrs(snapshot_store, keep)
    _filter_num_sampled_edges(original_store, snapshot_store, keep)

def extract_rdl_hidden(model, batch, entity_table):
    """Reproduce the native RDL feature path and return task-head input."""
    seed_time = batch[entity_table].seed_time
    x_dict = model.encoder(batch.tf_dict)
    temporal_encoder = getattr(model, 'temporal_encoder', None)
    use_temporal = bool(getattr(model, 'use_temporal', temporal_encoder is not None))
    if use_temporal and temporal_encoder is not None:
        relative_time = temporal_encoder(seed_time, batch.time_dict, batch.batch_dict)
        for (node_type, value) in relative_time.items():
            x_dict[node_type] = x_dict[node_type] + value
    for (node_type, embedding) in model.embedding_dict.items():
        x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)
    x_dict = model.gnn(x_dict, batch.edge_index_dict, batch.num_sampled_nodes_dict, batch.num_sampled_edges_dict)
    hidden = x_dict[entity_table][:seed_time.size(0)]
    if hidden.dim() != 2:
        raise RuntimeError(f'Explicit RDL hidden must be [B,C], got {tuple(hidden.shape)}')
    if not bool(torch.isfinite(hidden).all().item()):
        raise RuntimeError('Non-finite RDL hidden representation')
    return hidden



def make_interval_batches(batch, entity_table, boundaries_days):
    """Partition timed relations into disjoint intervals; retain static relations.

    The caller supplies the history boundaries and causally valid sampled graph.
    Boundaries use days; input timestamps use seconds. Intervals follow
    [0, b1], then (b_previous, b]. No dataset or task settings are embedded.
    """
    boundaries=tuple(float(value) for value in boundaries_days)
    if not boundaries or any(value<=0 for value in boundaries):
        raise ValueError("Positive interval boundaries are required")
    if any(right<=left for left,right in zip(boundaries,boundaries[1:])):
        raise ValueError("Interval boundaries must increase strictly")
    seeds=batch[entity_table].seed_time.reshape(-1)
    cuts=torch.tensor(boundaries,dtype=torch.float64,device=seeds.device)
    assignments={}
    counts=torch.zeros((seeds.numel(),len(boundaries)),dtype=torch.long,device=seeds.device)
    for edge in batch.edge_types:
        index=batch[edge].edge_index
        if index.size(1)==0:continue
        times,_=_edge_time_with_source(batch,edge)
        if times is None:continue
        query=_edge_prediction_time(batch,edge,seeds)
        age=(query.to(torch.float64)-times.to(torch.float64))/86400
        valid=torch.isfinite(age)&(age>=0)&(age<=cuts[-1])
        buckets=torch.bucketize(age.contiguous(),cuts,right=False)
        buckets=torch.where(valid,buckets,torch.full_like(buckets,-1))
        assignments[edge]=buckets
        source=batch[edge[0]].batch[index[0]].long()
        target=batch[edge[2]].batch[index[1]].long()
        if not torch.equal(source,target):raise RuntimeError("An edge crosses prediction queries")
        if bool(valid.any()):
            linear=source[valid]*len(boundaries)+buckets[valid]
            counts+=torch.bincount(linear,minlength=counts.numel()).reshape_as(counts)
    views=[]
    for k in range(len(boundaries)):
        view=batch.clone()
        for edge,buckets in assignments.items():_apply_edge_keep(batch,view,edge,buckets==k)
        views.append(view)
    return views,counts>0
