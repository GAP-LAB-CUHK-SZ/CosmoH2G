# Adapted from https://github.com/POSTECH-CVLab/point-transformer
import torch
from torch.autograd import Function

import pointops_cuda


class FurthestSampling(Function):
    @staticmethod
    def forward(ctx, xyz, offset, new_offset):
        assert xyz.is_contiguous()
        n, b, n_max = xyz.shape[0], offset.shape[0], offset[0]
        for i in range(1, b):
            n_max = max(offset[i] - offset[i - 1], n_max)
        idx = torch.zeros(new_offset[b - 1].item(), dtype=torch.int32, device=xyz.device)
        tmp = torch.full((n,), 1e10, dtype=torch.float32, device=xyz.device)
        pointops_cuda.furthestsampling_cuda(b, n_max, xyz, offset, new_offset, tmp, idx)
        del tmp
        return idx


furthestsampling = FurthestSampling.apply


class KNNQuery(Function):
    @staticmethod
    def forward(ctx, nsample, xyz, new_xyz, offset, new_offset):
        if new_xyz is None:
            new_xyz = xyz
        assert xyz.is_contiguous() and new_xyz.is_contiguous()
        m = new_xyz.shape[0]
        idx = torch.zeros((m, nsample), dtype=torch.int32, device=xyz.device)
        dist2 = torch.zeros((m, nsample), dtype=torch.float32, device=xyz.device)
        pointops_cuda.knnquery_cuda(m, nsample, xyz, new_xyz, offset, new_offset, idx, dist2)
        return idx, torch.sqrt(dist2)


knnquery = KNNQuery.apply


def queryandgroup(nsample, xyz, new_xyz, feat, idx, offset, new_offset, use_xyz=True):
    assert xyz.is_contiguous() and new_xyz.is_contiguous() and feat.is_contiguous()
    if new_xyz is None:
        new_xyz = xyz
    if idx is None:
        idx, _ = knnquery(nsample, xyz, new_xyz, offset, new_offset)
    m, c = new_xyz.shape[0], feat.shape[1]
    grouped_xyz = xyz[idx.view(-1).long(), :].view(m, nsample, 3)
    grouped_xyz -= new_xyz.unsqueeze(1)
    grouped_feat = feat[idx.view(-1).long(), :].view(m, nsample, c)
    if use_xyz:
        return torch.cat((grouped_xyz, grouped_feat), -1)
    return grouped_feat
