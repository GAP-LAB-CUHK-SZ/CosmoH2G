# Adapted from https://github.com/POSTECH-CVLab/point-transformer
import torch
import torch.nn as nn
from einops import rearrange

from cosmoh2g import pointops


class PointTransformerLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p = nn.Sequential(
            nn.Linear(3, 3), nn.BatchNorm1d(3), nn.ReLU(inplace=True), nn.Linear(3, out_planes)
        )
        self.linear_w = nn.Sequential(
            nn.BatchNorm1d(mid_planes),
            nn.ReLU(inplace=True),
            nn.Linear(mid_planes, mid_planes // share_planes),
            nn.BatchNorm1d(mid_planes // share_planes),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes // share_planes, out_planes // share_planes),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pxo):
        p, x, o = pxo
        x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)
        x_k = pointops.queryandgroup(self.nsample, p, p, x_k, None, o, o, use_xyz=True)
        x_v = pointops.queryandgroup(self.nsample, p, p, x_v, None, o, o, use_xyz=False)
        p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]
        for i, layer in enumerate(self.linear_p):
            p_r = layer(p_r.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(p_r)
        w = x_k - x_q.unsqueeze(1) + p_r.view(
            p_r.shape[0], p_r.shape[1], self.out_planes // self.mid_planes, self.mid_planes
        ).sum(2)
        for i, layer in enumerate(self.linear_w):
            w = layer(w.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i % 3 == 0 else layer(w)
        w = self.softmax(w)
        n, nsample, c = x_v.shape
        s = self.share_planes
        return ((x_v + p_r).view(n, nsample, s, c // s) * w.unsqueeze(2)).sum(1).view(n, c)


class TransitionDown(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        if stride != 1:
            self.linear = nn.Linear(3 + in_planes, out_planes, bias=False)
            self.pool = nn.MaxPool1d(nsample)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo
        if self.stride != 1:
            n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
            for i in range(1, o.shape[0]):
                count += (o[i].item() - o[i - 1].item()) // self.stride
                n_o.append(count)
            n_o = torch.tensor(n_o, dtype=torch.int32, device=p.device)
            idx = pointops.furthestsampling(p, o, n_o)
            n_p = p[idx.long(), :]
            x = pointops.queryandgroup(self.nsample, p, n_p, x, None, o, n_o, use_xyz=True)
            x = self.relu(self.bn(self.linear(x).transpose(1, 2).contiguous()))
            x = self.pool(x).squeeze(-1)
            p, o = n_p, n_o
        else:
            x = self.relu(self.bn(self.linear(x)))
        return [p, x, o]


class PointTransformerBlock(nn.Module):
    def __init__(self, in_planes, planes, share_planes=8, nsample=16):
        super().__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer2 = PointTransformerLayer(planes, planes, share_planes, nsample)
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes, bias=False)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo
        identity = x
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.transformer2([p, x, o])))
        x = self.bn3(self.linear3(x))
        x += identity
        x = self.relu(x)
        return [p, x, o]


class PointEncoder(nn.Module):
    """Four-level Point Transformer encoder; returns the mean-pooled global feature [B, 1, C]."""

    def __init__(self, point_feat_dim, planes=(32, 64, 128, 256), blocks=(2, 2, 2, 2)):
        super().__init__()
        self.c = point_feat_dim + 3
        self.in_planes = self.c
        share_planes = 8
        stride, nsample = [1, 4, 4, 4], [8, 16, 16, 16]
        self.enc1 = self._make_enc(planes[0], blocks[0], share_planes, stride[0], nsample[0])
        self.enc2 = self._make_enc(planes[1], blocks[1], share_planes, stride[1], nsample[1])
        self.enc3 = self._make_enc(planes[2], blocks[2], share_planes, stride[2], nsample[2])
        self.enc4 = self._make_enc(planes[3], blocks[3], share_planes, stride[3], nsample[3])

    def _make_enc(self, planes, blocks, share_planes, stride, nsample):
        layers = [TransitionDown(self.in_planes, planes, stride, nsample)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(PointTransformerBlock(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def forward(self, p, x):
        offset, count = [], 0
        for item in p:
            count += item.shape[0]
            offset.append(count)
        p0 = rearrange(p, "b n c -> (b n) c").contiguous()
        x0 = rearrange(x, "b n c -> (b n) c")
        o0 = torch.IntTensor(offset).to(p0.device)
        x0 = p0 if self.c == 3 else torch.cat((p0, x0), 1)
        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        _, x4, _ = self.enc4([p3, x3, o3])
        local_feat = rearrange(x4, "(b n) c -> b n c", b=len(offset))
        return local_feat.mean(dim=1, keepdim=True)
