import torch
import numpy as np
import torch.nn as nn
from .clip import clip
from .region_awareness import get_backbone


class LipFD(nn.Module):
    def __init__(self, name, num_classes=1):
        super(LipFD, self).__init__()

        self.conv1 = nn.Conv2d(
            3, 3, kernel_size=5, stride=5
        )  # (1120, 1120) -> (224, 224)
        self.encoder, self.preprocess = clip.load(name, device="cpu")
        self.backbone = get_backbone()

    def forward(self, x, feature):
        return self.backbone(x, feature)

    def get_features(self, x):
        x = self.conv1(x)
        features = self.encoder.encode_image(x)
        return features


class RALoss(nn.Module):
    def __init__(self):
        super(RALoss, self).__init__()

    def forward(self, alphas_max, alphas_org):
        # 将列表中的张量堆叠成一个张量
        alphas_max_stack = torch.stack(alphas_max, dim=0)  # shape: (num_regions, batch_size, 1)
        alphas_org_stack = torch.stack(alphas_org, dim=0)  # shape: (num_regions, batch_size, 1)

        # 向量化计算差异
        diff = alphas_max_stack - alphas_org_stack  # shape: (num_regions, batch_size, 1)

        # 添加数值稳定性保护
        diff = torch.clamp(diff, min=0.0, max=100.0)

        # 向量化计算损失权重
        loss_wt = 10 / torch.exp(diff)  # shape: (num_regions, batch_size, 1)

        # 对batch和区域取平均
        return loss_wt.mean()