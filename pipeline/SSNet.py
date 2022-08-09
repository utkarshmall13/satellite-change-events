import torch.nn as nn
from torchvision import models
import torch
import numpy as np
import torch.nn.functional as F

class Backbone(nn.Module):
    def __init__(self, backbone='r18', pretrained=True):
        super(Backbone, self).__init__()
        # print(pretrained, backbone)
        if backbone=='r101':
            resnet_model = models.resnet101(pretrained=pretrained)
            self.nout = 2048
        elif backbone=='r18':
            resnet_model = models.resnet18(pretrained=pretrained)
            self.nout = 512
        self.scale_factor = 2
        for i, child in enumerate(resnet_model.children()):
            if i == 0:
                self.conv1 = child
            if i == 1:
                self.bn1 = child
            if i == 2:
                self.relu = child
            if i == 3:
                self.maxpool = child
            if i == 4:
                self.layer1 = child
            if i == 5:
                self.layer2 = child
            if i == 6:
                self.layer3 = child
            if i == 7:
              self.layer4 = child
            if i == 8:
                self.ap = child
            if i == 9:
                self.fcc = child

        self.conv1x1_1 = nn.Conv2d(64, 128, 1)
        self.conv1x1_2 = nn.Conv2d(64*2, 128, 1)
        self.conv1x1_3 = nn.Conv2d(64*4, 128, 1)
        self.conv1x1_4 = nn.Conv2d(64*8, 128, 1)
        self.conv3x3_4 = nn.Conv2d(128, 128, 3, padding=1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x4scaled = F.interpolate(self.conv1x1_4(x4), scale_factor=(2, 2), mode='nearest')
        x3scaled = F.interpolate(x4scaled+self.conv1x1_3(x3), scale_factor=(2, 2), mode='nearest')
        x2scaled = F.interpolate(x3scaled+self.conv1x1_2(x2), scale_factor=(2, 2), mode='nearest')
        feature = x2scaled+self.conv1x1_1(x1)
        output = self.conv3x3_4(feature)
        return output

class BackboneSmall(nn.Module):
    def __init__(self, backbone='r18', pretrained=True):
        super(BackboneSmall, self).__init__()
        # print(pretrained, backbone)
        if backbone=='r101':
            resnet_model = models.resnet101(pretrained=pretrained)
            self.nout = 2048
        elif backbone=='r18':
            resnet_model = models.resnet18(pretrained=pretrained)
            self.nout = 512
        self.scale_factor = 2
        for i, child in enumerate(resnet_model.children()):
            if i == 0:
                self.conv1 = child
            if i == 1:
                self.bn1 = child
            if i == 2:
                self.relu = child
            if i == 3:
                self.maxpool = child
            if i == 4:
                self.layer1 = child
            if i == 5:
                self.layer2 = child

        self.conv1x1_1 = nn.Conv2d(64, 128, 1)
        self.conv1x1_2 = nn.Conv2d(64*2, 128, 1)
        self.conv3x3_1 = nn.Conv2d(128, 128, 3, padding=1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)

        x2scaled = F.interpolate(self.conv1x1_2(x2), scale_factor=(2, 2), mode='nearest')
        feature = x2scaled+self.conv1x1_1(x1)
        output = self.conv3x3_1(feature)
        return output


class BackboneFiner(nn.Module):
    def __init__(self, backbone='r18', pretrained=True):
        super(BackboneFiner, self).__init__()
        # print(pretrained, backbone)
        if backbone=='r101':
            resnet_model = models.resnet101(pretrained=pretrained)
            self.nout = 2048
        elif backbone=='r18':
            resnet_model = models.resnet18(pretrained=pretrained)
            self.nout = 512
        self.scale_factor = 2
        for i, child in enumerate(resnet_model.children()):
            if i == 0:
                self.conv1 = child
            if i == 1:
                self.bn1 = child
            if i == 2:
                self.relu = child
            if i == 3:
                self.maxpool = child
            if i == 4:
                self.layer1 = child
            if i == 5:
                self.layer2 = child

        self.conv1x1_1 = nn.Conv2d(64, 128, 1)
        self.conv1x1_2 = nn.Conv2d(64*2, 128, 1)
        self.conv3x3_1 = nn.Conv2d(128, 128, 3, padding=1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)

        x2scaled = F.interpolate(self.conv1x1_2(x2), scale_factor=(2, 2), mode='nearest')
        feature = x2scaled+self.conv1x1_1(x1)
        output = self.conv3x3_1(feature)
        return output


class NTXentLoss(torch.nn.Module):
    def __init__(self, device, temperature):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self._cosine_similarity = torch.nn.CosineSimilarity(dim=2)
        self.similarity_function = self._cosine_simililarity
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_uncorrelated_mask(self, batch_size):
        diag = np.eye(2 * batch_size)
        l1 = np.eye((2 * batch_size), 2 * batch_size, k=-batch_size)
        l2 = np.eye((2 * batch_size), 2 * batch_size, k=batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    def _get_correlated_mask(self, batch_size):
        l1 = np.eye(2 * batch_size, k=batch_size)
        l2 = np.eye(2 * batch_size, k=-batch_size)
        mask = torch.from_numpy(l1 + l2)
        mask = mask.type(torch.bool)
        return mask.to(self.device)

    def _cosine_simililarity(self, x, y):
        # x shape: (2N, 1, C, H, W)
        # y shape: (1, 2N, C, H, W)
        # v shape: (2N, 2N, H, W)

        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        batch_size = zis.size(0)
        H, W = zis.size(2), zis.size(3)
        representations = torch.cat([zjs, zis], dim=0)

        positive_mask = self._get_correlated_mask(batch_size).type(torch.bool)
        negative_mask = self._get_uncorrelated_mask(batch_size).type(torch.bool)

        similarity_matrix = self.similarity_function(representations, representations)
        positives = similarity_matrix[positive_mask].view(2 * batch_size, 1, H, W)
        negatives = similarity_matrix[negative_mask].view(2*batch_size, -1, H, W)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros((2*batch_size, H, W)).to(self.device).long()
        loss = self.criterion(logits, labels)
        return loss/(2*batch_size*H*W)

class projector_SIMCLR(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(projector_SIMCLR, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.fc1 = nn.Conv2d(in_dim, in_dim, 1)
        self.fc2 = nn.Conv2d(in_dim, out_dim, 1)
        self.relu = torch.nn.ReLU(inplace=False)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class Model(nn.Module):
    def __init__(self, receptive='largest'):
        super(Model, self).__init__()
        self.receptive = receptive
        if self.receptive=='largest':
            self.backbone = Backbone()
        elif self.receptive=='small':
            self.backbone = BackboneSmall()
        elif self.receptive=='finer':
            self.backbone = BackboneFiner()
        self.projector = projector_SIMCLR(128, 64)

    def forward_feat(self, v):
        return self.backbone(v)

    def forward_proj(self, v):
        return self.projector(v)

    def forward(self, v1 , v2):
        f1 = self.forward_feat(v1)
        f2 = self.forward_feat(v2)

        z1 = self.forward_proj(f1)
        z2 = self.forward_proj(f2)
        return z1, z2

