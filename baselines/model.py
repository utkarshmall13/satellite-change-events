import torch.nn as nn
from torchvision import models
import torch
import numpy as np
from torch import optim
from copy import deepcopy
from torch.optim import lr_scheduler

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
		elif backbone=='r10':
			resnet_model = models.resnet10(pretrained=pretrained)
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
			# if i == 7:
			# 	self.layer4 = child
			if i == 8:
				self.ap = child
			if i == 9:
				self.fcc = child


	def forward(self, x, mask):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x1 = self.layer1(x)
		x2 = self.layer2(x1)
		x3 = self.layer3(x2)
		# print(x3.size())

		x3 = x3*torch.unsqueeze(mask, dim=1)
		# x4 = self.layer4(x3)
		nume = torch.sum(x3, dim=(2, 3))
		deno = torch.unsqueeze(torch.sum(mask, dim=(1, 2)), dim=1)
		feats = (nume/deno)
		return feats


class NTXentLoss(torch.nn.Module):
	def __init__(self, device, batch_size, temperature, use_cosine_similarity):
		super(NTXentLoss, self).__init__()
		self.batch_size = batch_size
		self.temperature = temperature
		self.device = device
		self.softmax = torch.nn.Softmax(dim=-1)
		self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
		self.similarity_function = self._get_similarity_function(
			use_cosine_similarity)
		self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

	def _get_similarity_function(self, use_cosine_similarity):
		if use_cosine_similarity:
			self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
			return self._cosine_simililarity
		else:
			return self._dot_simililarity

	def _get_correlated_mask(self):
		diag = np.eye(2 * self.batch_size)
		l1 = np.eye((2 * self.batch_size), 2 *
					self.batch_size, k=-self.batch_size)
		l2 = np.eye((2 * self.batch_size), 2 *
					self.batch_size, k=self.batch_size)
		mask = torch.from_numpy((diag + l1 + l2))
		mask = (1 - mask).type(torch.bool)
		return mask.to(self.device)

	@staticmethod
	def _dot_simililarity(x, y):
		v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
		# x shape: (N, 1, C)
		# y shape: (1, C, 2N)
		# v shape: (N, 2N)
		return v

	def _cosine_simililarity(self, x, y):
		# x shape: (N, 1, C)
		# y shape: (1, 2N, C)
		# v shape: (N, 2N)
		v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
		return v

	def forward(self, zis, zjs):
		representations = torch.cat([zjs, zis], dim=0)

		similarity_matrix = self.similarity_function(
			representations, representations)

		# filter out the scores from the positive samples
		l_pos = torch.diag(similarity_matrix, self.batch_size)
		r_pos = torch.diag(similarity_matrix, -self.batch_size)
		positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

		negatives = similarity_matrix[self.mask_samples_from_same_repr].view(
			2 * self.batch_size, -1)

		logits = torch.cat((positives, negatives), dim=1)
		logits /= self.temperature

		labels = torch.zeros(2 * self.batch_size).to(self.device).long()
		loss = self.criterion(logits, labels)

		return loss/(2*self.batch_size)

class projector_SIMCLR(nn.Module):
	def __init__(self, in_dim, out_dim):
		super(projector_SIMCLR, self).__init__()
		self.in_dim = in_dim
		self.out_dim = out_dim

		self.fc1 = nn.Linear(in_dim, in_dim)
		self.fc2 = nn.Linear(in_dim, out_dim)
		self.relu = torch.nn.ReLU(inplace=False)

	def forward(self, x):
		return self.fc2(self.relu(self.fc1(x)))


class Model(nn.Module):
	def __init__(self, pretrained=False):
		super(Model, self).__init__()
		self.backbone = Backbone(pretrained=pretrained)
		self.projector = projector_SIMCLR(512, 64)

	def forward_feat(self, v, m):
		return self.backbone(v, m)

	def forward_proj(self, v):
		return self.projector(v)

	def forward(self, v11 , v12, v21 , v22, mask1, mask2):
		f11 = self.forward_feat(v11, mask1)
		f12 = self.forward_feat(v12, mask1)

		f21 = self.forward_feat(v21, mask2)
		f22 = self.forward_feat(v22, mask2)

		f1 = torch.cat((f11, f12), dim=1)
		f2 = torch.cat((f21, f22), dim=1)

		z1 = self.forward_proj(f1)
		z2 = self.forward_proj(f2)
		return z1, z2

	def forward_single(self, v11 , v12, mask1):
		f11 = self.forward_feat(v11, mask1)
		f12 = self.forward_feat(v12, mask1)
		f1 = torch.cat((f11, f12), dim=1)
		return f1

