from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from os.path import join, isdir
from os import listdir
import numpy as np
import torchvision.transforms.functional as TF

import torch
from torch.utils.data.sampler import Sampler
################################################################################
class SentinelDataset(Dataset):
	def __init__(self, data_dir, mode='train', transform=None):
		self.data_dir = data_dir
		self.mode = mode
		self.dirs = [tmp for tmp in sorted(listdir(self.data_dir)) if isdir(join(self.data_dir, tmp))]
		self.images = [[join(dir, img) for img in sorted(listdir(join(self.data_dir, dir)))[:60]] for dir in self.dirs if len(listdir(join(self.data_dir, dir)))>=60]
		self.images = [tmp for dir in self.images for tmp in dir]
		self.transform = transform[mode]
		print('data loaded', len(self.images))

	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):
		sample = np.array(Image.open(join(self.data_dir, self.images[idx])).convert('RGB'))
		aug = self.transform(sample)
		return aug

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transform = transforms.Compose([transforms.ToTensor(), transforms.RandomCrop(256), transforms.RandomHorizontalFlip()])
test_transform = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(256)])
transform = {'train': train_transform, 'test': test_transform}

class DistinctSampler(Sampler):
	def __init__(self, data_source, images_per_time=60):
		self.data_source = data_source
		self.images_per_time = images_per_time

	def __len__(self):
		return len(self.data_source)

	def smart_perm(self):
		spaceind = []
		for i in range(self.images_per_time):
			#could be fixed mode
			spaceind.append(np.random.permutation(len(self.data_source)//self.images_per_time))
		spaceind = np.concatenate(spaceind)

		timeind = []
		for i in range(len(self.data_source)//self.images_per_time):
			timeind.append(np.random.permutation(self.images_per_time))
		timeind = np.concatenate(timeind)

		ind = (spaceind*self.images_per_time +timeind).tolist()
		return ind

	def __iter__(self):
		yield from self.smart_perm()
################################################################################

class SentinelDatasetTemp(Dataset):
	def __init__(self, data_dir, mode='train', transform=None):
		self.data_dir = data_dir
		self.mode = mode
		self.dirs = [tmp for tmp in sorted(listdir(self.data_dir)) if isdir(join(self.data_dir, tmp))]
		self.images = [[join(dir, img) for img in sorted(listdir(join(self.data_dir, dir)))[:60]] for dir in self.dirs if len(listdir(join(self.data_dir, dir)))>=60]
		self.images = [(dir[i], dir[i+1]) for dir in self.images for i in range(len(dir)-1)]
		self.transform = transform[mode]
		print('data loaded', len(self.images))

	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):
		sample1 = np.array(Image.open(join(self.data_dir, self.images[idx][0])).convert('RGB'))
		sample2 = np.array(Image.open(join(self.data_dir, self.images[idx][1])).convert('RGB'))

		sample = np.concatenate([sample1, sample2], axis=2)
		aug = self.transform(sample)		
		return aug[:3], aug[3:]


class DistinctSamplerTemp(Sampler):
	def __init__(self, data_source, images_per_time=59):
		self.data_source = data_source
		self.images_per_time = images_per_time

	def __len__(self):
		return len(self.data_source)

	def smart_perm(self):
		spaceind = []
		for i in range(self.images_per_time):
			#could be fixed mode
			spaceind.append(np.random.permutation(len(self.data_source)//self.images_per_time))
		spaceind = np.concatenate(spaceind)

		timeind = []
		for i in range(len(self.data_source)//self.images_per_time):
			timeind.append(np.random.permutation(self.images_per_time))
		timeind = np.concatenate(timeind)

		ind = (spaceind*self.images_per_time +timeind).tolist()
		return ind

	def __iter__(self):
		yield from self.smart_perm()

################################################################################
class InferDataset(Dataset):
	def __init__(self, data_dir, transform=None):
		self.data_dir = data_dir
		self.images = [join(self.data_dir, img) for img in sorted(listdir(self.data_dir))]
		self.transform = transform
		print('data loaded', len(self.images))

	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):
		sample = np.array(Image.open(join(self.data_dir, self.images[idx])).convert('RGB'))
		aug = self.transform(sample)
		return self.images[idx], aug

transform_infer = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(1024)])

