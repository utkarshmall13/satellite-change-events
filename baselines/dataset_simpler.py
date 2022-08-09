from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from os.path import join
from os import listdir
from skimage.transform import rescale, resize
from skimage.io import imsave
import numpy as np
from skimage.transform import rotate
import torch
import matplotlib.pyplot as plt
from copy import deepcopy
################################################################################
class ApplyTwice:
	def __init__(self, transform):
		self.transform = transform

	def __call__(self, img):
		return self.transform(img), self.transform(img)

class RandomCrop(object):
	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size

	def __call__(self, sample):
		first, second, mask = sample['first'], sample['second'], sample['mask']
		h, w = first.shape[:2]
		new_h, new_w = self.output_size

		top = np.random.randint(0, h - new_h)
		left = np.random.randint(0, w - new_w)

		topsm = top//16
		leftsm = left//16
		output_sizesm = (self.output_size[0]//16, self.output_size[1]//16)

		first = first[top: top + new_h, left: left + new_w]
		second = second[top: top + new_h, left: left + new_w]
		mask = mask[topsm: topsm + output_sizesm[0], leftsm: leftsm + output_sizesm[1]]
		return {'first': first, 'second': second, 'mask': mask}

class RandomRotation(object):
	def __init__(self):
		return

	def __call__(self, sample):
		first, second, mask = sample['first'], sample['second'], sample['mask']
		angle = np.random.randint(0, 360)
		first = rotate(first, angle, mode='reflect', order=1)
		second = rotate(second, angle, mode='reflect', order=1)
		mask = rotate(mask, angle, mode='constant', cval=0, order=0)
		
		return {'first': first, 'second': second, 'mask': mask}

class Resize(object):
	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size
		return

	def __call__(self, sample):
		first, second, mask = sample['first'], sample['second'], sample['mask']

		first = resize(first, self.output_size, mode='reflect', order=1)
		second = resize(second, self.output_size, mode='reflect', order=1)
		mask = resize(mask, (self.output_size[0]//16, self.output_size[0]//16), mode='constant', cval=0, order=0)
		
		return {'first': first, 'second': second, 'mask': mask}

class CenterCrop(object):
	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size

	def __call__(self, sample):
		first, second, mask = sample['first'], sample['second'], sample['mask']
		h, w = first.shape[:2]
		new_h, new_w = self.output_size

		top = (h - new_h)//2
		left = (w - new_w)//2

		first = first[top: top + new_h, left: left + new_w]
		second = second[top: top + new_h, left: left + new_w]


		topsm = top//16
		leftsm = left//16
		output_sizesm = (self.output_size[0]//16, self.output_size[1]//16)
		mask = mask[topsm: topsm + output_sizesm[0], leftsm: leftsm + output_sizesm[0]]

		return {'first': first, 'second': second, 'mask': mask}

class RandomHorizontalFlip(object):
	def __init__(self, p=0.5):
		assert isinstance(p, (int, float))
		assert p>=0 and p<=1
		self.p_flip = p

	def __call__(self, sample):
		first, second, mask = sample['first'], sample['second'], sample['mask']
		predp = np.random.random()
		if predp<self.p_flip:
			first = first[:, ::-1].copy()
			second = second[:, ::-1].copy()
			mask = mask[:, ::-1].copy()

		return {'first': first, 'second': second, 'mask': mask}

class RandomVerticalFlip(object):
	def __init__(self, p=0.5):
		assert isinstance(p, (int, float))
		assert p>=0 and p<=1
		self.p_flip = p

	def __call__(self, sample):
		first, second, mask = sample['first'], sample['second'], sample['mask']
		predp = np.random.random()
		if predp<self.p_flip:
			first = first[::-1, :].copy()
			second = second[::-1, :].copy()
			mask = mask[::-1, :].copy()

		return {'first': first, 'second': second, 'mask': mask}


class Normalize(object):
	def __init__(self, mean, std):
		self.normalizer = transforms.Normalize(mean, std)

	def __call__(self, sample):
		first, second, mask = sample['first'], sample['second'], sample['mask']
		first = self.normalizer(first)
		second = self.normalizer(second)
		return {'first': first, 'second': second, 'mask': mask}

class ToTensor(object):
	def __call__(self, sample):
		first, second, mask = sample['first'], sample['second'], sample['mask']
		first = first.transpose((2, 0, 1))
		second = second.transpose((2, 0, 1))
		# print(first.shape, second.shape)
		return {'first': torch.from_numpy(first).float(), 'second': torch.from_numpy(second).float(), 'mask': torch.from_numpy(mask).float()}



class ResizeSeg(object):
	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size
		return

	def __call__(self, sample):
		first, second, mask = sample['first'], sample['second'], sample['mask']

		first = resize(first, self.output_size, mode='reflect', order=1)
		second = resize(second, self.output_size, mode='reflect', order=1)
		mask = resize(mask, (self.output_size[0]//9, self.output_size[0]//9), mode='constant', cval=0, order=0)
		
		return {'first': first, 'second': second, 'mask': mask}

class CenterSeg(object):
	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size

	def __call__(self, sample):
		first, second, mask = sample['first'], sample['second'], sample['mask']
		h, w = first.shape[:2]
		new_h, new_w = self.output_size

		top = (h - new_h)//2
		left = (w - new_w)//2

		first = first[top: top + new_h, left: left + new_w]
		second = second[top: top + new_h, left: left + new_w]


		topsm = top//9
		leftsm = left//9
		output_sizesm = (self.output_size[0]//9, self.output_size[1]//9)
		mask = mask[topsm: topsm + output_sizesm[0], leftsm: leftsm + output_sizesm[0]]

		return {'first': first, 'second': second, 'mask': mask}

class ResizeTile(object):
	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size
		return

	def __call__(self, sample):
		first, second, mask = sample['first'], sample['second'], sample['mask']

		first = resize(first, self.output_size, mode='reflect', order=1)
		second = resize(second, self.output_size, mode='reflect', order=1)
		mask = resize(mask, (self.output_size[0]//14, self.output_size[0]//14), mode='constant', cval=0, order=0)
		
		return {'first': first, 'second': second, 'mask': mask}

class CenterTile(object):
	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size

	def __call__(self, sample):
		first, second, mask = sample['first'], sample['second'], sample['mask']
		h, w = first.shape[:2]
		new_h, new_w = self.output_size

		top = (h - new_h)//2
		left = (w - new_w)//2

		first = first[top: top + new_h, left: left + new_w]
		second = second[top: top + new_h, left: left + new_w]


		topsm = top//14
		leftsm = left//14
		output_sizesm = (self.output_size[0]//14, self.output_size[1]//14)
		mask = mask[topsm: topsm + output_sizesm[0], leftsm: leftsm + output_sizesm[0]]

		return {'first': first, 'second': second, 'mask': mask}
################################################################################
class SentinelDataset(Dataset):
	def __init__(self, data_dir, mode='train', transform=None, time=True, change=True):
		self.data_dir = data_dir
		self.mode = mode
		self.time = time
		self.change = change
		self.files1 = sorted(listdir(join(data_dir, 'img1')))
		self.files2 = sorted(listdir(join(data_dir, 'img2')))
		self.masks = sorted(listdir(join(data_dir, 'mask')))
		self.transform = transform
		print('data loaded', len(self.files1))

	def __len__(self):
		return len(self.masks)

	def __getitem__(self, idx):
		sample1 = np.array(Image.open(join(self.data_dir, 'img1', self.files1[idx])).convert('RGB'))
		if self.time:
			sample2 = np.array(Image.open(join(self.data_dir, 'img2', self.files1[idx])).convert('RGB'))
		else:
			sample2 = deepcopy(sample1)
		if self.change:
			mask = (np.array(Image.open(join(self.data_dir, 'mask', self.files1[idx])))>0.5).astype(np.float32)
		else:
			mask = (np.array(Image.open(join(self.data_dir, 'mask', self.files1[idx])))>-1).astype(np.float32)

		epsilon = 0.01
		mask+=epsilon

		main = {'first': sample1, 'second': sample2, 'mask': mask}
		if self.mode=='train':
			aug1 = self.transform(main)
			aug2 = self.transform(main)

			# print(aug1['first'].size())
			# print(aug1['second'].size())
			# print(aug1['mask'].size())
			# print(aug2['first'].size())
			# print(aug2['second'].size())
			# print(aug2['mask'].size())

			# plt.figure(figsize=(15, 10))
			# plt.subplot(2, 3, 1)
			# plt.imshow(aug1['first'].numpy().transpose((1, 2, 0)))
			# plt.subplot(2, 3, 2)
			# plt.imshow(aug1['second'].numpy().transpose((1, 2, 0)))
			# plt.subplot(2, 3, 3)
			# plt.imshow(aug1['mask'].numpy())

			# plt.subplot(2, 3, 4)
			# plt.imshow(aug2['first'].numpy().transpose((1, 2, 0)))
			# plt.subplot(2, 3, 5)
			# plt.imshow(aug2['second'].numpy().transpose((1, 2, 0)))
			# plt.subplot(2, 3, 6)
			# plt.imshow(aug2['mask'].numpy())
			# plt.savefig('egbib')
			# plt.close()

			return aug1, aug2
		elif self.mode=='test':
			transformed = self.transform(main)
			return self.files1[idx], transformed
################################################################################

class SentinelDatasetCE(Dataset):
	def __init__(self, data_dir, mode='train', transform=None, time=True, change=True):
		self.data_dir = data_dir
		self.mode = mode
		self.time = time
		self.change = change

		self.dirs = sorted(listdir(join(data_dir)))

		self.files1 = []
		self.files2 = []
		self.masks = []

		for dir in self.dirs:
			filetmp = sorted(listdir(join(data_dir, dir)))
			imgs = [tmp for tmp in filetmp if '_img' in tmp]
			masks = [tmp for tmp in filetmp if '_mask' in tmp]

			for i in range(len(masks)):
				self.files1.append(join(dir, imgs[i]))
				self.files2.append(join(dir, imgs[i+1]))
				self.masks.append(join(dir, masks[i]))
		self.transform = transform
		print('data loaded', len(self.files1))

	def __len__(self):
		return len(self.masks)

	def __getitem__(self, idx):
		# print(self.files1[idx], self.files2[idx], self.masks[idx])
		sample1 = np.array(Image.open(join(self.data_dir, self.files1[idx])).convert('RGB'))
		if self.time:
			sample2 = np.array(Image.open(join(self.data_dir, self.files2[idx])).convert('RGB'))
		else:
			sample2 = deepcopy(sample1)
		if self.change:
			mask = (np.array(Image.open(join(self.data_dir, self.masks[idx])))>0.5).astype(np.float32)
		else:
			mask = (np.array(Image.open(join(self.data_dir, self.masks[idx])))>-1).astype(np.float32)

		epsilon = 0.001
		mask+=epsilon

		main = {'first': sample1, 'second': sample2, 'mask': mask}
		if self.mode=='train':
			aug1 = self.transform(main)
			aug2 = self.transform(main)

			# print(aug1['first'].size())
			# print(aug1['second'].size())
			# print(aug1['mask'].size())
			# print(aug2['first'].size())
			# print(aug2['second'].size())
			# print(aug2['mask'].size())

			# plt.figure(figsize=(15, 10))
			# plt.subplot(2, 3, 1)
			# plt.imshow(aug1['first'].numpy().transpose((1, 2, 0)))
			# plt.subplot(2, 3, 2)
			# plt.imshow(aug1['second'].numpy().transpose((1, 2, 0)))
			# plt.subplot(2, 3, 3)
			# plt.imshow(aug1['mask'].numpy())

			# plt.subplot(2, 3, 4)
			# plt.imshow(aug2['first'].numpy().transpose((1, 2, 0)))
			# plt.subplot(2, 3, 5)
			# plt.imshow(aug2['second'].numpy().transpose((1, 2, 0)))
			# plt.subplot(2, 3, 6)
			# plt.imshow(aug2['mask'].numpy())
			# plt.savefig('egbib')
			# plt.close()

			return aug1, aug2
		elif self.mode=='test':
			transformed = self.transform(main)
			return self.files1[idx], transformed



################################################################################
class FullDataset(Dataset):
	def __init__(self, data_dir, mode='train', transform=None):
		self.data_dir = data_dir
		self.mode = mode

		self.dirs = sorted([join(self.data_dir, tmp) for tmp in listdir(self.data_dir)])
		self.filess = [[join(dir, tmp) for tmp in listdir(dir) if '_cloud' not in tmp] for dir in self.dirs]
		self.files = [(files[i], files[i+1]) for files in self.filess for i in range(len(files)-1)]
		self.transform = transform
		print('data loaded', len(self.files))

	def __len__(self):
		return len(self.files)

	def __getitem__(self, idx):
		dim = np.random.randint(0, 896, 2)
		sample1 = np.array(Image.open(self.files[idx][0]).convert('RGB'))[dim[0]:dim[0]+128, dim[1]:dim[1]+128]
		sample2 = np.array(Image.open(self.files[idx][1]).convert('RGB'))[dim[0]:dim[0]+128, dim[1]:dim[1]+128]
		mask = np.ones((sample1.shape[0], sample1.shape[1])).astype(np.float32)
		epsilon = 0.01
		mask+=epsilon

		main = {'first': sample1, 'second': sample2, 'mask': mask}
		if self.mode=='train':
			aug1 = self.transform(main)
			aug2 = self.transform(main)
			return aug1, aug2
		elif self.mode=='test':
			transformed = self.transform(main)
			return self.files[idx], transformed
################################################################################
class EurosatDataset(Dataset):
	def __init__(self, data_dir, mode='train', transform=None):
		self.data_dir = data_dir
		self.mode = mode
		self.dirs = sorted([join(self.data_dir, tmp) for tmp in listdir(self.data_dir)])
		self.filess = [[join(dir, tmp) for tmp in listdir(dir)] for dir in self.dirs]
		self.files = [files[i] for files in self.filess for i in range(len(files))]
		self.transform = transform
		# print(self.files)
		print('data loaded', len(self.files))

	def __len__(self):
		return len(self.files)

	def __getitem__(self, idx):
		sample1 = np.array(Image.open(self.files[idx]).convert('RGB'))
		sample2 = np.array(Image.open(self.files[idx]).convert('RGB'))
		mask = np.ones((sample1.shape[0], sample1.shape[1])).astype(np.float32)
		epsilon = 0.01
		mask+=epsilon

		main = {'first': sample1, 'second': sample2, 'mask': mask}
		if self.mode=='train':
			aug1 = self.transform(main)
			aug2 = self.transform(main)
			return aug1, aug2
		elif self.mode=='test':
			transformed = self.transform(main)
			return self.files[idx], transformed
################################################################################

normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# para visualizacion
# main_transform = transforms.Compose([RandomRotation(), Resize(128), RandomCrop(112), RandomVerticalFlip(), RandomHorizontalFlip(), ToTensor()])
main_transform = transforms.Compose([RandomRotation(), Resize(128), RandomCrop(112), RandomVerticalFlip(), RandomHorizontalFlip(), ToTensor(), normalize])
test_transform = transforms.Compose([Resize(112), CenterCrop(112), ToTensor(), normalize])
seg_transform = transforms.Compose([ResizeSeg(63), CenterSeg(63), ToTensor(), normalize])
tile_transform = transforms.Compose([ResizeTile(100), CenterTile(100), ToTensor(), normalize])
