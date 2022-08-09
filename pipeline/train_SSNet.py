import argparse
from SSNet import Model, NTXentLoss
from torch import optim
from torch.optim import lr_scheduler
import torch
from torch.utils.data import DataLoader
from dataset import SentinelDataset, DistinctSampler, transform, normalize
from dataset import SentinelDatasetTemp, DistinctSamplerTemp
from dataset import InferDataset, transform_infer
import torchvision.transforms.functional as TF
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from functools import reduce
from os import mkdir, listdir
from os.path import isdir, join
import imageio
from datetime import datetime

################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--learning-rate', '-lr', default=0.001, type=float)
parser.add_argument('--backbone', '-bb', default='r18')
parser.add_argument('--epochs', '-e', default=20, type=int)
parser.add_argument('--byfour', '-bf', default=False, action='store_true')
parser.add_argument('--temporal', '-t', default=False, action='store_true')
parser.add_argument('--small-receptive', '-sr', default=False, action='store_true')
parser.add_argument('--finer-receptive', '-fr', default=False, action='store_true')
parser.add_argument('--cloud-mask', '-cm', default=False, action='store_true')
parser.add_argument('--patch-aug', '-pa', default=False, action='store_true')
parser.add_argument('--batch-size', '-bs', default=4, type=int)
parser.add_argument('--mode', '-m', default='train')
args = parser.parse_args()
################################################################################
division_factor = 4
if args.byfour:
	ext = '_byfour'
else:
	ext = '_'
if args.temporal:
	ext += '_temporal'
else:
	ext += ''
if args.small_receptive:
	ext += '_small'
else:
	ext += ''

if args.finer_receptive:
	ext += '_finer'
	division_factor = 2
else:
	ext += ''

if args.cloud_mask:
	ext += '_cm'
else:
	ext += ''

if args.patch_aug:
	ext += '_pa'
else:
	ext += ''

ext = ext.replace('__', '_')
################################################################################
data_dir = '../sentinel_cairo'
DATA_PATH = '../sentinel_cairo/'
SAVE_PATH = '../changes_cairo/'
################################################################################
cali = True
if cali:
	data_dir = '../sentinel_calif'
	DATA_PATH = '../sentinel_calif'
	SAVE_PATH = '../changes_cali/'
################################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.small_receptive:
	model = Model(receptive='small').to(device)
elif args.finer_receptive:
	model = Model(receptive='finer').to(device)
else:
	model = Model().to(device)
################################################################################
simclr_criterion = NTXentLoss('cuda', 0.07)
optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
################################################################################

def color_jitter(inp, factor=(0.1, 0.1, 0.1, 0.05)):
	rands = np.random.uniform(size=3)
	rands = ((rands-0.5)*np.array(factor)[:3]*2)+1

	inp = TF.adjust_brightness(inp, rands[0])
	inp = TF.adjust_contrast(inp, rands[1])
	inp = TF.adjust_saturation(inp, rands[2])
	inp = TF.adjust_hue(inp, (np.random.uniform()*factor[3]*2)-factor[3])
	return inp

def random_rotate(inp):
	angle = np.random.randint(0, 360)
	rotated = TF.rotate(inp, angle, interpolation=TF.InterpolationMode.NEAREST)
	return angle, rotated

def rotate(inp, angle):
	rotated = TF.rotate(inp, angle, interpolation=TF.InterpolationMode.NEAREST)
	return rotated

def random_crop(inp):
	size = np.random.randint(150, 180)
	topleft = np.random.randint(38, size-150+39, 2)
	cropped = TF.crop(inp, topleft[0], topleft[1], size, size)
	return (topleft[0], topleft[1], size), cropped

def crop(inp, cropfactor):
	cropped = TF.crop(inp, cropfactor[0]//division_factor, cropfactor[1]//division_factor, cropfactor[2]//division_factor, cropfactor[2]//division_factor)
	return cropped


################################################################################
def save_model(model, fname):
	torch.save(model.state_dict(), fname)
	print('model saved')

def load_model(model, fname):
	model.load_state_dict(torch.load(fname))
	print('model loaded')
################################################################################

if args.mode=='train':
	image_size = 256
	if args.temporal:
		dataset = SentinelDatasetTemp(data_dir, 'train', transform)
		dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=16, sampler=DistinctSamplerTemp(dataset), drop_last=True)
	else:
		dataset = SentinelDataset(data_dir, 'train', transform)
		dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=16, sampler=DistinctSampler(dataset), drop_last=True)

	for epoch in range(args.epochs):
		epoch_loss = []
		iter_loss = []
		model.train()
		for ite, data in enumerate(dataloader):
			optimizer.zero_grad()
			if args.temporal:
				data1 = color_jitter(data[0].to(device))
				data2 = color_jitter(data[1].to(device))
			else:
				data = data.to(device)
				data1 = color_jitter(data)
				data2 = color_jitter(data)

			rotfactor, data1 = random_rotate(data1)
			cropfactor, data1 = random_crop(data1)
			data1 = TF.resize(data1, [image_size, image_size], TF.InterpolationMode.NEAREST)

			# data1 = data1.cpu().numpy().transpose(0, 2, 3, 1)
			# data2 = data2.cpu().numpy().transpose(0, 2, 3, 1)
			# plt.figure(figsize=(args.batch_size*5, 10))
			# for j in range(args.batch_size):
			# 	plt.subplot(2, args.batch_size, j+1)
			# 	plt.imshow(data1[j])
			# 	plt.subplot(2, args.batch_size, args.batch_size+j+1)
			# 	plt.imshow(data2[j])
			# plt.savefig('egbib2')
			# exit()

			data1 = normalize(data1)
			data2 = normalize(data2)

			output1, output2 = model(data1, data2)

			output2 = rotate(output2, rotfactor)
			output2 = crop(output2, cropfactor)
			output2 = TF.resize(output2, [image_size//division_factor, image_size//division_factor], TF.InterpolationMode.NEAREST)

			# print(output1.size())
			# print(output2.size())
			if args.byfour:
				output1 = torch.cat(torch.tensor_split(torch.cat(torch.tensor_split(output1, 4, dim=-1), dim=0), 4, dim=-2), dim=0)
				output2 = torch.cat(torch.tensor_split(torch.cat(torch.tensor_split(output2, 4, dim=-1), dim=0), 4, dim=-2), dim=0)
				if division_factor==2:
					indices = torch.randperm(output1.size(0))[:output1.size(0)//4]
					output1 = output1[indices]
					output2 = output2[indices]

			# print(output1.size())
			# print(output2.size())
			loss_SIMCLR = simclr_criterion(output1, output2)
			loss_SIMCLR.backward()
			optimizer.step()
			# print(loss_SIMCLR.item())
			epoch_loss.append(loss_SIMCLR.item())
			iter_loss.append(loss_SIMCLR.item())

			if ite%20==19:
				print('Epoch:', epoch+1,'Iteration:', ite+1, 'Loss:', np.mean(iter_loss))
				iter_loss = []
			if ite%100==99:
				save_model(model, 'models/v1_'+str(args.learning_rate)+ext+'.pth.tar')
		print('Epoch:', epoch+1, 'Loss:', np.mean(epoch_loss))
		save_model(model, 'models/v1_'+str(args.learning_rate)+ext+'.pth.tar')
		exp_lr_scheduler.step()

if args.mode=='test':
	def otsu(data, num=200):
		max_value = np.max(data)
		min_value = np.min(data)

		total_num = reduce(lambda x, y: x*y, list(data.shape))
		step_value = (max_value - min_value)/num
		value = min_value
		best_threshold = min_value
		best_inter_class_var = 0
		while value < max_value:
			data_1 = data <= value
			data_2 = data > value
			w1 = np.sum(data_1)/total_num
			w2 = np.sum(data_2)/total_num
			if np.sum(data_2)==0 or np.sum(data_1)==0:
				inter_class_var = 0
			else:
				mean_1 = np.sum(data*(data_1))/np.sum(data_1)
				mean_2 = np.sum(data*(data_2))/np.sum(data_2)
				inter_class_var = w1*w2*np.power((mean_1-mean_2), 2)
			if best_inter_class_var < inter_class_var:
				best_inter_class_var = inter_class_var
				best_threshold = value
			value += step_value
			# print(value, inter_class_var)

		bwp = np.zeros(data.shape)
		bwp = data > best_threshold
		bwp = bwp.astype(int)
		return bwp, best_threshold


	load_model(model, 'models/v1_'+str(args.learning_rate)+ext+'.pth.tar')
	if args.temporal:
		dataset = SentinelDatasetTemp(data_dir, 'test', transform)
		dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=16, sampler=range(10561, 10620))
	else:
		dataset = SentinelDataset(data_dir, 'test', transform)
		dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=16, sampler=range(10740, 10800))

	for ite, data in enumerate(dataloader):
		with torch.no_grad():
			if args.temporal:
				data = data[0].to(device)
			else:
				data = data.to(device)

			# radiometric correction
			whole_mean = torch.mean(data, dim=(0, 2, 3), keepdim=True)
			whole_std = torch.std(data, dim=(0, 2, 3), keepdim=True)

			slice_means = torch.mean(data, dim=(2, 3), keepdim=True)
			slice_stds = torch.std(data, dim=(2, 3), keepdim=True)

			data = ((data-slice_means)/slice_stds)*whole_std+whole_mean
			data = torch.clamp(data, min=0, max=1.0)
			# data = torch.max

			output = model.forward_feat(normalize(data)).cpu().numpy().transpose(0, 2, 3, 1)
			diff = np.sqrt(np.sum((output[:-1]-output[1:])**2, axis=-1))
			
			output = output/norm(output, axis=3, keepdims=True)
			diff = 1-np.sum((output[:-1]*output[1:]), axis=-1)
			print(diff.shape)
			bmap, thres = otsu(diff)
			print(thres)
			vmin, vmax = np.min(diff), np.max(diff)

			data = data.cpu().numpy().transpose(0, 2, 3, 1)
			for i in range(len(output)-1):
				plt.figure(figsize=(15, 5))
				plt.subplot(1, 4, 1)
				plt.imshow(data[i])
				plt.subplot(1, 4, 2)
				plt.imshow(data[i+1])
				plt.subplot(1, 4, 3)
				plt.imshow(diff[i], vmin=vmin, vmax=vmax)
				plt.subplot(1, 4, 4)
				plt.imshow(bmap[i], vmin=0, vmax=1.01)
				# plt.imshow(1-np.sum((output[i+1]*output[i]), axis=-1))
				if not isdir('test/'+ext):
					mkdir('test/'+ext)
				plt.savefig('test/'+ext+'/_change'+str(ite*args.batch_size+i).zfill(3))
				plt.close()


if args.mode=='inference':
	load_model(model, 'models/v1_'+str(args.learning_rate)+ext+'.pth.tar')

	def otsu(data, num=25):
		max_value = torch.max(data).item()
		min_value = torch.min(data).item()

		total_num = reduce(lambda x, y: x*y, list(data.size()))
		step_value = (max_value - min_value)/num
		value = min_value
		best_threshold = min_value
		best_inter_class_var = 0
		while value < max_value:
			data_1 = (data <= value)
			data_2 = (data > value)
			w1 = (torch.sum(data_1)/total_num).item()
			w2 = (torch.sum(data_2)/total_num).item()
			if torch.sum(data_2).item()==0 or torch.sum(data_1).item()==0:
				inter_class_var = 0
			else:
				mean_1 = (torch.sum(data*(data_1))/torch.sum(data_1)).item()
				mean_2 = (torch.sum(data*(data_2))/torch.sum(data_2)).item()
				inter_class_var = w1*w2*np.power((mean_1-mean_2), 2)
			if best_inter_class_var < inter_class_var:
				best_inter_class_var = inter_class_var
				best_threshold = value
			value += step_value
			# print(value, inter_class_var)

		bwp = (data > best_threshold).cpu().numpy()
		return bwp, best_threshold

	dirs = sorted(listdir(DATA_PATH))
	for dir  in dirs:
		before = datetime.now()
		dataset = InferDataset(join(DATA_PATH, dir), transform_infer)
		dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=32)

		F = []
		Dirname = []
		Feats = []
		with torch.no_grad():
			for i, (f, data) in enumerate(dataloader):
				data = data.to(device)
				whole_mean = torch.mean(data, dim=(0, 2, 3), keepdim=True)
				whole_std = torch.std(data, dim=(0, 2, 3), keepdim=True)
				slice_means = torch.mean(data, dim=(2, 3), keepdim=True)
				slice_stds = torch.std(data, dim=(2, 3), keepdim=True)
				data = ((data-slice_means)/slice_stds)*whole_std+whole_mean
				data = torch.clamp(data, min=0, max=1.0)
				output = model.forward_feat(normalize(data)).permute(0, 2, 3, 1)
				F.append(f)
				Dirname.append(np.array([tmp.split('/')[-2] for tmp in f]))
				output = (output/torch.norm(output, dim=3, keepdim=True)).cpu().numpy()
				Feats.append(output)
				print(i)

		Feats = np.concatenate(Feats, axis=0)
		Dirname = np.concatenate(Dirname, axis=0)
		print(Dirname)
		F = np.concatenate(F, axis=0)
		after = datetime.now()
		Feats = torch.from_numpy(Feats).float()
		after = datetime.now()
		print("time taken: ", after-before)
	
		Diffs = 1-torch.sum((Feats*torch.roll(Feats, -1, dims=0)), dim=-1)	
		Diffs2 = 1-torch.sum((Feats*torch.roll(Feats, -2, dims=0)), dim=-1)	
		after = datetime.now()
		print("time taken: ", after-before)

		binary_change_maps, thresh = otsu(Diffs)
		binary_change_maps2 = (Diffs2>thresh).cpu().numpy()
		after = datetime.now()
		print("time taken: ", after-before)

		binary_change_maps = np.logical_and(binary_change_maps, binary_change_maps2).astype(int)*255

		print(thresh)
		for i in range(len(binary_change_maps)):
			if not isdir(join(SAVE_PATH, Dirname[i])):
				mkdir(join(SAVE_PATH, Dirname[i]))
			imageio.imwrite(join(SAVE_PATH, Dirname[i], 'KPCAMNet_BCM_'+str(i).zfill(2)+'.png'), binary_change_maps[i].astype(np.uint8))
	
		after = datetime.now()
		print("time taken: ", after-before)

