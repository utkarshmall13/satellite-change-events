import torch
import torch.nn as nn
import argparse
import numpy as np
from model import Model, NTXentLoss
from dataset_simpler import main_transform, test_transform, seg_transform, tile_transform, SentinelDataset, FullDataset, EurosatDataset, SentinelDatasetCE
from torch import optim
from torch.optim import lr_scheduler
from os.path import join
# from scipy.spatial.distance import cosine
################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--learning-rate', '-lr', default=0.001, type=float)
parser.add_argument('--backbone', '-bb', default='r18')
parser.add_argument('--epochs', '-e', default=5, type=int)
parser.add_argument('--batch-size', '-bs', default=512, type=int)
parser.add_argument('--mode', '-m', default='train')
parser.add_argument('--model-type', '-mt', default='all')
parser.add_argument('--dataset-name', '-dn', default='CaiRoad')
parser.add_argument('--run', '-r', default=0, type=int)
args = parser.parse_args()
################################################################################
data_dir = join('../../../../neuripsdata/', args.dataset_name, 'events')
full_data_dir = join('../../../../neuripsdata/', args.dataset_name, 'fulldata')
eurosatdir = '../../../../neuripsdata/Eurosat/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model().to(device)
################################################################################
simclr_criterion = NTXentLoss('cuda', args.batch_size, 0.07, True)
optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

################################################################################
def save_model(model, fname):
	torch.save(model.state_dict(), fname)

def load_model(model, fname):
	model.load_state_dict(torch.load(fname))
################################################################################

if args.mode=='train':
	if args.model_type=='all':
		sen_dataset = SentinelDatasetCE(data_dir, transform=main_transform)
	elif args.model_type=='notime':
		sen_dataset = SentinelDatasetCE(data_dir, transform=main_transform, time=False)
	elif args.model_type=='nochange':
		sen_dataset = SentinelDatasetCE(data_dir, transform=main_transform, change=False)
	elif args.model_type=='fulldata':
		data_dir = full_data_dir
		sen_dataset = FullDataset(data_dir, transform=main_transform)
	elif args.model_type=='eurosat':
		data_dir = eurosatdir
		sen_dataset = EurosatDataset(data_dir, transform=main_transform)
	elif args.model_type=='imnet':
		exit()

	if args.model_type=='imnet':
		pass
	train_dataloader = torch.utils.data.DataLoader(sen_dataset, batch_size=args.batch_size, num_workers=32, shuffle=True, drop_last=True)
	# exit()

	for epoch in range(args.epochs):
		epoch_loss = []
		iter_loss = []
		model.train()
		for ite, (sample1, sample2) in enumerate(train_dataloader):
			optimizer.zero_grad()
			z1, z2 = model(sample1['first'].to(device), sample1['second'].to(device), sample2['first'].to(device), sample2['second'].to(device), sample1['mask'].to(device), sample2['mask'].to(device))
			loss_SIMCLR = simclr_criterion(z1, z2)
			loss_SIMCLR.backward()
			optimizer.step()
			epoch_loss.append(loss_SIMCLR.item())
			iter_loss.append(loss_SIMCLR.item())
			if ite%1==0:
				print('Iteration:', ite+1, 'Loss:', np.mean(iter_loss))
				iter_loss = []
		print('Epoch:', epoch, 'Loss:', np.mean(epoch_loss))

		save_model(model, 'models/saved_model_'+args.dataset_name+'_'+args.model_type+'_'+str(args.learning_rate)+'_'+str(epoch).zfill(2)+'.pth.tar')
		exp_lr_scheduler.step()

################################################################################
elif args.mode=='infer':
	if args.model_type=='imnet':
		pass
	elif args.model_type=='3dseg':
		import sys
		sys.path.append('/phoenix/S5a/ukm4/2122projects/disco/src/baselines/3D_SITS_Clustering/simpler')
		from models import Encoder
		model = Encoder(3, 9, 10, 3, [32, 32, 64, 64]).to(device)
		print(model)
		model.load_state_dict(torch.load('/phoenix/S5a/ukm4/2122projects/disco/src/baselines/3D_SITS_Clustering/simpler/saved_models/encoder_cairo.pth'))
	elif args.model_type=='3dsegcha':
		import sys
		sys.path.append('/phoenix/S5a/ukm4/2122projects/disco/src/baselines/3D_SITS_Clustering/simpler')
		from models import Encoder
		model = Encoder(3, 9, 10, 3, [32, 32, 64, 64]).to(device)
		print(model)
		model.load_state_dict(torch.load('/phoenix/S5a/ukm4/2122projects/disco/src/baselines/3D_SITS_Clustering/simpler/saved_models/encoder_cairo_cha.pth'))
	elif args.model_type=='tile2vec':
		import sys
		sys.path.append('/phoenix/S5a/ukm4/2122projects/disco/src/baselines/tile2vec/src')
		from tilenet import make_tilenet
		model = make_tilenet(in_channels=3, z_dim=512)
		print(model)
		model.load_state_dict(torch.load('/phoenix/S5a/ukm4/2122projects/disco/src/baselines/tile2vec/models/TileNet_epoch50.pth'))
		model = model.to(device)
	else:
		if args.dataset_name=='CaiRoad':
			num_epochs = 9-args.run
		elif args.dataset_name=='CalFire':
			num_epochs = 19-args.run
		load_model(model, 'models/saved_model_'+args.dataset_name+'_'+args.model_type+'_'+str(args.learning_rate)+'_'+str(num_epochs).zfill(2)+'.pth.tar')

	if args.model_type=='all':
		sen_dataset = SentinelDatasetCE(data_dir, mode='test', transform=test_transform)
	elif args.model_type=='notime':
		sen_dataset = SentinelDatasetCE(data_dir, mode='test', transform=test_transform, time=False)
	elif args.model_type=='nochange':
		sen_dataset = SentinelDatasetCE(data_dir, mode='test', transform=test_transform, change=False)
	elif args.model_type=='3dseg' or args.model_type=='3dsegcha':
		sen_dataset = SentinelDatasetCE(data_dir, mode='test', transform=seg_transform)
	elif args.model_type=='tile2vec':
		sen_dataset = SentinelDatasetCE(data_dir, mode='test', transform=tile_transform)
	else:
		sen_dataset = SentinelDatasetCE(data_dir, mode='test', transform=test_transform)
	test_dataloader = torch.utils.data.DataLoader(sen_dataset, batch_size=args.batch_size, num_workers=32)

	model.eval()
	feats = []
	fnames = []
	for ite, (fname, sample) in enumerate(test_dataloader):
		if args.model_type=='3dseg' or args.model_type=='3dsegcha':
			f = sample['first'].to(device).unsqueeze(2)
			s = sample['second'].to(device).unsqueeze(2)
			m = sample['mask'].to(device)

			inp = torch.cat([f, s, s], 2)  # b * 3 * t* h* w

			b, c, t, h, w = inp.size(0), inp.size(1), inp.size(2), inp.size(3), inp.size(4)

			inp = inp.unfold(3, 9, 9).unfold(4, 9, 9).permute((0, 3, 4, 1, 2, 5, 6)).reshape(b*h//9*w//9, c, t, 9, 9)

			with torch.no_grad():
				feat, _ = model(inp)
			# b*h*w, 10
			feat = feat.reshape(b, h//9, w//9, 10)
			feat = (feat*m.unsqueeze(3)).sum((1, 2))/m.unsqueeze(3).sum((1, 2))
			feats.append(feat.detach().cpu().numpy())
			fnames+=fname
			print(len(feats))
		elif args.model_type=='tile2vec':
			f = sample['first'].to(device)
			s = sample['second'].to(device)
			m = sample['mask'].to(device)

			f = model.forward_pre(f)
			s = model.forward_pre(s)

			f = (f*m.unsqueeze(1)).sum((2, 3))/m.unsqueeze(1).sum((2, 3))
			s = (s*m.unsqueeze(1)).sum((2, 3))/m.unsqueeze(1).sum((2, 3))

			feat = torch.cat([f, s], dim=1)
			feats.append(feat.detach().cpu().numpy())
			fnames+=fname
			print(len(feats))

		else:
			feat = model.forward_single(sample['first'].to(device), sample['second'].to(device), sample['mask'].to(device))
			feat = feat.detach().cpu().numpy()
			print(feat.shape)
			feats.append(feat)
			fnames+=fname
	feats = np.concatenate(feats, axis=0)
	np.savez_compressed('info/info'+args.dataset_name+'_'+args.model_type+'_'+str(args.run)+'.npz', fnames=fnames, feats=feats)
################################################################################
elif args.mode=='infeuro':
	if args.model_type=='imnet':
		pass
	elif args.model_type=='3dseg':
		import sys
		sys.path.append('/phoenix/S5a/ukm4/2122projects/disco/src/baselines/3D_SITS_Clustering/simpler')
		from models import Encoder
		model = Encoder(3, 9, 10, 3, [32, 32, 64, 64]).to(device)
		print(model)
		model.load_state_dict(torch.load('/phoenix/S5a/ukm4/2122projects/disco/src/baselines/3D_SITS_Clustering/simpler/saved_models/encoder_cairo.pth'))
	elif args.model_type=='3dsegcha':
		import sys
		sys.path.append('/phoenix/S5a/ukm4/2122projects/disco/src/baselines/3D_SITS_Clustering/simpler')
		from models import Encoder
		model = Encoder(3, 9, 10, 3, [32, 32, 64, 64]).to(device)
		print(model)
		model.load_state_dict(torch.load('/phoenix/S5a/ukm4/2122projects/disco/src/baselines/3D_SITS_Clustering/simpler/saved_models/encoder_cairo_cha.pth'))
	elif args.model_type=='tile2vec':
		import sys
		sys.path.append('/phoenix/S5a/ukm4/2122projects/disco/src/baselines/tile2vec/src')
		from tilenet import make_tilenet
		model = make_tilenet(in_channels=3, z_dim=512)
		print(model)
		model.load_state_dict(torch.load('/phoenix/S5a/ukm4/2122projects/disco/src/baselines/tile2vec/models/TileNet_epoch50.pth'))
		model = model.to(device)
	else:
		load_model(model, 'models/v'+version+'_'+args.model_type+'_'+str(args.learning_rate)+'.pth.tar')

	data_dir = '/phoenix/S5a/ukm4/datasets/Eurosat'
	if args.model_type=='3dseg' or args.model_type=='3dsegcha':
		sen_dataset = EurosatDataset(data_dir, mode='test', transform=seg_transform)
	elif args.model_type=='tile2vec':
		sen_dataset = EurosatDataset(data_dir, mode='test', transform=tile_transform)
	else:
		sen_dataset = EurosatDataset(data_dir, mode='test', transform=test_transform)

	test_dataloader = torch.utils.data.DataLoader(sen_dataset, batch_size=args.batch_size, num_workers=32)

	model.eval()
	feats = []
	fnames = []
	for ite, (fname, sample) in enumerate(test_dataloader):
		if args.model_type=='3dseg' or args.model_type=='3dsegcha':
			f = sample['first'].to(device).unsqueeze(2)
			s = sample['second'].to(device).unsqueeze(2)
			m = sample['mask'].to(device)

			inp = torch.cat([f, s, s], 2)  # b * 3 * t* h* w

			b, c, t, h, w = inp.size(0), inp.size(1), inp.size(2), inp.size(3), inp.size(4)

			inp = inp.unfold(3, 9, 9).unfold(4, 9, 9).permute((0, 3, 4, 1, 2, 5, 6)).reshape(b*h//9*w//9, c, t, 9, 9)

			with torch.no_grad():
				feat, _ = model(inp)
			# b*h*w, 10
			feat = feat.reshape(b, h//9, w//9, 10)
			feat = (feat*m.unsqueeze(3)).sum((1, 2))/m.unsqueeze(3).sum((1, 2))
			feats.append(feat.detach().cpu().numpy())
			fnames+=fname
			print(len(feats))
		elif args.model_type=='tile2vec':
			f = sample['first'].to(device)
			s = sample['second'].to(device)
			m = sample['mask'].to(device)

			f = model.forward_pre(f)
			s = model.forward_pre(s)

			f = (f*m.unsqueeze(1)).sum((2, 3))/m.unsqueeze(1).sum((2, 3))
			s = (s*m.unsqueeze(1)).sum((2, 3))/m.unsqueeze(1).sum((2, 3))

			feat = torch.cat([f, s], dim=1)
			feats.append(feat.detach().cpu().numpy())
			fnames+=fname
			print(len(feats))

		else:
			print(fname)
			feat = model.forward_single(sample['first'].to(device), sample['second'].to(device), sample['mask'].to(device))
			feat = feat.detach().cpu().numpy()
			print(feat.shape)
			feats.append(feat)
			fnames+=fname
	feats = np.concatenate(feats, axis=0)
	np.savez_compressed('euro/info'+version+'_'+args.model_type+'.npz', fnames=fnames, feats=feats)




