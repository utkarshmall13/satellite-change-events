from multiprocessing import Pool
from os import listdir, mkdir
from os.path import join, isdir
import numpy as np
from skimage.io import imread, imsave
from skimage.color import label2rgb
from sklearn.metrics import pairwise_distances
from dataset import InferDataset, transform_infer
from SSNet import Model
from sklearn.neighbors import KDTree
from dataset import normalize
from torch.utils.data import DataLoader
import torch
import fbpca

from scipy.sparse.csgraph import connected_components
from scipy import sparse 
from datetime import datetime

################################################################################
imdir = '../sentinel_cairo/'
idir = '../changes'

time_dim = 4
threshold = 20
odir = '../changes_apfeatfast_cairo_'+str(threshold)+'_'+str(time_dim)

K = 20
offset = 1
batch_size = 8

start = 0
end = 400

################################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(receptive='finer').to(device)
def load_model(model, fname):
	model.load_state_dict(torch.load(fname))
	print('model loaded')
load_model(model, 'models/v1_0.001_byfour_finer.pth.tar')
################################################################################
dirs = sorted(listdir(idir))
dirs = dirs[start:end]
# dirs = sorted(listdir(idir))
for dir in dirs:
	files = sorted(listdir(join(idir, dir)))
	before = datetime.now()
	dataset = InferDataset(join(imdir, dir), transform_infer)
	dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=32)

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

	transformed_images = np.concatenate(Feats, axis=0)
	Dirname = np.concatenate(Dirname, axis=0)
	print(Dirname)
	F = np.concatenate(F, axis=0)
	after = datetime.now()
	print("time taken: ", after-before)

	print(transformed_images.shape)

	images = []
	label_images = []
	for find, file in enumerate(files):
		images.append(imread(join(idir, dir, file)).astype(int))

	images = np.array(images)
	for i in range(len(images)):
		if np.sum(images[i]>0.5)/np.sum(images[i]>-0.5)>0.1:
			images[i, : ,:] = 0


	feats = np.argwhere(images)
	img_feat = np.array([transformed_images[tmp[0], tmp[1], tmp[2]] for tmp in feats])
	img_feat, _, _ = fbpca.pca(img_feat, k=16)
	print(img_feat.shape)

	feats[:, 0] = feats[:, 0]*time_dim

	if len(img_feat)>5000:
		temp = img_feat[np.random.choice(len(img_feat), 5000, replace=False)]
	else:
		temp = img_feat
	threshold_img = np.mean(pairwise_distances(temp))*0.5
	print(threshold_img)

	nbrs = KDTree(feats)
	inds = nbrs.query_radius(feats, r=threshold)
	print(len(inds))

	Is = []
	Js = []
	# for i, ind in enumerate(inds):
	def get_indices(i):
		argind = np.argwhere(pairwise_distances([img_feat[i]], img_feat[inds[i]])[0]<threshold_img)[:, 0]
		ind = inds[i][argind]
		return ind
	pool = Pool(64)
	Js = pool.map(get_indices, list(range(len(feats))))
	pool.close()
	Is = [np.full_like(ind, i) for i, ind in enumerate(Js)]
	Js = np.concatenate(Js)
	Is = np.concatenate(Is)

	print(len(inds))
	print(len(Is))
	data = np.ones_like(Is)
	print(Is, Js, data)
	adj = sparse.coo_matrix((data, (Is, Js)))
	print(len(inds))
	n_components, labels = connected_components(adj, directed=False)
	print(n_components, labels)

	label_images = np.zeros(images.shape).astype(int)
	for feat, label in zip(feats, labels):
		label_images[feat[0]//time_dim, feat[1], feat[2]] = label
	
	if not isdir(join(odir, dir)):
		mkdir(join(odir, dir))
	for label_image, file in zip(label_images, files):
		color_image = label2rgb(label_image)*255
		color_image = color_image*np.expand_dims((label_image>0), axis=2).astype(np.uint8)
		imsave(join(odir, dir, file), color_image.astype(np.uint8), check_contrast=False)
	# break
	np.savez_compressed(join(odir, dir+'.npz'), label_images=label_images)
	# break			 
	after = datetime.now()
	print("time taken: ", after-before)

