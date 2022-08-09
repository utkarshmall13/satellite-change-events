from os.path import join, isfile, isdir
from os import listdir, mkdir
import pickle
from skimage.io import imread, imsave
from skimage.transform import rescale
import numpy as np
from multiprocessing import Pool
################################################################################
# Pickle functions


def unpickle(file):
	with open(file, 'rb') as fo:
		dic = pickle.load(fo)
	return dic


def inpickle(diction, file):
	with open(file, 'wb') as fo:
		pickle.dump(diction, fo)
################################################################################


idir = '../changes_apfeatfast_cairo_20_4/'
odir = '../cairoadevents/'
data_dir = '../sentinel_cairo/'
cloud_dir = '..//sentinel_cairomask'


files = sorted([tmp for tmp in listdir(idir) if isfile(join(idir, tmp))])
cuboids = [tmp.split('.')[0] for tmp in files]
imgfiles = [[join(cuboid, file) for file in sorted(listdir(join(data_dir, cuboid)))] for cuboid in cuboids]


def filter(file):
	print(file)
	data = np.load(join(idir, file))['label_images'][:-2]

	unique = np.unique(data)
	for label in unique:
		if label==0:
			continue
		size = (512, 512)
		inds = np.argwhere(data==label)
		# print(len(inds))
		if len(inds)>400:
			bbox = np.array([np.min(inds, axis=0), np.max(inds, axis=0)+1])
			tslic = bbox[:, 0]
			bbox = bbox[:, 1:]
			if bbox[1][0]-bbox[0][0]<bbox[1][1]-bbox[0][1]:
				x1 = bbox[0][0]-((bbox[1][1]-bbox[0][1])-(bbox[1][0]-bbox[0][0]))//2
				x2 = x1+(bbox[1][1]-bbox[0][1])

				if x1<0:
					x1 = 0
					x2 = (bbox[1][1]-bbox[0][1])
				elif x2>size[0]:
					x1 = size[0]-(bbox[1][1]-bbox[0][1])
					x2 = size[0]
				bbox[0, 0] = x1
				bbox[1, 0] = x2
			else:
				y1 = bbox[0][1]-((bbox[1][0]-bbox[0][0])-(bbox[1][1]-bbox[0][1]))//2
				y2 = y1+(bbox[1][0]-bbox[0][0])

				if y1<0:
					y1 = 0
					y2 = (bbox[1][0]-bbox[0][0])
				elif y2>size[1]:
					y1 = size[1]-(bbox[1][0]-bbox[0][0])
					y2 = size[1]
				bbox[0, 1] = y1
				bbox[1, 1] = y2

			mask = np.zeros((tslic[1]-tslic[0], size[0], size[1])).astype(np.uint8)
			for ind in inds:
				mask[ind[0]-tslic[0], ind[1], ind[2]] = 255
			mask = mask[:, bbox[0, 0]: bbox[1, 0], bbox[0, 1]: bbox[1, 1]]
			mask = rescale(mask, 2, anti_aliasing=False, order=0, channel_axis=0)
			bbox2 = bbox*2

			if len(np.argwhere(mask>0.5))>1600:
				print(tslic)
				masktmps = []
				idx = cuboids.index(file.split('.')[0])
				for t, i in enumerate(range(tslic[0], tslic[1])):
					print(i)
					cm1 = imread(join(cloud_dir, imgfiles[idx][i]))>(255/2)
					cm2 = imread(join(cloud_dir, imgfiles[idx][i+1]))>(255/2)
					cm = np.logical_or(cm1, cm2)[bbox2[0, 0]: bbox2[1, 0], bbox2[0, 1]: bbox2[1, 1]]
					masktmp = mask[t]*(1-cm)
					masktmps.append(masktmp)
				masktmps = np.array(masktmps)

				if len(np.argwhere(masktmps>0.5))>400:
					dirname = file.split('.')[0]+'_'+str(label).zfill(5)
					if not isdir(join(odir, dirname)):
						mkdir(join(odir, dirname))
					for t, i in enumerate(range(tslic[0], tslic[1])):
						print(i)
						imsave(join(odir, dirname, str(t).zfill(2)+'_mask.png'), masktmps[t])
						im1 = imread(data_dir+imgfiles[idx][i])[bbox2[0, 0]: bbox2[1, 0], bbox2[0, 1]: bbox2[1, 1]]
						imsave(join(odir, dirname, str(t).zfill(2)+'_img.png'), im1)
					im1 = imread(data_dir+imgfiles[idx][i+1])[bbox2[0, 0]: bbox2[1, 0], bbox2[0, 1]: bbox2[1, 1]]
					imsave(join(odir, dirname, str(t+1).zfill(2)+'_img.png'), im1)

pool = Pool()
ret = pool.map(filter, files)
pool.close()


# filter(files[1])
