import ee
import datetime
from subprocess import call

from PIL import Image
from os.path import join, isfile, isdir
from os import listdir, remove, mkdir
import numpy as np
import urllib.request
from multiprocessing.dummy import Pool

pool = Pool(processes=16)

def cloudmask457(image):
	qa = image.select('QA60')
	mask = (qa.bitwiseAnd(1 << 10).eq(0)).And(qa.bitwiseAnd(1 << 11)).eq(0)
	return image.updateMask(mask).divide(10000)

def correction(image, min=0, max=30000, gamma=1.0):
	image = (np.float_power(np.maximum(np.minimum(image, max-1), min)/max, 1/gamma)*256).astype(np.uint8)
	return image

longi = 31.75
lati = 30.05
year = 2016
quad = 0
yeardiv = 12

res = 0.05
odir = '../sentinel_cairo'

ee.Initialize()

# downloading dataset for each tile
def get_images(latlon):
	print(latlon)
	lati = latlon[0]
	longi = latlon[1]
	outdir = join(odir, str(int(lati*100))+'_'+str(int(longi*100)))
	if(not isdir(outdir)):
		mkdir(outdir)
	# for longi in np.arange (-180, 180, res):
	for year in range(2015, 2021):
		for quad in range(yeardiv):
			print(datetime.datetime(year, quad+1, 1), datetime.datetime(year+(quad+1)//yeardiv, (quad+1)%yeardiv+1, 1))
			collection = ee.ImageCollection('COPERNICUS/S2').filterDate(datetime.datetime(year, quad+1, 1), datetime.datetime(year+(quad+1)//yeardiv, (quad+1)%yeardiv+1, 1)).filterBounds(ee.Geometry.Point(longi, lati)).filterBounds(ee.Geometry.Point(longi+res, lati)).filterBounds(ee.Geometry.Point(longi, lati+res)).filterBounds(ee.Geometry.Point(longi+res, lati+res)).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
			collection = collection.map(cloudmask457)
			image = collection.median().select('B4', 'B3', 'B2')
			# image = image.visualize(})
			output = 'bbox'+str(int(longi*1000))+'_'+str(int(lati*1000))+'_'+str(year)+'_'+str(quad).zfill(2)
			print(output)
			try:
				url = image.getThumbURL(
					{'name': output, 'format': 'jpg', 'region': [[longi, lati+res], [longi+res, lati+res], [longi+res, lati], [longi, lati]], 'min': 0, 'max': 0.3, 'gamma': 1.0, 'dimensions': (1024, 1024)})
			except ee.ee_exception.EEException as e:
				print(e)
				# OUTER_BREAK = True
				continue
			print(url)

			if(isfile(join(outdir, output+'.jpg'))):
				continue
			urllib.request.urlretrieve(url, join(outdir, output+'.jpg'))


			print(datetime.datetime(year, quad+1, 1), datetime.datetime(year+(quad+1)//yeardiv, (quad+1)%yeardiv+1, 1))
			collection = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY').filterDate(datetime.datetime(year, quad+1, 1), datetime.datetime(year+(quad+1)//yeardiv, (quad+1)%yeardiv+1, 1)).filterBounds(ee.Geometry.Point(longi, lati)).filterBounds(ee.Geometry.Point(longi+res, lati)).filterBounds(ee.Geometry.Point(longi, lati+res)).filterBounds(ee.Geometry.Point(longi+res, lati+res))
			image = collection.median().select('probability')
			# image = image.visualize(})
			output = 'bbox'+str(int(longi*1000))+'_'+str(int(lati*1000))+'_'+str(year)+'_'+str(quad).zfill(2)
			print(output)
			try:
				url = image.getThumbURL(
					{'name': output, 'format': 'jpg', 'region': [[longi, lati+res], [longi+res, lati+res], [longi+res, lati], [longi, lati]], 'min': 0, 'max': 100, 'gamma': 1.0, 'dimensions': (1024, 1024)})
			except ee.ee_exception.EEException as e:
				print(e)
				# OUTER_BREAK = True
				continue
			print(url)

			if(isfile(join(outdir, output+'.jpg'))):
				continue
			urllib.request.urlretrieve(url, join(outdir, output+'.jpg'))

longs = list(np.arange(longi-(res*10), longi+(res*10), res))
lats = list(np.arange(lati-(res*10), lati+(res*10), res))
rows = []
for lon in longs:
	for lat in lats:
		rows.append((lat, lon))
print(rows)
pool.map(get_images, rows)


