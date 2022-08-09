import datetime
from os.path import join, isfile, isdir
from os import mkdir
import numpy as np
import urllib.request
from multiprocessing.dummy import Pool
import csv
from dateutil.relativedelta import relativedelta
import wget
import ee

longi = -106.75
lati = 31.35

start = datetime.datetime(2015, 8, 1)
res = 0.05/2
odir = '../sentinel_calif'

metadata = "https://www.cs.cornell.edu/projects/satellite-change-events/static/data/all_forest_fires.csv"
wget.download(metadata, out='../')

ee.Initialize()

# downloading dataset for each row
def get_images(latlontime):
	print(latlontime)
	lati = latlontime[1]
	longi = latlontime[0]
	starttime = datetime.datetime(int(latlontime[2].split('-')[0]), int(latlontime[2].split('-')[1]), int(latlontime[2].split('-')[2]))
	endtime = datetime.datetime(int(latlontime[3].split('-')[0]), int(latlontime[3].split('-')[1]), int(latlontime[3].split('-')[2]))
	incid = latlontime[4]

	monthbefore = (starttime.replace(day=1)-datetime.timedelta(days=1)).replace(day=1)
	monthafter = (endtime+datetime.timedelta(days=30)).replace(day=1)
	print(monthbefore)
	print(monthafter)

	if starttime<start:
		return
	outdir = join(odir, incid)
	print(monthbefore, monthbefore+relativedelta(months=+1))

	months_start = monthbefore+relativedelta(months=-3)
	months_end = monthafter+relativedelta(months=+3)
	monthnow = months_start
	while True:
		collection = ee.ImageCollection('COPERNICUS/S2').filterDate(monthnow, monthnow+relativedelta(months=+1)).filterBounds(ee.Geometry.Point(longi-res, lati-res)).filterBounds(ee.Geometry.Point(longi+res, lati-res)).filterBounds(ee.Geometry.Point(longi-res, lati+res)).filterBounds(ee.Geometry.Point(longi+res, lati+res))
		image = collection.median().select('B4', 'B3', 'B2')
		output = 'bbox'+'_'+str(monthnow.year)+'_'+str(monthnow.month).zfill(2)
		print(output)
		try:
			url = image.getThumbURL({'name': output, 'format': 'jpg', 'region': [[longi-res, lati+res], [longi+res, lati+res], [longi+res, lati-res], [longi-res, lati-res]], 'min': 0, 'max': 4000, 'gamma': 1.0, 'dimensions': (1024, 1024)})
		except ee.ee_exception.EEException as e:
			print(e)
			monthnow = monthnow+relativedelta(months=+1)
			if monthnow==months_end:
				break
			continue
		if(not isdir(outdir)):
			mkdir(outdir)
		print(url)
		if(isfile(join(outdir, output+'.jpg'))):
			monthnow = monthnow+relativedelta(months=+1)
			if monthnow==months_end:
				break
			continue
		urllib.request.urlretrieve(url, join(outdir, output+'.jpg'))

		collection = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY').filterDate(monthnow, monthnow+relativedelta(months=+1)).filterBounds(ee.Geometry.Point(longi-res, lati-res)).filterBounds(ee.Geometry.Point(longi+res, lati-res)).filterBounds(ee.Geometry.Point(longi-res, lati+res)).filterBounds(ee.Geometry.Point(longi+res, lati+res))
		image = collection.median().select('probability')
		output = 'bbox_cloud'+'_'+str(monthnow.year)+'_'+str(monthnow.month).zfill(2)
		print(output)
		try:
			url = image.getThumbURL({'name': output, 'format': 'jpg', 'region': [[longi-res, lati+res], [longi+res, lati+res], [longi+res, lati-res], [longi-res, lati-res]], 'min': 0, 'max': 100, 'gamma': 1.0, 'dimensions': (1024, 1024)})
		except ee.ee_exception.EEException as e:
			print(e)
			monthnow = monthnow+relativedelta(months=+1)
			if monthnow==months_end:
				break
			continue
		if(not isdir(outdir)):
			mkdir(outdir)
		print(url)
		if(isfile(join(outdir, output+'.jpg'))):
			monthnow = monthnow+relativedelta(months=+1)
			if monthnow==months_end:
				break
			continue
		urllib.request.urlretrieve(url, join(outdir, output+'.jpg'))

		monthnow = monthnow+relativedelta(months=+1)
		if monthnow==months_end:
			break

allrows = []
with open('../all_forest_fires.csv') as ifd:
	reader = csv.reader(ifd, delimiter=',')
	for i, row in enumerate(reader):
		if i!=0:
			if len(row[8])>0:
				allrows.append([float(row[12]), float(row[13]), row[19], row[18], row[15]])

print(allrows)
pool = Pool(processes=16)
pool.map(get_images, allrows)
pool.close()


