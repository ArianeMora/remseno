from tqdm import tqdm
import pandas as pd
# Build the polygon
data_dir = '../data/harvard/'
df = pd.DataFrame(data=[['plot', 'harvard', 42.540000, -72.180000]], columns=['id', 'label', 'lat', 'lon'])
df.to_csv(f'{data_dir}plot.csv', index=False)

from remseno import *
import pandas as pd

c = Coords(f'{data_dir}plot.csv', x_col='lon', y_col='lat', label_col='label',
                   id_col='id', sep=',', class1='harvard', class2='harvard', crs='EPSG:4326')

ys = df['lat'].values
bbs = []
meters = 5000
for i, x in enumerate(df['lon'].values):
    bbs.append(c.build_polygon_from_centre_point(x, ys[i], meters, meters, crs='EPSG:4326'))
polygons = []
x0, x1, y0, y1 = [], [], [], []
for bs in bbs:
    cs = []
    for b in bs:
        cs.append([b[0], b[1]])
    polygons.append(cs)
    x0.append(min([x[1] for x in cs]))
    x1.append(max([x[1] for x in cs]))
    y0.append(min([x[0] for x in cs]))
    y1.append(max([x[0] for x in cs]))
df['x0'] = x0
df['x1'] = x1
df['y0'] = y0
df['y1'] = y1


image_ids = []
tree_ids = df['label'].values
labels = ['summer_2022', 'winter_2022', 'spring_2022', 'autumn_2022']
for xi, x in enumerate([summer_2022, winter_2022, spring_2022, autumn_2022]):
    for i in tqdm(range(0, len(polygons))):
        image_ids.append(select_image_ids(f'{data_dir}{labels[xi]}_{tree_ids[i]}.csv', polygons[i], x,
                                          max_cloud_cover=0.1, visible_percent=95))

data = []
image_ids = image_ids
lat = df['lat'].values[0]
lon = df['lon'].values[0]
for i, v in enumerate(image_ids):
    aoi = c.build_polygon_from_centre_point(lat, lon, 1000, 1000, "EPSG:4326")
    # For some reason need to swap it around classic no idea why...
    aoi = [[p[1], p[0]] for p in aoi]
    data.append([aoi, image_ids[i]])

asyncio.run(download(data))
