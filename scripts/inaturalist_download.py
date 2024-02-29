import pandas as pd

neon_colour = '#AD67E4'
tallo_colour = '#B8E467'

data_dir = '../data/to_publish/'
fig_dir = f'{data_dir}figs/'
# Read in Tallo and Neon

summer_2022 = ["2022-06-01T00:00:00.000Z", "2022-08-30T00:00:00.000Z",
               "2022-12-01T00:00:00.000Z", "2023-02-26T00:00:00.000Z"]

winter_2022 = ["2022-01-12T00:00:00.000Z", "2023-02-26T00:00:00.000Z",
               "2022-06-01T00:00:00.000Z", "2022-08-30T00:00:00.000Z"]

spring_2022 = ["2022-04-01T00:00:00.000Z", "2023-05-30T00:00:00.000Z",
               "2022-09-01T00:00:00.000Z", "2022-10-30T00:00:00.000Z"]

autumn_2022 = ["2022-09-01T00:00:00.000Z", "2022-10-30T00:00:00.000Z",
               "2022-04-01T00:00:00.000Z", "2023-05-30T00:00:00.000Z"]

summer_2023 = ["2023-06-01T00:00:00.000Z", "2023-08-30T00:00:00.000Z",
               "2023-12-01T00:00:00.000Z", "2023-02-26T00:00:00.000Z"]

winter_2023 = ["2023-01-12T00:00:00.000Z", "2023-02-26T00:00:00.000Z",
               "2023-06-01T00:00:00.000Z", "2023-08-30T00:00:00.000Z"]

spring_2023 = ["2023-04-01T00:00:00.000Z", "2023-05-30T00:00:00.000Z",
               "2023-09-01T00:00:00.000Z", "2023-10-30T00:00:00.000Z"]

autumn_2023 = ["2023-09-01T00:00:00.000Z", "2023-10-30T00:00:00.000Z",
               "2023-04-01T00:00:00.000Z", "2023-05-30T00:00:00.000Z"]

summer_2021 = ["2021-06-01T00:00:00.000Z", "2021-08-30T00:00:00.000Z",
               "2021-12-01T00:00:00.000Z", "2021-02-26T00:00:00.000Z"]

winter_2021 = ["2021-01-12T00:00:00.000Z", "2021-02-26T00:00:00.000Z",
               "2021-06-01T00:00:00.000Z", "2021-08-30T00:00:00.000Z"]

spring_2021 = ["2021-04-01T00:00:00.000Z", "2021-05-30T00:00:00.000Z",
               "2021-09-01T00:00:00.000Z", "2021-10-30T00:00:00.000Z"]

autumn_2021 = ["2021-09-01T00:00:00.000Z", "2021-10-30T00:00:00.000Z",
               "2021-04-01T00:00:00.000Z", "2021-05-30T00:00:00.000Z"]


import pandas as pd
from remseno import *

c = Coords(f'{data_dir}combined_df.csv', x_col='longitude', y_col='latitude', label_col='class',
                   id_col='id', sep=',', class1='A', class2='B', crs='EPSG:4326')
df = c.df
ys = df['latitude'].values
bbs = []
meters = 3
for i, x in enumerate(df['longitude'].values):
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
tree_ids = df['id'].values
# We want to get from summer 2022
labels = [summer_2022]
# , spring_2021, winter_2021, autumn_2021,
#           summer_2022, spring_2022, winter_2022, autumn_2022,
#           summer_2023, spring_2023, winter_2023, autumn_2023
#          ]
max_count = 100000
for xi, x in enumerate(labels):
    for i in tqdm(range(0, len(polygons))):
        try:
            image_ids.append(select_image_ids(f'{data_dir}tree_files/{labels[xi]}_{tree_ids[i]}.csv', polygons[i], x,
                                          max_cloud_cover=0.1, visible_percent=95))
        except Exception as e:
            image_ids.append('NA')

        if i > max_count:
            break
df['image_ids'] = image_ids
df.to_csv(f'{data_dir}combined_df_image_ids.csv', index=False)
filename = f'{data_dir}combined_df_image_ids.csv'
meters = 3

c = Coords(f'{data_dir}combined_df.csv', x_col='longitude', y_col='latitude', label_col='class',
                   id_col='id', sep=',', class1='A', class2='B', crs='EPSG:4326')


df = pd.read_csv(filename)
image_ids = df['image_ids']
lat = df['latitude'].values
lon = df['longitude'].values
tree_ids = [f'tree_{i}' for i in df['id'].values]
data = []
for i, v in enumerate(image_ids):
    aoi = c.build_polygon_from_centre_point(lat[i], lon[i], meters, meters, "EPSG:4326")
    # For some reason need to swap it around classic no idea why...
    aoi = [[p[1], p[0]] for p in aoi]
    data.append([aoi, image_ids[i], tree_ids[i]])

print(len(data))
print(data[0])
print(data[-1])
asyncio.run(download(data))

