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

label = 'water'
filename = f'{data_dir}not_trees_df_image_ids_spring_{label}.csv'
meters = 3

c = Coords(filename, x_col='longitude', y_col='latitude', label_col='class',
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

