# Group by image_id and see how many of them get plotted on the same image...
import pandas as pd
from remseno import *

data_dir = '../data/harvard/'
filename = f'{data_dir}NEON_all.csv'
df = pd.read_csv(filename)

c = Coords(filename, x_col='Longitude', y_col='Latitude', label_col='family',
           id_col='uid.x', sep=',', class1='Angiosperm', class2='Gymnosperm', crs='EPSG:4326')

image_ids = df['image_ids']
lat = df['Latitude'].values
lon = df['Longitude'].values
tree_ids = df['uid.x'].values

data = []
grouped = df.groupby('image_ids')
summer = ['2022-06-01T00:00:00.000Z',
          '2022-08-30T00:00:00.000Z',
          '2022-12-01T00:00:00.000Z',
          '2023-02-26T00:00:00.000Z']

image_map = []
for image_id, sub_df in grouped:
    # Make a df on the coordinates and see how many fall on the 500m2 that I downloaded... might need to make more
    # or we could use for validation.
    # Get xmin, xmax, ymin and ymax of the region and padd by 100m
    lon_min = np.min(sub_df['Longitude'].values)
    lon_max = np.max(sub_df['Longitude'].values)
    lat_min = np.min(sub_df['Latitude'].values)
    lat_max = np.max(sub_df['Latitude'].values)
    aoi = [[lon_min, lat_min],
            [lon_max, lat_min],
            [lon_max, lat_max],
            [lon_min, lat_max],
            [lon_min, lat_min]]
    try:
        # Get the proper image ID...
        image_id_new = select_image_ids(f'{data_dir}tree_ids/{image_id}.csv', aoi, summer,
                         max_cloud_cover=0.1, visible_percent=95)
        data.append([aoi, image_id_new, image_id])
        image_map.append([image_id, image_id_new])
    except:
        print(image_id)

with open('udpated_images.csv', 'w+') as fout:
    fout.write('old,new\n')
    for im in image_map:
        fout.write(f'{im[0]},{im[1]}\n')

asyncio.run(download(data))