from remseno import *

# Build the polygon
data_dir = '../data/to_publish/'
filename = f'{data_dir}inat_grouped_locations_image_ids.csv'
meters = 30

c = Coords(f'{data_dir}inat_grouped_locations_image_ids.csv', x_col='longitude', y_col='latitude', label_col='family',
                   id_col='id', sep=',', class1='Angiosperm', class2='Gymnosperm', crs='EPSG:4326')

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
    if i > 1:
        break

print(len(data))
print(data[0])
print(data[-1])
asyncio.run(download(data))

