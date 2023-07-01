from remseno import *

# Build the polygon
data_dir = '../data/harvard/'
filename = f'{data_dir}tallo_neon_species_dedup_subsample.csv'
meters = 500

c = Coords(filename, x_col='longitude', y_col='latitude', label_col='division',
           id_col='tree_id', sep=',', class1='Angiosperm', class2='Gymnosperm', crs='EPSG:4326')

df = pd.read_csv(filename)
image_ids = df['image_ids']
lat = df['latitude'].values
lon = df['longitude'].values
tree_ids = df['tree_id'].values
data = []
for i, v in enumerate(image_ids):
    aoi = c.build_polygon_from_centre_point(lat[i], lon[i], meters, meters, "EPSG:4326")
    # For some reason need to swap it around classic no idea why...
    aoi = [[p[1], p[0]] for p in aoi]
    data.append([aoi, image_ids[i], tree_ids[i]])

asyncio.run(download(data))

