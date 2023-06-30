import pandas as pd

from remseno import *
from remseno.indices import *

data_dir = '../data/silver_fir/'
output_dir = data_dir
run_label = 'test'
all_dfs = pd.read_csv(f'{data_dir}info.csv')
df = all_dfs[all_dfs['Season'] == 'Autumn']

source_coords = f'{data_dir}corsica_trees_annotated.csv'
source_img = f'{df[df["label"] == "corsica"]["Path"].values[0]}'
dest_img = f'{df[df["label"] == "levie"]["Path"].values[0]}'

c = Coords(source_coords, x_col='Y', y_col='X', label_col='label',
                   id_col='Name', sep=',', class1='silver_fir', class2='silver_fir', crs="EPSG:4326")
o = Image()
o.load_image(image_path=source_img)
o_dest = Image()
o_dest.load_image(image_path=dest_img)

img_crs = str(o.image.crs)
# Plot the image
c.transform_coords(tree_coords="EPSG:4326", image_coords=img_crs, plot=True)

# Plot on the sat image
c.plot_on_image(image=o, band=1)

# Plot bounding box around each tree
# Plot each tree individually and the class that it belongs to
df = c.df
ax = o.plot(2, show_plot=False)
colour = df['colour'].values
for i in range(0, len(df)):
    x = df[c.x_col].values[i]
    y = df[c.y_col].values[i]
    bb = c.build_polygon_from_centre_point(x, y, 3, 3, img_crs)
    bb = [o.image.index(b[0], b[1]) for b in bb]
    xs = [b[1] for b in bb]
    ys = [b[0] for b in bb]
    ax.plot(xs, ys, c=colour[i])
plt.title("Bounding box plot")
plt.show()

# Plot RBG
#o.plot_rbg(show_plot=True)

# Calculate values for the location of interest

# Image bands is something like below, we leave it up to the user to define i.e. could be indicies
image_bands = []
bands = [1, 2, 3, 4, 5, 6, 7, 8]
for band in bands: # Always normalise so that it is easier for
    normed = o.image.read(band)
    image_bands.append(normed) # #normed-np.mean(normed)) #(normed - np.min(normed)) / (np.max(normed) - np.min(normed)))
# Also calculate it
img = o.image


nitian, ndvi, sr, tvi, gi, gndvi, pri, osavi, tcari, redge, redge2, siredge, normg, schl, schlcar = get_all_planetscope(img)
xs = c.df[c.x_col].values
ys = c.df[c.y_col].values
rows = []
info_cols = image_bands + [nitian, ndvi, sr, tvi, gi, gndvi, pri, osavi, tcari, redge, redge2, siredge, normg, schl, schlcar]
for i, id_val in enumerate(df[c.id_col].values):
    rows.append([id_val] + get_values_for_location(o, xs[i], ys[i], info_cols))

columns = ['id'] + [f'band_{i}' for i in bands]
columns += ['nitian', 'ndvi', 'sr', 'tvi', 'gi', 'gndvi', 'pri', 'osavi', 'tcari', 'redge', 'redge2',
            'siredge', 'normg', 'schl', 'schlcar']

ind_df = pd.DataFrame(data=rows, columns=columns)
ind_df.to_csv(f'{output_dir}calculated_indicies_{run_label}.csv', index=False)

# Create a mask
masking_cols = [f'band_{i}' for i in bands] + ['nitian', 'ndvi', 'sr', 'tvi', 'gi', 'gndvi', 'pri', 'osavi', 'tcari', 'redge', 'redge2',
            'siredge', 'normg', 'schl', 'schlcar']
info_cols = image_bands + [nitian, ndvi, sr, tvi, gi, gndvi, pri, osavi, tcari, redge, redge2, siredge, normg, schl, schlcar]
masks = []
for i, col in enumerate(masking_cols):
    # Now we want to build a total mask...
    mean_v = np.mean(ind_df[col].values)
    std_v = np.std(ind_df[col].values)
    lower_bound = np.min(ind_df[col].values) # - std_v #mean_v - std_v
    upper_bound = np.max(ind_df[col].values) # + std_v #mean_v + std_v
    masks.append(mask_values(info_cols[i], lower_bound, upper_bound))

total_mask = masks[0]
for mi, mask in enumerate(masks[1:]):
    total_mask *= mask

    # Plot each tree individually and the class that it belongs to
    df = c.df
    ax = o.plot(2, show_plot=False)

    ax.imshow(total_mask*ndvi, vmin=0, vmax=1)
    for i in range(0, len(df)):
        x = df[c.x_col].values[i]
        y = df[c.y_col].values[i]
        bb = c.build_polygon_from_centre_point(x, y, 20, 20, "EPSG:32614")
        bb = [o.image.index(b[0], b[1]) for b in bb]
        xs = [b[1] for b in bb]
        ys = [b[0] for b in bb]
        ax.plot(xs, ys)
    plt.title(f'Masked: {masking_cols[mi]}')
    plt.show()

# Plot each tree individually and the class that it belongs to
df = c.df
ax = o.plot(2, show_plot=False)

ax.imshow(total_mask*ndvi, vmin=0, vmax=1)
for i in range(0, len(df)):
    x = df[c.x_col].values[i]
    y = df[c.y_col].values[i]
    bb = c.build_polygon_from_centre_point(x, y, 20, 20, "EPSG:32614")
    bb = [o.image.index(b[0], b[1]) for b in bb]
    xs = [b[1] for b in bb]
    ys = [b[0] for b in bb]
    ax.plot(xs, ys)
plt.title("Masked plot")
plt.show()

# Plot each tree individually and the class that it belongs to
df = c.df
xs, ys = [], []
for i in range(0, len(df)):
    x = df[c.x_col].values[i]
    y = df[c.y_col].values[i]
    y, x = o.image.index(x, y)
    xs.append(x)
    ys.append(y)
pixel_buffer = 20
roi = {'x1': min(xs) - pixel_buffer, 'x2': max(xs) + pixel_buffer,
       'y1': min(ys) - pixel_buffer, 'y2': max(ys) + pixel_buffer}
ax = o.plot_idx(total_mask*ndvi, roi=roi, show_plot=False)
xs = [pixel_buffer + (x - min(xs)) for x in xs]
ys = [pixel_buffer + (y - min(ys)) for y in ys]
ax.scatter(xs, ys, s=8)

plt.title(f'Masked!')
plt.show()


# Image bands is something like below, we leave it up to the user to define i.e. could be indicies
image_bands = []
bands = [1, 2, 3, 4, 5, 6, 7, 8]
for band in bands: # Always normalise so that it is easier for
    normed = o_dest.image.read(band)
    image_bands.append(normed) # #normed-np.mean(normed)) #(normed - np.min(normed)) / (np.max(normed) - np.min(normed)))
# Also calculate it
img = o_dest.image

nitian, ndvi, sr, tvi, gi, gndvi, pri, osavi, tcari, redge, redge2, siredge, normg, schl, schlcar = get_all_planetscope(img)


# Create a mask
masking_cols = [f'band_{i}' for i in bands] + ['nitian', 'ndvi', 'sr', 'tvi', 'gi', 'gndvi', 'pri', 'osavi', 'tcari', 'redge', 'redge2',
            'siredge', 'normg', 'schl', 'schlcar']
info_cols = image_bands + [nitian, ndvi, sr, tvi, gi, gndvi, pri, osavi, tcari, redge, redge2, siredge, normg, schl, schlcar]
masks = []
for i, col in enumerate(masking_cols):
    # Now we want to build a total mask...
    mean_v = np.mean(ind_df[col].values)
    std_v = np.std(ind_df[col].values)
    lower_bound = np.min(ind_df[col].values) # - std_v #mean_v - std_v
    upper_bound = np.max(ind_df[col].values) # + std_v #mean_v + std_v
    masks.append(mask_values(info_cols[i], lower_bound, upper_bound))

total_mask = masks[0]
for mi, mask in enumerate(masks[1:]):
    total_mask *= mask

    # Plot each tree individually and the class that it belongs to
    df = c.df
    ax = o_dest.plot(2, show_plot=False)

    ax.imshow(total_mask*ndvi, vmin=0, vmax=1)
    plt.title(f'Masked: {masking_cols[mi]}')
    plt.show()


df = all_dfs[all_dfs['Season'] == 'Autumn']

dest_img = f'{df[df["label"] == "levie"]["Path"].values[1]}'
o_dest = Image()
o_dest.load_image(image_path=dest_img)



# Image bands is something like below, we leave it up to the user to define i.e. could be indicies
image_bands = []
bands = [1, 2, 3, 4, 5, 6, 7, 8]
for band in bands: # Always normalise so that it is easier for
    normed = o_dest.image.read(band)
    image_bands.append(normed) # #normed-np.mean(normed)) #(normed - np.min(normed)) / (np.max(normed) - np.min(normed)))
# Also calculate it
img = o_dest.image

nitian, ndvi, sr, tvi, gi, gndvi, pri, osavi, tcari, redge, redge2, siredge, normg, schl, schlcar = get_all_planetscope(img)


# Create a mask
masking_cols = [f'band_{i}' for i in bands] + ['nitian', 'ndvi', 'sr', 'tvi', 'gi', 'gndvi', 'pri', 'osavi', 'tcari', 'redge', 'redge2',
            'siredge', 'normg', 'schl', 'schlcar']
info_cols = image_bands + [nitian, ndvi, sr, tvi, gi, gndvi, pri, osavi, tcari, redge, redge2, siredge, normg, schl, schlcar]
masks = []
for i, col in enumerate(masking_cols):
    # Now we want to build a total mask...
    mean_v = np.mean(ind_df[col].values)
    std_v = np.std(ind_df[col].values)
    lower_bound = np.min(ind_df[col].values) # - std_v #mean_v - std_v
    upper_bound = np.max(ind_df[col].values) # + std_v #mean_v + std_v
    masks.append(mask_values(info_cols[i], lower_bound, upper_bound))

total_mask = masks[0]
for mi, mask in enumerate(masks[1:]):
    total_mask *= mask

    # Plot each tree individually and the class that it belongs to
    df = c.df
    ax = o_dest.plot(2, show_plot=False)

    ax.imshow(total_mask*ndvi, vmin=0, vmax=1)
    plt.title(f'Masked: {masking_cols[mi]}')
    plt.show()