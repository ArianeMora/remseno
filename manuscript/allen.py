import pandas as pd

from remseno import *
import numpy as np
from sklearn.model_selection import cross_validate
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Import classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

np.seterr(divide='ignore', invalid='ignore')

coordinate_file_path = 'data/Allen_all_genotyped_trees_2024.csv'

training_times = ['data/20230421_Allen.tif', 'data/20230518_Allen.tif', 'data/20230915_Allen.tif',
                  'data/20230708_Allen.tif', 'data/20230816_Allen.tif']

test_times = ['data/20230708_Allen.tif']

training_data = []


def get_all_planetscope(img):
    ndvi = get_ndvi(image=img, red_band=6, nir_band=8)
    sr = get_sr(image=img, red_band=6, nir_band=8)
    pri = get_pri(image=img, green_band=4, greeni_band=3)
    tcari = get_tcari(image=img, rededge_band=7, greeni_band=3, red_band=6)
    redge = get_redge(image=img, nir_band=8, green=4, r_edge=7)
    schl = get_schl(image=img, red_band=6, rededge_band=7, nir_band=8)
    schlcar = get_schlcar(image=img, red_band=6, greeni_band=3)
    return ndvi, sr, pri, tcari, redge, schl, schlcar

rows = []
images = []
coords = []
for data_file in training_times:
    o = Image()
    o.load_image(image_path=data_file)
    # Get the indicies from the image
    indices = get_all_planetscope(o.image)
    bands = [o.image.read(b) for b in [1, 2, 3, 4, 5, 6, 7, 8]] + list(indices)
    bands_dict = {}
    for i, b in enumerate(bands):
        bands_dict[i] = b
    images.append({'image': o, 'indexs': bands_dict})
    rows += bands

c = Coords(coordinate_file_path, x_col='X', y_col='Y', label_col='class',
           id_col='id', sep=',', class1='Sylvatica', class2='Orientals', crs="EPSG:32632")

c.transform_coords(tree_coords="EPSG:32632", image_coords="EPSG:32632", plot=False)
coords.append(c)

clf = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.2,
    max_depth=3, random_state=42
)

ml = ML()

ml.train_ml_on_multiple_images(clf, images, coords=c, validation_percent=5, test_percent=30, max_pixel_padding=1,
                       normalise=False)

ml.train_df.to_csv('train.csv')

# Generate a synthetic dataset
cols = [c for c in ml.train_df.columns if c not in ['id', 'class', 'X', 'Y', 'predicted_label']]
X = ml.train_df[cols]
y_labels = ml.train_df['class'].values
y = [1 if label == 'Orientalis' else 0 for label in y_labels]
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# List of classifiers to evaluate
classifiers = [
    ("Logistic Regression", LogisticRegression(max_iter=1000)),
    ("K-Nearest Neighbors", KNeighborsClassifier()),
    ("Random Forest", RandomForestClassifier()),
    ("Gradient Boosting", GradientBoostingClassifier()),
    ("Support Vector Machine", make_pipeline(StandardScaler(), SVC())),
    ("MLP Classifier", MLPClassifier(max_iter=1000))
]
csv_data = [["Classifier", "Accuracy", "Precision", "Recall", "F1 Score"]]
scoring = ['accuracy', 'precision', 'recall', 'f1']

k_folds = 10

for name, clf in classifiers:
    # Optionally, wrap the classifier with a StandardScaler in a pipeline
    if not isinstance(clf, make_pipeline(StandardScaler(), SVC()).__class__):
        clf = make_pipeline(StandardScaler(), clf)

    # Perform k-fold cross-validation and compute scores
    scores = cross_validate(clf, X, y, cv=k_folds, scoring=scoring, return_train_score=False)

    # Store per-fold scores
    for fold_index in range(k_folds):
        fold_data = [
            name,
            fold_index + 1,  # Fold number (starting from 1)
            scores['test_accuracy'][fold_index],
            scores['test_precision'][fold_index],
            scores['test_recall'][fold_index],
            scores['test_f1'][fold_index]
        ]
        csv_data.append(fold_data)

    # Calculate and print the mean of each metric across all folds
    accuracy_avg = np.mean(scores['test_accuracy'])
    precision_avg = np.mean(scores['test_precision'])
    recall_avg = np.mean(scores['test_recall'])
    f1_avg = np.mean(scores['test_f1'])
    print(
        f"{name} - Avg Accuracy: {accuracy_avg:.2f}, Avg Precision: {precision_avg:.2f}, Avg Recall: {recall_avg:.2f}, Avg F1: {f1_avg:.2f}")

# Write the metrics (including per-fold) to a CSV file
csv_file = "classifier_metrics_with_per_fold_kfold.csv"
with open(csv_file, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(csv_data)

