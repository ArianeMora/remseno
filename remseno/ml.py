###############################################################################
#                                                                             #
#    This program is free software: you can redistribute it and/or modify     #
#    it under the terms of the GNU General Public License as published by     #
#    the Free Software Foundation, either version 3 of the License, or        #
#    (at your option) any later version.                                      #
#                                                                             #
#    This program is distributed in the hope that it will be useful,          #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of           #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            #
#    GNU General Public License for more details.                             #
#                                                                             #
#    You should have received a copy of the GNU General Public License        #
#    along with this program. If not, see <http://www.gnu.org/licenses/>.     #
#                                                                             #
###############################################################################

"""
Machine learning component of the project.
"""
import math
import os
from sklearn.model_selection import StratifiedKFold

import pandas as pd
from sklearn.model_selection import cross_validate
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# import xgboost as xgb
# from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# List of classifiers to evaluate
classifiers = [
    ("Logistic Regression", LogisticRegression(
        max_iter=1000,
        C=1,
        solver='liblinear',  # Good choice for small datasets
        penalty='l2'  # L2 regularization
    )),
    ("K-Nearest Neighbors", KNeighborsClassifier(
        n_neighbors=5,
        metric='euclidean'
    )),
    ("Random Forest", RandomForestClassifier(
        n_estimators=100,  # Fewer trees
        max_depth=10,  # Limit depth of each tree
        bootstrap=True,  # Use bootstrap samples in the construction of trees
        random_state=42
    )),
    ("Gradient Boosting", GradientBoostingClassifier(
        learning_rate=0.1,
        n_estimators=100,
        max_depth=10,  # Shallower trees
        subsample=0.8,  # Stochastic Gradient Boosting
        random_state=42
    )),
    ("Support Vector Machine", make_pipeline(
        StandardScaler(),
        SVC(
            C=1,
            kernel='rbf',  # Radial Basis Function (RBF) kernel
            gamma='scale'  # Automatic gamma value
        )
    )),
    ("MLP Classifier", MLPClassifier(max_iter=1000)),
    # ("XGBoost", xgb.XGBClassifier(
    #     use_label_encoder=False,
    #     eval_metric='logloss',
    #     n_estimators=300,  # Start with fewer trees
    #     max_depth=8,  # Shallower trees to prevent overfitting
    #     learning_rate=0.1,  # Smaller learning rate for gradual improvements
    #     subsample=0.8,  # Use 80% of data to prevent overfitting
    #     colsample_bytree=0.8,  # Use 80% of features to prevent overfitting
    #     min_child_weight=1,  # Minimum sum of instance weight needed in a child
    #     reg_alpha=0.01,  # L1 regularization term on weights (increases model generalization)
    #     reg_lambda=1.0  # L2 regularization term on weights
    # )),
    # ("CatBoost", CatBoostClassifier(
    #     learning_rate=0.1,
    #     depth=8,  # Shallower trees for small datasets
    #     iterations=300,  # Fewer iterations to start, adjust based on CV
    #     random_seed=42,
    #     l2_leaf_reg=3,  # Regularization rate
    #     border_count=128,  # Default is fine, adjust if necessary
    #     subsample=0.8,  # Consider subsampling for small datasets
    #     logging_level='Silent',  # Keeps the output clean
    #     early_stopping_rounds=30  # Use early stopping to prevent overfitting
    # ))
]


class ML:

    def __init__(self):
        self.train_samples = None
        self.validation_df = None
        self.train_df = None
        self.clf = None

    def save(self, output_dir, label=''):
        pickle.dumps(os.path.join(output_dir, f'classifier_{label}.pkl'))
        self.train_df.to_csv(os.path.join(output_dir, f'trainDF_{label}.csv'), index=False)
        self.validation_df.to_csv(os.path.join(output_dir, f'validDF_{label}.csv'), index=False)

    def classify(self, test_df, clf, csv_file='classify.csv'):
        # Generate predictions
        cols = [c for c in test_df.columns if c not in ['id', 'class', 'X', 'Y', 'predicted_label']]
        X = test_df[cols]
        y_pred = clf.predict(X)
        test_df['Y_prediction'] = y_pred
        test_df.to_csv(csv_file, index=False)


    def validate_clf(self, test_df, classifiers, class1_label, csv_file='classifers_performance_validation_set.csv'):
        """ Validate the trained classifiers on a test set. """
        cols = [c for c in test_df.columns if c not in ['id', 'class', 'X', 'Y', 'predicted_label', 'Y_prediction']]
        X = test_df[cols]
        y_labels = test_df['class'].values
        y_test = [1 if label == class1_label else 0 for label in y_labels]
        # Since we're validating the whole thing is the y-test
        csv_data = [["Classifier", "Kth-fold", "Accuracy", "Precision", "Recall", "F1 Score"]]

        for name in classifiers:
            clf = classifiers[name]
            scoring = ['accuracy', 'precision', 'recall', 'f1']

            # Generate predictions
            y_pred = clf.predict(X)

            # Compute scores
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro')
            recall = recall_score(y_test, y_pred, average='macro')
            f1 = f1_score(y_test, y_pred, average='macro')

            # Store per-fold score
            fold_data = [
                name,
                1,
                accuracy,
                precision,
                recall,
                f1,
            ]
            csv_data.append(fold_data)
            # Write the metrics (including per-fold) to a CSV file
        with open(csv_file, "w+", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(csv_data)

    def train_clf(self, train_df, class1_label, class2_label, csv_file="classifier_metrics_for_test_dataset.csv"):
        """  train classifiers using a test set """
        cols = [c for c in train_df.columns if c not in ['id', 'class', 'X', 'Y', 'predicted_label']]
        X = train_df[cols]
        y_labels = train_df['class'].values
        y = [1 if label == class1_label else 0 for label in y_labels]
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        csv_data = [["Classifier", "Kth-fold", "Accuracy", "Precision", "Recall", "F1 Score"]]
        scoring = ['accuracy', 'precision', 'recall', 'f1']
        trained_classifiers = {}
        for name, clf in classifiers:
            # Optionally, wrap the classifier with a StandardScaler in a pipeline
            if not isinstance(clf, make_pipeline(StandardScaler(), SVC()).__class__):
                clf = make_pipeline(StandardScaler(), clf)

            # Perform k-fold cross-validation and compute scores
            # Fit the classifier
            clf = clf.fit(X_train, y_train)
            trained_classifiers[name] = clf
            # Generate predictions
            y_pred = clf.predict(X_test)

            # Compute scores
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro')
            recall = recall_score(y_test, y_pred, average='macro')
            f1 = f1_score(y_test, y_pred, average='macro')
            
            # Store per-fold score
            fold_data = [
                name,
                1,
                accuracy,
                precision,
                recall,
                f1,
            ]
            csv_data.append(fold_data)

            # Save the classifier to a file
            with open(f'{name}.pkl', 'wb') as file:
                pickle.dump(clf, file)

        # Write the metrics (including per-fold) to a CSV file
        with open(csv_file, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(csv_data)

        return trained_classifiers

    def perform_k_fold_cv(self, train_df, class1_label, class2_label,
                          csv_file="classifier_metrics_with_per_fold_kfold.csv", k_folds=10):
        """ Perform CV validation. """
        cols = [c for c in train_df.columns if c not in ['id', 'class', 'X', 'Y', 'predicted_label']]
        X = train_df[cols]
        y_labels = train_df['class'].values
        y = [1 if label == class1_label else 0 for label in y_labels]
        # Split the dataset into training and testing sets
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        csv_data = [["Classifier", "Kth-fold", "Accuracy", "Precision", "Recall", "F1 Score"]]
        scoring = ['accuracy', 'precision', 'recall', 'f1']
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

        for name, clf in classifiers:
            # Optionally, wrap the classifier with a StandardScaler in a pipeline
            if not isinstance(clf, make_pipeline(StandardScaler(), SVC()).__class__):
                clf = make_pipeline(StandardScaler(), clf)

            # Perform k-fold cross-validation and compute scores

            scores = cross_validate(clf, X, y, cv=skf, scoring=scoring, return_train_score=False)

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
        with open(csv_file, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(csv_data)


    def build_train_df(self, df, image, coords, image_bands, max_pixel_padding=1, normalise=True):
        """
        Build a training dataframe from image and coordinate files!

        :param image:
        :param coords:
        :param image_bands:
        :param max_pixel_padding:
        :param band_labels:
        :return:
        """
        xs = df[coords.x_col].values
        ys = df[coords.y_col].values
        rows = []
        # Image bands is something like below, we leave it up to the user to define i.e. could be indicies
        classes = df[coords.label_col].values
        band_labels = [f'band_{band_i}' for band_i, c in enumerate(image_bands)]

        for i, tid in enumerate(df[coords.id_col].values):
            y, x = image.image.index(xs[i], ys[i])
            # Now for each bounding area make a training point
            bb = coords.build_circle_from_centre_point(x, y, max_pixel_padding)
            for xy in bb:
                # We now build the feature set which is
                data_row = [tid, classes[i], xy[0], xy[1]]
                for band_i, image_band in image_bands.items():
                    # Here we are extracting the feature i.e. the value from the image bands we're interested in
                    data_row.append(image_band[xy[1], xy[0]])
                rows.append(data_row)
        # Now create a dataframe from this
        train_df = pd.DataFrame(rows, columns=[coords.id_col, coords.label_col, coords.x_col, coords.y_col] +
                                              band_labels)
        # If normalise we normalise each row
        if normalise:
            train_df = train_df.fillna(0)

            # Use a standard scaler on each row
            for b in band_labels:
                vals = train_df[b]
                train_df[b] = (vals - np.mean(vals))/np.std(vals)  # Just do the usual standard scaling!
            train_df = train_df.fillna(0)

        return train_df, band_labels

    def build_train_df_from_indexs(self, df, images, coords, max_pixel_padding=1, normalise=True):
        """
        Build a training dataframe from a list of images! NOTE: this assumes that all the images have the
        same coord system!

        :param image:
        :param coords:
        :param image_bands:
        :param max_pixel_padding:
        :param band_labels:
        :return:
        """
        xs = df[coords.x_col].values
        ys = df[coords.y_col].values
        rows = []
        # Image bands is something like below, we leave it up to the user to define i.e. could be indicies
        classes = df[coords.label_col].values
        for i, tid in enumerate(df[coords.id_col].values):
            # We now build the feature set which is
            data_row = [tid, classes[i]]
            for image_values in images:
                image = image_values['image']
                y, x = image.image.index(xs[i], ys[i])
                # Now for each bounding area make a training point
                bb = coords.build_circle_from_centre_point(x, y, max_pixel_padding)
                for xy in bb:
                    for label, image_band in image_values['indexs'].items():
                        # Here we are extracting the feature i.e. the value from the image bands we're interested in
                        data_row.append(image_band[xy[1], xy[0]])
            rows.append(data_row)
        # Now create a dataframe from this
        train_df = pd.DataFrame(rows)
        cols = [c for i, c in enumerate(train_df.columns) if i > 1]
        # If normalise we normalise each row
        if normalise:
            # Use a standard scaler on each column
            for b in cols:
                vals = train_df[b]
                train_df[b] = (vals - np.mean(vals))/np.std(vals)  # Just do the usual standard scaling!
        cols = [coords.id_col, coords.label_col] + [c for i, c in enumerate(train_df.columns) if i > 1]
        train_df.columns = cols
        return train_df, [c for i, c in enumerate(train_df.columns) if i > 1]

    def validate(self, clf, image, coords, image_bands, max_pixel_padding=1, normalise=False):
        """
        Validate a pretrained classifier
        :param clf:
        :param image:
        :param coords:
        :param bands:
        :param max_pixel_padding:
        :param normalise:
        :return:
        """
        """
               Train a ML classifier for the image and coords.

               :return: VAE or something
               """
        # Build training DF from the coords
        # First we want to hold some trees out, so we select a random sample from the dataset
        # Now use the other as the dataframe
        df, training_cols = self.build_train_df(coords.df, image, coords, image_bands,
                                                max_pixel_padding=max_pixel_padding,
                                                normalise=normalise)

        X = df[training_cols].values
        y = df[coords.label_col]
        # Train model
        test_score = clf.score(X, y)
        y_pred = clf.predict(X)

        print(f"test score {test_score}, {y_pred}")
        print(f"clf score {clf.score(X, y)}")
        print(f"balanced a s {balanced_accuracy_score(y, clf.predict(X))}")
        # The equation of the separating plane is given by all x so that np.dot(svc.coef_[0], x) + b = 0.
        # Solve for w3 (z)
        z = lambda x, y: (-clf.intercept_[0] - clf.coef_[0][0] * x - clf.coef_[0][1] * y) / clf.coef_[0][2]
        Y = y
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # For each class add as a point
        for yval in set(Y):
            ax.plot3D(X[Y == yval, 0], X[Y == yval, 1], X[Y == yval, 2], 'o')
        plt.show()

        # Also do a decision tree classifier
        pred = clf.predict(X)
        cm = confusion_matrix(df[coords.label_col], pred, labels=clf.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
        disp.plot()
        plt.show()


        # Add the predicted label in for each
        df['predicted_label'] = pred
        self.clf = clf  # Save to the model
        return self.get_overall_tree_pred(coords.df, coords, df, df)

    def train_on_image_list(self, clf, images, coords, validation_percent=30, test_percent=30,
                            max_pixel_padding=1, normalise=False, pretrained=False):
        # Build training DF from the coords
        # First we want to hold some trees out, so we select a random sample from the dataset
        valid_df = coords.df.sample(math.ceil(len(coords.df) * (validation_percent / 100)))
        print(f"{valid_df}")
        # Now use the other as the dataframe
        df = coords.df[~coords.df[coords.id_col].isin(list(valid_df[coords.id_col].values))]
        train_df = pd.DataFrame()
        for i in range(0, len(images) - 1):
            image = images[i]
            tmp_df, training_cols = self.build_train_df(df, image['image'], coords, image['indexs'],
                                                        max_pixel_padding, normalise)
            train_df = pd.concat([train_df, tmp_df])
        train_df = train_df.fillna(0)
        train_df.to_csv('train.csv')
        self.train_df = train_df

        self.valid_df, training_cols = self.build_train_df(df, images[-1]['image'], coords, images[-1]['indexs'],
                                                           max_pixel_padding, normalise)
        df = train_df.copy()
        X = df[training_cols].values
        y = df[coords.label_col]
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percent / 100,
                                                            random_state=42)
        # Get the pixels from the orthomosaic
        if not pretrained:
            clf = clf.fit(X_train, y_train)
        # if it's already been trained we can just run it!
        clf.score(X_test, y_test)
        test_score = clf.score(X_test, y_test)
        y_pred = clf.predict(X_test)

        print(f"test score {test_score}, {y_pred}")
        # Then also do the final ones in the validation df i.e. unseen trees
        X = self.valid_df[training_cols].values
        y = self.valid_df[coords.label_col]
        print(f"clf score {clf.score(X, y)}")
        print(f"balanced score {balanced_accuracy_score(y, clf.predict(X))}")
        print(len(self.valid_df), len(self.train_df))
        Y = y
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # For each class add as a point
        for yval in set(Y):
            ax.plot3D(X[Y == yval, 0], X[Y == yval, 1], X[Y == yval, 2], 'o')
        plt.show()

        # Also do a decision tree classifier
        pred = clf.predict(X_test)
        cm = confusion_matrix(y_test, pred, labels=clf.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
        disp.plot()
        plt.show()

        # Add the predicted label in for each
        self.train_df['predicted_label'] = clf.predict(self.train_df[training_cols].values)
        self.valid_df['predicted_label'] = clf.predict(self.valid_df[training_cols].values)
        self.clf = clf  # Save to the model
        #return self.get_overall_tree_pred(coords.df, coords, self.train_df, self.valid_df)

    def train_ml_on_multiple_images(self, clf, images, coords, validation_percent=30, test_percent=30,
                                    max_pixel_padding=1, normalise=False, pretrained=False):
        """
        Train a ML classifier for multiple images with different indicies.
        """
        # Build training DF from the coords
        # First we want to hold some trees out, so we select a random sample from the dataset
        valid_df = coords.df.sample(math.ceil(len(coords.df) * (validation_percent / 100)))
        print(f"{valid_df}")
        # Now use the other as the dataframe
        df = coords.df[~coords.df[coords.id_col].isin(list(valid_df[coords.id_col].values))]
        df, training_cols = self.build_train_df_from_indexs(df, images, coords, max_pixel_padding, normalise)
        self.train_df = df
        self.valid_df, training_cols = self.build_train_df_from_indexs(valid_df, images, coords)

        X = df[training_cols].values
        y = df[coords.label_col]
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percent / 100,
                                                            random_state=42)
        # Get the pixels from the orthomosaic
        if not pretrained:
            clf = clf.fit(X_train, y_train)
        # if it's already been trained we can just run it!
        clf.score(X_test, y_test)
        test_score = clf.score(X_test, y_test)
        y_pred = clf.predict(X_test)
        print(f"test score {test_score}, {y_pred}")
        # Then also do the final ones in the validation df i.e. unseen trees
        X = self.valid_df[training_cols].values
        y = self.valid_df[coords.label_col]
        print(f"clf score {clf.score(X, y)}")
        print(f"balanced score {balanced_accuracy_score(y, clf.predict(X))}")
        print(len(self.valid_df), len(self.train_df))
        Y = y
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # For each class add as a point
        for yval in set(Y):
            ax.plot3D(X[Y == yval, 0], X[Y == yval, 1], X[Y == yval, 2], 'o')
        plt.show()

        # Also do a decision tree classifier
        pred = clf.predict(X_test)
        cm = confusion_matrix(y_test, pred, labels=clf.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
        disp.plot()
        plt.show()

        # Add the predicted label in for each
        self.train_df['predicted_label'] = clf.predict(self.train_df[training_cols].values)
        self.valid_df['predicted_label'] = clf.predict(self.valid_df[training_cols].values)
        self.clf = clf  # Save to the model
        return self.get_overall_tree_pred(coords.df, coords, self.train_df, self.valid_df)

    def train_ml(self, clf, image, coords, image_bands, validation_percent=30, test_percent=30,
                 max_pixel_padding=1, normalise=True):
        """
        Train a ML classifier for the image and coords.

        :return: VAE or something
        """
        # Build training DF from the coords
        # First we want to hold some trees out, so we select a random sample from the dataset
        valid_df = coords.df.sample(math.ceil(len(coords.df)*(validation_percent/100)))
        # Now use the other as the dataframe
        df = coords.df[~coords.df[coords.id_col].isin(list(valid_df[coords.id_col].values))]
        df, training_cols = self.build_train_df(df, image, coords, image_bands, max_pixel_padding=max_pixel_padding,
                                                normalise=normalise)
        self.train_df = df
        self.valid_df, training_cols = self.build_train_df(valid_df, image, coords, image_bands,
                                                           max_pixel_padding=max_pixel_padding, normalise=normalise)

        X = df[training_cols].values
        y = df[coords.label_col]
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percent/100,
                                                            random_state=42)
        # Get the pixels from the orthomosaic
        clf = clf.fit(X_train, y_train)
        clf.score(X_test, y_test)
        test_score = clf.score(X_test, y_test)
        y_pred = clf.predict(X_test)
        print(f"test score = {test_score}, {y_pred}")
        # Then also do the final ones in the validation df i.e. unseen trees
        X = self.valid_df[training_cols].values
        y = self.valid_df[coords.label_col]
        print(f"clf={clf.score(X, y)}")
        print(f"balanced= {balanced_accuracy_score(y, clf.predict(X))}")
        print(len(self.valid_df), len(self.train_df))
        # The equation of the separating plane is given by all x so that np.dot(svc.coef_[0], x) + b = 0.
        # Solve for w3 (z)
        Y = y

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # For each class add as a point
        for yval in set(Y):
            ax.plot3D(X[Y == yval, 0], X[Y == yval, 1], X[Y == yval, 2], 'o')
        plt.show()

        # Also do a decision tree classifier
        pred = clf.predict(X_test)
        cm = confusion_matrix(y_test, pred, labels=clf.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
        disp.plot()
        plt.show()

        # Add the predicted label in for each
        self.train_df['predicted_label'] = clf.predict(self.train_df[training_cols].values)
        self.valid_df['predicted_label'] = clf.predict(self.valid_df[training_cols].values)
        self.clf = clf # Save to the model

    def group_results(self, train_df, coords, correct, incorrect, id_value_map, data_type):
        """
        Group results.

        :param train_df:
        :param coords:
        :param correct:
        :param incorrect:
        :param id_value_map:
        :return:
        """
        grped_df = train_df.groupby(coords.id_col)
        for gid, g_df in grped_df:
            # Get the most common colour
            true_label = g_df[coords.label_col].values[0]
            # Now check what the average prediction was
            pred_correct = len(g_df[g_df['predicted_label'] == true_label])
            pred_incorrect = len(g_df[g_df['predicted_label'] != true_label])
            # average prediction is going to just be the most common prediction
            if pred_correct > pred_incorrect:
                pred_value = true_label
                correct += 1
            else:
                # Only works for the binary class problem
                pred_value = None
                incorrect += 1
            overall_pred = 'correct' if correct > incorrect else 'incorrect'
            id_value_map[gid] = {'pred': pred_value, 'pred_prob': pred_correct / (pred_correct + pred_incorrect),
                                 'type': data_type, 'overall_pred': overall_pred}
        return id_value_map, correct, incorrect

    def get_overall_tree_pred(self, df, coords, train_df, valid_df):
        """
        :return:
        """
        # Do it for both the training and the validation dataframe
        # Map the ID to the predicted label
        id_value_map, correct, incorrect = {}, 0, 0
        id_value_map, correct, incorrect = self.group_results(train_df, coords, correct, incorrect, id_value_map, 'train')
        id_value_map, correct, incorrect = self.group_results(valid_df, coords, correct, incorrect, id_value_map, 'valid')
        # Do the same for validation
        total_prob = correct / (correct + incorrect)
        print(f"Overall Accuracy =  {total_prob}")
        preds = []
        pred_probs = []
        data_types = []
        overall_pred = []
        for tid in df[coords.id_col].values:
            preds.append(id_value_map[tid]['pred'])
            pred_probs.append(id_value_map[tid]['pred_prob'])
            data_types.append(id_value_map[tid]['type'])
            overall_pred.append(id_value_map[tid]['overall_pred'])

        df['pred'] = preds
        df['pred_probs'] = pred_probs
        df['data_type'] = data_types
        df['overall_pred'] = overall_pred

    def test_ml(self, clf, image, coords, image_bands, max_pixel_padding=1, normalise=False):
        # Preprocess the testing image and coordinates
        # Similar to the training phase, build the testing DataFrame
        test_df, test_cols = self.build_train_df(coords.df, image, coords, image_bands,
                                                 max_pixel_padding=max_pixel_padding, normalise=normalise)

        # Extract the features and labels
        X_test = test_df[test_cols].values
        y_test = test_df[coords.label_col].values

        # Predict using the trained classifier
        y_pred = clf.predict(X_test)

        # Calculate evaluation metrics
        accuracy = balanced_accuracy_score(y_test, y_pred)
        print(f"Testing Accuracy: {accuracy}")

        # Display a confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
        disp.plot()
        plt.show()

        # Return the testing DataFrame with predictions for further analysis
        test_df['predicted_label'] = y_pred
        df = pd.DataFrame()



        return df, test_df

