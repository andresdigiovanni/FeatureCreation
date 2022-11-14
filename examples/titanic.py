import numpy as np
import pandas as pd
import sklearn.datasets
import sklearn.metrics
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

from examples.shared import plot_feature_importance
from feature_creation import FeatureCreation


def train_and_predict(train_x, valid_x, train_y, valid_y):
    gbm = LGBMClassifier()
    gbm.fit(train_x, train_y)
    preds = gbm.predict(valid_x)
    pred_labels = np.rint(preds)

    return gbm, sklearn.metrics.accuracy_score(valid_y, pred_labels)


if __name__ == "__main__":
    # read data
    df = pd.read_csv("examples/datasets/titanic.csv")
    data = df.drop(columns="Survived")
    target = df["Survived"]

    # pre-process: remove no valid columns
    data = data.drop(columns=["PassengerId", "Name", "Ticket"])

    categorical_columns = list(data.select_dtypes(include=["object"]).columns)
    numerial_columns = list(data.select_dtypes(include=[np.number]).columns)

    # pre-process: categorical columns
    for cat in categorical_columns:
        data = pd.get_dummies(data, columns=[cat], prefix=cat, drop_first=True)

    # split train/test
    train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.25)

    # score original
    gbm, score = train_and_predict(train_x, valid_x, train_y, valid_y)
    print(f"Original score: {score}")
    plot_feature_importance(gbm, valid_x)

    # score new features
    feature_creation = FeatureCreation()
    transformed_train_x = feature_creation.fit(
        train_x, train_y, LGBMClassifier(), verbose=2, numerial_columns=numerial_columns
    )
    transformed_valid_x = feature_creation.transform(valid_x)

    gbm, score = train_and_predict(
        transformed_train_x, transformed_valid_x, train_y, valid_y
    )
    print(f"New features score: {score}")
    plot_feature_importance(gbm, transformed_valid_x)
