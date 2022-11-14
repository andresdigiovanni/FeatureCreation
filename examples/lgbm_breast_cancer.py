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
    dataset = sklearn.datasets.load_breast_cancer()
    df = pd.DataFrame(
        data=np.c_[dataset["data"], dataset["target"]],
        columns=list(dataset["feature_names"]) + ["target"],
    )

    data = df.drop(columns="target")
    target = df["target"]
    train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.25)

    # original
    gbm, score = train_and_predict(train_x, valid_x, train_y, valid_y)
    print(f"Original score: {score}")
    plot_feature_importance(gbm, valid_x)

    # new features
    feature_creation = FeatureCreation()
    transformed_train_x = feature_creation.fit(
        train_x, train_y, LGBMClassifier(), verbose=2
    )
    transformed_valid_x = feature_creation.transform(valid_x)

    gbm, score = train_and_predict(
        transformed_train_x, transformed_valid_x, train_y, valid_y
    )
    print(f"New features score: {score}")
    plot_feature_importance(gbm, transformed_valid_x)
