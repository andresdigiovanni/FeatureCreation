import numpy as np
import pandas as pd

from .shared import select_best_features


def bins_features_fit(X, y, numerial_columns, clf, verbose):
    if verbose > 0:
        print(f"Computing bins features with {len(numerial_columns)} columns")

    transformations = {}
    best_columns = []

    idx_columns = [X.columns.get_loc(c) for c in numerial_columns if c in X]
    name_columns = list(X.columns)

    # calculate best features
    for i in range(len(idx_columns)):
        trans_X = X.copy()
        col_1 = X.columns[idx_columns[i]]

        if verbose > 1:
            print(f"  #{i}: {col_1}")

        for op, bins in _operations():
            trans_X, transformations = _add_transformation(
                col_1, op, bins, trans_X, transformations
            )

        partial_best_columns = select_best_features(trans_X, y, name_columns, clf)
        best_columns += partial_best_columns

    transformations = {k: v for k, v in transformations.items() if k in best_columns}

    # select best features
    trans_X = _bins_features_transform(X, transformations)
    best_columns = select_best_features(
        trans_X, y, name_columns, clf, pre_select_features=False
    )
    transformations = {k: v for k, v in transformations.items() if k in best_columns}

    # final X
    new_columns = list(transformations.keys())
    X = X[name_columns + new_columns]

    if verbose > 0:
        print(f"  created {len(new_columns)} features: {new_columns}")

    return X, transformations


def bins_features_transform(df, transformations):
    return _bins_features_transform(df, transformations)


def _bins_features_transform(df, transformations):
    for k, v in transformations.items():
        df[k], _ = _op_columns(df[v["col_1"]], v["op"], intervals=v["intervals"])

    return df


def _operations():
    return [
        ("cut", 3),
        ("cut", 5),
        ("cut", 10),
        ("cut", 16),
        ("qcut", 3),
        ("qcut", 5),
        ("qcut", 10),
        ("qcut", 16),
    ]


def _op_columns(col_1, op, bins=-1, intervals=[]):

    if len(intervals) == 0:
        if op == "cut":
            _, intervals = pd.cut(col_1, bins=bins, retbins=True)

        elif op == "qcut":
            _, intervals = pd.qcut(col_1.rank(method="first"), q=bins, retbins=True)

    intervals = intervals[1:-1]

    def calc_group(x, intervals):
        for i in range(len(intervals)):
            if x <= intervals[i]:
                return i
        return len(intervals)

    series = col_1.apply(lambda x: calc_group(x, intervals))
    return series, intervals


def _add_transformation(col_1, op, bins, trans_X, transformations):
    name = f"{col_1}_{op}_{bins}"

    trans_X[name], intervals = _op_columns(trans_X[col_1], op, bins=bins)
    trans_X = trans_X.copy()

    transformations[name] = {
        "col_1": col_1,
        "op": op,
        "intervals": intervals,
    }

    return trans_X, transformations
