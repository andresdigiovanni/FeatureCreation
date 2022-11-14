import numpy as np

from .shared import select_best_features


def cyclical_features_fit(X, y, numerial_columns, clf, verbose):
    if verbose > 0:
        print(f"Computing cyclical features with {len(numerial_columns)} columns")

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

        for op in _operations():
            trans_X, transformations = _add_transformation(
                col_1, op, trans_X, transformations
            )

        partial_best_columns = select_best_features(trans_X, y, name_columns, clf)
        best_columns += partial_best_columns

    transformations = {k: v for k, v in transformations.items() if k in best_columns}

    # select best features
    trans_X = _cyclical_features_transform(X, transformations)
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


def cyclical_features_transform(df, transformations):
    return _cyclical_features_transform(df, transformations)


def _cyclical_features_transform(df, transformations):
    for k, v in transformations.items():
        df[k] = _op_columns(df[v["col_1"]], v["op"])

    return df


def _operations():
    return ["sin", "cos"]


def _op_columns(col_1, op):

    if op == "sin":
        return np.sin(col_1 * (2.0 * np.pi / col_1).max())

    elif op == "cos":
        return np.cos(col_1 * (2.0 * np.pi / col_1).max())


def _add_transformation(col_1, op, trans_X, transformations):
    name = f"{col_1}_{op}"

    trans_X[name] = _op_columns(trans_X[col_1], op)
    trans_X = trans_X.copy()

    transformations[name] = {
        "col_1": col_1,
        "op": op,
    }

    return trans_X, transformations
