from .shared import select_best_features


def math_features_fit(X, y, numerial_columns, clf, verbose):
    if verbose > 0:
        print(f"Computing math features with {len(numerial_columns)} columns")

    transformations = {}
    best_columns = []

    idx_columns = [X.columns.get_loc(c) for c in numerial_columns if c in X]
    name_columns = list(X.columns)

    # calculate best features
    for i_1 in range(len(idx_columns)):
        trans_X = X.copy()
        col_1 = X.columns[idx_columns[i_1]]

        if verbose > 1:
            print(f"  #{i_1}: {col_1}")

        for i_2 in range(len(idx_columns)):
            if i_1 == i_2:
                continue

            col_2 = X.columns[idx_columns[i_2]]

            for op in _operations(i_1 < i_2):
                trans_X, transformations = _add_transformation(
                    col_1, col_2, op, trans_X, transformations
                )

        partial_best_columns = select_best_features(trans_X, y, name_columns, clf)
        best_columns += partial_best_columns

    transformations = {k: v for k, v in transformations.items() if k in best_columns}

    # select best features
    trans_X = _math_features_transform(X, transformations)
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


def math_features_transform(df, transformations):
    return _math_features_transform(df, transformations)


def _math_features_transform(df, transformations):
    for k, v in transformations.items():
        df[k] = _op_columns(df[v["col_1"]], df[v["col_2"]], v["op"])

    return df


def _operations(add_operations_without_order):
    ops = ["truediv", "floordiv", "mod", "pow"]

    if add_operations_without_order:
        ops += ["sum", "rest", "prod", "mean", "hypotenuse"]

    return ops


def _op_columns(col_1, col_2, op):
    if op == "sum":
        return col_1 + col_2

    elif op == "rest":
        return col_1 - col_2

    elif op == "prod":
        return col_1 * col_2

    elif op == "mean":
        return (col_1 + col_2) / 2

    elif op == "hypotenuse":
        return (col_1**2 + col_2**2) ** 0.5

    elif op == "truediv":
        return col_1 / col_2

    elif op == "floordiv":
        return col_1 // col_2

    elif op == "mod":
        return col_1 % col_2

    elif op == "pow":
        return col_1**col_2


def _add_transformation(col_1, col_2, op, trans_X, transformations):
    name = f"{col_1}_{op}_{col_2}"

    trans_X[name] = _op_columns(trans_X[col_1], trans_X[col_2], op)
    trans_X = trans_X.copy()

    transformations[name] = {
        "col_1": col_1,
        "col_2": col_2,
        "op": op,
    }

    return trans_X, transformations
