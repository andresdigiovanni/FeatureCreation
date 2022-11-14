from sklearn.feature_selection import RFE
from sklearn.feature_selection import SequentialFeatureSelector as SFS


def select_best_features(X, y, orig_columns, clf, pre_select_features=True):

    if pre_select_features:
        best_columns = _pre_select_features(clf, X, y)
        filtered_X = X[best_columns]
    else:
        filtered_X = X

    best_columns = _select_features(clf, filtered_X, y)
    return [c for c in best_columns if c not in orig_columns]


def _pre_select_features(clf, X, y):
    selector = RFE(clf, n_features_to_select=7, step=0.05)
    selector.fit(X, y)

    feature_idx = selector.get_support()
    return list(X.columns[feature_idx])


def _select_features(clf, X, y):
    selector = SFS(
        clf,
        n_features_to_select="auto",
        tol=1e-3,
        cv=3,
        direction="forward",
        scoring="accuracy",
        n_jobs=-1,
    )
    selector.fit(X, y)

    feature_idx = selector.get_support()
    return list(X.columns[feature_idx])
