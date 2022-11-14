import plotly.graph_objects as go


def plot_feature_importance(model, X):
    feature_imp = list(zip(model.feature_importances_, X.columns))
    feature_imp = sorted(feature_imp, key=lambda x: x[0])
    x, y = zip(*feature_imp)

    fig = go.Figure(go.Bar(x=x, y=y, orientation="h"))
    fig.show()
