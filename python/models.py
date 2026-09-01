"""
Model registry for yield prediction.
Instantiate with a model type string; the class wraps fit/predict/score
and exposes feature_importances_ when available.

Supported types:
    rf    — RandomForestRegressor
    gbm   — GradientBoostingRegressor
    xgb   — XGBRegressor (requires xgboost package)
    ridge — RidgeCV, standardised (linear, heavily regularised)
    enet  — ElasticNetCV, standardised (linear, sparse + regularised)

WHY LINEAR MODELS BELONG HERE

The ensembles calibrate their flexibility to the ROW count, and for seasonal
climate features that count is misleading. The part of yield that weather moves
is largely a common year effect -- districts within a year share their weather,
so their residuals are correlated (r ~ 0.28 for Zambian maize). The number of
independent realisations is therefore closer to the number of YEARS (tens) than
the number of district-years (thousands). A 200-tree forest against ~33
effective points fits idiosyncrasy and carries it into the held-out year as
noise, which shows up as a strongly negative anomaly R2 next to a healthy
absolute R2.

Ridge is the natural counterpart: its capacity is set by a penalty chosen on the
data rather than by the row count, and where there is no signal it shrinks
towards the mean instead of inventing one. Both linear models standardise first,
inside the pipeline, so the penalty is applied on comparable scales and the fold
statistics are learned on training rows only.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SUPPORTED_MODELS = ["rf", "gbm", "xgb", "ridge", "enet"]

# Spanning ten orders of magnitude: the useful penalty is tiny when the features
# genuinely carry the signal and enormous when they do not, and which regime a
# given AOI/feature-space lands in is exactly what these runs are measuring.
RIDGE_ALPHAS = np.logspace(-3, 7, 41)


class GroupedRidgeCV(RegressorMixin, BaseEstimator):
    """Ridge whose penalty is chosen by grouped CV, falling back to row-wise.

    Inherits BaseEstimator so Pipeline can introspect it: from scikit-learn 1.6
    `check_is_fitted` goes through `get_tags`, which needs `__sklearn_tags__`,
    and a plain class raises AttributeError at predict time rather than at fit.
    RegressorMixin goes first so BaseEstimator sits rightmost in the MRO, as
    sklearn requires.

    Why not sklearn's RidgeCV directly: its default generalised CV holds out one
    ROW at a time, which assumes rows are independent. Seasonal climate features
    are shared across the districts of a country in a given year, so holding out
    one district-year leaves 70 near-duplicates of it in the training set. The CV
    error therefore looks far better than it is at every alpha, and the selected
    penalty comes out much too small -- the same effective-sample-size mistake
    the forest makes, now inside the regularisation that was supposed to fix it.

    Measured on a synthetic panel with a target unrelated to the features,
    row-wise GCV picked alpha=178 and left coefficients up to 0.28; holding out
    whole years drives alpha to the top of the grid and the coefficients to zero,
    which is the honest answer.

    So when `groups` are supplied at fit time (the training rows' YEARS), alpha
    is chosen by holding out whole groups. Without groups it falls back to
    RidgeCV, which is correct for genuinely independent rows.
    """

    def __init__(self, alphas=RIDGE_ALPHAS, n_splits: int = 5):
        # Stored verbatim, not coerced: sklearn's get_params/clone contract is
        # that __init__ records its arguments untouched.
        self.alphas = alphas
        self.n_splits = n_splits

    def fit(self, X, y, groups=None):
        from sklearn.linear_model import Ridge
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        alphas = np.asarray(self.alphas, dtype=float)

        n_groups = 0 if groups is None else len(np.unique(groups))
        if groups is None or n_groups < 3:
            base = RidgeCV(alphas=alphas).fit(X, y)
            self.alpha_ = float(base.alpha_)
            self.n_groups_ = n_groups
        else:
            from sklearn.model_selection import GroupKFold
            splits = list(GroupKFold(n_splits=min(self.n_splits, n_groups)).split(X, y, groups))
            err = np.zeros(len(alphas))
            for tr, te in splits:
                for i, a in enumerate(alphas):
                    r = Ridge(alpha=a).fit(X[tr], y[tr])
                    err[i] += float(((y[te] - r.predict(X[te])) ** 2).sum())
            self.alpha_ = float(alphas[int(np.argmin(err))])
            self.n_groups_ = n_groups
            self.cv_curve_ = err / len(y)

        self._est = Ridge(alpha=self.alpha_).fit(X, y)
        self.coef_ = self._est.coef_
        self.intercept_ = self._est.intercept_
        return self

    def predict(self, X):
        return self._est.predict(np.asarray(X, dtype=float))

    def score(self, X, y):
        return self._est.score(np.asarray(X, dtype=float), np.asarray(y, dtype=float))


class YieldModel:
    def __init__(self, model_type: str, random_state: int = 42):
        if model_type not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                f"Choose from: {SUPPORTED_MODELS}"
            )
        self.model_type = model_type
        self._model = self._build(model_type, random_state)

    def _build(self, model_type, random_state):
        if model_type == "rf":
            return RandomForestRegressor(
                n_estimators=200,
                max_features="sqrt",
                min_samples_leaf=3,
                random_state=random_state,
                n_jobs=-1,
            )
        if model_type == "gbm":
            return GradientBoostingRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                min_samples_leaf=3,
                random_state=random_state,
            )
        if model_type == "xgb":
            try:
                from xgboost import XGBRegressor
            except ImportError as exc:
                raise ImportError(
                    "xgboost is not installed. Run: pip install xgboost"
                ) from exc
            return XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                n_jobs=-1,
                verbosity=0,
            )
        if model_type == "ridge":
            # Leave-one-out generalised CV over RIDGE_ALPHAS, refit on the whole
            # training fold. The alpha is chosen inside each outer LOO fold, so
            # the held-out year never informs the penalty.
            return Pipeline([
                ("scale", StandardScaler()),
                ("model", GroupedRidgeCV(alphas=RIDGE_ALPHAS)),
            ])
        if model_type == "enet":
            # l1_ratio grid rather than a fixed value: with correlated seasonal
            # aggregates, pure lasso arbitrarily keeps one of a colinear pair,
            # and the ratio that works is an empirical question per feature set.
            return Pipeline([
                ("scale", StandardScaler()),
                ("model", ElasticNetCV(
                    l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
                    n_alphas=50, cv=5, random_state=random_state,
                    max_iter=20000, n_jobs=-1)),
            ])

    def fit(self, X, y, groups=None):
        """Fit, passing `groups` to estimators that select hyperparameters by CV.

        `groups` should be the training rows' years. Only the linear CV models
        use it -- for the ensembles it is silently ignored, since their capacity
        is set by hyperparameters rather than chosen on the data.
        """
        if groups is not None and self._uses_groups():
            self._model.fit(X, y, model__groups=np.asarray(groups))
        else:
            self._model.fit(X, y)
        return self

    def _uses_groups(self) -> bool:
        if not isinstance(self._model, Pipeline):
            return False
        return isinstance(self._model.named_steps.get("model"), GroupedRidgeCV)

    def predict(self, X):
        return self._model.predict(X)

    def score(self, X, y):
        return self._model.score(X, y)

    @property
    def feature_importances_(self):
        """Tree importances, or |standardised coefficient| for the linear models.

        Not the same quantity -- one is impurity reduction, the other is effect
        size on standardised inputs -- but both answer "which columns is this
        model actually using", which is what the callers want. Absolute value
        because the caller ranks them; sign is available via .coef_.
        """
        imp = getattr(self._model, "feature_importances_", None)
        if imp is not None:
            return imp
        coef = self.coef_
        return None if coef is None else np.abs(coef)

    @property
    def coef_(self):
        """Signed standardised coefficients for the linear models, else None.

        The pipeline standardises first, so these are already on a common scale:
        yield units per one standard deviation of each feature.
        """
        est = self._model
        if isinstance(est, Pipeline):
            est = est.named_steps.get("model")
        return getattr(est, "coef_", None)

    @property
    def alpha_(self):
        """The penalty the CV actually selected, or None. Worth logging: an alpha
        pinned at the top of RIDGE_ALPHAS means the fold found no signal and
        shrank everything to the mean."""
        est = self._model
        if isinstance(est, Pipeline):
            est = est.named_steps.get("model")
        return getattr(est, "alpha_", None)
